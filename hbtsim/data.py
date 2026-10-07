"""`hbtsim data`: where the NewEra model tables live and how to fetch them.

The binned NewEra (PHOENIX) tables that `hbtsim.sed` reads are not part
of the package: they are hosted at NERSC
(https://portal.nersc.gov/project/newera/binned/, the `remote` block of
hbtsim/configs/resources/*.json) and cached on the user's machine.

    hbtsim data path                       # where tables are looked for
    hbtsim data fetch --target betaaur     # the grid corners this binary needs
    hbtsim data fetch --model lte04800-4.50-0.0 --resource newera_redclump
    hbtsim data fetch --all                # every 380-1000 nm table
    hbtsim data list                       # what the cache holds
    hbtsim data manifest DIR --write       # build manifest.json for a hosted directory

Resolution order of a resource's directory (hbtsim.catalog Resource):
its env override ($HBTSIM_NEWERA_DIR, ...), its `path` if that exists
relative to the current directory (a repository checkout with data/),
then `<data_dir>/<resource name>` where data_dir() is $HBTSIM_DATA_DIR
or the platform cache directory (~/Library/Caches/hbtsim on macOS,
$XDG_CACHE_HOME/hbtsim or ~/.cache/hbtsim on Linux, %LOCALAPPDATA%/hbtsim/Cache
on Windows).  $HBTSIM_DATA_URL overrides every resource's base URL;
HBTSIM_AUTO_FETCH=1 (or --fetch on the tools) lets attach_atmosphere pull
missing tables on demand.  Downloads use only the standard library
(urllib; file:// URLs work for local mirrors and tests) and are verified
against the manifest's sha256.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ENV_DATA_DIR = "HBTSIM_DATA_DIR"
ENV_DATA_URL = "HBTSIM_DATA_URL"
ENV_AUTO_FETCH = "HBTSIM_AUTO_FETCH"
DEFAULT_RANGE = "380-1000nm_0.02nm"
MANIFEST_VERSION = 1

# newera_lte09200-4.00-0.0_380-1000nm_0.02nm.npz
TABLE_RE = re.compile(r"^newera_(lte(\d{5})-(\d\.\d\d)([-+]\d\.\d))_(\d+)-(\d+)nm_([0-9.]+)nm\.npz$")


class DataError(Exception):
    pass


# ---------------------------------------------------------------------------
# locations
# ---------------------------------------------------------------------------
def data_dir() -> Path:
    """$HBTSIM_DATA_DIR or the platform cache directory for hbtsim."""
    env = os.environ.get(ENV_DATA_DIR)
    if env:
        return Path(env).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "hbtsim"
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "hbtsim" / "Cache"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "hbtsim"


def auto_fetch_enabled() -> bool:
    return os.environ.get(ENV_AUTO_FETCH, "").strip().lower() in ("1", "true", "yes", "on")


def parse_table_name(name: str) -> dict | None:
    """{model, teff, logg, z, lo_nm, hi_nm, step_nm, range} for a binned
    table file name, None for anything else."""
    m = TABLE_RE.match(os.path.basename(name))
    if not m:
        return None
    lo, hi, step = int(m.group(5)), int(m.group(6)), m.group(7)
    return dict(model=m.group(1), teff=float(m.group(2)), logg=float(m.group(3)), z=float(m.group(4)),
                lo_nm=lo, hi_nm=hi, step_nm=float(step), range=f"{lo}-{hi}nm_{step}nm")


def table_name(model: str, range_token: str = DEFAULT_RANGE) -> str:
    return f"newera_{model}_{range_token}.npz"


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(directory, resource_of=None, progress=None) -> dict:
    """The manifest of every binned table in `directory`.  resource_of(name)
    -> resource name is informational (default "newera")."""
    directory = Path(directory)
    files = []
    for p in sorted(directory.glob("newera_lte*.npz")):
        info = parse_table_name(p.name)
        if info is None:
            continue
        if progress:
            progress(f"  hashing {p.name}")
        files.append({"name": p.name, "size": p.stat().st_size, "sha256": sha256_file(p),
                      "resource": resource_of(p.name) if resource_of else "newera", **info})
    return {"schema_version": MANIFEST_VERSION,
            "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "files": files}


def _join_url(base: str, name: str) -> str:
    if not base.endswith("/"):
        base += "/"
    return urllib.parse.urljoin(base, name)


def base_url_for(resource, override: str | None = None) -> str:
    """Explicit override, $HBTSIM_DATA_URL, else the resource's remote base URL."""
    url = override or os.environ.get(ENV_DATA_URL) or resource.remote_dict.get("base_url")
    if not url:
        raise DataError(f"resource {resource.name!r} has no remote base URL "
                        f"(set {ENV_DATA_URL} or the file's `remote.base_url`)")
    return url


def load_manifest(base_url: str, manifest_name: str = "manifest.json", timeout: float = 30.0) -> dict:
    url = _join_url(base_url, manifest_name)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            d = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise DataError(f"cannot read the table manifest at {url}: {e}") from None
    if d.get("schema_version") != MANIFEST_VERSION or "files" not in d:
        raise DataError(f"{url} is not an hbtsim table manifest (schema_version {MANIFEST_VERSION})")
    return d


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------
def _present(path: Path, entry: dict, verify: bool) -> bool:
    if not path.exists() or path.stat().st_size != entry["size"]:
        return False
    return (not verify) or sha256_file(path) == entry["sha256"]


def _download(url: str, dest: Path, expect_sha256: str, progress=None, timeout: float = 60.0) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(part, "wb") as out:
            for block in iter(lambda: resp.read(1 << 20), b""):
                out.write(block)
                h.update(block)
    except (urllib.error.URLError, OSError) as e:
        part.unlink(missing_ok=True)
        raise DataError(f"download of {url} failed: {e}") from None
    if h.hexdigest() != expect_sha256:
        part.unlink(missing_ok=True)
        raise DataError(f"{dest.name}: sha256 mismatch after download (corrupt transfer or stale manifest)")
    os.replace(part, dest)


def fetch_entries(entries, base_url: str, dest, progress=print, verify_existing: bool = True) -> list:
    """Download the manifest entries missing from dest; returns the local paths."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for e in entries:
        path = dest / e["name"]
        if _present(path, e, verify_existing):
            if progress:
                progress(f"  have   {e['name']}")
        else:
            if progress:
                progress(f"  fetch  {e['name']} ({e['size'] / 1e6:.1f} MB)")
            _download(_join_url(base_url, e["name"]), path, e["sha256"], progress=progress)
        out.append(path)
    return out


def select_entries(manifest: dict, *, models=None, ranges=(DEFAULT_RANGE,), all_models: bool = False) -> list:
    """Manifest entries for the given models (or all) in the given ranges
    (None: every range)."""
    want = None if (all_models or not models) else set(models)
    if want is None and not all_models:
        return []
    out = [e for e in manifest["files"]
           if (want is None or e["model"] in want) and (ranges is None or e["range"] in ranges)]
    if want is not None:
        missing = sorted(want - {e["model"] for e in out})
        if missing:
            raise DataError(f"model(s) {missing} not in the manifest for range(s) {list(ranges) if ranges else 'any'}")
    return out


def corners_for(teff: float, logg: float, z: float, entries, allow_extrapolation: bool = False,
                max_clamp_teff: float = 1000.0, max_clamp_logg: float = 0.5) -> list:
    """The grid models NewEraGrid.interpolate would use for (teff, logg) given
    the models available in `entries` (bracketing T_eff and log g pairs,
    clamped to the edge when allow_extrapolation and within the clamp
    limits); [] when the grid cannot serve the star."""
    keys = sorted({(e["teff"], e["logg"]) for e in entries if e["z"] == z})
    if not keys:
        return []
    teffs = sorted({k[0] for k in keys})
    loggs = sorted({k[1] for k in keys})
    t_use = min(max(teff, teffs[0]), teffs[-1])
    g_use = min(max(logg, loggs[0]), loggs[-1])
    if (t_use != teff or g_use != logg):
        if not allow_extrapolation or abs(teff - t_use) > max_clamp_teff or abs(logg - g_use) > max_clamp_logg:
            return []

    def bracket(vals, x):
        lo = max(v for v in vals if v <= x)
        hi = min(v for v in vals if v >= x)
        return lo, hi
    t0, t1 = bracket(teffs, t_use)
    g0, g1 = bracket(loggs, g_use)
    wt = 0.0 if t1 == t0 else (t_use - t0) / (t1 - t0)
    wg = 0.0 if g1 == g0 else (g_use - g0) / (g1 - g0)
    corners = []
    for t, w_t in ((t0, 1.0 - wt), (t1, wt)):
        for g, w_g in ((g0, 1.0 - wg), (g1, wg)):
            if w_t * w_g > 0.0 and (t, g) in keys:
                corners.append((t, g))
    if len({c for c in corners}) < len([1 for t, w_t in ((t0, 1 - wt), (t1, wt))
                                        for g, w_g in ((g0, 1 - wg), (g1, wg)) if w_t * w_g > 0]):
        return []           # a needed corner is not on the grid
    by_key = {}
    for e in entries:
        by_key.setdefault((e["teff"], e["logg"], e["z"]), e["model"])
    return sorted({by_key[(t, g, z)] for t, g in corners})


def fetch(resource, *, models=None, ranges=(DEFAULT_RANGE,), all_models: bool = False,
          base_url: str | None = None, dest=None, progress=print) -> list:
    """Fetch the given models (or all) of a catalog Resource into dest
    (default: the resource's cache directory); returns the local paths."""
    url = base_url_for(resource, base_url)
    manifest = load_manifest(url, resource.remote_dict.get("manifest", "manifest.json"))
    entries = select_entries(manifest, models=models, ranges=ranges, all_models=all_models)
    if progress:
        progress(f"{resource.name}: {len(entries)} table(s), {sum(e['size'] for e in entries) / 1e6:.0f} MB from {url}")
    dest = Path(dest) if dest is not None else data_dir() / resource.name
    return fetch_entries(entries, url, dest, progress=progress)


def fetch_for_target(cat, name: str, *, base_url: str | None = None, dest=None,
                     allow_extrapolation: bool | None = None, progress=print) -> list:
    """Fetch what `name`'s atmosphere pointer needs: the explicit stand-in
    model, or the grid corners bracketing each star's (T_eff, log g)."""
    ptr = dict(cat.registry.raw("target", name).get("atmosphere") or {})
    if not ptr:
        if progress:
            progress(f"{name}: no atmosphere pointer, nothing to fetch")
        return []
    res = cat.load_resource(ptr["resource"])
    rng = ptr.get("range", res.remote_dict.get("default_range", DEFAULT_RANGE))
    if ptr.get("model"):
        return fetch(res, models=[ptr["model"]], ranges=(rng,), base_url=base_url, dest=dest, progress=progress)
    url = base_url_for(res, base_url)
    manifest = load_manifest(url, res.remote_dict.get("manifest", "manifest.json"))
    entries = [e for e in manifest["files"] if e["range"] == rng]
    target = cat.load_target(name)
    stars = ([target.primary, target.secondary] if hasattr(target, "primary")
             else [target.star] if hasattr(target, "star") else [])
    allow = bool(ptr.get("allow_extrapolation", False)) if allow_extrapolation is None else allow_extrapolation
    models = set()
    for s in stars:
        c = corners_for(s.teff, s.log_g, s.metallicity, entries, allow)
        if progress:
            progress(f"  {s.name}: T_eff {s.teff:.0f} K, log g {s.log_g:.2f} -> "
                     + (", ".join(c) if c else "not covered by the grid"))
        models.update(c)
    if not models:
        return []
    chosen = [e for e in entries if e["model"] in models]
    if progress:
        progress(f"{res.name}: {len(chosen)} table(s), {sum(e['size'] for e in chosen) / 1e6:.0f} MB from {url}")
    return fetch_entries(chosen, url, Path(dest) if dest is not None else data_dir() / res.name, progress=progress)


def inventory(directory) -> list:
    """Parsed entries of the tables present in a directory."""
    d = Path(directory)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("newera_lte*.npz")):
        info = parse_table_name(p.name)
        if info:
            out.append({"name": p.name, "size": p.stat().st_size, "path": str(p), **info})
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    from .catalog import Catalog
    from .catalog.cli import add_config_dir_option
    p = argparse.ArgumentParser(prog="hbtsim data", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_dir_option(p)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fetch", help="download tables into the cache")
    s.add_argument("--target", action="append", default=[], help="catalog target(s): fetch the corners they need")
    s.add_argument("--model", action="append", default=[], help="model name(s), e.g. lte09200-4.00-0.0")
    s.add_argument("--resource", default="newera", help="resource for --model / --all (default newera)")
    s.add_argument("--all", action="store_true", help="every table of the resource in --range")
    s.add_argument("--range", default=DEFAULT_RANGE, help=f"wavelength-range token (default {DEFAULT_RANGE}; 'any')")
    s.add_argument("--base-url", default=None, help=f"override the remote base URL (also ${ENV_DATA_URL})")
    s.add_argument("--dest", default=None, help="download directory (default: the resource's cache directory)")
    s.add_argument("--allow-extrapolation", action="store_true", help="for --target: also fetch the edge model")
    s = sub.add_parser("list", help="tables present locally, per resource")
    s = sub.add_parser("path", help="the directories searched for each resource")
    s = sub.add_parser("manifest", help="build manifest.json for a directory of tables")
    s.add_argument("directory")
    s.add_argument("--write", action="store_true", help="write <directory>/manifest.json (default: print)")
    args = p.parse_args(argv)
    cat = Catalog(paths=args.config_dir)

    if args.cmd == "path":
        print(f"data_dir: {data_dir()}  (${ENV_DATA_DIR} overrides)")
        for n in cat.list_names("resource"):
            r = cat.load_resource(n)
            found = r.resolved_path if r.exists() else None
            print(f"{n}:")
            for c in r.candidates():
                print(f"  {'*' if c == found else ' '} {c}")
            print(f"    remote: {r.remote_dict.get('base_url', '-')}")
        return 0
    if args.cmd == "list":
        for n in cat.list_names("resource"):
            r = cat.load_resource(n)
            inv = inventory(r.resolved_path)
            print(f"{n}: {len(inv)} table(s), {sum(e['size'] for e in inv) / 1e6:.0f} MB in {r.resolved_path}")
            for e in inv:
                print(f"  {e['name']}  ({e['teff']:.0f} K, log g {e['logg']:.2f}, [M/H] {e['z']:+.1f}, {e['range']})")
        return 0
    if args.cmd == "manifest":
        m = build_manifest(args.directory, progress=print if args.write else None)
        text = json.dumps(m, indent=1)
        if args.write:
            out = Path(args.directory) / "manifest.json"
            out.write_text(text + "\n")
            print(f"wrote {out}: {len(m['files'])} table(s)")
        else:
            print(text)
        return 0
    # fetch
    ranges = None if args.range == "any" else (args.range,)
    got = []
    try:
        for t in args.target:
            got += fetch_for_target(cat, t, base_url=args.base_url, dest=args.dest,
                                    allow_extrapolation=True if args.allow_extrapolation else None)
        if args.model or args.all:
            got += fetch(cat.load_resource(args.resource), models=args.model or None, ranges=ranges,
                         all_models=args.all, base_url=args.base_url, dest=args.dest)
    except DataError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not (args.target or args.model or args.all):
        p.error("give --target, --model or --all")
    print(f"{len(got)} table(s) in place")
    return 0


if __name__ == "__main__":
    sys.exit(main())
