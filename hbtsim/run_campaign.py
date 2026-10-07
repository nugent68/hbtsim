"""`hbtsim run` / `hbtsim-run`: execute a catalog campaign.

    hbtsim run g3_spica_vlt                       # a shipped campaign
    hbtsim run my_campaign.json --config-dir cfg  # your own
    hbtsim run suite_phase6 --jobs 4              # every member campaign

Each campaign names a runner (hbtsim/runners/<runner>.py: g3, g2,
g3_campaign, chromatic, montecarlo, scale, suite) and the catalog objects
it works on; the runner returns a dict of results.  Everything lands in
output/campaigns/<name>[_<suffix>]/: results.json (the numbers, the
resolved campaign definition, its content hash and the git revision),
table.md / table.tex when the runner makes tables, figures, and log.txt
with everything the runner printed.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

from .catalog import Campaign, Catalog
from .catalog.cli import add_config_dir_option
from .serialize import canonical_json, content_hash, to_json


@dataclass(frozen=True)
class RunOptions:
    out_dir: Path
    figures: bool = True
    track: bool = True
    latex: bool = True
    newera_dir: str | None = None        # None: the target's atmosphere resource
    no_newera: bool = False
    allow_extrapolation: bool | None = None
    fetch: bool | None = None            # None: $HBTSIM_AUTO_FETCH
    phase: float | None = None
    jobs: int = 1
    quiet: bool = False
    overrides: tuple = ()                # ((dotted.key, value), ...) applied to the spec


def _set(spec: dict, dotted: str, value) -> dict:
    """A copy of spec with spec[a][b][c] = value for "a.b.c"."""
    out = json.loads(json.dumps(spec))
    d = out
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value
    return out


def _parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_overrides(cat: Catalog, camp: Campaign, overrides) -> Campaign:
    """Re-build a campaign from its definition with --set key=value edits."""
    if not overrides:
        return camp
    spec = camp.spec
    for k, v in overrides:
        spec = _set(spec, k, v)
    from .catalog.build import build_campaign
    from .catalog.schema import check_kind
    check_kind(f"campaigns/{camp.name} (+ --set)", "campaign", spec)
    return build_campaign(cat, spec)


def attach_targets(cat: Catalog, camp: Campaign, opts: RunOptions, log=print) -> Campaign:
    """The campaign with model atmospheres on its targets (per the campaign's
    `atmosphere` block and the command-line overrides)."""
    atm = camp.spec.get("atmosphere") or {}
    use = atm.get("use", True) and not opts.no_newera
    if not use:
        log("atmosphere: blackbody + linear limb darkening")
        return camp
    names = ([camp.spec["target"]] if "target" in camp.spec else []) + list(camp.spec.get("targets", ()))
    allow = opts.allow_extrapolation if opts.allow_extrapolation is not None else atm.get("allow_extrapolation")
    targets = []
    for name, t in zip(names, camp.targets):
        newera_dir = opts.newera_dir
        if newera_dir is None and atm.get("resource"):
            newera_dir = cat.load_resource(atm["resource"]).resolved_path
        if atm.get("stand_in"):
            # an explicit stand-in model for every target (red-clump giants)
            ptr = dict(cat.registry.raw("target", name).get("atmosphere") or {})
            ptr["model"] = atm["stand_in"]
            cat_ptr = cat.registry._raw.get(("target", name))
            t2, report = _attach_with_pointer(cat, t, name, ptr, newera_dir, allow)
        else:
            t2, report = cat.attach_atmosphere(t, name=name, newera_dir=newera_dir, allow_extrapolation=allow,
                                               fetch=opts.fetch)
        log(f"atmosphere [{name}]: {report}")
        targets.append(t2)
    return replace(camp, targets=tuple(targets))


def _attach_with_pointer(cat, target, name, ptr, newera_dir, allow):
    """attach_atmosphere with a pointer given explicitly (stand-in models)."""
    from . import sed
    if newera_dir is None:
        res = cat.load_resource(ptr["resource"])
        newera_dir = res.resolved_path
    rng = ptr.get("range", "380-1000nm_0.02nm")
    path = os.path.join(newera_dir, f"newera_{ptr['model']}_{rng}.npz")
    if not os.path.exists(path):
        return target, f"stand-in table {path} not found: blackbody + linear law"
    ft, ld = sed.load_star_tables(path)
    return replace(target, star=sed.with_tables(target.star, ft, ld)), f"stand-in {ptr['model']} ({rng})"


def git_revision() -> str:
    """The git revision of a source checkout, else the installed version."""
    root = Path(__file__).resolve().parents[1]
    if (root / ".git").exists():
        try:
            return subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:          # noqa: BLE001
            pass
    from . import __version__
    return f"hbtsim {__version__}"


class _Tee:
    """Write to several streams (stdout and the log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            if not st.closed:
                st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            if not st.closed:
                st.flush()

    def isatty(self):
        return False


def run(cat: Catalog, camp: Campaign, opts: RunOptions, suffix: str = "") -> dict:
    """Run one campaign; returns the results dict (also written to disk)."""
    import importlib
    from . import runners
    if camp.runner not in runners.RUNNERS:
        raise SystemExit(f"runner {camp.runner!r} is not implemented; have {sorted(runners.RUNNERS)}")
    name = camp.name + (f"_{suffix}" if suffix else "")
    out = opts.out_dir / name
    out.mkdir(parents=True, exist_ok=True)
    mod = importlib.import_module(f"hbtsim.runners.{runners.RUNNERS[camp.runner]}")
    t0 = time.time()
    with open(out / "log.txt", "w") as logf:
        tee = _Tee(sys.stdout, logf) if not opts.quiet else logf
        with contextlib.redirect_stdout(tee):
            print(f"# hbtsim run {camp.name} ({camp.runner}) -- {_dt.datetime.now():%Y-%m-%d %H:%M}, "
                  f"git {git_revision()}")
            camp_run = camp if camp.runner == "suite" else attach_targets(cat, camp, opts)
            results = mod.run(camp_run, cat, opts, out) or {}
    results = dict(results)
    results.setdefault("campaign", camp.name)
    results["runner"] = camp.runner
    results["suffix"] = suffix
    results["campaign_definition"] = camp.spec
    results["campaign_hash"] = content_hash(camp.spec, mode="key")
    results["git"] = git_revision()
    results["elapsed_s"] = round(time.time() - t0, 1)
    results["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
    with open(out / "results.json", "w") as fh:
        json.dump(to_json(results, mode="key"), fh, indent=2, allow_nan=False, default=str)
    if not opts.quiet:
        print(f"-> {out}/results.json")
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="hbtsim run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("campaign", help="catalog campaign name or path to a campaign .json")
    add_config_dir_option(p)
    p.add_argument("--out", default="output/campaigns", help="output root (default output/campaigns)")
    p.add_argument("--suffix", default="", help="appended to the output directory name")
    p.add_argument("--newera-dir", default=None, help="NewEra tables (default: the campaign's resource)")
    p.add_argument("--no-newera", action="store_true", help="blackbody + linear limb darkening")
    p.add_argument("--allow-extrapolation", action="store_true")
    p.add_argument("--fetch", action="store_true",
                   help="download missing NewEra tables first (hbtsim data; also HBTSIM_AUTO_FETCH=1)")
    p.add_argument("--phase", type=float, default=None, help="orbital phase of the snapshot (runner g3)")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-track", action="store_true")
    p.add_argument("--no-latex", action="store_true")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a campaign field, dotted (e.g. night.block_minutes=30, "
                        "options.layout_scan.sides_m=[8,12])")
    p.add_argument("--jobs", type=int, default=None, help="parallel members of a suite")
    p.add_argument("--quiet", action="store_true", help="log to file only")
    args = p.parse_args(argv)

    cat = Catalog(paths=args.config_dir)
    camp = cat.load_campaign(args.campaign)
    overrides = []
    for item in args.set:
        k, _, v = item.partition("=")
        overrides.append((k, _parse_value(v)))
    camp = apply_overrides(cat, camp, overrides)
    opts = RunOptions(out_dir=Path(args.out), figures=not args.no_figures, track=not args.no_track,
                      latex=not args.no_latex, newera_dir=args.newera_dir, no_newera=args.no_newera,
                      allow_extrapolation=True if args.allow_extrapolation else None,
                      fetch=True if args.fetch else None, phase=args.phase,
                      jobs=args.jobs or camp.spec.get("jobs", 1), quiet=args.quiet,
                      overrides=tuple(overrides))
    run(cat, camp, opts, suffix=args.suffix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
