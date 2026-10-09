#!/usr/bin/env python
"""Render the parameter tables of README.md and docs/*.md from the catalog.

Each table sits between `<!-- catalog:NAME -->` and `<!-- /catalog -->`
markers; `--write` replaces the blocks in place, `--check` (the default,
run by tests/test_docs_tables.py) exits 1 when a block is stale.  The
numbers in prose stay hand-written; the tables never drift from
hbtsim/configs again.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hbtsim.catalog import Catalog  # noqa: E402

cat = Catalog(env=False)
BINARIES = ("betaaur", "algol", "spica", "deltavel")


def _fmt(x, nd=2):
    return f"{x:.{nd}f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)


def _summary(name, key):
    return cat.raw("target", name).get("summary", {}).get(key, "")


def systems_table() -> str:
    rows = []
    hdr = ["", *[f"**{cat.raw('target', n)['summary'].get('short', n)}** (`{n}`)" for n in BINARIES]]
    rows.append("| " + " | ".join(hdr) + " |")
    rows.append("|" + "---|" * len(hdr))
    T = {n: cat.load_target(n) for n in BINARIES}

    def row(label, fn):
        rows.append(f"| {label} | " + " | ".join(fn(n, T[n]) for n in BINARIES) + " |")
    row("Components", lambda n, s: _summary(n, "components"))
    row("Masses", lambda n, s: f"{_fmt(s.primary.mass_msun, 3)} / {_fmt(s.secondary.mass_msun, 3)} M☉")
    row("Radii", lambda n, s: f"{_fmt(s.primary.radius_rsun, 3)} / {_fmt(s.secondary.radius_rsun, 3)} R☉")
    row("T_eff", lambda n, s: f"{s.primary.teff:.0f} / {s.secondary.teff:.0f} K")
    row("Period", lambda n, s: f"{s.period_days!r} d" + (f", e = {s.eccentricity:g}" if s.eccentricity else ""))
    row("Inclination", lambda n, s: f"{s.inclination_deg!r}°")
    row("Distance", lambda n, s: f"{s.distance_pc:g} pc" + (" (orbital parallax)" if n == "deltavel" else ""))
    row("Angular semi-major axis", lambda n, s: f"{s.angular_semimajor_mas:.2f} mas")
    row("Angular diameters", lambda n, s: f"{2 * s.angular_radius_mas(s.primary):.2f} / "
                                          f"{2 * s.angular_radius_mas(s.secondary):.2f} mas")
    row("Ω (ascending node)", lambda n, s: f"{s.node_pa_deg:g}°")
    row("Eclipses", lambda n, s: _summary(n, "eclipses"))
    row("Anchored photometry", lambda n, s: ", ".join(f"{'g' if lam == 477.0 else 'i' if lam == 763.0 else f'{lam:g} nm'} {m:.2f}"
                                                     for lam, m in s.mag_anchors)
        + (f" ({_summary(n, 'anchor_note')})" if _summary(n, "anchor_note") else ""))
    row("NewEra tables", lambda n, s: _summary(n, "newera"))
    row("Sources", lambda n, s: _summary(n, "sources"))
    return "\n".join(rows)


def hardware_list() -> str:
    out = ["Telescopes, detectors, spectrographs, sites and arrays are JSON files in",
           "`hbtsim/configs/` (`hbtsim catalog list`); the shipped ones:", ""]
    tel = []
    for n in cat.list_names("telescope"):
        t = cat.load_telescope(n)
        area = f", {t.collecting_area_m2:g} m²" if t.collecting_area_m2 is not None else ""
        tel.append(f"`{n}` ({t.diameter_m:g} m{area}, throughput {t.throughput:g})")
    out.append("- **Telescopes**: " + "; ".join(tel) + ".")
    det = []
    for n in cat.list_names("detector"):
        d = cat.load_detector(n)
        ceiling = (f", link ≤ {d.max_total_cps:.2g} cps" if d.max_total_cps else "") + \
                  (f", ≤ {d.max_cps_per_pixel:.2g} cps/pixel" if d.max_cps_per_pixel else "")
        det.append(f"`{n}` ({d.jitter_fwhm_ps:.0f} ps FWHM, {d.dead_time_ns:g} ns dead, "
                   f"{d.dark_cps_per_pixel:g} cps dark, {d.readout}{ceiling})")
    out.append("- **Detectors**: " + "; ".join(det) + ".")
    sp = []
    for n in cat.list_names("spectrograph"):
        s = cat.load_spectrograph(n)
        res = f"R = {s.resolving_power:g}" if s.resolving_power else f"{s.channel_width_nm:.2f} nm channels"
        sp.append(f"`{n}` ({s.lambda_min_nm:.0f}–{s.lambda_max_nm:.0f} nm, {s.n_channels} ch, {res}, "
                  f"throughput {s.throughput:g})")
    out.append("- **Spectrographs**: " + "; ".join(sp) + ".")
    arr = []
    for n in cat.list_names("array"):
        d = cat.raw("array", n)
        a = cat.load_array(n)
        lengths = sorted(float(np.hypot(*b)) for _, _, b in a.pairs())
        site = f" at {a.site.name}" if a.site else ""
        gen = " (generator: `load_array(name, side_m=…)`)" if "generator" in d else ""
        span = f"{lengths[0]:.0f} m" if len(lengths) == 1 else f"{lengths[0]:.0f}–{lengths[-1]:.0f} m"
        arr.append(f"`{n}` ({len(a.stations)} stations, {span}{site}){gen}")
    out.append("- **Arrays**: " + "; ".join(arr) + ".")
    out.append("- **Sites**: " + "; ".join(f"`{n}` ({cat.load_site(n).name})" for n in cat.list_names("site")) + ".")
    return "\n".join(out)


def maunakea_baselines() -> str:
    tri = cat.load_triangle("maunakea_subaru_keck")
    names = [s.name for s in tri.stations]
    pairs = [(names[0], names[1]), (names[1], names[2]), (names[2], names[0])]
    rows = ["| Baseline | Length |", "|---|---|"]
    for (a, b), L in zip(pairs, tri.baseline_lengths()):
        rows.append(f"| {a} – {b} | {L:.1f} m |")
    return "\n".join(rows)


def iact_parameters() -> str:
    ver, mag = cat.load_array("veritas"), cat.load_array("magic_lst1")
    vt, mt, lt = (cat.load_telescope(n) for n in ("veritas_12m", "magic_17m", "lst1_23m"))
    vb, mb = cat.load_backend("veritas_sii"), cat.load_backend("magic_sii")
    from hbtsim.iact import calibrated
    q_cal = calibrated(vb, vt.area_m2).q
    asm = {n: set(cat.raw("telescope", n).get("assumptions", ())) for n in ("veritas_12m", "magic_17m", "lst1_23m")}

    def site(s):
        return f"{s.name} ({s.latitude_deg:g}°, {s.longitude_deg:g}°, {s.elevation_m:g} m)"

    def area(t, n):
        return f"{t.collecting_area_m2:g} m²" + (" (**assumed**)" if "collecting_area_m2" in asm[n] else "")
    vl = sorted(round(float(np.hypot(*b)), 1) for _, _, b in ver.pairs())
    ml = sorted(round(float(np.hypot(*b)), 1) for _, _, b in mag.pairs())
    rows = [
        ("", "VERITAS", "MAGIC", "LST-1"),
        ("dishes", f"{len(ver.stations)} × {vt.diameter_m:g} m, {area(vt, 'veritas_12m')}",
         f"2 × {mt.diameter_m:g} m, {area(mt, 'magic_17m')}", f"{lt.diameter_m:g} m, {area(lt, 'lst1_23m')}"),
        ("site", site(ver.site), site(mag.site), "same"),
        ("baselines", ", ".join(f"{x:g}" for x in vl) + " m (fitted to the published values)",
         f"MAGIC-I–II {ml[0]:g} m", f"{ml[1]:g}, {ml[2]:g} m to LST-1; **orientation assumed**"),
        ("filter", f"{vb.lambda_nm:g} nm, {vb.dlambda_nm:g} nm effective",
         f"{mb.lambda_nm:g} nm / {mb.dlambda_nm:g} nm", "as MAGIC (assumed)"),
        ("QE α", f"{vb.alpha:g}", f"{mb.alpha:g}", "as MAGIC"),
        ("optical q", f"**calibrated: {q_cal:.3f}** (file: {vb.q:g})", f"{mb.q:g}", "as MAGIC"),
        ("b_el", f"{vb.b_el_hz / 1e6:g} MHz; {vb.time_resolution_ns:g} ns time resolution",
         f"{mb.b_el_hz / 1e6:g} MHz effective; {mb.time_resolution_ns:g} ns", "as MAGIC"),
        ("F, σ_spec", "absorbed into q" if vb.anchor else f"{vb.noise_factor:g}, {vb.sigma_spec:g}",
         f"{mb.noise_factor:g}, {mb.sigma_spec:g}", "as MAGIC"),
        ("precision anchor", f"{vb.anchor.star}: σ(|V|²) = {vb.anchor.sigma_vis2:g} per pair in "
                             f"{vb.anchor.t_s / 3600:g} h at AB {vb.anchor.mag_ab:.2f}" if vb.anchor else "—", "—", "—"),
    ]
    out = ["| " + " | ".join(rows[0]) + " |", "|---|---|---|---|"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def lpqi_parameters() -> str:
    pf, orm = cat.load_array("lpqi_pathfinder"), cat.load_array("lpqi_orm")
    d, n, m = (cat.load_detector(k) for k in ("lpqi_spad64_i2cass", "lpqi_spad64_i2cass_nextgen", "lpqi_spad1_mpd"))
    (_, _, b), = pf.pairs()
    pa = np.degrees(np.arctan2(b[0], b[1])) % 360
    lengths = {f"{orm.stations[i].name}–{orm.stations[j].name}": float(np.hypot(*bb)) for i, j, bb in orm.pairs()}
    filters = [cat.load_spectrograph(nm) for nm in cat.list_names("spectrograph") if nm.startswith("filter_lpqi_")]
    rows = [("", "value", "source / status")]
    rows += [("site", f"{pf.site.name}: {pf.site.latitude_deg:g}°, {pf.site.longitude_deg:g}°, {pf.site.elevation_m:g} m", "Wikipedia (NOT infobox)"),
             ("Pathfinder pair", f"{pf.stations[0].name} {pf.stations[0].telescope.diameter_m:g} m + {pf.stations[1].name} "
                                 f"{pf.stations[1].telescope.diameter_m:g} m, B = {np.hypot(*b):.0f} m at PA {pa:.0f}°",
              "surveyed positions (NOT GPS; TNG geodetic survey, ~5 m); the project quotes 550 m"),
             ("five-telescope network", ", ".join(f"{s.name} {s.telescope.diameter_m:g} m" for s in orm.stations)
              + f"; baselines {min(lengths.values()):.0f}–{max(lengths.values()):.0f} m", "NOT, WHT, INT GPS pages; TNG survey; GTC web page (arc-second)"),
             ("telescope throughput", ", ".join(f"{s.telescope.throughput:g}" for s in pf.stations), "hbtsim default (**assumed**)"),
             ("detector (final LPQI: lensed IMSE-LPQI 64×64 + R = 10 000 spectrograph)",
              f"PDE {n.pde(550.0):.2f} at 400–550 nm, {n.pde(650.0):.2f} at 650 nm, {n.pde(950.0):.3f} at 950 nm (microlenses); "
              f"{n.jitter_fwhm_ps:.0f} ps FWHM, {n.n_pixels} pixels per channel, dead {n.dead_time_ns:g} ns, "
              f"dark {n.dark_cps_per_pixel:g} cps/pixel, ≤ {n.max_cps_per_pixel:.1e} cps per pixel",
              "F. Prada 2026-10-07 / 2026-10-09; PDE curve beyond 650 nm **assumed**"),
             ("detector (Pathfinder primary: MPD single pixel)", f"PDE {m.pde(550.0):.2f} flat 400–700 nm, {m.jitter_fwhm_ps:.0f} ps FWHM (275 ps below 470 nm), "
                                                   f"dead {m.dead_time_ns:g} ns, dark {m.dark_cps_per_pixel:g} cps, 1 pixel, max detection rate {m.max_total_cps:.0e} cps",
              "F. Prada 2026-10-07 and 2026-10-09 (12e6 cps, primary detector); flat PDE **assumed**"),
             ("bare IMSE array (reference only)", f"PDE {d.pde(550.0):.3f} = fill factor 3.5 % × PDP 75 %, otherwise as the lensed array with {d.n_pixels} pixels",
              "Quintana et al. 2026, Sensors 26, 5757; not a Pathfinder option"),
             ("filters", "; ".join(f"{f.channel_centers_nm[0]:.1f} nm ({f.channel_width_nm:.1f} nm)" for f in filters)
              + f"; throughput {filters[0].throughput:g}; one per night", "wavelengths **to be confirmed**; width and one-per-night from the user")]
    out = ["| " + " | ".join(rows[0]) + " |", "|---|---|---|"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def campaigns_table() -> str:
    rows = ["| Campaign | Runner | What it computes |", "|---|---|---|"]
    for n in cat.list_names("campaign"):
        d = cat.raw("campaign", n)
        rows.append(f"| `{n}` | {d['runner']} | {d.get('label', '')} |")
    return "\n".join(rows)


BLOCKS = {
    "README.md": {"systems": systems_table, "hardware": hardware_list, "campaigns": campaigns_table},
    "docs/three_telescope_feasibility.md": {"maunakea_baselines": maunakea_baselines},
    "docs/iact_targets.md": {"iact_parameters": iact_parameters},
    "docs/lpqi_pathfinder.md": {"lpqi_parameters": lpqi_parameters},
}
MARK = re.compile(r"(<!-- catalog:(\w+) -->\n)(.*?)(\n<!-- /catalog -->)", re.S)


def render(path: Path, blocks: dict) -> tuple:
    text = path.read_text()
    missing = set(blocks)

    def sub(m):
        name = m.group(2)
        if name not in blocks:
            raise SystemExit(f"{path}: no renderer for block {name!r}")
        missing.discard(name)
        return m.group(1) + blocks[name]() + m.group(4)
    new = MARK.sub(sub, text)
    if missing:
        raise SystemExit(f"{path}: marker(s) {sorted(missing)} not found")
    return text, new


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="rewrite the blocks (default: check only)")
    args = ap.parse_args(argv)
    stale = []
    for rel, blocks in BLOCKS.items():
        path = ROOT / rel
        old, new = render(path, blocks)
        if old != new:
            stale.append(rel)
            if args.write:
                path.write_text(new)
    if args.write:
        print("updated: " + (", ".join(stale) if stale else "nothing"))
        return 0
    if stale:
        print("stale catalog tables in: " + ", ".join(stale) + "  (run scripts/catalog_tables.py --write)")
        return 1
    print("catalog tables up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
