"""Runner `g3_campaign`: a multi-night closure-phase campaign of a binary
with an equilateral triangle of identical units (formerly
scripts/deltavel_eonsii.py: delta Vel with three EON-SII 4 m units at
Paranal).  Prints

  (1) diagnostics: rho(phase), the conjunctions, the eclipse windows,
      the hour-angle window at the array's site (and at a comparison
      site), and the fraction of the orbit (uniform in phase) over which
      4 m and 8.2 m pupils keep > 50 % fringe contrast at 400 nm;
  (2) a layout scan of the triangle side (one-hour transit snapshots,
      the scan backend, uniform phases plus the eclipse windows);
  (3) the campaign's backends on the same geometry;
  (4) a campaign of n_uniform nights at uniform phases plus every
      eclipse-window night (auto block length, analytic outside and
      rendered inside the eclipse guard, per-epoch grids), cached per
      night so it resumes;
  (5) per-night table, campaign totals, nights to precision, figures,
      table rows, and a blackbody maximum-separation snapshot.

Statistics (hbtsim.snr3): "amplitude" detects the model's cos phi_c
pattern (a template fit); "phase" is the Fisher information on a
closure-phase shift, sqrt(sum (snr sin phi_c)^2), i.e. the part that
carries the image asymmetry.  g3 measures |T| cos phi_c only, so near
phi_c = 0 or pi the phase information is second order.

Campaign knobs (all under `options`): layout_scan.sides_m, layout_scan.backend
(a catalog backend name, the label of one of the campaign's backends, or
an inline backend object), layout_scan.scan_phases, side_m (fixed side;
null: from the scan), n_uniform, precision_targets.{phase_rad, amplitude},
compare_site, diagnostics_spectrograph; `night.{block_minutes, vis_method,
render_grid, min_alt_deg}` pass through to campaign_g3_snr.
"""

from __future__ import annotations

import math
import time
import warnings
from pathlib import Path

import numpy as np

from ..aperture import fringe_smearing_factor
from ..catalog.build import BUILDERS
from ..catalog.schema import check_kind
from ..geometry import hour_angle_window
from ..orbit import max_separation_phase, sky_positions
from ..params import MAS, in_eclipse_rho
from ..snr3 import (campaign_g3_snr, campaign_nights_to_precision,
                    geometry_samples, spectral_g3_snr)
from . import write_tables

TARGETS = (("phase", 0.1, "rad on a closure-phase shift"),
           ("amplitude", 0.1, "on cos phi_c (template)"))


def _finite(x):
    """A float for JSON, None where the number is not finite (inf nights)."""
    x = float(x)
    return x if math.isfinite(x) else None


def _runs(mask):
    """(start, end) index pairs of True runs in a boolean array (no wrap)."""
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def _backend(cat, campaign, value):
    """The scan backend: the label of one of the campaign's backends, a
    catalog backend name, or an inline backend object."""
    if value is None:
        return campaign.backends[0]
    if isinstance(value, dict):
        check_kind(f"campaigns/{campaign.name}: options.layout_scan.backend", "backend", value)
        return BUILDERS["backend"](cat, value)
    for b in campaign.backends:
        if b.name == value:
            return b
    return cat.load_backend(value)


def triangle_array(cat, array_name, side_m, detector=None):
    """The catalog's equilateral triangle at the given side (and detector)."""
    kw = {} if detector is None else {"detector": detector}
    return cat.load_array(array_name, side_m=float(side_m), **kw)


def diagnostics(system, site, compare_site=None):
    ph = np.linspace(0.0, 1.0, 20001)[:-1]
    rho = np.asarray(sky_positions(2 * np.pi * ph, system).rho)
    print(f"\n(1) Diagnostics: {system.name}, P = {system.period_days} d, e = {system.eccentricity}")
    print(f"  rho = {rho.min():.2f}-{rho.max():.2f} mas (max at phase {max_separation_phase(system):.3f}); "
          f"disks {2 * system.drawn_radius_mas(system.primary):.2f} / "
          f"{2 * system.drawn_radius_mas(system.secondary):.2f} mas")
    minima = [k for k in range(ph.size) if rho[k] < rho[k - 1] and rho[k] <= rho[(k + 1) % ph.size]]
    for k in minima:
        print(f"  conjunction at phase {ph[k]:.3f}: rho = {rho[k]:.2f} mas")
    ecl = in_eclipse_rho(system, rho)
    windows = [(float(ph[a]), float(ph[b])) for a, b in _runs(ecl)]
    for a, b in windows:
        print(f"  eclipse-guard window {a:.4f}-{b:.4f} ({(b - a) * system.period_days:.2f} d)")
    print(f"  in the guard: {100 * ecl.mean():.1f} % of the orbit")
    h = hour_angle_window(system.dec_deg, site.latitude_deg)
    line = f"  above 30 deg: {site.name} H = {h[0]:+.2f}..{h[1]:+.2f} h"
    if compare_site is not None:
        ht = hour_angle_window(system.dec_deg, compare_site.latitude_deg)
        line += f"; {compare_site.name} {ht}"
    print(line)
    for d in (4.0, 8.2):
        f = np.array([fringe_smearing_factor(d, d, r * MAS, 400e-9) for r in rho[::10]])
        f_max_sep = fringe_smearing_factor(d, d, rho.max() * MAS, 400e-9)
        print(f"  {d:.1f} m pupils, 400 nm: fringe contrast > 0.5 for {100 * np.mean(f > 0.5):.0f} % "
              f"of the orbit (uniform in phase); {f_max_sep:.3f} at maximum separation")
    return windows


def layout_scan(system, sides, phases, scan_backend, cat, array_name):
    """One-hour transit snapshots per side and phase (scan_backend).
    Returns (best side, per-side rows)."""
    print(f"\n(2) Layout scan: equilateral side, {scan_backend.name}, 1 h at transit, "
          f"{len(phases)} phases (uniform + eclipse windows); SNR per sqrt(h), rms over phases")
    print(f"  {'side':>6s} {'phase rms':>10s} {'phase max':>10s} {'ampl rms':>10s}")
    best = None
    rows = []
    for side in sides:
        arr = triangle_array(cat, array_name, side, detector=scan_backend.detector)
        (tri,) = arr.projected(0.0, system.dec_deg).triangles()
        sp, sa = [], []
        for ph in phases:
            r = spectral_g3_snr(system, tri, spectrograph=scan_backend.spectrograph,
                                orbital_phase=ph, vis_method="auto", grid="epoch",
                                polarization_mode=scan_backend.polarization_mode)
            sp.append(r.snr_phase)
            sa.append(r.snr_amplitude)
        sp, sa = np.array(sp), np.array(sa)
        rms_p = float(np.sqrt(np.mean(sp**2)))
        rms_a = float(np.sqrt(np.mean(sa**2)))
        k_max = int(np.argmax(sp))
        print(f"  {side:5.0f}m {rms_p:10.3g} {sp.max():10.3g} {rms_a:10.3g}  "
              f"(phase max at {phases[k_max]:.3f})")
        rows.append({"side_m": float(side), "phase_rms": rms_p, "phase_max": float(sp.max()),
                     "phase_max_at": float(phases[k_max]), "amplitude_rms": rms_a})
        if best is None or rms_p > best[1]:
            best = (float(side), rms_p)
    print(f"  -> best side for the phase statistic: {best[0]:.0f} m")
    return best[0], rows


def eclipse_nights(windows, period_days):
    """Night centres covering each eclipse window at <= one night's phase step."""
    step = 1.0 / period_days
    out = []
    for a, b in windows:
        n = int(np.ceil((b - a) / step)) + 1
        out.extend(np.linspace(a, b, n))
    return out


def campaign(system, arr, backends, side, n_uniform, windows, cache_dir, **night_kw):
    uniform = list(np.arange(n_uniform) / n_uniform)
    ecl = eclipse_nights(windows, system.period_days)
    phases = uniform + ecl
    site = arr.site.name if arr.site is not None else "?"
    print(f"\n(4) Campaign: {side:.0f} m triangle at {site}, {len(uniform)} uniform nights + "
          f"{len(ecl)} eclipse-window nights (these fall in different orbital cycles), "
          f"auto blocks and method, cache {cache_dir}")
    t0 = time.time()

    def progress(k, n):
        print(f"    night {k + 1:2d}/{len(phases)} phase {n.phase_mid:.4f} rho {n.rho_mas:5.2f} mas: "
              f"{n.n_blocks} x {n.block_minutes:.0f} min ({n.n_render_blocks} rendered); "
              + "; ".join(f"{b.name.split(',')[-1].strip()}: ph {n.snr[b.name]['phase']:.3g} "
                          f"amp {n.snr[b.name]['amplitude']:.3g}" for b in backends)
              + f"  [{time.time() - t0:.0f} s]", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = campaign_g3_snr(system, arr, backends, phases_mid=phases, cache_dir=cache_dir,
                              progress=progress, **night_kw)
    return res, len(uniform)


def report(res, backends, n_uniform, targets, out_dir, latex):
    print(f"\n(5) Campaign totals ({res.n_nights} nights; SNR adds in quadrature over nights)")
    rows = []
    summary = {}
    for b in backends:
        tot = {st: res.snr(b.name, st) for st in ("phase", "amplitude", "total")}
        uni = {st: float(np.sqrt(np.sum(res.per_night(b.name, st)[:n_uniform] ** 2)))
               for st in ("phase", "amplitude")}
        ecl = {st: float(np.sqrt(np.sum(res.per_night(b.name, st)[n_uniform:] ** 2)))
               for st in ("phase", "amplitude")}
        need = {st: campaign_nights_to_precision(res, b.name, tgt, st) for st, tgt, _ in targets}
        print(f"  {b.name:32s} phase {tot['phase']:.3g} (uniform {uni['phase']:.3g}, eclipse "
              f"{ecl['phase']:.3g}); amplitude {tot['amplitude']:.3g} (uniform {uni['amplitude']:.3g}, "
              f"eclipse {ecl['amplitude']:.3g}); nights to 0.1: phase {need['phase']:.3g}, "
              f"amplitude {need['amplitude']:.3g}")
        rows.append((b.name, tot, need))
        summary[b.name] = {
            "template_snr": tot["amplitude"], "phase_snr": tot["phase"], "total_snr": tot["total"],
            "nights_amplitude": _finite(need["amplitude"]), "nights_phase": _finite(need["phase"]),
            "amplitude_uniform": uni["amplitude"], "amplitude_eclipse": ecl["amplitude"],
            "phase_uniform": uni["phase"], "phase_eclipse": ecl["phase"],
            "per_night_phase": [float(x) for x in res.per_night(b.name, "phase")],
            "per_night_amplitude": [float(x) for x in res.per_night(b.name, "amplitude")],
        }
    for st, tgt, what in targets:
        print(f"  ('{st}' target: {tgt} {what}; nights = campaign nights x (1/target/SNR)^2)")
    table = [[name, f"{tot['phase']:.3g}", f"{tot['amplitude']:.3g}",
              f"{need['phase']:.3g}", f"{need['amplitude']:.3g}"] for name, tot, need in rows]
    write_tables(out_dir, table, ["backend", "SNR phase", "SNR amplitude", "nights phase",
                                  "nights amplitude"], latex=latex)
    if latex:
        print("\n  LaTeX rows (backend & SNR phase & SNR amplitude & nights phase & nights amplitude):")
        for name, tot, need in rows:
            print(f"  {name} & {tot['phase']:.3g} & {tot['amplitude']:.3g} & "
                  f"{need['phase']:.3g} & {need['amplitude']:.3g} \\\\")
    return summary


def figures(res, system, side, backends, tri, spec, site_name, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ph = np.array([n.phase_mid for n in res.nights])
    order = np.argsort(ph)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    for b in backends:
        a1.semilogy(ph[order], res.per_night(b.name, "phase")[order], "o-", ms=3, label=b.name)
        a2.semilogy(ph[order], res.per_night(b.name, "amplitude")[order], "o-", ms=3)
    a1.set_ylabel("SNR per night, phase")
    a2.set_ylabel("SNR per night, amplitude")
    a2.set_xlabel("Orbital phase (night centre)")
    a1.legend(fontsize=7)
    a1.set_title(f"{system.name}: three units, {side:.0f} m triangle, {site_name}")
    for a in (a1, a2):
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    f1 = out_dir / "g3_campaign_per_night.png"
    fig.savefig(f1, dpi=140)
    plt.close(fig)
    print(f"  wrote {f1}")

    phases = np.linspace(0.0, 1.0, 121)
    cmap = np.array([geometry_samples(system, tri, spec, p, "auto", "epoch").ts.cos_phi_c
                     for p in phases])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.pcolormesh(spec.channel_centers_nm, phases, cmap, cmap="RdBu_r", vmin=-1, vmax=1,
                       shading="auto")
    fig.colorbar(im, label=r"$\cos\varphi_c$")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Orbital phase")
    ax.set_title(f"{system.name}: {side:.0f} m triangle at transit (aperture-averaged)")
    fig.tight_layout()
    f2 = out_dir / "g3_cosphi_campaign.png"
    fig.savefig(f2, dpi=140)
    plt.close(fig)
    print(f"  wrote {f2}")


def run(campaign_def, cat, opts, out_dir: Path) -> dict:
    camp = campaign_def
    system = camp.target
    backends = tuple(camp.backends)
    array_name = camp.spec["array"]
    target_name = camp.spec["target"] if "target" in camp.spec else camp.spec["targets"][0]
    opt = camp.option
    sides = list(opt("options.layout_scan.sides_m", [8, 12, 20, 30, 45, 60, 90, 120]))
    scan_backend = _backend(cat, camp, opt("options.layout_scan.backend"))
    n_scan = int(opt("options.layout_scan.scan_phases", 16))
    n_uniform = int(opt("options.n_uniform", 32))
    side = opt("options.side_m")
    pt = opt("options.precision_targets") or {}
    targets = tuple((st, float(pt.get(key, tgt)), what)
                    for (st, tgt, what), key in zip(TARGETS, ("phase_rad", "amplitude")))
    cache_dir = camp.spec.get("cache_dir")
    compare = opt("options.compare_site")
    compare_site = None if compare is None else cat.load_site(compare)
    night_kw = {k: opt(f"night.{k}") for k in ("block_minutes", "vis_method", "render_grid", "min_alt_deg")
                if opt(f"night.{k}") is not None}
    site = camp.array.site
    warnings.filterwarnings("ignore", message=".*extrapolated.*")
    warnings.filterwarnings("ignore", message=".*dead-time.*")

    windows = diagnostics(system, site, compare_site)
    scan_ph = list(np.arange(n_scan) / n_scan) + [0.5 * (a + b) for a, b in windows]
    scan_rows = []
    if side is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            side, scan_rows = layout_scan(system, sides, scan_ph, scan_backend, cat, array_name)
    side = float(side)
    print("\n(3) Backends: " + "; ".join(b.name for b in backends) + " (1 GHz time-tag links)")
    arr = triangle_array(cat, array_name, side)
    res, n_uni = campaign(system, arr, backends, side, n_uniform, windows, cache_dir, **night_kw)
    summary = report(res, backends, n_uni, targets, out_dir, opts.latex)

    # blackbody comparison: one-hour transit snapshot at maximum separation
    blackbody = cat.load_target(target_name)
    ph_max = max_separation_phase(blackbody)
    print(f"\n  1 h transit snapshot at maximum separation (phase {ph_max:.3f}), blackbody vs NewEra:")
    bb_vs = {}
    for b in backends:
        (tri_b,) = triangle_array(cat, array_name, side, detector=b.detector).projected(
            0.0, blackbody.dec_deg).triangles()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = [spectral_g3_snr(s, tri_b, spectrograph=b.spectrograph, orbital_phase=ph_max,
                                   polarization_mode=b.polarization_mode)
                   for s in (blackbody, system)]
        print(f"    {b.name:32s} amplitude {out[0].snr_amplitude:.3g} -> {out[1].snr_amplitude:.3g}; "
              f"phase {out[0].snr_phase:.3g} -> {out[1].snr_phase:.3g}")
        bb_vs[b.name] = {"amplitude": [float(out[0].snr_amplitude), float(out[1].snr_amplitude)],
                         "phase": [float(out[0].snr_phase), float(out[1].snr_phase)]}
    if opts.figures:
        (tri,) = arr.projected(0.0, system.dec_deg).triangles()
        spec = cat.load_spectrograph(opt("options.diagnostics_spectrograph", "eonsii_60ch"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # the D/P > 0.7 quadrature notices
            figures(res, system, side, backends, tri, spec,
                    site.name if site is not None else "?", out_dir)
    return {
        "side_m": side,
        "layout_scan": {"backend": scan_backend.name, "sides_m": [float(s) for s in sides],
                        "scan_phases": [float(p) for p in scan_ph], "sides": scan_rows},
        "windows": [[a, b] for a, b in windows],
        "n_nights": res.n_nights,
        "campaign": summary,
        "n_uniform": n_uni,
        "phases_mid": [float(n.phase_mid) for n in res.nights],
        "blackbody_vs_newera": {"phase": float(ph_max), "backends": bb_vs},
    }
