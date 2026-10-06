"""delta Vel closure phases with three EON-SII 4 m units at Paranal / CTAO-South.

delta Vel (A2 IV + A4 V, P = 45.15 d, a = 16.6 mas at 25.1 pc, disks
1.10 / 0.93 mas) never rises usefully at Teide (dec -54.7), so the
EON-SII pair plus a third identical unit is placed at Paranal on an
equilateral triangle (the catalog's eonsii_triangle_paranal array).  The script prints

  (1) diagnostics: rho(phase), the conjunctions, the eclipse windows,
      the Paranal hour-angle window, and the fraction of the orbit
      (uniform in phase) over which 4 m and 8.2 m pupils keep > 50 %
      fringe contrast at 400 nm;
  (2) a layout scan of the triangle side (one-hour transit snapshots,
      SPAD + PBS backend, uniform phases plus the eclipse windows);
  (3) four backends on the same geometry (1000 ch MCP-PMT, 1000 ch SPAD,
      SPAD + PBS, R = 7500 SPAD; 1 GHz time-tag links);
  (4) a campaign of n_uniform nights at uniform phases plus every
      eclipse-window night (auto block length, analytic outside and
      rendered inside the eclipse guard, per-epoch grids), cached per
      night so it resumes;
  (5) per-night table, campaign totals, nights to precision, figures,
      LaTeX rows, and a blackbody maximum-separation snapshot.

Statistics (hbtsim.snr3): "amplitude" detects the model's cos phi_c
pattern (a template fit); "phase" is the Fisher information on a
closure-phase shift, sqrt(sum (snr sin phi_c)^2), i.e. the part that
carries the image asymmetry.  g3 measures |T| cos phi_c only, so near
phi_c = 0 or pi the phase information is second order.

Orbit caveat: the conjunction phases depend on the adopted omega
convention; the campaign samples both conjunctions, so its totals do
not depend on which is which.

    .venv/bin/python scripts/deltavel_eonsii.py [--side 20] [--scan-only] [--table]
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np

from hbtsim.aperture import fringe_smearing_factor
from hbtsim.catalog import Catalog
from hbtsim.geometry import hour_angle_window
from hbtsim.orbit import max_separation_phase, positions_at, sky_positions
from hbtsim.params import DAY, MAS, in_eclipse_rho
from hbtsim.sed import attach_from_cli
from hbtsim.snr3 import (campaign_g3_snr, campaign_nights_to_precision,
                         geometry_samples, spectral_g3_snr)

CAT = Catalog(env=False)
DELTA_VEL = CAT.load_target("deltavel")
PARANAL = CAT.load_site("paranal")
TEIDE = CAT.load_site("teide")
BACKENDS = tuple(CAT.load_backend(n) for n in
                 ("eonsii_mcp", "eonsii_spad", "eonsii_spad_pbs", "eonsii_r7500_spad"))
SCAN_BACKEND = BACKENDS[2]


def eonsii_triangle(side_m, detector=None):
    """Three EON-SII 4 m units on an equilateral triangle at Paranal."""
    kw = {} if detector is None else {"detector": detector}
    return CAT.load_array("eonsii_triangle_paranal", side_m=float(side_m), **kw)
TARGETS = (("phase", 0.1, "rad on a closure-phase shift"),
           ("amplitude", 0.1, "on cos phi_c (template)"))


def _runs(mask):
    """(start, end) index pairs of True runs in a boolean array (no wrap)."""
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def diagnostics(system):
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
    windows = [(ph[a], ph[b]) for a, b in _runs(ecl)]
    for a, b in windows:
        print(f"  eclipse-guard window {a:.4f}-{b:.4f} ({(b - a) * system.period_days:.2f} d)")
    print(f"  in the guard: {100 * ecl.mean():.1f} % of the orbit")
    h = hour_angle_window(system.dec_deg, PARANAL.latitude_deg)
    ht = hour_angle_window(system.dec_deg, TEIDE.latitude_deg)
    print(f"  above 30 deg: Paranal H = {h[0]:+.2f}..{h[1]:+.2f} h; Teide {ht}")
    for d in (4.0, 8.2):
        f = np.array([fringe_smearing_factor(d, d, r * MAS, 400e-9) for r in rho[::10]])
        f_max_sep = fringe_smearing_factor(d, d, rho.max() * MAS, 400e-9)
        print(f"  {d:.1f} m pupils, 400 nm: fringe contrast > 0.5 for {100 * np.mean(f > 0.5):.0f} % "
              f"of the orbit (uniform in phase); {f_max_sep:.3f} at maximum separation")
    return windows


def layout_scan(system, sides, phases):
    """One-hour transit snapshots per side and phase (SCAN_BACKEND)."""
    print(f"\n(2) Layout scan: equilateral side, {SCAN_BACKEND.name}, 1 h at transit, "
          f"{len(phases)} phases (uniform + eclipse windows); SNR per sqrt(h), rms over phases")
    print(f"  {'side':>6s} {'phase rms':>10s} {'phase max':>10s} {'ampl rms':>10s}")
    best = None
    for side in sides:
        arr = eonsii_triangle(side, detector=SCAN_BACKEND.detector)
        (tri,) = arr.projected(0.0, system.dec_deg).triangles()
        sp, sa = [], []
        for ph in phases:
            r = spectral_g3_snr(system, tri, spectrograph=SCAN_BACKEND.spectrograph,
                                orbital_phase=ph, vis_method="auto", grid="epoch",
                                polarization_mode=SCAN_BACKEND.polarization_mode)
            sp.append(r.snr_phase)
            sa.append(r.snr_amplitude)
        sp, sa = np.array(sp), np.array(sa)
        rms_p = float(np.sqrt(np.mean(sp**2)))
        print(f"  {side:5.0f}m {rms_p:10.3g} {sp.max():10.3g} {np.sqrt(np.mean(sa**2)):10.3g}  "
              f"(phase max at {phases[int(np.argmax(sp))]:.3f})")
        if best is None or rms_p > best[1]:
            best = (side, rms_p)
    print(f"  -> best side for the phase statistic: {best[0]:.0f} m")
    return best[0]


def eclipse_nights(windows, period_days):
    """Night centres covering each eclipse window at <= one night's phase step."""
    step = 1.0 / period_days
    out = []
    for a, b in windows:
        n = int(np.ceil((b - a) / step)) + 1
        out.extend(np.linspace(a, b, n))
    return out


def campaign(system, side, n_uniform, windows, cache_dir):
    arr = eonsii_triangle(side)
    uniform = list(np.arange(n_uniform) / n_uniform)
    ecl = eclipse_nights(windows, system.period_days)
    phases = uniform + ecl
    print(f"\n(4) Campaign: {side:.0f} m triangle at Paranal, {len(uniform)} uniform nights + "
          f"{len(ecl)} eclipse-window nights (these fall in different orbital cycles), "
          f"auto blocks and method, cache {cache_dir}")
    t0 = time.time()

    def progress(k, n):
        print(f"    night {k + 1:2d}/{len(phases)} phase {n.phase_mid:.4f} rho {n.rho_mas:5.2f} mas: "
              f"{n.n_blocks} x {n.block_minutes:.0f} min ({n.n_render_blocks} rendered); "
              + "; ".join(f"{b.name.split(',')[-1].strip()}: ph {n.snr[b.name]['phase']:.3g} "
                          f"amp {n.snr[b.name]['amplitude']:.3g}" for b in BACKENDS)
              + f"  [{time.time() - t0:.0f} s]", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = campaign_g3_snr(system, arr, BACKENDS, phases_mid=phases, cache_dir=cache_dir,
                              progress=progress)
    return res, len(uniform)


def report(res, n_uniform, latex):
    print(f"\n(5) Campaign totals ({res.n_nights} nights; SNR adds in quadrature over nights)")
    rows = []
    for b in BACKENDS:
        tot = {st: res.snr(b.name, st) for st in ("phase", "amplitude", "total")}
        uni = {st: float(np.sqrt(np.sum(res.per_night(b.name, st)[:n_uniform] ** 2)))
               for st in ("phase", "amplitude")}
        ecl = {st: float(np.sqrt(np.sum(res.per_night(b.name, st)[n_uniform:] ** 2)))
               for st in ("phase", "amplitude")}
        need = {st: campaign_nights_to_precision(res, b.name, tgt, st) for st, tgt, _ in TARGETS}
        print(f"  {b.name:32s} phase {tot['phase']:.3g} (uniform {uni['phase']:.3g}, eclipse "
              f"{ecl['phase']:.3g}); amplitude {tot['amplitude']:.3g} (uniform {uni['amplitude']:.3g}, "
              f"eclipse {ecl['amplitude']:.3g}); nights to 0.1: phase {need['phase']:.3g}, "
              f"amplitude {need['amplitude']:.3g}")
        rows.append((b.name, tot, need))
    for st, tgt, what in TARGETS:
        print(f"  ('{st}' target: {tgt} {what}; nights = campaign nights x (1/target/SNR)^2)")
    if latex:
        print("\n  LaTeX rows (backend & SNR phase & SNR amplitude & nights phase & nights amplitude):")
        for name, tot, need in rows:
            print(f"  {name} & {tot['phase']:.3g} & {tot['amplitude']:.3g} & "
                  f"{need['phase']:.3g} & {need['amplitude']:.3g} \\\\")


def figures(res, system, side, out_dir="output"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ph = np.array([n.phase_mid for n in res.nights])
    order = np.argsort(ph)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    for b in BACKENDS:
        a1.semilogy(ph[order], res.per_night(b.name, "phase")[order], "o-", ms=3, label=b.name)
        a2.semilogy(ph[order], res.per_night(b.name, "amplitude")[order], "o-", ms=3)
    a1.set_ylabel("SNR per night, phase")
    a2.set_ylabel("SNR per night, amplitude")
    a2.set_xlabel("Orbital phase (night centre)")
    a1.legend(fontsize=7)
    a1.set_title(f"{system.name}: three EON-SII 4 m units, {side:.0f} m triangle, Paranal")
    for a in (a1, a2):
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    f1 = os.path.join(out_dir, "g3_deltavel_eonsii_per_night.png")
    fig.savefig(f1, dpi=140)
    plt.close(fig)
    print(f"  wrote {f1}")

    (tri,) = eonsii_triangle(side).projected(0.0, system.dec_deg).triangles()
    spec = CAT.load_spectrograph("eonsii_60ch")
    phases = np.linspace(0.0, 1.0, 121)
    cmap = np.array([geometry_samples(system, tri, spec, p, "auto", "epoch").ts.cos_phi_c
                     for p in phases])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.pcolormesh(spec.channel_centers_nm, phases, cmap, cmap="RdBu_r", vmin=-1, vmax=1,
                       shading="auto")
    fig.colorbar(im, label=r"$\cos\varphi_c$")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Orbital phase")
    ax.set_title(f"{system.name}: EON-SII {side:.0f} m triangle at transit (aperture-averaged)")
    fig.tight_layout()
    f2 = os.path.join(out_dir, "g3_cosphi_deltavel_eonsii.png")
    fig.savefig(f2, dpi=140)
    plt.close(fig)
    print(f"  wrote {f2}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", type=float, default=None, help="triangle side [m] (default: from the scan)")
    ap.add_argument("--sides", type=float, nargs="+", default=[8, 12, 20, 30, 45, 60, 90, 120])
    ap.add_argument("--n-uniform", type=int, default=32)
    ap.add_argument("--scan-phases", type=int, default=16)
    ap.add_argument("--scan-only", action="store_true")
    ap.add_argument("--cache-dir", default="output/cache/deltavel_eonsii")
    ap.add_argument("--newera-dir", default="data/newera")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message=".*extrapolated.*")
    warnings.filterwarnings("ignore", message=".*dead-time.*")

    system = attach_from_cli(DELTA_VEL, args.newera_dir)
    windows = diagnostics(system)
    scan_ph = list(np.arange(args.scan_phases) / args.scan_phases) + \
        [0.5 * (a + b) for a, b in windows]
    side = args.side
    if side is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            side = layout_scan(system, args.sides, scan_ph)
    if args.scan_only:
        return
    print("\n(3) Backends: " + "; ".join(b.name for b in BACKENDS) + " (1 GHz time-tag links)")
    res, n_uni = campaign(system, side, args.n_uniform, windows, args.cache_dir)
    report(res, n_uni, args.table)

    # blackbody comparison: one-hour transit snapshot at maximum separation
    ph_max = max_separation_phase(DELTA_VEL)
    (tri,) = eonsii_triangle(side).projected(0.0, DELTA_VEL.dec_deg).triangles()
    print(f"\n  1 h transit snapshot at maximum separation (phase {ph_max:.3f}), blackbody vs NewEra:")
    for b in BACKENDS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = [spectral_g3_snr(s, tri.__class__(tuple(st.__class__(**{**st.__dict__, "detector": b.detector})
                                                          for st in tri.stations), tri.site),
                                   spectrograph=b.spectrograph, orbital_phase=ph_max,
                                   polarization_mode=b.polarization_mode)
                   for s in (DELTA_VEL, system)]
        print(f"    {b.name:32s} amplitude {out[0].snr_amplitude:.3g} -> {out[1].snr_amplitude:.3g}; "
              f"phase {out[0].snr_phase:.3g} -> {out[1].snr_phase:.3g}")
    if not args.no_figures:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # the D/P > 0.7 quadrature notices
            figures(res, system, side)


if __name__ == "__main__":
    main()
