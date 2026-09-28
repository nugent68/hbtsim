"""Feasibility of three-telescope closure-phase intensity interferometry.

Generates the figures and every number quoted in
docs/three_telescope_feasibility.md and the paper's Table 3:

  (a) per-pair |gamma|(lambda) (point and aperture-averaged) and the
      per-triangle bispectrum amplitudes at quadrature
  (b) the cos(phi_c)(lambda, orbital phase) map on one triangle
  (c) snapshot sensitivities at quadrature for several backends, the
      time to reach Delta cos phi_c <= 0.1 for the three statistics
      (global template amplitude; R = 100 binned closure phases; a
      single channel), and one realistic night along the uv track
  (d) the two-telescope g2 numbers of the paper's Section 3

    python scripts/feasibility_g3.py --system spica --array vlt --table
    python scripts/feasibility_g3.py --system deltavel --array vlt
    python scripts/feasibility_g3.py --system algol --array maunakea
    python scripts/feasibility_g3.py --g2

Instrument model (hbtsim.snr): telescope throughput 0.3 x spectrograph
0.5 x PDE; aperture smearing on; sky orientation from each system's
Omega; the current SPAD Lambda is time-tag limited (1e8 cps) while the
next-generation detector has a correlator readout.
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np

from hbtsim.aperture import fringe_smearing_factor
from hbtsim.bispectrum import (MAUNAKEA_SUBARU_KECK, VLT_UT, Array, Triangle,
                               binary_vis_complex_analytic, spectral_triple)
from hbtsim.geometry import hour_angle_window
from hbtsim.orbit import max_separation_phase, positions_at, sky_positions
from hbtsim.params import MAS, SYSTEMS, GridConfig
from hbtsim.sed import attach_from_cli
from hbtsim.snr import (C2PU, EON_SII_TELESCOPE, EONSII_MCP_PMT, EONSII_SPAD,
                        EONSII_SPECTROGRAPH, EONSII_SPECTROGRAPH_R7500, KECK,
                        SPAD_LAMBDA, SPAD_LAMBDA_NG, Observation, Spectrograph,
                        g2_snr, spectral_g2_snr, system_ab_mag)
from hbtsim.snr3 import (array_g3_snr, nights_to_precision, spectral_g3_snr,
                         time_to_precision, track_g3_snr)

NIGHT_H = 8.0
ARRAYS = {"vlt": VLT_UT, "maunakea": MAUNAKEA_SUBARU_KECK}

# backends: (label, spectrograph, detector, polarization)
SPEC_320 = Spectrograph(n_channels=320)
SPEC_R5000 = Spectrograph.from_resolving_power(5000.0)
BACKENDS = [
    ("current SPAD Lambda, 320 ch, time-tag", SPEC_320, SPAD_LAMBDA, "unpolarized"),
    ("current SPAD Lambda, 320 ch, correlator", SPEC_320, SPAD_LAMBDA_NG, "unpolarized"),
    ("next-gen R = 5000, correlator", SPEC_R5000, SPAD_LAMBDA_NG, "unpolarized"),
    ("next-gen R = 5000, correlator + PBS", SPEC_R5000, SPAD_LAMBDA_NG, "pbs"),
]


def _with_detector(arr, det):
    from dataclasses import replace
    st = tuple(replace(s, detector=det) for s in arr.stations)
    return (Array(st, arr.site) if isinstance(arr, Array) else Triangle(st, arr.site))


def _fmt_time(hours: float) -> str:
    if not np.isfinite(hours):
        return "inf"
    if hours < 1.0:
        return f"{hours * 60:.1f} min"
    if hours < 3 * NIGHT_H:
        return f"{hours:.2f} h"
    return f"{hours / NIGHT_H:.3g} nights"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_gammas(system, arr, out, phase):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nm = np.linspace(400.0, 950.0, 200)
    pos = positions_at(system, phase)
    tris = arr.triangles() if isinstance(arr, Array) else [arr]
    pairs = arr.pairs() if isinstance(arr, Array) else [
        (i, j, b) for (i, j), b in zip(((0, 1), (1, 2), (2, 0)), arr.baseline_vectors())]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)
    bv = np.array([b for _, _, b in pairs])
    gam = binary_vis_complex_analytic(bv, nm, system, pos)
    from hbtsim.aperture import pupil_pair_quadrature
    for k, (i, j, b) in enumerate(pairs):
        d1, d2 = arr.stations[i].telescope.diameter_m, arr.stations[j].telescope.diameter_m
        q = pupil_pair_quadrature(d1, d2)
        pts = q.points(b[None, :]).reshape(-1, 2)
        sm = np.sqrt(q.reduce(np.abs(binary_vis_complex_analytic(pts, nm, system, pos)) ** 2))
        line, = ax1.plot(nm, np.abs(gam[:, k]), lw=1.0, alpha=0.5)
        ax1.plot(nm, sm, color=line.get_color(), lw=1.8,
                 label=f"{arr.stations[i].name}-{arr.stations[j].name} ({np.hypot(*b):.0f} m)")
    ax1.set_ylabel(r"$|\gamma_{ij}|$  (thin: point; bold: aperture-averaged)")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)
    ax1.set_title(f"{system.name}, orbital phase {phase:.2f}")
    for tri in tris:
        ts = spectral_triple(pos, tri, nm, system, pupils=True)
        ax2.plot(nm, ts.triple_amp, label=tri.name)
    ax2.set_xlabel("Wavelength [nm]")
    ax2.set_ylabel(r"$|\langle\gamma_{12}\gamma_{23}\gamma_{31}\rangle|$ (3-pupil average)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_cosphi_map(system, tri, out, n_phase=51):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nm = np.linspace(400.0, 950.0, 120)
    phases = np.linspace(0.0, 1.0, n_phase)
    cosmap = np.full((phases.size, nm.size), np.nan)
    for k, ph in enumerate(phases):
        pos = positions_at(system, ph)
        try:
            ts = spectral_triple(pos, tri, nm, system, pupils=True)
        except ValueError:
            ts = spectral_triple(pos, tri, nm, system, GridConfig().fit_orbit(system), method="render",
                                 pupils=True)
        cosmap[k] = ts.cos_phi_c
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.pcolormesh(nm, phases, cosmap, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
    fig.colorbar(im, label=r"$\cos\varphi_c$")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Orbital phase")
    ax.set_title(f"{system.name}: closure-phase cosine on {tri.name} (aperture-averaged)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def snapshot_rows(system, arr, phase, latex=False):
    """Per backend: snapshot sensitivities at one orbital phase and the
    time to Delta cos phi_c <= 0.1 for the three statistics."""
    rows = []
    for label, spec, det, pol in BACKENDS:
        a = _with_detector(arr, det)
        is_arr = isinstance(a, Array)
        kw = dict(spectrograph=spec, polarization_mode=pol, orbital_phase=phase)
        res = (array_g3_snr(system, a, **kw) if is_arr
               else spectral_g3_snr(system, a, **kw))
        per = res.per_triangle if is_arr else [res]
        tkw = dict(array=a) if is_arr else dict(triangle=a)
        t = {s: time_to_precision(system, 0.1, statistic=s, R_bin=100.0, **tkw, **kw)
             / 3600.0 for s in ("amplitude", "binned", "channel")}
        rates = per[0].total_rate_cps[0]
        row = dict(label=label, n_ch=spec.n_channels, snr_amp=res.snr_amplitude,
                   snr_total=res.snr_total, t_amp=t["amplitude"], t_bin=t["binned"],
                   t_ch=t["channel"], rate=rates, limited=per[0].readout_limited,
                   scale=per[0].readout_scale, load=per[0].dead_time_load_max,
                   ridge=float(np.median(per[0].ridge_ratio)),
                   triple_max=max(float(p.triple_amp.max()) for p in per))
        rows.append(row)
        print(f"  {label:42s} SNR3_amp/sqrt(h) = {row['snr_amp']:8.3g}  "
              f"[total {row['snr_total']:8.3g}]  d<=0.1: amplitude {_fmt_time(t['amplitude'])}, "
              f"binned R=100 {_fmt_time(t['binned'])}, channel {_fmt_time(t['channel'])}; "
              f"rate {rates:.2e} cps/tel{' READOUT-LIMITED x%.1e' % row['scale'] if row['limited'] else ''}, "
              f"dead-time load {row['load']:.2f}, ridge ratio ~{row['ridge']:.0f}, "
              f"max |triple| {row['triple_max']:.2f}")
    if latex:
        print("\n  LaTeX rows (target & backend & channels & SNR3/sqrt(h) & "
              "amplitude & binned R=100 & channel):")
        for r in rows:
            print(f"  {system.name.split()[0]} & {r['label']} & {r['n_ch']} & "
                  f"{r['snr_amp']:.3g} & {_fmt_time(r['t_amp'])} & "
                  f"{_fmt_time(r['t_bin'])} & {_fmt_time(r['t_ch'])} \\\\")
    return rows


def track_rows(system, arr, phase0, block_minutes=15.0):
    """One night along the uv track for the next-generation backends."""
    print(f"\n  uv track from {arr.site.name}: window "
          f"{hour_angle_window(system.dec_deg, arr.site.latitude_deg)} h, "
          f"{block_minutes:.0f}-min blocks, phase0 = {phase0:.3f}")
    out = []
    for label, spec, det, pol in BACKENDS[2:]:
        a = _with_detector(arr, det)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tr = track_g3_snr(system, a, spec, block_minutes=block_minutes,
                              phase0=phase0, polarization_mode=pol)
        if tr.n_blocks == 0:
            print(f"  {label}: source never above 30 deg")
            continue
        drift = max(b.drift_cycles_max for b in tr.blocks)
        n_amp = nights_to_precision(tr, 0.1, "amplitude")
        n_bin = nights_to_precision(tr, 0.1, "binned", R_bin=100.0)
        n_ch = nights_to_precision(tr, 0.1, "channel")
        print(f"  {label:42s} {tr.n_blocks} blocks, {tr.t_total_s / 3600:.1f} h: "
              f"SNR3_amp/night = {tr.snr_amplitude:.3g}; nights to d<=0.1: "
              f"amplitude {n_amp:.3g}, binned R=100 {n_bin:.3g}, channel {n_ch:.3g}; "
              f"max drift {drift:.2f} cycles (flagged: {tr.drift_flagged})")
        out.append((label, tr, n_amp, n_bin, n_ch))
    return out


def smearing_summary(system, arr, phase):
    pos = positions_at(system, phase)
    rho = float(pos.rho) * MAS
    d = arr.stations[0].telescope.diameter_m
    print(f"  fringe contrast retained by {d:.1f} m pupils at 400 nm, rho = "
          f"{float(pos.rho):.2f} mas: {fringe_smearing_factor(d, d, rho, 400e-9):.2f} "
          f"(800 nm: {fringe_smearing_factor(d, d, rho, 800e-9):.2f})")


def g2_numbers():
    """The two-telescope numbers of the paper's Section 3."""
    from hbtsim.params import ALGOL, BETA_AUR
    print("\n=== Two-telescope g2 (Section 3) ===")
    pos = positions_at(BETA_AUR, 0.0)
    for lam in (400.0, 800.0):
        obs = Observation(wavelength_nm=lam, filter_width_nm=10.0, t_int_s=3600.0)
        from hbtsim.hbt import binary_vis2_analytic
        for b in (15.0, 50.0):
            v2 = float(binary_vis2_analytic(b, lam, BETA_AUR, float(pos.rho))[0])
            r = g2_snr(v2, system_ab_mag(BETA_AUR, lam), obs, telescope1=C2PU)
            print(f"  Beta Aur, C2PU, 10 nm filter at {lam:.0f} nm, B = {b:.0f} m: "
                  f"|V|^2 = {v2:.3f}, SNR2/h = {r.snr:.2f}")
    for det, lab in ((SPAD_LAMBDA, "time-tag"), (SPAD_LAMBDA_NG, "correlator")):
        r = spectral_g2_snr(BETA_AUR, 50.0, spectrograph=SPEC_320, telescope1=C2PU,
                            detector1=det, vis2_method="analytic")
        print(f"  Beta Aur, C2PU, 320 ch, B = 50 m, {lab}: SNR2/h = {r.snr_total:.2f} "
              f"(rate {r.total_rate_cps[0]:.2e} cps/tel, limited={r.readout_limited}, "
              f"scale {r.readout_scale:.2f})")
    for sysm in (ALGOL, BETA_AUR):
        for det, lab in ((SPAD_LAMBDA, "time-tag"), (SPAD_LAMBDA_NG, "correlator")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = spectral_g2_snr(sysm, 85.0, spectrograph=SPEC_320, telescope1=KECK,
                                    detector1=det, vis2_method="analytic")
            sig = 1.0 / r.snr * r.vis2
            sig_unit = 1.0 / (r.snr / np.maximum(r.vis2, 1e-12))
            print(f"  {sysm.name.split()[0]}, Keck 85 m, 320 ch, {lab}: SNR2/h = "
                  f"{r.snr_total:.1f}; sigma(|V|^2) per channel-hour "
                  f"{np.median(sig_unit):.3g} (median), {sig_unit.min():.3g}-{sig_unit.max():.3g}; "
                  f"rate {r.total_rate_cps[0]:.2e} cps/tel, limited={r.readout_limited}, "
                  f"dead-time load {r.dead_time_load_max:.2f}")


# two-telescope instruments: telescope, [(label, spectrograph, detector, polarization)]
G2_INSTRUMENTS = {
    "c2pu": (C2PU, [
        ("320 ch, time-tag", SPEC_320, SPAD_LAMBDA, "unpolarized"),
        ("320 ch, correlator", SPEC_320, SPAD_LAMBDA_NG, "unpolarized"),
        ("R = 5000, correlator", SPEC_R5000, SPAD_LAMBDA_NG, "unpolarized"),
        ("R = 5000, correlator + PBS", SPEC_R5000, SPAD_LAMBDA_NG, "pbs")]),
    "keck": (KECK, [
        ("320 ch, time-tag", SPEC_320, SPAD_LAMBDA, "unpolarized"),
        ("320 ch, correlator", SPEC_320, SPAD_LAMBDA_NG, "unpolarized"),
        ("R = 5000, correlator", SPEC_R5000, SPAD_LAMBDA_NG, "unpolarized")]),
    "eonsii": (EON_SII_TELESCOPE, [
        ("1000 ch, MCP-PMT", EONSII_SPECTROGRAPH, EONSII_MCP_PMT, "unpolarized"),
        ("1000 ch, MCP-PMT + PBS", EONSII_SPECTROGRAPH, EONSII_MCP_PMT, "pbs"),
        ("1000 ch, QUASAR SPAD", EONSII_SPECTROGRAPH, EONSII_SPAD, "unpolarized"),
        ("1000 ch, QUASAR SPAD + PBS", EONSII_SPECTROGRAPH, EONSII_SPAD, "pbs"),
        ("R = 7500 (2388 ch), QUASAR SPAD", EONSII_SPECTROGRAPH_R7500, EONSII_SPAD, "unpolarized")]),
}


def g2_table(instrument: str, newera_dir=None, allow_extrapolation=False,
             baselines=None, latex=False):
    """Two-telescope g2 sensitivity of every system on one instrument: the
    baseline (along the separation, scanned) that maximizes the total SNR
    of the instrument's first backend at quadrature, then SNR/h for each
    backend at that baseline, the total rate per telescope and the pupil
    smearing D/P."""
    tel, backends = G2_INSTRUMENTS[instrument]
    b_scan = np.arange(10.0, 301.0, 5.0) if baselines is None else np.asarray(baselines, float)
    print(f"\n=== Two-telescope g2: {instrument} (2 x {tel.diameter_m:g} m, "
          f"{tel.area_m2:.1f} m^2 each, throughput {tel.throughput:.2f}) ===")
    rows = []
    for key in ("betaaur", "algol", "deltavel", "spica"):
        system = attach_from_cli(SYSTEMS[key], newera_dir, allow_extrapolation, verbose=False)
        phase = max_separation_phase(system)
        pos = positions_at(system, phase)
        label0, spec0, det0, pol0 = backends[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tot = [spectral_g2_snr(system, b, spectrograph=spec0, telescope1=tel, detector1=det0,
                                   polarization_mode=pol0, orbital_phase=phase,
                                   vis2_method="analytic").snr_total for b in b_scan]
        b_best = float(b_scan[int(np.argmax(tot))])
        d_over_p = tel.diameter_m * float(pos.rho) * MAS / (spec0.lambda_min_nm * 1e-9)
        tabled = "NewEra" if system.has_sed_tables else ("NewEra(A)" if system.primary.flux_table is not None else "blackbody")
        print(f"\n  {system.name} [{tabled}]: rho = {float(pos.rho):.2f} mas at phase {phase:.3f}; "
              f"best baseline {b_best:.0f} m (first backend), D/P = {d_over_p:.2f} at "
              f"{spec0.lambda_min_nm:.0f} nm, fringe contrast retained "
              f"{fringe_smearing_factor(tel.diameter_m, tel.diameter_m, float(pos.rho) * MAS, spec0.lambda_min_nm * 1e-9):.2f}")
        for label, spec, det, pol in backends:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = spectral_g2_snr(system, b_best, spectrograph=spec, telescope1=tel,
                                    detector1=det, polarization_mode=pol, orbital_phase=phase,
                                    vis2_method="analytic")
            print(f"    {label:34s} SNR2/h = {r.snr_total:8.1f}; rate {r.total_rate_cps[0]:.2e} cps/tel"
                  f"{' READOUT-LIMITED x%.2f' % r.readout_scale if r.readout_limited else ''}, "
                  f"dead-time load {r.dead_time_load_max:.2f}, |V|^2 median {np.median(r.vis2):.3f}")
            rows.append((system.name.split()[0], instrument, label, b_best, r.snr_total,
                         r.total_rate_cps[0], r.readout_limited))
    if latex:
        print("\n  LaTeX rows (target & instrument & backend & B & SNR2/sqrt(h)):")
        for r in rows:
            print(f"  {r[0]} & {r[1]} & {r[2]} & {r[3]:.0f} m & {r[4]:.1f} \\\\")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", choices=sorted(SYSTEMS), default="spica")
    p.add_argument("--array", choices=sorted(ARRAYS), default="vlt")
    p.add_argument("--phase", type=float, default=None,
                   help="orbital phase for the snapshot (default: max separation)")
    p.add_argument("--table", action="store_true", help="print LaTeX rows")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-track", action="store_true")
    p.add_argument("--block-minutes", type=float, default=15.0)
    p.add_argument("--g2", action="store_true", help="only the g2 numbers")
    p.add_argument("--instrument", choices=sorted(G2_INSTRUMENTS), default="c2pu",
                   help="two-telescope instrument for the --g2 table")
    p.add_argument("--newera-dir", default=os.environ.get("HBTSIM_NEWERA_DIR") or
                   ("data/newera" if os.path.isdir("data/newera") else None),
                   help="directory of binned NewEra tables to attach to every star "
                        "the grid covers (default: $HBTSIM_NEWERA_DIR or data/newera)")
    p.add_argument("--no-newera", action="store_true", help="blackbody + Claret only")
    p.add_argument("--allow-extrapolation", action="store_true",
                   help="clamp stars beyond the NewEra grid edge to it (Algol A)")
    args = p.parse_args()
    if args.no_newera:
        args.newera_dir = None
    warnings.filterwarnings("ignore", message=".*extrapolated.*")
    warnings.filterwarnings("ignore", message=".*dead-time.*")

    if args.g2:
        if args.instrument == "c2pu":
            g2_numbers()
        g2_table(args.instrument, args.newera_dir, args.allow_extrapolation, latex=args.table)
        return

    system, arr = SYSTEMS[args.system], ARRAYS[args.array]
    system = attach_from_cli(system, args.newera_dir, args.allow_extrapolation)
    phase = max_separation_phase(system) if args.phase is None else args.phase
    print(f"=== {system.name} on {args.array} ({len(arr.stations)} stations, "
          f"Omega = {system.node_pa_deg} deg, dec = {system.dec_deg} deg), "
          f"phase {phase:.3f} ===")
    smearing_summary(system, arr, phase)
    if not args.no_figures:
        tag = f"{args.system}_{args.array}"
        fig_gammas(system, arr, f"output/g3_gammas_{tag}.png", phase)
        tri = arr.triangles()[1] if isinstance(arr, Array) else arr
        fig_cosphi_map(system, tri, f"output/g3_cosphi_{tag}.png")
    print("\n  snapshot at quadrature (per hour):")
    snapshot_rows(system, arr, phase, latex=args.table)
    if args.system == "deltavel":
        for ph in (0.25, 0.5, 0.9):
            print(f"\n  delta Vel at phase {ph}:")
            snapshot_rows(system, arr, ph, latex=args.table)
    if not args.no_track:
        track_rows(system, arr, phase, args.block_minutes)


if __name__ == "__main__":
    main()
