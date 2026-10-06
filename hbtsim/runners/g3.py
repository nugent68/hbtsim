"""Runner `g3`: feasibility of three-telescope closure-phase intensity
interferometry for one binary on one array (the former
scripts/feasibility_g3.py).

Generates the figures and every number quoted in
docs/three_telescope_feasibility.md and the paper's Table 3:

  (a) per-pair |gamma|(lambda) (point and aperture-averaged) and the
      per-triangle bispectrum amplitudes at quadrature (g3_gammas.png)
  (b) the cos(phi_c)(lambda, orbital phase) map on one triangle
      (g3_cosphi.png)
  (c) snapshot sensitivities at quadrature for several backends, the
      time to reach Delta cos phi_c <= 0.1 for the three statistics
      (global template amplitude; R = 100 binned closure phases; a
      single channel), and one realistic night along the uv track

Campaign fields used: target, array, backends (snapshot), track_backends,
phases.snapshot ("max_separation" or a number; --phase overrides),
phases.extra (false or a list of further snapshot phases),
night.block_minutes, outputs.track.  Instrument model (hbtsim.snr):
telescope throughput x spectrograph throughput x PDE; aperture smearing
on; sky orientation from each system's Omega.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from hbtsim.aperture import fringe_smearing_factor
from hbtsim.bispectrum import Array, Triangle, binary_vis_complex_analytic, spectral_triple
from hbtsim.geometry import hour_angle_window
from hbtsim.orbit import max_separation_phase, positions_at, sky_positions
from hbtsim.params import MAS, GridConfig
from hbtsim.snr3 import (array_g3_snr, nights_to_precision, spectral_g3_snr,
                         time_to_precision, track_g3_snr)

from . import write_tables

NIGHT_H = 8.0


def _with_detector(arr, det):
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


def _finite(x):
    """A plain float, or None where the number is not finite (JSON)."""
    x = float(x)
    return x if np.isfinite(x) else None


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


def fig_cosphi_map(system, tri, out, n_phase=51, sky_panels=11):
    """cos(phi_c) over wavelength and orbital phase, with a column of small
    sky images (the two disks at their projected positions, centre-of-mass
    frame, north up and east left) aligned with the phase axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

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
    fig = plt.figure(figsize=(9.6, 7.0))
    ax = fig.add_axes([0.07, 0.09, 0.60, 0.76])
    im = ax.pcolormesh(nm, phases, cosmap, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
    cax = fig.add_axes([0.685, 0.09, 0.018, 0.76])
    fig.colorbar(im, cax=cax, label=r"$\cos\varphi_c$")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Orbital phase")
    ax.set_title(f"{system.name}: closure-phase cosine on {tri.name} (aperture-averaged)",
                 fontsize=10, loc="left")

    # sky panels: one small square axis per phase, centred on that phase's y
    r1 = system.drawn_radius_mas(system.primary)
    r2 = system.drawn_radius_mas(system.secondary)
    all_pos = sky_positions(np.linspace(0.0, 2 * np.pi, 721), system)
    reach = max(float(np.max(np.abs(np.asarray(v)))) for v in
                (all_pos.x1, all_pos.y1, all_pos.x2, all_pos.y2)) + max(r1, r2)
    half = 1.08 * reach
    x0, y0, w, h = 0.80, 0.09, 0.60, 0.76
    fig_w, fig_h = fig.get_size_inches()
    size_in = min(0.95 * h * fig_h / sky_panels, 1.0)         # square panels [inches]
    sw, sh = size_in / fig_w, size_in / fig_h
    ph_panels = np.linspace(0.0, 1.0, sky_panels)
    for j, ph in enumerate(ph_panels):
        yc = y0 + h * ph
        sax = fig.add_axes([x0, yc - sh / 2, sw, sh])
        pos = positions_at(system, ph)
        # east to the left: plot -x
        sax.add_patch(Circle((-float(pos.x1), float(pos.y1)), r1, color="#c9973a", lw=0))
        sax.add_patch(Circle((-float(pos.x2), float(pos.y2)), r2, color="#3f6fb5", lw=0))
        for xx, yy in ((all_pos.x1, all_pos.y1), (all_pos.x2, all_pos.y2)):
            sax.plot(-np.asarray(xx), np.asarray(yy), color="0.8", lw=0.4, zorder=0)
        sax.set_xlim(half, -half)
        sax.set_ylim(-half, half)
        sax.set_aspect("equal")
        sax.set_xticks([]); sax.set_yticks([])
        for sp in sax.spines.values():
            sp.set_linewidth(0.4); sp.set_color("0.6")
        sax.text(1.08, 0.5, f"{ph:.1f}", transform=sax.transAxes, va="center", fontsize=7)
    # compass and scale bar in their own box above the column
    lax = fig.add_axes([x0, y0 + h + sh / 2 + 0.01, sw, sh * 0.8])
    lax.set_xlim(half, -half); lax.set_ylim(-half * 0.8, half * 0.8); lax.set_aspect("equal")
    lax.axis("off")
    lax.annotate("", xy=(half * 0.7, half * 0.55), xytext=(half * 0.7, -half * 0.35),
                 arrowprops=dict(arrowstyle="-|>", lw=0.8))
    lax.annotate("", xy=(-half * 0.2, -half * 0.35), xytext=(half * 0.7, -half * 0.35),
                 arrowprops=dict(arrowstyle="-|>", lw=0.8))
    lax.text(half * 0.7, half * 0.6, "N", ha="center", va="bottom", fontsize=6)
    lax.text(-half * 0.25, -half * 0.35, "E", ha="right", va="center", fontsize=6)
    bar = 1.0 if half > 1.5 else 0.5
    lax.plot([-half * 0.35, -half * 0.35 - bar], [half * 0.45] * 2, color="k", lw=1.2)
    lax.text(-half * 0.35 - bar / 2, half * 0.5, f"{bar:g} mas", ha="center", va="bottom", fontsize=6)
    fig.text(x0 + sw / 2, y0 + h + sh / 2 + 0.01 + sh * 0.8 + 0.006,
             f"sky: A {2 * r1:.2f} mas, B {2 * r2:.2f} mas", ha="center", fontsize=7)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def snapshot_rows(system, arr, phase, backends, latex=False):
    """Per backend (label, spectrograph, detector, polarization): snapshot
    sensitivities at one orbital phase and the time to Delta cos phi_c <= 0.1
    for the three statistics."""
    rows = []
    for label, spec, det, pol in backends:
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


def track_rows(system, arr, phase0, backends, block_minutes=15.0):
    """One night along the uv track for the track backends."""
    print(f"\n  uv track from {arr.site.name}: window "
          f"{hour_angle_window(system.dec_deg, arr.site.latitude_deg)} h, "
          f"{block_minutes:.0f}-min blocks, phase0 = {phase0:.3f}")
    out = []
    for label, spec, det, pol in backends:
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


def smearing_summary(system, arr, phase) -> dict:
    pos = positions_at(system, phase)
    rho = float(pos.rho) * MAS
    d = arr.stations[0].telescope.diameter_m
    f400 = fringe_smearing_factor(d, d, rho, 400e-9)
    f800 = fringe_smearing_factor(d, d, rho, 800e-9)
    print(f"  fringe contrast retained by {d:.1f} m pupils at 400 nm, rho = "
          f"{float(pos.rho):.2f} mas: {f400:.2f} "
          f"(800 nm: {f800:.2f})")
    return {"pupil_m": float(d), "rho_mas": float(pos.rho),
            "retained_400nm": float(f400), "retained_800nm": float(f800)}


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------
def _backend_tuples(backends) -> list:
    return [(b.name, b.spectrograph, b.detector, b.polarization_mode) for b in backends]


def _snapshot_json(rows) -> list:
    out = []
    for r in rows:
        out.append({"backend": r["label"], "n_channels": int(r["n_ch"]),
                    "snr_amplitude": _finite(r["snr_amp"]), "snr_total": _finite(r["snr_total"]),
                    "t_amplitude_h": _finite(r["t_amp"]), "t_binned_h": _finite(r["t_bin"]),
                    "t_channel_h": _finite(r["t_ch"]), "rate_cps": _finite(r["rate"]),
                    "readout_limited": bool(r["limited"]), "readout_scale": _finite(r["scale"]),
                    "dead_time_load": _finite(r["load"]), "ridge_ratio": _finite(r["ridge"]),
                    "triple_max": _finite(r["triple_max"])})
    return out


def _resolve_phase(campaign, opts, system) -> float:
    if opts.phase is not None:
        return float(opts.phase)
    ph = campaign.option("phases.snapshot", "max_separation")
    if ph == "max_separation" or ph is None:
        return float(max_separation_phase(system))
    return float(ph)


def run(campaign, cat, opts, out_dir: Path) -> dict:
    warnings.filterwarnings("ignore", message=".*extrapolated.*")
    warnings.filterwarnings("ignore", message=".*dead-time.*")

    system = campaign.target
    arr = campaign.array
    if arr is None:
        raise SystemExit(f"campaign {campaign.name}: runner g3 needs an `array`")
    # a three-station array is handled as the Triangle the former script used
    if isinstance(arr, Array) and len(arr.stations) == 3:
        arr = arr.as_triangle()
    array_name = campaign.spec.get("array") or campaign.option("instrument.array") or "array"
    phase = _resolve_phase(campaign, opts, system)
    block_minutes = float(campaign.option("night.block_minutes", 15.0))
    snap_b = _backend_tuples(campaign.backends)
    track_b = _backend_tuples(campaign.track_backends)
    latex = bool(opts.latex)

    print(f"=== {system.name} on {array_name} ({len(arr.stations)} stations, "
          f"Omega = {system.node_pa_deg} deg, dec = {system.dec_deg} deg), "
          f"phase {phase:.3f} ===")
    smearing = smearing_summary(system, arr, phase)
    figures = {}
    if opts.figures:
        f1 = out_dir / "g3_gammas.png"
        fig_gammas(system, arr, str(f1), phase)
        tri = arr.triangles()[1] if isinstance(arr, Array) else arr
        f2 = out_dir / "g3_cosphi.png"
        fig_cosphi_map(system, tri, str(f2))
        figures = {"gammas": f1.name, "cosphi": f2.name}

    print("\n  snapshot at quadrature (per hour):")
    rows = snapshot_rows(system, arr, phase, snap_b, latex=latex)

    extra = campaign.option("phases.extra") or []
    extra_out = {}
    for ph in extra:
        ph = float(ph)
        print(f"\n  {system.name.split()[0]} at phase {ph}:")
        extra_out[f"{ph:g}"] = _snapshot_json(snapshot_rows(system, arr, ph, snap_b, latex=latex))

    track_out = []
    if opts.track and campaign.option("outputs.track", True) and track_b:
        for label, tr, n_amp, n_bin, n_ch in track_rows(system, arr, phase, track_b, block_minutes):
            track_out.append({"backend": label, "n_blocks": int(tr.n_blocks),
                              "t_total_h": float(tr.t_total_s / 3600.0),
                              "snr_amplitude": _finite(tr.snr_amplitude),
                              "nights_amplitude": _finite(n_amp), "nights_binned": _finite(n_bin),
                              "nights_channel": _finite(n_ch),
                              "drift_max": float(max(b.drift_cycles_max for b in tr.blocks)),
                              "drift_flagged": bool(tr.drift_flagged)})

    short = system.name.split()[0]
    table = [[short, r["label"], r["n_ch"], f"{r['snr_amp']:.3g}", _fmt_time(r["t_amp"]),
              _fmt_time(r["t_bin"]), _fmt_time(r["t_ch"])] for r in rows]
    write_tables(out_dir, table, ["target", "backend", "channels", "SNR3_amp/sqrt(h)",
                                  "amplitude", "binned R=100", "channel"], latex=latex)

    return {"target": system.name, "array": array_name, "phase": phase,
            "block_minutes": block_minutes, "snapshot": _snapshot_json(rows),
            "extra_phases": extra_out, "track": track_out, "smearing": smearing,
            "figures": figures}
