"""Spica on the four VERITAS telescopes: what the six baselines should see
around the 4.01-day orbit (hbtsim.iact).

Model: the catalog's spica (blackbody + Spica linear limb darkening, anchored
g/i; circular-orbit approximation, a = 1.71 mas) with the analytic
two-disk visibility on the projected VERITAS baselines, averaged over the
12 m pupils, at 416 nm.  Sensitivity: VERITAS's own analog S/N form with
the optical efficiency calibrated to their published precision (0.016 on
|V|^2 per pair in 4.25 h on epsilon Ori).  One night per orbital phase,
17-minute blocks as in their analysis.

    .venv/bin/python scripts/spica_veritas.py [--nights 8] [--block-minutes 17]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

from hbtsim.aperture import fringe_smearing_factor
from hbtsim.catalog import Catalog
from hbtsim.geometry import hour_angle_window
from hbtsim.iact import binary_vis2_fn, calibrated, pair_track, track_sigma
from hbtsim.orbit import positions_at, sky_positions
from hbtsim.params import MAS
from hbtsim.snr import system_ab_mag

CAT = Catalog(env=False)
SPICA = CAT.load_target("spica")
VERITAS = CAT.load_array("veritas")
FLWO = VERITAS.site
VERITAS_TELESCOPE = CAT.load_telescope("veritas_12m")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nights", type=int, default=8, help="nights spread over one orbit")
    ap.add_argument("--block-minutes", type=float, default=17.0)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    lam = 416.0
    backend = calibrated(CAT.load_backend("veritas_sii"), VERITAS_TELESCOPE.area_m2)
    anchor = backend.anchor
    mag = float(system_ab_mag(SPICA, lam))
    h0, h1 = hour_angle_window(SPICA.dec_deg, FLWO.latitude_deg)
    ph = np.linspace(0, 1, 2001)[:-1]
    rho = np.asarray(sky_positions(2 * np.pi * ph, SPICA).rho)
    print(f"Spica on VERITAS at {lam:.0f} nm: AB {mag:.2f}; separation {rho.min():.2f}-{rho.max():.2f} mas, "
          f"fringe period {lam * 1e-9 / (rho.max() * MAS):.0f}-{lam * 1e-9 / (rho.min() * MAS):.0f} m; "
          f"12 m pupils keep {fringe_smearing_factor(12, 12, rho.max() * MAS, lam * 1e-9):.2f} of the contrast at maximum separation")
    print(f"  FLWO window H = {h0:+.2f}..{h1:+.2f} h ({h1 - h0:.1f} h above 30 deg); blocks of {args.block_minutes:g} min")
    print(f"  VERITAS sensitivity: q calibrated to {backend.q:.3f} from {anchor.star} "
          f"(sigma(|V|^2) = {anchor.sigma_vis2} per pair in {anchor.t_s / 3600:.2f} h at AB mag {anchor.mag_ab:.2f})")

    tracks = []
    for k in range(args.nights):
        tr = pair_track(VERITAS, SPICA.dec_deg, lam, binary_vis2_fn(SPICA, pupils=True),
                        block_minutes=args.block_minutes, phase0=k / args.nights, period_days=SPICA.period_days)
        tracks.append(tr)
    tr = tracks[0]
    sig_blk = track_sigma(tr, mag, backend)
    sig_h = track_sigma(tr, mag, backend, 3600.0)
    print(f"  per pair: sigma(|V|^2) = {sig_blk[0]:.4f} per {tr.block_s / 60:.0f}-min block, {sig_h[0]:.4f} per hour, "
          f"{(sig_h[0] / 0.01) ** 2:.1f} h to 0.01; {tr.hour_angle_h.size} blocks per night")
    for p, name in enumerate(tr.pair_names):
        allv = np.concatenate([t.vis2[p] for t in tracks]); allb = np.concatenate([t.baseline_len_m[p] for t in tracks])
        print(f"    {name}: projected {allb.min():.0f}-{allb.max():.0f} m, |V|^2 {allv.min():.3f}-{allv.max():.3f} over the orbit, "
              f"swing / sigma(1 h) = {(allv.max() - allv.min()) / sig_h[p]:.0f}")
    print("  (orbit: hbtsim spica a = 1.71 mas circular; Raiola et al. 2025 use Herbison-Evans 1971: a = 1.54 mas, "
          "theta 0.90/0.40 mas, brightness ratio 6.4, e = 0.133)")

    if args.no_figure:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), width_ratios=[1.35, 1])
    markers = ["o", "s", "^", "v", "D", "P"]
    for p, name in enumerate(tr.pair_names):
        for t in tracks:
            sc = a1.scatter(t.baseline_len_m[p], t.vis2[p], c=t.phase % 1, cmap="twilight", vmin=0, vmax=1,
                            s=14, marker=markers[p], edgecolors="none")
    for p, name in enumerate(tr.pair_names):
        a1.scatter([], [], marker=markers[p], c="0.3", s=20, label=name)
    a1.errorbar([30.0], [0.9], yerr=[sig_blk[0]], fmt="none", ecolor="k", capsize=3)
    a1.text(34.0, 0.9, f"±σ per {tr.block_s / 60:.0f} min", fontsize=8, va="center")
    a1.errorbar([30.0], [0.78], yerr=[sig_h[0]], fmt="none", ecolor="k", capsize=3)
    a1.text(34.0, 0.78, "±σ per hour", fontsize=8, va="center")
    fig.colorbar(sc, ax=a1, label="orbital phase")
    a1.set_xlabel("Projected baseline [m]")
    a1.set_ylabel(r"$|V|^2$ at 416 nm (12 m pupils)")
    a1.set_title(f"Spica on VERITAS: {args.nights} nights around the orbit", fontsize=10)
    a1.legend(fontsize=7, ncol=2, loc="upper right")
    a1.grid(alpha=0.3)
    # right: |V|^2 vs orbital phase at transit on the longest and shortest baselines
    phases = np.linspace(0, 1, 241)
    fn = binary_vis2_fn(SPICA, pupils=True)
    proj = VERITAS.projected(0.0, SPICA.dec_deg).pairs()
    for p in (int(np.argmax(tr.baseline_len_m[:, 0])), int(np.argmin(tr.baseline_len_m[:, 0]))):
        b = proj[p][2]
        v = np.array([float(fn(b[None, :], lam, q, tr.diameters_m[p])[0]) for q in phases])
        a2.plot(phases, v, label=f"{tr.pair_names[p]} at transit ({np.hypot(*b):.0f} m)")
    a2.set_xlabel("Orbital phase")
    a2.set_ylabel(r"$|V|^2$")
    a2.set_title("The binary signature on fixed baselines", fontsize=10)
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    out = "output/spica_veritas.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
