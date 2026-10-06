"""gamma Cas on MAGIC-I, MAGIC-II and LST-1: oblateness and a circumstellar
disk in the 425 nm squared visibility (hbtsim.iact, hbtsim.single).

Models: the MAGIC circular fit (theta_LD = 0.532 mas); the VERITAS 2025
ellipse (minor 0.43 mas, axis ratio 1.28, PA 116 deg); each with a Gaussian
disk (FWHM 2.9 mas, the literature envelope) carrying 0, 10 or 20 % of the
flux at 425 nm.  Sensitivity: MAGIC's published S/N form and constants,
LST-1 scaled by its mirror area.  One night from Roque de los Muchachos
in 20-minute blocks; Fisher estimate of the axis-ratio precision from the
three baselines' rotation; bias of a circular uniform-disk fit when a disk
is present.  Gravity darkening (Roche-von Zeipel) is not modeled.

    .venv/bin/python scripts/gammacas_magic.py [--block-minutes 20]
"""

from __future__ import annotations

import argparse
import warnings
from functools import partial

import numpy as np

from hbtsim.catalog import Catalog
from hbtsim.geometry import hour_angle_window
from hbtsim.iact import pair_track, single_vis2_fn, track_sigma
from hbtsim.params import MAS
from hbtsim.single import composite_vis, ud_vis2

LAM = 425.0
CAT = Catalog(env=False)
GAMMA_CAS = CAT.load_target("gammacas")
GAMMA_CAS_ELLIPSE = GAMMA_CAS.ellipse          # single.Ellipse (VERITAS 2025)
MAGIC_LST1 = CAT.load_array("magic_lst1")
MAGIC_SII = CAT.load_backend("magic_sii")
ORM = MAGIC_LST1.site


def model_fn(theta_major, axis_ratio, pa, f):
    return single_vis2_fn(lambda bv, nm: composite_vis(bv, nm, GAMMA_CAS.star, theta_major, axis_ratio=axis_ratio,
                                                       pa_deg=pa, disk_fraction=f, disk_fwhm_mas=2.9))


def fit_ud(track, sigma_pair):
    """Weighted least-squares circular uniform-disk diameter to a track's |V|^2
    (with a free zero-baseline normalization, as in the groups' fits)."""
    b = track.baseline_len_m.ravel(); v = track.vis2.ravel()
    w = np.repeat(1.0 / sigma_pair**2, track.vis2.shape[1])
    best = None
    for th in np.linspace(0.30, 0.80, 501):
        m = ud_vis2(th, b, LAM)[0]
        n0 = np.sum(w * m * v) / np.sum(w * m * m)
        chi2 = np.sum(w * (v - n0 * m) ** 2)
        if best is None or chi2 < best[1]:
            best = (th, chi2, n0)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block-minutes", type=float, default=20.0)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    mag = float(GAMMA_CAS.ab_mag(LAM))
    h0, h1 = hour_angle_window(GAMMA_CAS.dec_deg, ORM.latitude_deg)
    E = GAMMA_CAS_ELLIPSE
    print(f"gamma Cas on MAGIC + LST-1 at {LAM:.0f} nm: AB {mag:.2f} (B = 2.29); window H = {h0:+.2f}..{h1:+.2f} h "
          f"({h1 - h0:.1f} h above 30 deg); blocks of {args.block_minutes:g} min")
    print(f"  models: circular theta_LD = {GAMMA_CAS.theta_ld_mas} mas (MAGIC 2024: 0.532 +/- 0.039 +/- 0.023); "
          f"ellipse major {E.theta_major_mas:.3f} mas, axis ratio {E.axis_ratio}, PA {E.pa_deg} deg (VERITAS 2025)")
    circ = pair_track(MAGIC_LST1, GAMMA_CAS.dec_deg, LAM, model_fn(GAMMA_CAS.theta_ld_mas, 1.0, 0.0, 0.0),
                      block_minutes=args.block_minutes)
    ell = pair_track(MAGIC_LST1, GAMMA_CAS.dec_deg, LAM, model_fn(E.theta_major_mas, E.axis_ratio, E.pa_deg, 0.0),
                     block_minutes=args.block_minutes)
    sig_blk = track_sigma(circ, mag, MAGIC_SII)
    sig_h = track_sigma(circ, mag, MAGIC_SII, 3600.0)
    print("  MAGIC sensitivity (their Eq. 4 with their constants; LST-1 area 390 m^2 assumed):")
    for p, name in enumerate(circ.pair_names):
        print(f"    {name:16s} projected {circ.baseline_len_m[p].min():.0f}-{circ.baseline_len_m[p].max():.0f} m, "
              f"PA {circ.position_angle_deg[p].min():.0f}..{circ.position_angle_deg[p].max():.0f} deg; "
              f"|V|^2 circ {circ.vis2[p].min():.2f}-{circ.vis2[p].max():.2f}, ellipse {ell.vis2[p].min():.2f}-{ell.vis2[p].max():.2f}; "
              f"sigma(|V|^2) {sig_blk[p]:.3f} per block, {sig_h[p]:.4f} per hour")
    # Fisher on the axis ratio (and PA) from one night, theta_major fixed
    def fisher(params):
        eps = 1e-3
        grads = []
        for i in range(len(params)):
            up = list(params); dn = list(params); up[i] += eps; dn[i] -= eps
            tu = pair_track(MAGIC_LST1, GAMMA_CAS.dec_deg, LAM, model_fn(up[0], up[1], up[2], 0.0), block_minutes=args.block_minutes)
            td = pair_track(MAGIC_LST1, GAMMA_CAS.dec_deg, LAM, model_fn(dn[0], dn[1], dn[2], 0.0), block_minutes=args.block_minutes)
            grads.append((tu.vis2 - td.vis2) / (2 * eps))
        g = np.stack(grads)                                   # (n_par, n_pair, n_blk)
        w = (1.0 / sig_blk**2)[None, :, None]
        return np.einsum("ipk,jpk,pk->ij", g, g, np.broadcast_to(w[0], g.shape[1:]))
    F = fisher([E.theta_major_mas, E.axis_ratio, E.pa_deg])
    cov = np.linalg.inv(F)
    s_th, s_r, s_pa = np.sqrt(np.diag(cov))
    print(f"  one night, three parameters free (major axis, axis ratio, PA): sigma(theta_major) = {s_th:.3f} mas, "
          f"sigma(axis ratio) = {s_r:.3f}, sigma(PA) = {s_pa:.0f} deg -> {(s_r / 0.02) ** 2:.1f} nights for +/- 0.02 on the ratio "
          f"(VERITAS 2025: +/- 0.04 +/- 0.02 from 160 pair-hours)")
    # disk fraction: bias of a circular UD fit
    print("  circumstellar disk (Gaussian, FWHM 2.9 mas) at 425 nm: circular-UD fit to the night's |V|^2")
    th0 = None
    for f in (0.0, 0.1, 0.2):
        t = pair_track(MAGIC_LST1, GAMMA_CAS.dec_deg, LAM, model_fn(GAMMA_CAS.theta_ld_mas, 1.0, 0.0, f),
                       block_minutes=args.block_minutes)
        th, chi2, n0 = fit_ud(t, sig_blk)
        th0 = th if th0 is None else th0
        print(f"    f = {f:.1f}: fitted theta_UD = {th:.3f} mas ({100 * (th / th0 - 1):+.1f} %), fitted zero-baseline "
              f"normalization {n0:.3f} (true 1), chi2/N = {chi2 / t.vis2.size:.2f}")
    print("  -> a 2.9 mas disk is fully resolved on 50-90 m baselines: it scales |V|^2 by (1 - f)^2 at every baseline, "
          "which the free zero-baseline normalization absorbs. The diameter is unbiased; the disk hides in the "
          "calibration (N0), so a calibrator with known N0, or shorter baselines, is needed to measure f.")
    print("  (f at 425 nm is unknown for gamma Cas: the disk dominates in H-alpha; a free parameter here)")
    print("  not modeled: gravity darkening (Roche-von Zeipel; VERITAS fit 0.604 mas equatorial), v sin i = 389 km/s")

    if args.no_figure:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    markers = ["o", "s", "^"]
    for p, name in enumerate(circ.pair_names):
        a1.plot(circ.baseline_len_m[p], circ.vis2[p], color="0.6", lw=1.0, marker=markers[p], ms=3,
                label=f"{name}, circular 0.53 mas" if p == 0 else None)
        sc = a1.scatter(ell.baseline_len_m[p], ell.vis2[p], c=ell.position_angle_deg[p] % 180, cmap="viridis",
                        vmin=0, vmax=180, s=22, marker=markers[p], label=f"{name}, ellipse" )
    a1.errorbar([30.0], [0.55], yerr=[sig_blk[0]], fmt="none", ecolor="k", capsize=3)
    a1.text(34.0, 0.55, f"±σ per {circ.block_s / 60:.0f} min (MAGIC-I/II)", fontsize=8, va="center")
    fig.colorbar(sc, ax=a1, label="baseline position angle [deg]")
    a1.set_xlabel("Projected baseline [m]")
    a1.set_ylabel(r"$|V|^2$ at 425 nm")
    a1.set_title("γ Cas, one night: circular 0.53 mas (grey) vs VERITAS ellipse (colour by PA)", fontsize=10)
    a1.legend(fontsize=7)
    a1.grid(alpha=0.3)
    b = np.linspace(10, 140, 200)
    bv = np.stack([b, np.zeros_like(b)], axis=1)
    for f, c in ((0.0, "C0"), (0.1, "C1"), (0.2, "C3")):
        v = np.abs(composite_vis(bv, LAM, GAMMA_CAS.star, GAMMA_CAS.theta_ld_mas, disk_fraction=f, disk_fwhm_mas=2.9)[0]) ** 2
        a2.plot(b, v, color=c, label=f"disk fraction {f:.0%} at 425 nm")
    for p in range(3):
        a2.axvspan(circ.baseline_len_m[p].min(), circ.baseline_len_m[p].max(), color="0.9", zorder=0)
    a2.set_xlabel("Baseline [m]")
    a2.set_ylabel(r"$|V|^2$")
    a2.set_title("A resolved disk scales |V|² by (1−f)²: it hides in the normalization", fontsize=10)
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    out = "output/gammacas_magic.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
