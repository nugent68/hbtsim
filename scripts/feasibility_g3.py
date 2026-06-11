"""Feasibility of three-telescope closure-phase intensity interferometry.

Generates the figures and tables behind docs/three_telescope_feasibility.md:

  (a) per-pair |gamma|(lambda) and the triple amplitude at quadrature
  (b) cos(phi_c)(lambda, orbital phase) map
  (c) per-channel SNR3 per hour, and nights to Delta cos(phi_c) targets
      for several channel widths
  (d) the real Maunakea triangle vs a compact 85 m equilateral
  (e) comparison against H.E.S.S. / CTA-LST (Zmija et al. 2025)

    python scripts/feasibility_g3.py --system algol
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hbtsim.bispectrum import (MAUNAKEA_SUBARU_KECK, Triangle,
                               binary_vis_complex_analytic,
                               equilateral_triangle, spectral_bispectrum)
from hbtsim.orbit import SkyPositions, sky_positions
from hbtsim.params import SYSTEMS, GridConfig
from hbtsim.snr import KECK, Spectrograph
from hbtsim.snr3 import spectral_g3_snr, time_to_cos_phi

NIGHT_S = 8 * 3600.0


def _pos(system, phase):
    return SkyPositions(*(np.asarray(v) for v in
                          sky_positions(2 * np.pi * phase, system)))


def fig_gammas(system, sysname, triangle, out):
    nm = np.linspace(400.0, 950.0, 200)
    pos = _pos(system, 0.0)
    bv = triangle.baseline_vectors()
    gam = np.array([binary_vis_complex_analytic(bv, float(l), system, pos)
                    for l in nm])
    names = [f"{triangle.stations[i].name}-{triangle.stations[j].name}"
             for i, j in ((0, 1), (1, 2), (2, 0))]
    lengths = triangle.baseline_lengths()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for k in range(3):
        ax1.plot(nm, np.abs(gam[:, k]),
                 label=f"{names[k]} ({lengths[k]:.0f} m)")
    ax1.set_ylabel(r"$|\gamma_{ij}|$")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_title(f"{system.name} at quadrature")

    bis = gam.prod(axis=1)
    ax2.plot(nm, np.abs(bis), "k", label=r"$|\gamma_{12}\gamma_{23}\gamma_{31}|$")
    ax2.plot(nm, np.real(bis), "C0--", lw=1,
             label=r"Re (with $\cos\varphi_c$ sign)")
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_xlabel("Wavelength [nm]")
    ax2.set_ylabel("triple product")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}  (peak |triple| = {np.abs(bis).max():.4f} at "
          f"{nm[np.abs(bis).argmax()]:.0f} nm)")
    return float(np.abs(bis).max())


def fig_cosphi_map(system, sysname, triangle, grid, out):
    """cos phi_c over (lambda, orbital phase) via the FFT path (valid in
    eclipse)."""
    nm = np.linspace(400.0, 950.0, 56)
    phases = np.linspace(0.0, 0.5, 26)  # symmetric over the half orbit
    cosmap = np.empty((phases.size, nm.size))
    for k, ph in enumerate(phases):
        gam = np.asarray(spectral_bispectrum(_pos(system, ph), triangle, nm,
                                             system, grid))
        bis = gam[:, 0] * gam[:, 1] * gam[:, 2]
        cosmap[k] = np.cos(np.angle(bis))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.pcolormesh(nm, phases, cosmap, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, label=r"$\cos\varphi_c$")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Orbital phase")
    ax.set_title(f"{system.name}: closure-phase cosine "
                 f"({triangle.stations[0].name} triangle)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def feasibility_table(system, sysname, triangle, label):
    print(f"\n=== {system.name} on {label} ===")
    print(f"{'channels':>16} {'dlam[nm]':>9} {'best SNR3/ch/h':>15} "
          f"{'SNR3 mux/h':>11} {'nights d<=0.3':>14} {'nights d<=0.1':>14}")
    rows = []
    for n_ch, note in ((320, "SPAD Lambda"), (5500, "R~5000 spectrograph")):
        spec = Spectrograph(lambda_min_nm=400.0, lambda_max_nm=950.0,
                            n_channels=n_ch)
        res = spectral_g3_snr(system, triangle, spectrograph=spec)
        t03 = time_to_cos_phi(system, triangle, 0.3, spectrograph=spec)
        t01 = time_to_cos_phi(system, triangle, 0.1, spectrograph=spec)
        print(f"{n_ch:>10} ({note[:9]}) {spec.channel_width_nm:>8.2f} "
              f"{res.snr.max():>15.2e} {res.snr_total:>11.2e} "
              f"{t03 / NIGHT_S:>14.3g} {t01 / NIGHT_S:>14.3g}")
        rows.append((n_ch, spec.channel_width_nm, res, t03, t01))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", choices=sorted(SYSTEMS), default="algol")
    args = p.parse_args()

    system = SYSTEMS[args.system]
    grid = GridConfig()
    maunakea = MAUNAKEA_SUBARU_KECK
    compact = equilateral_triangle(85.0, KECK)

    print(f"Maunakea triangle baselines: "
          f"{np.round(maunakea.baseline_lengths(), 1)} m")
    fig_gammas(system, args.system, maunakea,
               f"output/g3_gammas_{args.system}.png")
    fig_cosphi_map(system, args.system, maunakea, grid,
                   f"output/g3_cosphi_{args.system}.png")

    feasibility_table(system, args.system, maunakea,
                      "Subaru + Keck I + Keck II (152/85/226 m)")
    feasibility_table(system, args.system, compact,
                      "compact equilateral 85 m (3 x 10 m)")

    print("\n=== context: Zmija et al. 2025 (Table 2) ===")
    print("  H.E.S.S. (3x100 m^2, 5 ns, 10 nm, 1 ch):   ~1100-2400 yr for "
          "dcos<=0.1 (Nunki/Dschubba)")
    print("  CTA LSTs (4x400 m^2, 0.1 ns, 0.1 nm, 1000 ch): ~2-5 months")


if __name__ == "__main__":
    main()
