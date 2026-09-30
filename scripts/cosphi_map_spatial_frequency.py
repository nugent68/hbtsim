"""The Spica cos(phi_c) map of the paper's Figure 4 with the x-axis as
spatial frequency B/lambda (in mega-wavelengths, for the longest side
UT1-UT4 of the triangle; the other two sides scale with it) instead of
wavelength.  Same model: analytic two-disk bispectrum averaged over the
three 8.2 m pupils, rendered inside eclipses.

    .venv/bin/python scripts/cosphi_map_spatial_frequency.py [--newera-dir data/newera]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

from hbtsim.bispectrum import VLT_UT, spectral_triple
from hbtsim.orbit import positions_at
from hbtsim.params import MAS, SPICA, GridConfig
from hbtsim.sed import attach_from_cli


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--newera-dir", default=None, help="Spica has no NewEra coverage; blackbody by default")
    ap.add_argument("--out", default="output/g3_cosphi_spica_vlt_spatial_frequency.png")
    ap.add_argument("--n-phase", type=int, default=51)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    system = attach_from_cli(SPICA, args.newera_dir, verbose=False)
    tri = VLT_UT.triangles()[1]                       # UT1-UT2-UT4
    b_max = float(tri.baseline_lengths().max())       # 130.2 m
    nm = np.linspace(400.0, 950.0, 120)
    phases = np.linspace(0.0, 1.0, args.n_phase)
    cosmap = np.full((phases.size, nm.size), np.nan)
    rho = np.empty(phases.size)
    for k, ph in enumerate(phases):
        pos = positions_at(system, ph)
        rho[k] = float(pos.rho)
        try:
            ts = spectral_triple(pos, tri, nm, system, pupils=True)
        except ValueError:
            ts = spectral_triple(pos, tri, nm, system, GridConfig().fit_orbit(system),
                                 method="render", pupils=True)
        cosmap[k] = ts.cos_phi_c

    # spatial frequency of the longest side, in mega-wavelengths (rising to the right)
    u = b_max / (nm * 1e-9) / 1e6
    order = np.argsort(u)
    u, cosmap_u = u[order], cosmap[:, order]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.pcolormesh(u, phases, cosmap_u, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
    fig.colorbar(im, label=r"$\cos\varphi_c$")
    ax.set_xlabel(r"Spatial frequency $B_{\rm UT1\text{-}UT4}/\lambda$  [M$\lambda$]")
    ax.set_ylabel("Orbital phase")
    ax.set_title(f"{system.name}: closure-phase cosine on {tri.name} (aperture-averaged)")
    # secondary axis: the wavelength that gives each spatial frequency
    sec = ax.secondary_xaxis("top", functions=(lambda uu: b_max / (uu * 1e6) * 1e9,
                                               lambda lam: b_max / (lam * 1e-9) / 1e6))
    sec.set_xlabel("Wavelength [nm]")
    sec.set_xticks([400, 450, 500, 600, 700, 800, 950])
    # fringe cycles across the binary separation at maximum separation, for scale
    k = int(np.argmax(rho))
    cycles = u * 1e6 * rho[k] * MAS
    ax.text(0.02, 0.97, f"at max. separation ({rho[k]:.2f} mas): "
            f"{cycles.min():.1f}-{cycles.max():.1f} fringe cycles across the pair on UT1-UT4",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
