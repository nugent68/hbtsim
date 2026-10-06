"""Blackbody vs NewEra per-channel photometry of the binaries: what the
model atmospheres change in the fringe contrast.

For each system the surface flux ratio F_NewEra / (pi B_lambda) per star,
the A/B (Aa/Ab) flux ratio f1/f2 = F1 theta1^2 / (F2 theta2^2) with
blackbodies and with NewEra tables, the fringe factor 2 f1 f2/(f1+f2)^2,
and the synthetic g/i magnitudes against the anchors, at a set of
continuum and line-core wavelengths.  Also the spherical extension
R_outer/R_edge and the linear limb-darkening coefficient (mu >= 0.1) of
each table against the Claret & Bloemen value in params.

    .venv/bin/python scripts/sed_compare.py [--newera-dir data/newera]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

from hbtsim.catalog import Catalog
from hbtsim.params import MAS
from hbtsim.sed import NewEraGrid, with_newera
from hbtsim.snr import model_ab_mag

CAT = Catalog(env=False)

LAM = np.array([400.0, 420.0, 450.0, 486.27, 500.0, 550.0, 650.0, 800.0, 900.0])  # vacuum nm; 486.27 = H-beta core


def linear_u(prof, lam):
    """Linear coefficient fitted INSIDE the spherical limb, in the inner
    mu' = sqrt(1 - (r/R_edge)^2) a plane-parallel law refers to (the
    outer-boundary mu of the table overstates the darkening), over
    mu' >= 0.1."""
    mu = np.asarray(prof.mu, dtype=float)
    r_edge = np.sqrt(np.clip(1 - mu**2, 0, 1)) * prof.r_outer        # r / R_edge
    inside = r_edge < 1.0
    mu_in = np.sqrt(np.clip(1 - r_edge**2, 0, 1))
    sel = inside & (mu_in >= 0.1)
    out = []
    for row in prof.rows(lam):
        out.append(np.polyfit(1 - mu_in[sel], 1 - row[sel], 1)[0])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--newera-dir", default="data/newera")
    args = ap.parse_args()
    grid = NewEraGrid.scan(args.newera_dir)
    print(f"NewEra grid: {grid.coverage()}")
    for key in ("betaaur", "deltavel", "algol"):
        sys0 = CAT.load_target(key)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sys1, rep = with_newera(sys0, grid, allow_extrapolation=True)
        print(f"\n=== {sys0.name} ===")
        for name, what in rep.items():
            print(f"  {name}: {what}")
        stars0 = (sys0.primary, sys0.secondary)
        stars1 = (sys1.primary, sys1.secondary)
        th = [sys1.drawn_radius_mas(s) * MAS for s in stars1]
        print(f"  drawn radii: {[round(s.radius_scale, 5) for s in stars1]} x catalogue "
              f"(spherical extension R_outer/R_edge)")
        hdr = "  lambda[nm]  F1/piB1  F2/piB2   f1/f2(BB)  f1/f2(NewEra)  fringe(BB)  fringe(NewEra)  u1(NewEra) u1(C11)  u2(NewEra) u2(C11)"
        print(hdr)
        for lam in LAM:
            f0 = [s.surface_flux(lam) * t**2 for s, t in zip(stars0, th)]
            f1 = [s.surface_flux(lam) * t**2 for s, t in zip(stars1, th)]
            ratio = [s1.surface_flux(lam) / s0.surface_flux(lam) for s0, s1 in zip(stars0, stars1)]
            fr = lambda f: 2 * f[0] * f[1] / (f[0] + f[1]) ** 2
            u_new = [float(linear_u(s.ld_profile, lam)[0]) if s.ld_profile is not None else np.nan for s in stars1]
            u_c11 = [s.ld_coeff(lam) for s in stars0]
            print(f"  {lam:9.2f}  {ratio[0]:7.3f}  {ratio[1]:7.3f}   {f0[0] / f0[1]:9.3f}  {f1[0] / f1[1]:13.3f}  "
                  f"{fr(f0):10.4f}  {fr(f1):14.4f}  {u_new[0]:9.3f} {u_c11[0]:7.3f}  {u_new[1]:9.3f} {u_c11[1]:7.3f}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m0 = model_ab_mag(sys0, np.array([477.0, 763.0]))
            m1 = model_ab_mag(sys1, np.array([477.0, 763.0]))
        anc = dict(sys0.mag_anchors)                 # {wavelength_nm: mag_ab}
        print(f"  g, i: blackbody {m0[0]:.3f}, {m0[1]:.3f}; NewEra {m1[0]:.3f}, {m1[1]:.3f}; "
              f"anchors {anc[477.0]:.2f}, {anc[763.0]:.2f}")


if __name__ == "__main__":
    main()
