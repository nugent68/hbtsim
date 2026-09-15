"""Model-atmosphere hooks: flux tables and centre-to-limb profiles.

A Star may carry a FluxTable (surface flux F_lambda) and an LDProfile
(I(mu, lambda)/I(1, lambda)) in place of the blackbody + linear
limb-darkening defaults (hbtsim.params).  This module builds them:

  * from NewEra PHOENIX HSR-RF files (Hauschildt et al. 2025, A&A 698,
    A47; the angle-resolved "RF" variant provided on request by the
    authors): read_newera_hsr_rf() reads /PHOENIX_RF/{wl, flux, mu,
    Intensities}, restricts to a wavelength range and bins to a common
    step; scripts/prepare_newera.py writes the result to a small .npz
    that load_star_tables() reads back (no h5py needed at run time).
  * synthetic tables for tests and cross-checks: planck_flux_table()
    (F = pi B_lambda), linear_ld_profile() (the linear law tabulated).

NewEra conventions (checked on the example HSR-RF file): wavelengths in
vacuum Angstrom; flux is log10 of F_lambda in erg s^-1 cm^-2 cm^-1 at
the surface (1 erg s^-1 cm^-2 cm^-1 = 0.1 W m^-2 m^-1); the 127 mu
values are 64 core rays plus 63 tangent rays of the spherical model,
with Intensities (n_wl, 127) in the same units per steradian.  The
intensity table is normalized here to I(mu = 1) so that the renderer's
central-intensity weights come from the flux table alone
(Star.central_intensity = F / (pi x 2 int I mu dmu)).
"""

from __future__ import annotations

import numpy as np

from .params import FluxTable, LDProfile, linear_ld_rows, planck

CGS_FLAM_TO_SI = 0.1     # erg s^-1 cm^-2 cm^-1 -> W m^-2 m^-1


# ---------------------------------------------------------------------------
# Synthetic tables
# ---------------------------------------------------------------------------
def planck_flux_table(teff: float, wavelength_nm) -> FluxTable:
    """F_lambda = pi B_lambda(T_eff) on the given grid [nm]."""
    lam = np.asarray(wavelength_nm, dtype=float)
    return FluxTable(wavelength_nm=lam, flux=np.pi * planck(lam * 1e-9, teff),
                     source=f"blackbody {teff:g} K")


def linear_ld_profile(ld_table_nm, wavelength_nm, n_mu: int = 64) -> LDProfile:
    """The linear law 1 - u(lambda)(1 - mu) tabulated on a mu grid."""
    lam = np.asarray(wavelength_nm, dtype=float)
    tab_lam, tab_u = zip(*ld_table_nm)
    u = np.interp(lam, tab_lam, tab_u)
    mu = np.linspace(0.0, 1.0, n_mu)
    return LDProfile(mu=mu, wavelength_nm=lam, intensity=linear_ld_rows(u, mu),
                     source="linear law")


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------
def bin_to_step(wavelength_nm, values, lam_min_nm, lam_max_nm, step_nm):
    """Mean of `values` (n_wl, ...) in bins of step_nm over
    [lam_min, lam_max]; returns (bin_centres, binned).  Empty bins are
    filled by interpolation from their neighbours."""
    lam = np.asarray(wavelength_nm, dtype=float)
    vals = np.asarray(values)
    edges = np.arange(lam_min_nm, lam_max_nm + step_nm / 2, step_nm)
    idx = np.searchsorted(edges, lam, side="right") - 1
    ok = (idx >= 0) & (idx < edges.size - 1)
    idx, v = idx[ok], vals[ok]
    n = edges.size - 1
    counts = np.bincount(idx, minlength=n)
    flat = v.reshape(v.shape[0], -1)
    sums = np.stack([np.bincount(idx, weights=flat[:, j], minlength=n)
                     for j in range(flat.shape[1])], axis=1)
    out = np.full(sums.shape, np.nan)
    good = counts > 0
    out[good] = sums[good] / counts[good, None]
    centres = 0.5 * (edges[:-1] + edges[1:])
    if not good.all():
        for j in range(out.shape[1]):
            out[~good, j] = np.interp(centres[~good], centres[good], out[good, j])
    return centres, out.reshape((n,) + v.shape[1:])


# ---------------------------------------------------------------------------
# NewEra HSR-RF files
# ---------------------------------------------------------------------------
def read_newera_hsr_rf(path: str, lam_min_nm: float = 380.0,
                       lam_max_nm: float = 1000.0, step_nm: float = 0.02,
                       chunk: int = 200_000) -> dict:
    """Read a NewEra HSR-RF HDF5 file (requires h5py) and bin the flux
    and the angle-resolved intensities to step_nm over the range.
    Returns dict(wavelength_nm, flux [W m^-2 m^-1], mu, intensity
    [normalized to mu = 1], teff/logg/meta when present)."""
    import h5py

    with h5py.File(path, "r") as h:
        g = h["PHOENIX_RF"]
        wl = np.asarray(g["wl"][()], dtype=float) / 10.0     # vacuum A -> nm
        mu = np.asarray(g["mu"][()], dtype=float)
        lo = int(np.searchsorted(wl, lam_min_nm))
        hi = int(np.searchsorted(wl, lam_max_nm, side="right"))
        cen = None
        flux_acc, int_acc, cnt_acc = None, None, None
        for a in range(lo, hi, chunk):
            b = min(a + chunk, hi)
            w = wl[a:b]
            f = 10.0 ** np.asarray(g["flux"][a:b], dtype=float) * CGS_FLAM_TO_SI
            inten = np.asarray(g["Intensities"][a:b, :], dtype=float)
            edges = np.arange(lam_min_nm, lam_max_nm + step_nm / 2, step_nm)
            idx = np.searchsorted(edges, w, side="right") - 1
            n = edges.size - 1
            ok = (idx >= 0) & (idx < n)
            idx = idx[ok]
            if cen is None:
                cen = 0.5 * (edges[:-1] + edges[1:])
                flux_acc = np.zeros(n)
                int_acc = np.zeros((n, mu.size))
                cnt_acc = np.zeros(n)
            flux_acc += np.bincount(idx, weights=f[ok], minlength=n)
            for j in range(mu.size):
                int_acc[:, j] += np.bincount(idx, weights=inten[ok, j], minlength=n)
            cnt_acc += np.bincount(idx, minlength=n)
        meta = {k: g.attrs[k] for k in g.attrs} if hasattr(g, "attrs") else {}
    good = cnt_acc > 0
    flux = np.full(cen.size, np.nan)
    flux[good] = flux_acc[good] / cnt_acc[good]
    inten = np.full((cen.size, mu.size), np.nan)
    inten[good] = int_acc[good] / cnt_acc[good, None]
    if not good.all():
        flux[~good] = np.interp(cen[~good], cen[good], flux[good])
        for j in range(mu.size):
            inten[~good, j] = np.interp(cen[~good], cen[good], inten[good, j])
    order = np.argsort(mu)
    mu = mu[order]
    inten = inten[:, order]
    centre = inten[:, np.argmax(mu)]
    norm = inten / np.maximum(centre[:, None], 1e-300)
    return dict(wavelength_nm=cen, flux=flux, mu=mu, intensity=norm,
                meta=meta, path=path)


def save_star_tables(path_npz: str, tables: dict) -> None:
    np.savez_compressed(path_npz, wavelength_nm=tables["wavelength_nm"],
                        flux=tables["flux"], mu=tables["mu"],
                        intensity=tables["intensity"].astype(np.float32),
                        source=np.str_(tables.get("path", "")))


def load_star_tables(path_npz: str) -> tuple:
    """(FluxTable, LDProfile) from a prepare_newera.py output."""
    d = np.load(path_npz, allow_pickle=False)
    src = str(d["source"]) if "source" in d else path_npz
    ft = FluxTable(wavelength_nm=d["wavelength_nm"], flux=d["flux"], source=src)
    ld = LDProfile(mu=d["mu"], wavelength_nm=d["wavelength_nm"],
                   intensity=np.asarray(d["intensity"], dtype=float), source=src)
    return ft, ld


def with_tables(star, flux_table: FluxTable | None = None,
                ld_profile: LDProfile | None = None):
    """A copy of a Star carrying the given tables."""
    from dataclasses import replace
    return replace(star, flux_table=flux_table, ld_profile=ld_profile)
