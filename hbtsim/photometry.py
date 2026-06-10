"""Band fluxes and magnitudes from the rendered images.

The lightcurve is obtained by summing the band-weighted image (Planck
surface brightness at the band effective wavelength times the limb-darkened,
occulted disks), so it is automatically consistent with the eclipse geometry
on the grid.

Apparent magnitudes are synthetic monochromatic AB magnitudes at the band
effective wavelength: the rendered image is in units of the secondary's
Planck surface brightness (render.py sets w2 = 1), so

    f_lambda = B_lambda(T2) * sum(img) * Omega_pixel   [W m^-2 m^-1]
    f_nu     = f_lambda * lambda^2 / c                 [W m^-2 Hz^-1]
    m_AB     = -2.5 log10(f_nu / 3631 Jy).

Stars are treated as blackbodies, so expect ~0.2-0.3 mag offsets from
observed photometry of real (line-blanketed) A-star spectra.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .params import AB_ZERO_FNU, C_LIGHT, BinarySystem, GridConfig, planck


def band_flux(img) -> float:
    """Total flux in arbitrary units (pixel solid angle omitted: constant)."""
    return float(jnp.sum(img))


def to_mag(flux: np.ndarray, flux_ref: float) -> np.ndarray:
    """Relative magnitude m = -2.5 log10(F / F_ref)."""
    return -2.5 * np.log10(np.asarray(flux) / flux_ref)


def apparent_ab_mag(img_flux: np.ndarray, wavelength_nm: float,
                    system: BinarySystem, grid: GridConfig) -> np.ndarray:
    """Apparent AB magnitude from image sums (see module docstring)."""
    lam_m = wavelength_nm * 1e-9
    f_lam = planck(lam_m, system.secondary.teff) * np.asarray(img_flux) \
        * grid.pixel_scale_rad**2
    f_nu = f_lam * lam_m**2 / C_LIGHT
    return -2.5 * np.log10(f_nu / AB_ZERO_FNU)
