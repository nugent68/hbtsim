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

Stars are treated as blackbodies, which for these A-type photospheres
makes the synthetic magnitudes too faint and too red: Balmer/Paschen
line blanketing and H- opacity are not modeled, giving offsets of
~0.46 mag in g and ~0.18 mag in i relative to the observed photometry
(g ~ 1.80, i ~ 2.10 from V = 1.90, B-V = 0.03 via Jester et al. 2005).
The lightcurves are therefore anchored: anchored_mags() shifts each
band's synthetic curve by a constant so its maximum light matches the
observed magnitude stored in BinarySystem.mag_anchors.  The eclipse
shapes and depths come entirely from the simulation; only the zero
point per band is set by observation.
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


def anchored_mags(synth_mags: np.ndarray, band: str,
                  system: BinarySystem) -> np.ndarray:
    """Shift a band's synthetic magnitude curve by a constant so that its
    maximum light equals the observed magnitude in system.mag_anchors,
    correcting the blackbody zero-point offset (see module docstring)."""
    anchors = dict(system.mag_anchors)
    m = np.asarray(synth_mags)
    return m - m.min() + anchors[band]
