"""Band fluxes and magnitudes from the rendered images.

The lightcurve is obtained by summing the band-weighted image (Planck
surface brightness at the band effective wavelength times the limb-darkened,
occulted disks), so it is automatically consistent with the eclipse geometry
on the grid.

Apparent magnitudes are synthetic monochromatic AB magnitudes at the band
effective wavelength: the rendered image is in units of the secondary's
central intensity I_2(1) (render.py sets w2 = 1; a blackbody unless the
star carries a model-atmosphere flux table), so

    f_lambda = I_2(1) * sum(img) * Omega_pixel   [W m^-2 m^-1]
    f_nu     = f_lambda * lambda^2 / c                 [W m^-2 Hz^-1]
    m_AB     = -2.5 log10(f_nu / 3631 Jy).

Without model atmospheres the stars are blackbodies, which for these
A-type photospheres makes the synthetic magnitudes too faint and too
red (Balmer/Paschen line blanketing and H- opacity are not modeled; a
few tenths of a magnitude for Beta Aur).  The lightcurves are therefore
anchored: anchored_mags() shifts each band's synthetic curve by a
constant so its maximum light matches the observed magnitude stored in
BinarySystem.mag_anchors.  The eclipse shapes and depths come entirely
from the simulation; only the zero point per band is set by observation.

When both stars carry model-atmosphere flux tables
(BinarySystem.has_sed_tables) the synthetic magnitudes are used as they
are -- the same convention as snr.system_ab_mag -- and the anchors only
raise a warning when the model misses them by more than
params.ANCHOR_CHECK_MAG.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import warnings

from .params import (AB_ZERO_FNU, ANCHOR_CHECK_MAG, C_LIGHT, BinarySystem,
                     GridConfig, planck)


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
    f_lam = (system.secondary.central_intensity(wavelength_nm)
             * np.asarray(img_flux) * grid.pixel_scale_rad**2)
    f_nu = f_lam * lam_m**2 / C_LIGHT
    return -2.5 * np.log10(f_nu / AB_ZERO_FNU)


def max_light_mag(system: BinarySystem, wavelength_nm: float,
                  grid: GridConfig) -> float:
    """Synthetic AB magnitude of the system at maximum light: rendered at
    the epoch of largest projected separation (out of eclipse by
    construction), on the given grid."""
    from .orbit import max_separation_phase, positions_at
    from .render import render_image

    pos = positions_at(system, max_separation_phase(system))
    flux = band_flux(render_image(pos, system, wavelength_nm, grid))
    return float(apparent_ab_mag(flux, wavelength_nm, system, grid))


def anchor_table(system: BinarySystem, band: str) -> dict:
    """{band: observed mag} for the requested band, accepting anchors keyed
    by band name or by wavelength [nm] (snr.BAND_LAMBDA_NM maps the two)."""
    from .snr import BAND_LAMBDA_NM
    out = {}
    for key, mag in system.mag_anchors:
        if isinstance(key, str):
            out[key] = mag
        elif band in BAND_LAMBDA_NM and float(key) == BAND_LAMBDA_NM[band]:
            out[band] = mag
    return out


def anchored_mags(synth_mags: np.ndarray, band: str, system: BinarySystem,
                  reference_mag: float | None = None) -> np.ndarray:
    """Shift a band's synthetic magnitude curve by a constant so that
    maximum light equals the observed magnitude in system.mag_anchors,
    correcting the blackbody zero-point offset (see module docstring).
    reference_mag is the synthetic magnitude at maximum light (from
    max_light_mag, independent of which epochs the curve samples); if
    None the curve's own minimum is used (legacy behaviour, which
    rectifies render jitter and depends on the phase window)."""
    anchors = anchor_table(system, band)
    m = np.asarray(synth_mags)
    ref = float(m.min()) if reference_mag is None else float(reference_mag)
    if system.has_sed_tables:
        if band in anchors and abs(ref - anchors[band]) > ANCHOR_CHECK_MAG:
            warnings.warn(f"{system.name}: the model-atmosphere {band}-band "
                          f"magnitude at maximum light ({ref:.2f}) misses the "
                          f"observed anchor ({anchors[band]:.2f}) by more than "
                          f"{ANCHOR_CHECK_MAG} mag (radii, distance or third "
                          f"light?); not anchoring", stacklevel=2)
        return m
    return m - ref + anchors[band]
