"""Band fluxes and magnitudes from the rendered images.

The lightcurve is obtained by summing the band-weighted image (Planck
surface brightness at the band effective wavelength times the limb-darkened,
occulted disks), so it is automatically consistent with the eclipse geometry
on the grid.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def band_flux(img) -> float:
    """Total flux in arbitrary units (pixel solid angle omitted: constant)."""
    return float(jnp.sum(img))


def to_mag(flux: np.ndarray, flux_ref: float) -> np.ndarray:
    """Relative magnitude m = -2.5 log10(F / F_ref)."""
    return -2.5 * np.log10(np.asarray(flux) / flux_ref)
