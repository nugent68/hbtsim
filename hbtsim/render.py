"""JAX rendering of the limb-darkened binary onto the sky grid.

The image is built in the center-of-mass frame with one pixel = pixel_scale
mas. Each star is a linearly limb-darkened disk whose rim is softened over
one pixel (a coverage factor) to suppress FFT ringing from a hard edge.
Occultation is handled by z-ordering: where the front disk covers a pixel,
the back disk is hidden in proportion to the coverage, which reproduces the
exact partial-eclipse geometry on the grid.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

import numpy as np

from .orbit import SkyPositions
from .params import C_LIGHT, H_PLANCK, K_BOLTZ, BinarySystem, GridConfig, planck


@partial(jax.jit, static_argnames=("n",))
def _render_kernel(x1, y1, x2, y2, front2, r1, r2, w1, w2, u, n):
    """All positions/radii in pixels relative to the grid center; w1, w2 are
    the central surface brightnesses (Planck weights) of each star."""
    c = (n - 1) / 2.0
    coord = jnp.arange(n, dtype=jnp.float32) - c
    xx = coord[None, :]
    yy = coord[:, None]

    def disk(xc, yc, rad):
        r = jnp.hypot(xx - xc, yy - yc)
        mu = jnp.sqrt(jnp.clip(1.0 - (r / rad) ** 2, 0.0, 1.0))
        cover = jnp.clip(rad - r + 0.5, 0.0, 1.0)  # 1-px soft rim
        return (1.0 - u * (1.0 - mu)) * cover, cover

    d1, cover1 = disk(x1, y1, r1)
    d2, cover2 = disk(x2, y2, r2)

    img_2front = w2 * d2 + w1 * d1 * (1.0 - cover2)
    img_1front = w1 * d1 + w2 * d2 * (1.0 - cover1)
    return jnp.where(front2, img_2front, img_1front)


def _planck_jnp(wavelength_m: jax.Array, teff: float) -> jax.Array:
    """JAX twin of params.planck, traceable in wavelength."""
    x = H_PLANCK * C_LIGHT / (wavelength_m * K_BOLTZ * teff)
    return 2.0 * H_PLANCK * C_LIGHT**2 / wavelength_m**5 / jnp.expm1(x)


def spectral_weights(wavelengths_nm, system: BinarySystem):
    """Per-channel limb-darkening coefficient and primary/secondary Planck
    surface-brightness ratio, (u, w1), as traceable float32 arrays — the
    JAX equivalent of what render_image computes per call in Python."""
    lam_nm = jnp.asarray(wavelengths_nm, dtype=jnp.float32)
    ld = np.asarray(system.ld_table_nm, dtype=np.float32)
    u = jnp.interp(lam_nm, jnp.asarray(ld[:, 0]), jnp.asarray(ld[:, 1]))
    lam_m = lam_nm * 1e-9
    w1 = (_planck_jnp(lam_m, system.primary.teff)
          / _planck_jnp(lam_m, system.secondary.teff))
    return u, jnp.asarray(w1, dtype=jnp.float32)


def render_image(pos: SkyPositions, system: BinarySystem, wavelength_nm: float,
                 grid: GridConfig) -> jax.Array:
    """Render the binary at a single epoch (scalar entries in `pos`)."""
    scale = grid.pixel_scale_mas
    lam_m = wavelength_nm * 1e-9
    return _render_kernel(
        jnp.float32(pos.x1 / scale), jnp.float32(pos.y1 / scale),
        jnp.float32(pos.x2 / scale), jnp.float32(pos.y2 / scale),
        jnp.bool_(pos.front2),
        jnp.float32(system.angular_radius_mas(system.primary) / scale),
        jnp.float32(system.angular_radius_mas(system.secondary) / scale),
        jnp.float32(planck(lam_m, system.primary.teff) / planck(lam_m, system.secondary.teff)),
        jnp.float32(1.0),
        jnp.float32(system.ld_coeff(wavelength_nm)),
        grid.n,
    )
