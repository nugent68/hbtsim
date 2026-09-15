"""JAX rendering of the limb-darkened binary onto the sky grid.

The image is built in the center-of-mass frame with one pixel = pixel_scale
mas, axis 0 = y (North) and axis 1 = x (East), the grid centre at pixel
(n-1)/2.  Each star is a linearly limb-darkened disk whose rim is
softened over one pixel (a coverage factor).  Occultation is handled by
z-ordering: where the front disk covers a pixel, the back disk is hidden
in proportion to the coverage, which reproduces the exact partial-eclipse
geometry on the grid.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

import numpy as np

from .orbit import SkyPositions
from .params import C_LIGHT, H_PLANCK, K_BOLTZ, BinarySystem, GridConfig, planck


@partial(jax.jit, static_argnames=("n",))
def render_kernel(x1, y1, x2, y2, front2, r1, r2, w1, w2, u1, u2, n):
    """All positions/radii in pixels relative to the grid center; w1, w2 are
    the central surface brightnesses (Planck weights) and u1, u2 the linear
    limb-darkening coefficients of each star."""
    c = (n - 1) / 2.0
    coord = jnp.arange(n, dtype=jnp.float32) - c
    xx = coord[None, :]
    yy = coord[:, None]

    def disk(xc, yc, rad, u):
        r = jnp.hypot(xx - xc, yy - yc)
        mu = jnp.sqrt(jnp.clip(1.0 - (r / rad) ** 2, 0.0, 1.0))
        cover = jnp.clip(rad - r + 0.5, 0.0, 1.0)  # 1-px soft rim
        return (1.0 - u * (1.0 - mu)) * cover, cover

    d1, cover1 = disk(x1, y1, r1, u1)
    d2, cover2 = disk(x2, y2, r2, u2)

    img_2front = w2 * d2 + w1 * d1 * (1.0 - cover2)
    img_1front = w1 * d1 + w2 * d2 * (1.0 - cover1)
    return jnp.where(front2, img_2front, img_1front)


_render_kernel = render_kernel  # backward-compatible private name


def check_extent(pos: SkyPositions, system: BinarySystem,
                 grid: GridConfig) -> None:
    """Raise ValueError if either disk (plus its soft rim) would leave the
    grid at any of the epochs in pos: a clipped disk silently corrupts the
    flux and the visibility."""
    half = grid.half_extent_mas - grid.pixel_scale_mas
    worst = 0.0
    for x, y, star in ((pos.x1, pos.y1, system.primary),
                       (pos.x2, pos.y2, system.secondary)):
        reach = np.maximum(np.abs(np.asarray(x, float)),
                           np.abs(np.asarray(y, float))) \
            + system.angular_radius_mas(star)
        worst = max(worst, float(np.max(reach)))
    if worst > half:
        raise ValueError(
            f"{system.name}: a disk reaches {worst:.2f} mas from the grid "
            f"centre but the {grid.n}-pixel grid at {grid.pixel_scale_mas} "
            f"mas/px only holds {half:.2f} mas; enlarge GridConfig.n or "
            f"coarsen pixel_scale_mas")


def kernel_args(pos: SkyPositions, system: BinarySystem, grid: GridConfig):
    """(x1, y1, x2, y2, front2, r1, r2) for render_kernel at ONE epoch
    (scalar entries in pos), as float32 pixel quantities, after the
    grid-extent check."""
    check_extent(pos, system, grid)
    s = grid.pixel_scale_mas
    return (jnp.float32(pos.x1 / s), jnp.float32(pos.y1 / s),
            jnp.float32(pos.x2 / s), jnp.float32(pos.y2 / s),
            jnp.bool_(pos.front2),
            jnp.float32(system.angular_radius_mas(system.primary) / s),
            jnp.float32(system.angular_radius_mas(system.secondary) / s))


def _planck_jnp(wavelength_m: jax.Array, teff: float) -> jax.Array:
    """JAX twin of params.planck, traceable in wavelength."""
    x = H_PLANCK * C_LIGHT / (wavelength_m * K_BOLTZ * teff)
    return 2.0 * H_PLANCK * C_LIGHT**2 / wavelength_m**5 / jnp.expm1(x)


def spectral_weights(wavelengths_nm, system: BinarySystem):
    """Per-channel per-star limb-darkening coefficients and the
    primary/secondary Planck surface-brightness ratio, (u1, u2, w1), as
    traceable float32 arrays — the JAX equivalent of what render_image
    computes per call in Python."""
    lam_nm = jnp.asarray(wavelengths_nm, dtype=jnp.float32)

    def interp_u(star):
        ld = np.asarray(star.ld_table_nm, dtype=np.float32)
        return jnp.interp(lam_nm, jnp.asarray(ld[:, 0]), jnp.asarray(ld[:, 1]))

    lam_m = lam_nm * 1e-9
    w1 = (_planck_jnp(lam_m, system.primary.teff)
          / _planck_jnp(lam_m, system.secondary.teff))
    return (interp_u(system.primary), interp_u(system.secondary),
            jnp.asarray(w1, dtype=jnp.float32))


def render_image(pos: SkyPositions, system: BinarySystem, wavelength_nm: float,
                 grid: GridConfig) -> jax.Array:
    """Render the binary at a single epoch (scalar entries in `pos`), in
    units of the secondary's central surface brightness (w2 = 1)."""
    lam_m = wavelength_nm * 1e-9
    return render_kernel(
        *kernel_args(pos, system, grid),
        jnp.float32(planck(lam_m, system.primary.teff) / planck(lam_m, system.secondary.teff)),
        jnp.float32(1.0),
        jnp.float32(system.primary.ld_coeff(wavelength_nm)),
        jnp.float32(system.secondary.ld_coeff(wavelength_nm)),
        grid.n,
    )
