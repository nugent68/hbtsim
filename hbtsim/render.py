"""JAX rendering of the limb-darkened binary onto the sky grid.

The image is built in the center-of-mass frame with one pixel = pixel_scale
mas, axis 0 = y (North) and axis 1 = x (East), the grid centre at pixel
(n-1)/2.  Each star is a limb-darkened disk -- its centre-to-limb
profile I(mu)/I(1) is a table on a uniform mu grid (GridConfig.n_mu),
which holds the linear law 1 - u(1 - mu) exactly and any model
atmosphere profile (Star.ld_profile) to interpolation accuracy -- whose
rim is softened over one (sub-)pixel by a coverage factor.  Occultation
is handled by z-ordering: where the front disk covers a pixel, the back
disk is hidden in proportion to the coverage, which reproduces the
exact partial-eclipse geometry on the grid.

Accuracy.  Sampling the sqrt(1 - r^2) limb at pixel centres biases a
disk's flux and its apparent diameter at the 1e-4 level for a 50-pixel
radius (up to 3e-3 at 15 pixels).  GridConfig.supersample = s renders
each pixel as the mean of s x s sub-pixel-shifted soft-rim renders
(scan-accumulated, so memory does not grow), which reduces the bias
roughly as s^-2; GridConfig.for_system() chooses a pixel scale that
resolves the smaller star with >= 50 pixels.  Images are in units of
the secondary's central intensity (w2 = 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

import numpy as np

from .orbit import SkyPositions
from .params import BinarySystem, GridConfig, linear_ld_rows


@partial(jax.jit, static_argnames=("n", "s"))
def render_kernel(x1, y1, x2, y2, front2, r1, r2, w1, w2, i1, i2, n, s=1):
    """All positions/radii in pixels relative to the grid center; w1, w2
    are the central surface brightnesses of each star and i1, i2 their
    limb-darkening profiles I(mu)/I(1) tabulated on n_mu points uniform
    in mu (n_mu = i1.shape[-1]); s = supersampling factor."""
    c = (n - 1) / 2.0
    coord = jnp.arange(n, dtype=jnp.float32) - c
    mu_grid = jnp.linspace(0.0, 1.0, i1.shape[-1], dtype=jnp.float32)
    inv_s = 1.0 / s

    def sub_render(dx, dy):
        xx = coord[None, :] + dx
        yy = coord[:, None] + dy

        def disk(xc, yc, rad, irow):
            r = jnp.hypot(xx - xc, yy - yc)
            mu = jnp.sqrt(jnp.clip(1.0 - (r / rad) ** 2, 0.0, 1.0))
            cover = jnp.clip((rad - r) * s + 0.5, 0.0, 1.0)  # 1-sub-px soft rim
            return jnp.interp(mu, mu_grid, irow) * cover, cover

        d1, cover1 = disk(x1, y1, r1, i1)
        d2, cover2 = disk(x2, y2, r2, i2)
        img_2front = w2 * d2 + w1 * d1 * (1.0 - cover2)
        img_1front = w1 * d1 + w2 * d2 * (1.0 - cover1)
        return jnp.where(front2, img_2front, img_1front)

    if s == 1:
        return sub_render(0.0, 0.0)
    offs = (jnp.arange(s, dtype=jnp.float32) + 0.5) * inv_s - 0.5
    dxy = jnp.stack(jnp.meshgrid(offs, offs, indexing="ij"), axis=-1).reshape(-1, 2)

    def body(acc, o):
        return acc + sub_render(o[0], o[1]), None

    acc, _ = jax.lax.scan(body, jnp.zeros((n, n), jnp.float32), dxy)
    return acc * (inv_s * inv_s)


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


@dataclass(frozen=True)
class ChannelWeights:
    """Per-channel render inputs: the primary/secondary central-intensity
    ratio w1 (n_lambda,) and the limb-darkening rows i1, i2
    (n_lambda, n_mu) on GridConfig.mu_grid, as float32 JAX arrays."""
    w1: jax.Array
    i1: jax.Array
    i2: jax.Array


def spectral_weights(wavelengths_nm, system: BinarySystem,
                     grid: GridConfig = GridConfig()) -> ChannelWeights:
    """Render inputs for every channel: central-intensity ratio (model
    SED or Planck) and tabulated limb-darkening rows (model profile or
    linear law)."""
    lam = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
    w1 = (system.primary.central_intensity(lam)
          / system.secondary.central_intensity(lam))
    mu = grid.mu_grid
    return ChannelWeights(
        w1=jnp.asarray(w1, dtype=jnp.float32),
        i1=jnp.asarray(system.primary.ld_rows(lam, mu), dtype=jnp.float32),
        i2=jnp.asarray(system.secondary.ld_rows(lam, mu), dtype=jnp.float32))


def render_image(pos: SkyPositions, system: BinarySystem, wavelength_nm: float,
                 grid: GridConfig) -> jax.Array:
    """Render the binary at a single epoch (scalar entries in `pos`), in
    units of the secondary's central intensity (w2 = 1)."""
    cw = spectral_weights([wavelength_nm], system, grid)
    return render_kernel(*kernel_args(pos, system, grid), cw.w1[0],
                         jnp.float32(1.0), cw.i1[0], cw.i2[0], grid.n,
                         grid.supersample)


def linear_rows_jnp(u, n_mu: int) -> jax.Array:
    """Convenience for tests: the linear law on the kernel's mu grid."""
    return jnp.asarray(linear_ld_rows(u, np.linspace(0.0, 1.0, n_mu)),
                       dtype=jnp.float32)
