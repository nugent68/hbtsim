"""Two-dimensional |V|^2 and V(u, v) maps by zero-padded FFT.

NOT on the science path.  Every quantity the interferometer model needs
is sampled exactly by hbtsim.hbt (K-point DFT); this module keeps the
older padded-FFT + bilinear-interpolation chain for full 2-D maps of the
visibility plane (plots, exploratory work) and as an independent
cross-check in the tests, which also document its interpolation error:
bilinear sampling of the ~1 m FFT grid damps the binary fringe by up to
a few 1e-3 in |V|^2 (tests/test_fftmap.py).

Chain: image -> zero-padded 2-D real FFT -> normalize by the zero-frequency
value (V(0, 0) = 1) -> central crop -> bilinear interpolation at
(B/lambda) dtheta pad bins along a position angle.  The padded FFT
samples baselines every dB = lambda / (pad dtheta) (~1 m at 400 nm with
the default grid) and keeps +/- crop_half bins: beyond the crop the
sampler returns ZERO (|V|^2 -> 0, g2 -> 1), i.e. it is silently wrong
above ~258 m at 400 nm -- another reason it is not used for science.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .params import GridConfig

CROP_HALF = 256  # central crop half-width of the shifted |V|^2 map (bins)


@partial(jax.jit, static_argnames=("pad", "crop_half"))
def vis2_map(img: jax.Array, pad: int, crop_half: int = CROP_HALF) -> jax.Array:
    """|V|^2 on the central (2*crop_half, crop_half) region of the padded
    real-FFT grid, zero frequency at index [crop_half, 0]; the column axis
    holds only non-negative frequencies (Hermitian symmetry).  Normalized
    by the image sum so the zero-frequency value is exactly 1."""
    F = jnp.fft.rfft2(img, s=(pad, pad))
    vis2 = (jnp.abs(F) / jnp.sum(img)) ** 2
    c = pad // 2
    return jnp.fft.fftshift(vis2, axes=0)[c - crop_half:c + crop_half,
                                          :crop_half]


@partial(jax.jit, static_argnames=("n", "pad", "crop_half"))
def vis_complex_map(img: jax.Array, n: int, pad: int,
                    crop_half: int = CROP_HALF) -> jax.Array:
    """Complex V(u, v) on the same region as vis2_map (|result|^2 equals
    vis2_map).  The phase origin is moved from FFT pixel [0, 0] to the
    image-grid centre ((n-1)/2) by demodulating the map before any
    interpolation."""
    F = jnp.fft.rfft2(img, s=(pad, pad)) / jnp.sum(img)
    c = pad // 2
    F = jnp.fft.fftshift(F, axes=0)[c - crop_half:c + crop_half, :crop_half]
    ctr = (n - 1) / 2.0
    ky = jnp.arange(-crop_half, crop_half, dtype=jnp.float32)[:, None]
    kx = jnp.arange(0, crop_half, dtype=jnp.float32)[None, :]
    return F * jnp.exp(2j * jnp.pi * ctr * (kx + ky) / pad)


def baseline_bin_offset(baselines_m, wavelength_m: float, grid: GridConfig):
    """FFT bin offset from zero frequency for baseline B: (B/lambda) dtheta pad."""
    b = jnp.asarray(baselines_m)
    return b / wavelength_m * grid.pixel_scale_rad * grid.pad


def uv_bins_of_baseline(bvecs_m, wavelength_m, grid: GridConfig):
    """FFT-bin offsets (fx, fy) for baseline vectors (..., 2) in metres."""
    b = jnp.asarray(bvecs_m)
    return b / wavelength_m * grid.pixel_scale_rad * grid.pad


def vis2_of_baseline(vis2: jax.Array, baselines_m, wavelength_m: float,
                     pa_rad: float, grid: GridConfig,
                     crop_half: int = CROP_HALF) -> jax.Array:
    """Bilinear sample of a vis2_map along position angle pa_rad
    (measured from +x)."""
    idx = baseline_bin_offset(baselines_m, wavelength_m, grid)
    dy = idx * jnp.sin(pa_rad)  # FFT axis 0 <-> sky y
    dx = idx * jnp.cos(pa_rad)  # FFT axis 1 <-> sky x (non-negative half)
    sign = jnp.where(dx < 0, -1.0, 1.0)  # Hermitian: |V(-u,-v)| = |V(u,v)|
    row = crop_half + sign * dy
    col = sign * dx
    return jax.scipy.ndimage.map_coordinates(vis2, [row, col], order=1)


def g2_of_baseline(vis2: jax.Array, baselines_m, wavelength_m: float,
                   pa_rad: float, grid: GridConfig,
                   crop_half: int = CROP_HALF) -> jax.Array:
    return 1.0 + vis2_of_baseline(vis2, baselines_m, wavelength_m, pa_rad,
                                  grid, crop_half)


def vis_complex_of_uv(vmap: jax.Array, fx, fy,
                      crop_half: int = CROP_HALF) -> jax.Array:
    """Bilinear sample of a (centred) vis_complex_map at bin offsets
    (fx, fy), Re and Im separately; fx < 0 uses V(-u) = V*(u)."""
    fx = jnp.asarray(fx)
    fy = jnp.asarray(fy)
    sign = jnp.where(fx < 0, -1.0, 1.0)
    row = crop_half + sign * fy
    col = sign * fx
    interp = lambda a: jax.scipy.ndimage.map_coordinates(a, [row, col], order=1)
    v = interp(jnp.real(vmap)) + 1j * interp(jnp.imag(vmap))
    return jnp.where(sign < 0, jnp.conj(v), v)
