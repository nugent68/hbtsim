"""Batched multi-wavelength |V|^2(B, lambda) via the FFT pipeline.

For spectrally multiplexed intensity interferometry every detector pixel
is its own wavelength channel, so the squared visibility is needed at
hundreds of wavelengths per epoch.  This module runs the existing chain

    render limb-darkened binary (lambda)  ->  zero-padded 2D real FFT
    ->  |V|^2(u, v)  ->  sample at (B/lambda) along the baseline PA

for all channels inside a single jitted JAX computation, so it batches
efficiently on a GPU (and still works, more slowly, on CPU).

Memory is the constraint, not compute: each channel's padded FFT touches
~0.75-1 GiB (8192^2 float32 input + 8192x4097 complex64 spectrum + FFT
workspace), so all 320 SPAD Lambda channels at once would need ~100 GiB.
jax.lax.map(..., batch_size=chunk) processes `chunk` channels as one
vmapped (batched cuFFT) step and scans across chunks, reusing the chunk
buffers: peak memory is one chunk (~16 GiB at chunk=16), while only the
small per-channel samples accumulate.  Default chunk: 16 on GPU
(fits a 40 GiB A100), 4 on CPU.

Everything is forced to float32/complex64, matching GPU behavior even
when the test suite enables x64.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .hbt import CROP_HALF, vis2_map, vis2_of_baseline
from .orbit import SkyPositions
from .params import BinarySystem, GridConfig
from .render import _render_kernel, spectral_weights


def _auto_chunk() -> int:
    return 16 if jax.default_backend() == "gpu" else 4


@partial(jax.jit, static_argnames=("grid", "chunk", "crop_half"))
def _spectral_vis2_jit(x1, y1, x2, y2, front2, r1, r2,
                       u1, u2, w1, lam_m,       # (n_lambda,)
                       baselines_m, pa_rad,     # (n_B,), scalar
                       grid: GridConfig, chunk: int, crop_half: int):
    def one_channel(ch):
        u1_k, u2_k, w1_k, lam_k = ch
        img = _render_kernel(x1, y1, x2, y2, front2, r1, r2,
                             w1_k, jnp.float32(1.0), u1_k, u2_k, grid.n)
        v2map = vis2_map(img, grid.pad, crop_half)
        samp = vis2_of_baseline(v2map, baselines_m, lam_k, pa_rad, grid,
                                crop_half)
        return samp, jnp.sum(img)

    return jax.lax.map(one_channel, (u1, u2, w1, lam_m), batch_size=chunk)


def spectral_vis2(pos: SkyPositions, baselines_m, wavelengths_nm,
                  system: BinarySystem, grid: GridConfig, *,
                  pa_rad: float | None = None,
                  chunk_size: int | None = None,
                  crop_half: int = CROP_HALF,
                  return_flux: bool = False) -> jax.Array:
    """|V|^2 for every (wavelength, baseline) pair at one epoch.

    pos entries must be scalars (a single epoch).  The baseline position
    angle defaults to the projected separation axis, matching the movie
    panel and the analytic binary visibility.  Returns (n_lambda, n_B);
    with return_flux=True also the (n_lambda,) total image flux per
    channel (arbitrary units, eclipse-dimmed -- useful for per-epoch
    photometry without re-rendering).
    """
    scale = grid.pixel_scale_mas
    u1, u2, w1 = spectral_weights(wavelengths_nm, system)
    lam_m = jnp.asarray(wavelengths_nm, dtype=jnp.float32) * 1e-9
    pa = float(pos.pa) if pa_rad is None else float(pa_rad)
    chunk = _auto_chunk() if chunk_size is None else int(chunk_size)
    vis2, flux = _spectral_vis2_jit(
        jnp.float32(pos.x1 / scale), jnp.float32(pos.y1 / scale),
        jnp.float32(pos.x2 / scale), jnp.float32(pos.y2 / scale),
        jnp.bool_(pos.front2),
        jnp.float32(system.angular_radius_mas(system.primary) / scale),
        jnp.float32(system.angular_radius_mas(system.secondary) / scale),
        u1, u2, w1, lam_m,
        jnp.asarray(np.atleast_1d(baselines_m), dtype=jnp.float32),
        jnp.float32(pa),
        grid, chunk, crop_half)
    return (vis2, flux) if return_flux else vis2
