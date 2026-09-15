"""Batched multi-wavelength complex visibilities V(B, lambda).

For spectrally multiplexed intensity interferometry every detector pixel
is its own wavelength channel, so the visibility is needed at hundreds
to thousands of wavelengths per epoch.  This module fuses, per channel,

    render limb-darkened binary (lambda)  ->  exact K-point DFT at the
    requested baseline vectors (hbtsim.hbt.dft_points)

inside one jitted computation: jax.lax.map(..., batch_size=chunk)
processes `chunk` channels as one vmapped step (batched renders and
matrix products, which is what makes a GPU efficient) and scans across
chunks, reusing the chunk buffers.  Per channel the working set is a
few n^2 float32 temporaries (~30 MB at n = 1024), so the default chunk
(64 on GPU, 8 on CPU) costs ~2 GB / 0.25 GB; only the small per-channel
samples accumulate.  The channel list is padded to a multiple of the
chunk (repeating the last channel) so the map body is traced once.

Everything is forced to float32/complex64, matching GPU behaviour even
when the test suite enables x64; the DFT phase is split-precision (see
hbtsim.hbt) so the 226 m Maunakea arm is still exact to ~1e-6 rad.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .aperture import resolve_pupils
from .hbt import baseline_vectors_along_pa, check_frequency, dft_points, split_frequency
from .orbit import SkyPositions
from .params import BinarySystem, GridConfig
from .render import kernel_args, render_kernel, spectral_weights


def _auto_chunk() -> int:
    return 64 if jax.default_backend() == "gpu" else 8


def _pad_to_chunk(n_lambda: int, chunk_size: int | None):
    chunk = _auto_chunk() if chunk_size is None else int(chunk_size)
    chunk = max(1, min(chunk, n_lambda))
    n_pad = -(-n_lambda // chunk) * chunk
    return chunk, n_pad


@partial(jax.jit, static_argnames=("n", "chunk"))
def _spectral_vis_jit(x1, y1, x2, y2, front2, r1, r2,
                      u1, u2, w1,                     # (n_pad,)
                      fx_hi, fx_lo, fy_hi, fy_lo,     # (n_pad, K)
                      n: int, chunk: int):
    def one_channel(ch):
        u1_k, u2_k, w1_k, fxh, fxl, fyh, fyl = ch
        img = render_kernel(x1, y1, x2, y2, front2, r1, r2,
                            w1_k, jnp.float32(1.0), u1_k, u2_k, n)
        return dft_points(img, fxh, fxl, fyh, fyl), jnp.sum(img)

    return jax.lax.map(one_channel, (u1, u2, w1, fx_hi, fx_lo, fy_hi, fy_lo),
                       batch_size=chunk)


def spectral_vis(pos: SkyPositions, bvecs_m, wavelengths_nm,
                 system: BinarySystem, grid: GridConfig, *,
                 chunk_size: int | None = None,
                 return_flux: bool = False):
    """Complex V for every (wavelength, baseline vector) at one epoch.

    pos entries must be scalars (a single epoch); bvecs_m is (K, 2)
    [m, (East, North)].  Returns complex64 (n_lambda, K); with
    return_flux=True also the (n_lambda,) total image flux per channel
    (arbitrary units, eclipse-dimmed -- useful for per-epoch photometry
    without re-rendering).  Valid at all phases including eclipses.
    """
    b = np.atleast_2d(np.asarray(bvecs_m, dtype=float))          # (K, 2)
    lam = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))  # (n_l,)
    n_l = lam.size
    f = b[None, :, :] / (lam[:, None, None] * 1e-9) * grid.pixel_scale_rad
    check_frequency(f, grid.n)

    chunk, n_pad = _pad_to_chunk(n_l, chunk_size)
    pad_idx = np.minimum(np.arange(n_pad), n_l - 1)
    u1, u2, w1 = spectral_weights(lam[pad_idx], system)
    fx_hi, fx_lo = split_frequency(f[pad_idx, :, 0])
    fy_hi, fy_lo = split_frequency(f[pad_idx, :, 1])
    f32 = lambda a: jnp.asarray(a, dtype=jnp.float32)

    vis, flux = _spectral_vis_jit(
        *kernel_args(pos, system, grid), u1, u2, w1,
        f32(fx_hi), f32(fx_lo), f32(fy_hi), f32(fy_lo), grid.n, chunk)
    vis, flux = vis[:n_l], flux[:n_l]
    return (vis, flux) if return_flux else vis


def spectral_vis2(pos: SkyPositions, baselines_m, wavelengths_nm,
                  system: BinarySystem, grid: GridConfig, *,
                  pa_rad: float | None = None,
                  chunk_size: int | None = None,
                  return_flux: bool = False,
                  pupils=None):
    """|V|^2 for every (wavelength, scalar baseline) pair at one epoch,
    (n_lambda, n_B).  The baseline position angle defaults to the
    projected separation axis (matching the movie panel and the analytic
    binary visibility).  pupils = (d1, d2) [m] or an aperture.
    PupilQuadrature averages |V|^2 over the two telescope apertures
    (what a correlator measures); None samples at a point.  See
    spectral_vis for return_flux."""
    pa = float(pos.pa) if pa_rad is None else float(pa_rad)
    bvecs = baseline_vectors_along_pa(baselines_m, pa)
    quad = resolve_pupils(pupils, None)
    pts = bvecs if quad is None else quad.points(bvecs).reshape(-1, 2)
    out = spectral_vis(pos, pts, wavelengths_nm, system, grid,
                       chunk_size=chunk_size, return_flux=return_flux)
    vis, flux = out if return_flux else (out, None)
    vis2 = np.asarray(jnp.abs(vis) ** 2)
    if quad is not None:
        vis2 = quad.reduce(vis2.reshape(vis2.shape[0], bvecs.shape[0], -1))
    return (vis2, np.asarray(flux)) if return_flux else vis2
