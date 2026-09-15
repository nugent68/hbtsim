"""Complex visibility and g2(B) from the rendered sky image: exact DFT.

An intensity interferometer needs V(u, v) at a handful of (u, v) points
per spectral channel -- one per baseline, or a few dozen per baseline
once the finite apertures are averaged over.  Those points are evaluated
EXACTLY by a separable K-point discrete Fourier transform of the n x n
image,

    V(u, v) = sum_{y,x} I[y, x] exp(-2 pi i (u theta_x + v theta_y)) / sum I,

    theta_k = (k - (n - 1)/2) dtheta,        (u, v) = (B_E, B_N) / lambda,

which costs two n x n x K real matrix products (~0.4 GFLOP for K = 176,
n = 1024): about a thousand times fewer operations than the zero-padded
8192^2 FFT it replaces (kept in hbtsim.fftmap for 2-D maps and
cross-checks), with no interpolation error, no crop limit on the
baseline, and ~4 MB of working memory per channel instead of ~0.5 GB.

Conventions (shared with render.py, bispectrum.py and the analytic
references): image axis 0 = y = North, axis 1 = x = East; the phase
origin is the grid centre, where render.py puts the centre of mass;
V(0, 0) = 1 to float32 rounding.  Baseline vectors are (East, North) in metres and
B_ij = p_j - p_i.

Precision.  The kernel runs in the image's dtype (float32 on GPU).  The
phase 2 pi f (k - c), with f the frequency in cycles per pixel, reaches
~15 cycles on a 226 m baseline at 400 nm, where a float32 frequency
alone would carry a ~5e-6 rad error; so the host splits f (float64)
into a coarse part f_hi -- an integer multiple of 2^-14, whose product
with the half-integer pixel offset is exact in float32 and is reduced
modulo one cycle -- and a small remainder f_lo.  The phase error is then
~1e-6 rad.  The matrix products use jax.lax.Precision.HIGHEST, which
disables the TF32 fast path on Ampere GPUs (1e-3 relative error
otherwise); tests/test_dft_core.py guards this at the 1e-6 level.

Siegert relation: g2(B) = 1 + |V(B)|^2 is the ideal (fully coherent
detection) normalization.  A real intensity interferometer measures
1 + (tau_c / Delta t) |V|^2 with a contrast set by the ratio of coherence
time to detector resolution (Rai, Basak & Saha 2021, eq. 6); the photon
budget in hbtsim.snr carries that factor.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np

from .params import MAS, GridConfig, require_out_of_eclipse

_SPLIT = 2.0 ** 14          # f_hi granularity (cycles / pixel)
_MAX_CYCLES_PER_PIXEL = 0.25  # guard: well below Nyquist (0.5)


# ---------------------------------------------------------------------------
# Frequencies
# ---------------------------------------------------------------------------
def check_frequency(f_cycles_per_px, n: int) -> None:
    """Reject sampling frequencies the DFT of an n-pixel image cannot
    represent accurately: beyond ~Nyquist the discrete image aliases, and
    beyond f n ~ 1024 the split-precision phase is no longer exact in
    float32.  Physical baselines are far inside both limits (0.03 cycles
    per pixel on 226 m at 400 nm with the default grid)."""
    f = np.abs(np.asarray(f_cycles_per_px, dtype=float))
    if f.size == 0:
        return
    fmax = float(f.max())
    limit = min(_MAX_CYCLES_PER_PIXEL, 1024.0 / n)
    if not fmax <= limit:
        raise ValueError(
            f"sampling frequency {fmax:.3g} cycles/pixel exceeds {limit:.3g}: "
            f"the baseline is too long for this pixel scale (coarsen "
            f"GridConfig.pixel_scale_mas or shorten the baseline)")


def split_frequency(f_cycles_per_px):
    """Host-side float64 split f = f_hi + f_lo with f_hi an integer
    multiple of 2^-14 (see the module docstring)."""
    f = np.asarray(f_cycles_per_px, dtype=np.float64)
    f_hi = np.round(f * _SPLIT) / _SPLIT
    return f_hi, f - f_hi


def _phasor(f_hi, f_lo, k):
    """cos and sin of -2 pi f (k - c) for frequencies (K,) and pixel
    offsets k (n,), with the exact modulo-one reduction of the coarse
    part.  Returns two (K, n) arrays."""
    cyc = f_hi[:, None] * k[None, :]            # exact in float32
    cyc = cyc - jnp.round(cyc) + f_lo[:, None] * k[None, :]
    ang = (-2.0 * jnp.pi) * cyc
    return jnp.cos(ang), jnp.sin(ang)


def dft_points(img, fx_hi, fx_lo, fy_hi, fy_lo, pixel_window: bool = False):
    """V at K frequency points (cycles per pixel along x = axis 1 and
    y = axis 0) of an (n, n) image, phase origin at the grid centre,
    normalized to V(0, 0) = 1.  Traceable: used directly inside the
    batched spectral kernels; vis_points() is the jitted host entry.

    pixel_window=True divides by sinc(pi fx) sinc(pi fy): a supersampled
    (pixel-INTEGRATED) image is the continuous source convolved with the
    pixel box, whose transform is that sinc (1 - 1.5e-3 at 0.03
    cycles/pixel); a plain midpoint-sampled render (GridConfig.
    supersample = 1) carries no such factor."""
    n = img.shape[-1]
    k = jnp.arange(n, dtype=img.dtype) - (n - 1) / 2.0
    # _phasor returns cos and sin of the SIGNED angle -2 pi f (k - c), so
    # the phasor e^{-2 pi i f (k - c)} is cx + i sx
    cx, sx = _phasor(fx_hi, fx_lo, k)          # (K, n)
    cy, sy = _phasor(fy_hi, fy_lo, k)
    hi = jax.lax.Precision.HIGHEST
    # row pass: sum_x I[y, x] (cx + i sx) = rr + i ri, both (n_y, K)
    rr = jnp.dot(img, cx.T, precision=hi)
    ri = jnp.dot(img, sx.T, precision=hi)
    # column pass: sum_y (cy + i sy)(rr + i ri)
    vr = jnp.sum(cy * rr.T - sy * ri.T, axis=1)
    vi = jnp.sum(cy * ri.T + sy * rr.T, axis=1)
    out = jax.lax.complex(vr, vi) / jnp.sum(img)
    if pixel_window:
        out = out / (jnp.sinc(fx_hi + fx_lo) * jnp.sinc(fy_hi + fy_lo))
    return out


_dft_points_jit = jax.jit(dft_points, static_argnames=("pixel_window",))


# ---------------------------------------------------------------------------
# Public sampling API
# ---------------------------------------------------------------------------
def vis_points(img, u, v, grid: GridConfig, origin_rad=None) -> jax.Array:
    """Complex V at spatial frequencies (u, v) [cycles/rad] (arrays of
    equal shape), conjugate to sky x (East) and y (North).  The phase is
    referenced to the grid centre; pass origin_rad = (x0, y0) if the grid
    centre sits at sky position (x0, y0) of a larger frame, in which case
    the result is multiplied by exp(-2 pi i (u x0 + v y0))."""
    u = np.atleast_1d(np.asarray(u, dtype=float))
    v = np.atleast_1d(np.asarray(v, dtype=float))
    shape = u.shape
    fx = u.ravel() * grid.pixel_scale_rad
    fy = v.ravel() * grid.pixel_scale_rad
    check_frequency(fx, img.shape[-1])
    check_frequency(fy, img.shape[-1])
    dt = img.dtype
    args = [jnp.asarray(a, dtype=dt)
            for a in (*split_frequency(fx), *split_frequency(fy))]
    out = _dft_points_jit(img, *args, pixel_window=grid.supersample > 1)
    if origin_rad is not None:
        x0, y0 = origin_rad
        out = out * jnp.exp(-2j * jnp.pi * jnp.asarray(
            u.ravel() * x0 + v.ravel() * y0, dtype=out.real.dtype))
    return out.reshape(shape)


def vis_of_baselines(img, bvecs_m, wavelength_m: float,
                     grid: GridConfig) -> jax.Array:
    """Complex V at baseline vectors (K, 2) [m, (East, North)]."""
    b = np.atleast_2d(np.asarray(bvecs_m, dtype=float))
    return vis_points(img, b[:, 0] / wavelength_m, b[:, 1] / wavelength_m,
                      grid)


def vis2_of_baselines(img, bvecs_m, wavelength_m: float,
                      grid: GridConfig) -> jax.Array:
    return jnp.abs(vis_of_baselines(img, bvecs_m, wavelength_m, grid)) ** 2


def baseline_vectors_along_pa(baselines_m, pa_rad: float) -> np.ndarray:
    """(K, 2) (East, North) vectors of scalar baselines at position angle
    pa_rad measured from +x (East) towards +y (North) -- the convention of
    orbit.SkyPositions.pa, so pa = pos.pa lies along the projected
    separation."""
    b = np.atleast_1d(np.asarray(baselines_m, dtype=float))
    return np.stack([b * np.cos(pa_rad), b * np.sin(pa_rad)], axis=1)


def vis2_along_pa(img, baselines_m, wavelength_m: float, pa_rad: float,
                  grid: GridConfig) -> jax.Array:
    """|V|^2 at scalar baselines along one position angle."""
    return vis2_of_baselines(img, baseline_vectors_along_pa(baselines_m, pa_rad),
                             wavelength_m, grid)


def g2_along_pa(img, baselines_m, wavelength_m: float, pa_rad: float,
                grid: GridConfig) -> jax.Array:
    """Siegert relation: g2(B) = 1 + |V(B)|^2 (ideal normalization)."""
    return 1.0 + vis2_along_pa(img, baselines_m, wavelength_m, pa_rad, grid)


# ---------------------------------------------------------------------------
# float64 numpy reference (tests)
# ---------------------------------------------------------------------------
def vis_points_np(img, u, v, grid: GridConfig) -> np.ndarray:
    """Direct float64 evaluation of vis_points (same conventions,
    including the pixel-window correction for supersampled grids), the
    reference the JAX kernel is tested against."""
    img = np.asarray(img, dtype=np.float64)
    n = img.shape[-1]
    coord = (np.arange(n) - (n - 1) / 2.0) * grid.pixel_scale_rad
    u = np.atleast_1d(np.asarray(u, dtype=float)).ravel()
    v = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
    px = np.exp(-2j * np.pi * np.outer(u, coord))  # (K, n)
    py = np.exp(-2j * np.pi * np.outer(v, coord))
    out = np.einsum("ky,yx,kx->k", py, img, px) / img.sum()
    if grid.supersample > 1:
        out = out / (np.sinc(u * grid.pixel_scale_rad) * np.sinc(v * grid.pixel_scale_rad))
    return out


# ---------------------------------------------------------------------------
# Analytic binary (validation / fast out-of-eclipse path)
# ---------------------------------------------------------------------------
def binary_vis2_analytic(baselines_m, wavelength_nm, system,
                         rho_mas: float) -> np.ndarray:
    """|V|^2 along the separation axis for two non-overlapping limb-darkened
    disks at projected separation rho (Rai, Basak & Saha 2021, eq. 5):

        |V|^2 = [f1^2 V1^2 + f2^2 V2^2 + 2 f1 f2 V1 V2 cos(2 pi B rho/lam)]
                / (f1 + f2)^2.

    wavelength_nm may be a scalar (returns (n_B,)) or an array (returns
    (n_lambda, n_B)).  Raises ValueError in (or near) eclipse, where the
    disks overlap and the formula is invalid."""
    from .limbdark import star_disk_visibility

    require_out_of_eclipse(system, rho_mas)
    scalar_lam = np.ndim(wavelength_nm) == 0
    lam_nm = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))[:, None]
    lam = lam_nm * 1e-9
    b = np.atleast_1d(np.asarray(baselines_m, dtype=float))[None, :]
    rho_rad = rho_mas * MAS

    stars = (system.primary, system.secondary)
    th = [2.0 * system.angular_radius_mas(s) * MAS for s in stars]
    # per-star flux weights F_s theta_s^2 (model SED or pi B_lambda): the
    # disk factors 2 int I mu dmu do not cancel between different stars
    f = [np.atleast_1d(s.surface_flux(lam_nm[:, 0]))[:, None] * t**2
         for s, t in zip(stars, th)]
    v = [star_disk_visibility(s, np.pi * t * b / lam, lam_nm[:, 0])
         for s, t in zip(stars, th)]
    fringe = np.cos(2.0 * np.pi * b * rho_rad / lam)
    out = (f[0]**2 * v[0]**2 + f[1]**2 * v[1]**2
           + 2.0 * f[0] * f[1] * v[0] * v[1] * fringe) / (f[0] + f[1])**2
    return out[0] if scalar_lam else out


# ---------------------------------------------------------------------------
# Deprecated FFT-map names (moved to hbtsim.fftmap)
# ---------------------------------------------------------------------------
_MOVED = ("CROP_HALF", "vis2_map", "baseline_bin_offset", "vis2_of_baseline",
          "g2_of_baseline")


def __getattr__(name):
    if name in _MOVED:
        warnings.warn(f"hbtsim.hbt.{name} moved to hbtsim.fftmap (2-D maps "
                      f"only); the science path is hbt.vis2_along_pa / "
                      f"hbt.vis_of_baselines", DeprecationWarning, stacklevel=2)
        from . import fftmap
        return getattr(fftmap, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
