"""Squared visibility and g2(B) from the rendered sky image.

Chain: image -> zero-padded 2D FFT -> normalize by the zero-frequency value
(so V(0,0) = 1 exactly) -> |V|^2(u, v) -> bilinear interpolation along the
chosen baseline direction -> Siegert relation g2(B) = 1 + |V(B)|^2.

The padded real FFT samples baselines every dB = lambda / (pad * dtheta)
(about 1 m at 400 nm with the default grid), and only the central
+/- crop_half frequency bins are kept: 160 m at 400 nm is ~160 bins.
Negative-frequency sample points are mapped onto the stored non-negative
half-plane via the Hermitian symmetry of the FFT of a real image,
|V(-u, -v)| = |V(u, v)|.

Note: g2 = 1 + |V|^2 is the ideal (fully coherent detection) normalization.
A real intensity interferometer measures 1 + (tau_c/Delta t)|V|^2 with a
contrast set by the ratio of coherence time to detector resolution
(e.g. Rai, Basak & Saha 2021, eq. 6).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .params import GridConfig

CROP_HALF = 256  # central crop half-width of the shifted |V|^2 map (bins)


@partial(jax.jit, static_argnames=("pad", "crop_half"))
def vis2_map(img: jax.Array, pad: int, crop_half: int = CROP_HALF) -> jax.Array:
    """|V|^2 on the central (2*crop_half, crop_half) region of the padded
    real-FFT grid, with zero frequency at index [crop_half, 0]; the column
    axis holds only non-negative frequencies (Hermitian symmetry)."""
    F = jnp.fft.rfft2(img, s=(pad, pad))
    vis2 = (jnp.abs(F) / jnp.sum(img)) ** 2
    c = pad // 2
    return jnp.fft.fftshift(vis2, axes=0)[c - crop_half:c + crop_half,
                                          :crop_half]


def baseline_bin_offset(baselines_m, wavelength_m: float, grid: GridConfig):
    """FFT bin offset from zero frequency for baseline B: (B/lambda)*dtheta*pad."""
    b = jnp.asarray(baselines_m)
    return b / wavelength_m * grid.pixel_scale_rad * grid.pad


def vis2_of_baseline(vis2: jax.Array, baselines_m, wavelength_m: float,
                     pa_rad: float, grid: GridConfig,
                     crop_half: int = CROP_HALF) -> jax.Array:
    """Sample |V|^2 along a baseline at position angle pa_rad (sky x-y plane,
    measured from the +x axis) by bilinear interpolation."""
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
    """Siegert relation: g2(B) = 1 + |V(B)|^2 (ideal normalization)."""
    return 1.0 + vis2_of_baseline(vis2, baselines_m, wavelength_m, pa_rad,
                                  grid, crop_half)


def binary_vis2_analytic(baselines_m, wavelength_nm: float, system,
                         rho_mas: float) -> np.ndarray:
    """|V|^2 along the separation axis for two non-overlapping limb-darkened
    disks at projected separation rho (Rai, Basak & Saha 2021, eq. 5):

        |V|^2 = [f1^2 V1^2 + f2^2 V2^2 + 2 f1 f2 V1 V2 cos(2 pi B rho/lam)]
                / (f1 + f2)^2.

    Much faster than the FFT pipeline (no rendering) and validated against
    it to <0.5% in tests/test_sanity.py; NOT valid during eclipses, where
    the disks overlap."""
    from .limbdark import visibility_ld_disk
    from .params import MAS, planck

    lam = wavelength_nm * 1e-9
    b = np.atleast_1d(np.asarray(baselines_m, dtype=float))
    rho_rad = rho_mas * MAS

    stars = (system.primary, system.secondary)
    th = [2.0 * system.angular_radius_mas(s) * MAS for s in stars]
    us = [s.ld_coeff(wavelength_nm) for s in stars]
    # per-star flux weights: with different u's the (1 - u/3) disk factors
    # no longer cancel in the normalization
    f = [planck(lam, s.teff) * t**2 * (1.0 - u_s / 3.0)
         for s, t, u_s in zip(stars, th, us)]
    v = [visibility_ld_disk(np.pi * t * b / lam, u_s)
         for t, u_s in zip(th, us)]
    fringe = np.cos(2.0 * np.pi * b * rho_rad / lam)
    return (f[0]**2 * v[0]**2 + f[1]**2 * v[1]**2
            + 2.0 * f[0] * f[1] * v[0] * v[1] * fringe) / (f[0] + f[1])**2


def vis2_direct(img: np.ndarray, baselines_m, wavelength_m: float,
                pa_rad: float, grid: GridConfig) -> np.ndarray:
    """Direct DFT evaluation of |V|^2 at exact (u, v) points; slow reference
    implementation used to validate the FFT + interpolation path in tests."""
    img = np.asarray(img, dtype=float)
    n = img.shape[0]
    coord = (np.arange(n) - (n - 1) / 2.0) * grid.pixel_scale_rad
    b = np.atleast_1d(np.asarray(baselines_m, dtype=float))
    u = b * np.cos(pa_rad) / wavelength_m  # cycles/rad, conjugate to sky x
    v = b * np.sin(pa_rad) / wavelength_m  # conjugate to sky y
    px = np.exp(-2j * np.pi * np.outer(u, coord))  # (nb, n)
    py = np.exp(-2j * np.pi * np.outer(v, coord))
    vis = np.einsum("by,yx,bx->b", py, img, px) / img.sum()
    return np.abs(vis) ** 2
