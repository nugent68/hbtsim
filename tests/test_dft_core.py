"""The exact K-point DFT sampling core (hbtsim.hbt.dft_points and the
public samplers built on it).

The float32 JAX kernel is compared with an independent float64 numpy
evaluation at the movie baselines and on the longest Maunakea arm
(226 m at 400 nm, ~15 cycles of phase across the grid).  The 1e-6
tolerances FAIL if the matrix products silently run in TF32 (Ampere
GPUs' default fast path, 1e-3 relative error) or if the split-precision
phase is dropped -- that is the point of the test.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hbtsim import hbt
from hbtsim.orbit import SkyPositions, sky_positions
from hbtsim.params import ALGOL, BETA_AUR, MAS, GridConfig, MovieConfig
from hbtsim.render import linear_rows_jnp, render_image, render_kernel

GRID = GridConfig()
LONGEST_ARM_M = 225.9


def _pos(system, psi):
    return SkyPositions(*(np.asarray(v) for v in sky_positions(psi, system)))


def _f32(img):
    """The kernel runs in the image dtype: force float32 so the test
    exercises the GPU path even when another module enabled x64."""
    return jnp.asarray(img, dtype=jnp.float32)


def test_matches_float64_reference_at_movie_baselines():
    pos = _pos(BETA_AUR, 0.3)
    img = _f32(render_image(pos, BETA_AUR, 400.0, GRID))
    b = np.asarray(MovieConfig().baselines_m)
    lam = 400e-9
    bv = hbt.baseline_vectors_along_pa(b, float(pos.pa))
    v = np.asarray(hbt.vis_of_baselines(img, bv, lam, GRID))
    ref = hbt.vis_points_np(img, bv[:, 0] / lam, bv[:, 1] / lam, GRID)
    assert v.dtype == np.complex64
    assert np.abs(np.abs(v) ** 2 - np.abs(ref) ** 2).max() < 1e-6
    assert np.abs(v - ref).max() < 2e-6
    ok = np.abs(ref) > 0.02
    assert np.abs(np.angle(v[ok] * np.conj(ref[ok]))).max() < 1e-5


def test_matches_float64_reference_on_longest_maunakea_arm():
    """226 m at 400 nm, arbitrary orientation: the largest phase the
    package ever asks for."""
    pos = _pos(ALGOL, 0.7)
    img = _f32(render_image(pos, ALGOL, 400.0, GRID))
    lam = 400e-9
    bv = np.array([[LONGEST_ARM_M * np.cos(0.4), LONGEST_ARM_M * np.sin(0.4)],
                   [-LONGEST_ARM_M, 0.0], [0.0, LONGEST_ARM_M]])
    v = np.asarray(hbt.vis_of_baselines(img, bv, lam, GRID))
    ref = hbt.vis_points_np(img, bv[:, 0] / lam, bv[:, 1] / lam, GRID)
    assert np.abs(v - ref).max() < 2e-6
    assert np.abs(np.angle(v * np.conj(ref))).max() < 1e-5


def test_zero_frequency_is_exactly_one():
    img = _f32(render_image(_pos(BETA_AUR, 1.0), BETA_AUR, 800.0, GRID))
    v = hbt.vis_points(img, [0.0], [0.0], GRID)
    assert complex(v[0]) == pytest.approx(1.0 + 0.0j, abs=1e-6)  # f32 rounding


def test_hermitian_symmetry():
    img = _f32(render_image(_pos(ALGOL, 0.2), ALGOL, 600.0, GRID))
    u = np.array([50.0, -120.0, 200.0]) / 600e-9
    v = np.array([-30.0, 80.0, 10.0]) / 600e-9
    a = np.asarray(hbt.vis_points(img, u, v, GRID))
    b = np.asarray(hbt.vis_points(img, -u, -v, GRID))
    assert np.allclose(a, np.conj(b), atol=1e-7)


def test_displaced_point_source_phase():
    """A single-pixel source at offset (dx, dy) px has V = e^{-2 pi i
    (fx dx + fy dy)} exactly."""
    n = GRID.n
    img = jnp.zeros((n, n), jnp.float32)
    dx, dy = 37, -101
    c = (n - 1) // 2  # (n-1)/2 is a half-integer; use the pixel at c + 0.5
    img = img.at[c + dy, c + dx].set(1.0)
    u = np.array([80.0, 150.0, 226.0]) / 400e-9
    v = np.array([-60.0, 20.0, 5.0]) / 400e-9
    got = np.asarray(hbt.vis_points(img, u, v, GRID))
    dth = GRID.pixel_scale_rad
    thx = (c + dx - (n - 1) / 2.0) * dth
    thy = (c + dy - (n - 1) / 2.0) * dth
    expect = np.exp(-2j * np.pi * (u * thx + v * thy))
    assert np.abs(got - expect).max() < 2e-6


def test_origin_shift():
    """origin_rad applies the plane-wave factor of the grid centre's sky
    position."""
    img = _f32(render_image(_pos(BETA_AUR, 0.0), BETA_AUR, 500.0, GRID))
    u = np.array([40.0, 90.0]) / 500e-9
    v = np.array([10.0, -70.0]) / 500e-9
    x0, y0 = 1.3 * MAS, -0.7 * MAS
    a = np.asarray(hbt.vis_points(img, u, v, GRID))
    b = np.asarray(hbt.vis_points(img, u, v, GRID, origin_rad=(x0, y0)))
    assert np.allclose(b, a * np.exp(-2j * np.pi * (u * x0 + v * y0)), atol=1e-6)


def test_frequency_guard():
    img = _f32(render_kernel(0.0, 0.0, 300.0, 300.0, False, 20.0, 1.0,
                             1.0, 0.0, linear_rows_jnp(0.3, GRID.n_mu),
                             linear_rows_jnp(0.0, GRID.n_mu), GRID.n))
    # 0.03 cycles/px is the physical maximum; 0.4 aliases
    with pytest.raises(ValueError, match="cycles/pixel"):
        hbt.vis_points(img, [0.4 / GRID.pixel_scale_rad], [0.0], GRID)


def test_vis2_along_pa_and_g2():
    pos = _pos(BETA_AUR, 0.0)
    img = _f32(render_image(pos, BETA_AUR, 400.0, GRID))
    b = np.array([0.0, 30.0, 60.0])
    v2 = np.asarray(hbt.vis2_along_pa(img, b, 400e-9, float(pos.pa), GRID))
    g2 = np.asarray(hbt.g2_along_pa(img, b, 400e-9, float(pos.pa), GRID))
    assert v2[0] == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(g2, 1.0 + v2)
    assert np.all((v2 >= -1e-7) & (v2 <= 1.0 + 1e-6))


def test_speed_vs_padded_fft(monkeypatch):
    """The DFT at 176 points must be far cheaper than one padded FFT
    (soft check: >= 5x on any machine; it is 50-100x in practice)."""
    import time

    from hbtsim import fftmap

    pos = _pos(BETA_AUR, 0.3)
    img = _f32(render_image(pos, BETA_AUR, 400.0, GRID))
    b = np.arange(0.0, 176.0)
    # warm up both
    hbt.vis2_along_pa(img, b, 400e-9, 0.3, GRID).block_until_ready()
    fftmap.vis2_map(img, GRID.pad).block_until_ready()

    t0 = time.perf_counter()
    for _ in range(3):
        hbt.vis2_along_pa(img, b, 400e-9, 0.3, GRID).block_until_ready()
    t_dft = (time.perf_counter() - t0) / 3
    t0 = time.perf_counter()
    fftmap.vis2_map(img, GRID.pad).block_until_ready()
    t_fft = time.perf_counter() - t0
    assert t_fft / t_dft > 5.0
