"""The padded-FFT 2-D maps (hbtsim.fftmap): cross-checks against the
exact DFT core, and documentation of the interpolation error and crop
limit that took them off the science path."""

import jax
import numpy as np
import pytest

from hbtsim import fftmap, hbt
from hbtsim.orbit import SkyPositions, sky_positions
from hbtsim.params import BETA_AUR, GridConfig, MovieConfig
from hbtsim.render import linear_rows_jnp, render_image, render_kernel

GRID = GridConfig()


def _pos(psi):
    return SkyPositions(*(np.asarray(v) for v in sky_positions(psi, BETA_AUR)))


def test_bilinear_interpolation_vs_exact_dft():
    """FFT + bilinear interpolation agrees with the exact DFT only to a
    few 1e-3 in |V|^2 (the fringe is damped by the interpolation) -- the
    old science-path tolerance, kept here as a record."""
    cfg = MovieConfig()
    pos = _pos(0.3)
    lam = 400e-9
    img = render_image(pos, BETA_AUR, 400.0, GRID)
    v2map = fftmap.vis2_map(img, GRID.pad)
    b = np.asarray(cfg.baselines_m)
    v2_interp = np.asarray(fftmap.vis2_of_baseline(v2map, b, lam, float(pos.pa), GRID))
    v2_exact = np.asarray(hbt.vis2_along_pa(img, b, lam, float(pos.pa), GRID))
    assert np.allclose(v2_interp, v2_exact, atol=5e-3)
    assert not np.allclose(v2_interp, v2_exact, atol=1e-5)


def test_complex_map_modulus_equals_vis2_map():
    img = render_image(_pos(0.0), BETA_AUR, 500.0, GRID)
    v2 = np.asarray(fftmap.vis2_map(img, GRID.pad))
    vc = np.asarray(fftmap.vis_complex_map(img, GRID.n, GRID.pad))
    assert np.allclose(np.abs(vc) ** 2, v2, atol=1e-6)


def test_complex_map_phase_vs_exact_dft():
    """The demodulated complex map, sampled bilinearly, tracks the exact
    DFT phase to ~0.2 deg on the Maunakea triangle."""
    from hbtsim.bispectrum import MAUNAKEA_SUBARU_KECK

    img = render_image(_pos(0.0), BETA_AUR, 500.0, GRID)
    bv = MAUNAKEA_SUBARU_KECK.baseline_vectors()
    vmap = fftmap.vis_complex_map(img, GRID.n, GRID.pad)
    f = np.asarray(fftmap.uv_bins_of_baseline(bv, 500e-9, GRID))
    g = np.asarray(fftmap.vis_complex_of_uv(vmap, f[:, 0], f[:, 1]))
    ref = np.asarray(hbt.vis_of_baselines(img, bv, 500e-9, GRID))
    assert np.degrees(np.abs(np.angle(g * np.conj(ref)))).max() < 0.2
    assert np.abs(np.abs(g) - np.abs(ref)).max() < 5e-3


def test_crop_limit_returns_zero():
    """Beyond the crop the FFT sampler silently returns |V|^2 = 0, while
    the DFT is unaffected: the reason the maps are not used for science."""
    img = render_kernel(0.0, 0.0, 300.0, 300.0, False, 3.0, 1.0,
                        1.0, 0.0, linear_rows_jnp(0.0, GRID.n_mu),
                        linear_rows_jnp(0.0, GRID.n_mu), GRID.n)  # near-point source
    v2map = fftmap.vis2_map(img, GRID.pad)
    b_far = GRID.baseline_step_m(400e-9) * (fftmap.CROP_HALF + 20)
    v2_fft = float(fftmap.vis2_of_baseline(v2map, np.array([b_far]), 400e-9,
                                           0.0, GRID)[0])
    v2_dft = float(hbt.vis2_along_pa(img, np.array([b_far]), 400e-9, 0.0, GRID)[0])
    assert v2_fft == 0.0
    assert v2_dft > 0.9


def test_deprecated_names_still_resolve():
    with pytest.warns(DeprecationWarning):
        fn = hbt.vis2_map
    assert fn is fftmap.vis2_map
