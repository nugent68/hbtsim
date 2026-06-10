"""Tests of the batched multi-wavelength FFT pipeline (hbtsim.spectral)."""

import numpy as np
import pytest

from hbtsim import hbt
from hbtsim.orbit import SkyPositions, sky_positions
from hbtsim.params import BETA_AUR, GridConfig
from hbtsim.render import render_image
from hbtsim.snr import Spectrograph, spectral_g2_snr
from hbtsim.spectral import spectral_vis2

GRID = GridConfig()
BASELINES = np.arange(10.0, 151.0, 10.0)


def _pos(psi):
    return SkyPositions(*(np.asarray(v) for v in sky_positions(psi, BETA_AUR)))


def test_batched_matches_single_channel_pipeline():
    """The lax.map path reproduces render_image + vis2_map + vis2_of_baseline
    channel by channel (same float32 chain, so near machine precision)."""
    pos = _pos(0.3)
    batched = np.asarray(spectral_vis2(pos, BASELINES, [400.0, 800.0],
                                       BETA_AUR, GRID, chunk_size=2))
    for j, lam_nm in enumerate((400.0, 800.0)):
        img = render_image(pos, BETA_AUR, lam_nm, GRID)
        v2map = hbt.vis2_map(img, GRID.pad)
        ref = np.asarray(hbt.vis2_of_baseline(v2map, BASELINES, lam_nm * 1e-9,
                                              float(pos.pa), GRID))
        assert np.allclose(batched[j], ref, atol=1e-4)


def test_batched_matches_analytic_out_of_eclipse():
    pos = _pos(0.0)
    nm = np.linspace(410.0, 940.0, 16)
    v2 = np.asarray(spectral_vis2(pos, BASELINES, nm, BETA_AUR, GRID,
                                  chunk_size=4))
    for j, lam_nm in enumerate(nm):
        ana = hbt.binary_vis2_analytic(BASELINES, float(lam_nm), BETA_AUR,
                                       float(pos.rho))
        assert np.allclose(v2[j], ana, atol=5e-3), f"channel {lam_nm} nm"


def test_batched_in_eclipse_sane():
    """Mid-eclipse (phase 0.25): overlapping disks, where the analytic
    formula is invalid -- the FFT path must still give |V|^2(0) = 1 and
    values within [0, 1]."""
    pos = _pos(np.pi / 2)
    b = np.concatenate([[0.0], BASELINES])
    v2 = np.asarray(spectral_vis2(pos, b, [450.0, 700.0], BETA_AUR, GRID,
                                  chunk_size=2))
    assert np.allclose(v2[:, 0], 1.0, atol=1e-5)
    assert np.all((v2 >= -1e-6) & (v2 <= 1.0 + 1e-5))


def test_chunk_size_invariance():
    pos = _pos(0.0)
    nm = np.linspace(420.0, 900.0, 10)
    a = np.asarray(spectral_vis2(pos, BASELINES, nm, BETA_AUR, GRID,
                                 chunk_size=1))
    b = np.asarray(spectral_vis2(pos, BASELINES, nm, BETA_AUR, GRID,
                                 chunk_size=5))
    assert np.array_equal(a, b)


def test_spectral_snr_methods_agree():
    spec = Spectrograph(lambda_min_nm=410.0, lambda_max_nm=940.0,
                        n_channels=16)
    fft = spectral_g2_snr(BETA_AUR, 50.0, spectrograph=spec,
                          vis2_method="fft")
    ana = spectral_g2_snr(BETA_AUR, 50.0, spectrograph=spec,
                          vis2_method="analytic")
    assert fft.vis2_method == "fft" and ana.vis2_method == "analytic"
    assert np.allclose(fft.vis2, ana.vis2, atol=5e-3)
    assert fft.snr_total == pytest.approx(ana.snr_total, rel=2e-2)


def test_spectral_snr_eclipse_dispatch():
    spec = Spectrograph(lambda_min_nm=450.0, lambda_max_nm=900.0, n_channels=4)
    # the analytic method must refuse to run during an eclipse
    with pytest.raises(ValueError, match="eclipse"):
        spectral_g2_snr(BETA_AUR, 50.0, spectrograph=spec,
                        orbital_phase=0.25, vis2_method="analytic")
    # the FFT method handles it
    res = spectral_g2_snr(BETA_AUR, 50.0, spectrograph=spec,
                          orbital_phase=0.25, vis2_method="fft")
    assert np.isfinite(res.snr_total) and res.snr_total > 0.0
