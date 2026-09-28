"""Tests of the batched multi-wavelength render + DFT pipeline (hbtsim.spectral)."""

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
    """The lax.map path reproduces render_image + vis2_along_pa channel by
    channel (same float32 chain; only summation order may differ)."""
    pos = _pos(0.3)
    batched = np.asarray(spectral_vis2(pos, BASELINES, [400.0, 800.0],
                                       BETA_AUR, GRID, chunk_size=2))
    for j, lam_nm in enumerate((400.0, 800.0)):
        img = render_image(pos, BETA_AUR, lam_nm, GRID)
        ref = np.asarray(hbt.vis2_along_pa(img, BASELINES, lam_nm * 1e-9,
                                           float(pos.pa), GRID))
        assert np.allclose(batched[j], ref, atol=1e-6)


def test_spectral_vis_complex_and_flux():
    """spectral_vis returns complex64 (n_lambda, K) at arbitrary baseline
    vectors, with Hermitian symmetry, and return_flux gives the rendered
    image sum per channel."""
    from hbtsim.spectral import spectral_vis

    pos = _pos(0.4)
    bv = np.array([[30.0, 10.0], [-30.0, -10.0], [0.0, 120.0]])
    nm = np.array([450.0, 700.0, 900.0])
    vis, flux = spectral_vis(pos, bv, nm, BETA_AUR, GRID, chunk_size=2,
                             return_flux=True)
    vis, flux = np.asarray(vis), np.asarray(flux)
    assert vis.shape == (3, 3) and vis.dtype == np.complex64
    assert np.allclose(vis[:, 0], np.conj(vis[:, 1]), atol=1e-6)
    for j, lam_nm in enumerate(nm):
        img = render_image(pos, BETA_AUR, float(lam_nm), GRID)
        assert flux[j] == pytest.approx(float(img.sum()), rel=1e-6)
        ref = np.asarray(hbt.vis_of_baselines(img, bv, lam_nm * 1e-9, GRID))
        assert np.allclose(vis[j], ref, atol=1e-6)


def test_batched_matches_analytic_out_of_eclipse():
    pos = _pos(0.0)
    nm = np.linspace(410.0, 940.0, 16)
    v2 = np.asarray(spectral_vis2(pos, BASELINES, nm, BETA_AUR, GRID,
                                  chunk_size=4))
    for j, lam_nm in enumerate(nm):
        ana = hbt.binary_vis2_analytic(BASELINES, float(lam_nm), BETA_AUR,
                                       float(pos.rho))
        assert np.allclose(v2[j], ana, atol=1e-3), f"channel {lam_nm} nm"


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
    """Including a chunk that does not divide n_lambda (padding path) and
    one larger than n_lambda (clamped)."""
    pos = _pos(0.0)
    nm = np.linspace(420.0, 900.0, 10)
    a = np.asarray(spectral_vis2(pos, BASELINES, nm, BETA_AUR, GRID,
                                 chunk_size=1))
    for chunk in (3, 5, 64):
        b = np.asarray(spectral_vis2(pos, BASELINES, nm, BETA_AUR, GRID,
                                     chunk_size=chunk))
        assert np.allclose(a, b, atol=1e-7), chunk


def test_spectral_snr_methods_agree():
    spec = Spectrograph(lambda_min_nm=410.0, lambda_max_nm=940.0,
                        n_channels=16)
    rnd = spectral_g2_snr(BETA_AUR, 50.0, spectrograph=spec,
                          vis2_method="render")
    ana = spectral_g2_snr(BETA_AUR, 50.0, spectrograph=spec,
                          vis2_method="analytic")
    assert rnd.vis2_method == "render" and ana.vis2_method == "analytic"
    assert np.allclose(rnd.vis2, ana.vis2, atol=1e-3)
    assert rnd.snr_total == pytest.approx(ana.snr_total, rel=5e-3)
    with pytest.warns(DeprecationWarning):
        old = spectral_g2_snr(BETA_AUR, 50.0, spectrograph=spec,
                              vis2_method="fft")
    assert old.vis2_method == "render"


def test_spectral_snr_eclipse_dispatch():
    spec = Spectrograph(lambda_min_nm=450.0, lambda_max_nm=900.0, n_channels=4)
    # the analytic method must refuse to run during an eclipse
    with pytest.raises(ValueError, match="eclipse"):
        spectral_g2_snr(BETA_AUR, 50.0, spectrograph=spec,
                        orbital_phase=0.25, vis2_method="analytic")
    # the rendered path handles it
    res = spectral_g2_snr(BETA_AUR, 50.0, spectrograph=spec,
                          orbital_phase=0.25, vis2_method="render")
    assert np.isfinite(res.snr_total) and res.snr_total > 0.0


def test_spectral_snr_applies_eclipse_dimming():
    """Inside an Algol eclipse the rendered flux dims the photon rates;
    out of eclipse the dimming factor is 1 to the render accuracy."""
    from hbtsim.params import ALGOL
    from hbtsim.snr import Spectrograph, spectral_g2_snr
    spec = Spectrograph(lambda_min_nm=470.0, lambda_max_nm=484.0, n_channels=2)
    out = spectral_g2_snr(ALGOL, 60.0, spectrograph=spec, orbital_phase=0.0,
                          vis2_method="render")
    assert np.allclose(out.dimming, 1.0, atol=3e-3)
    ecl = [spectral_g2_snr(ALGOL, 60.0, spectrograph=spec, orbital_phase=ph,
                           vis2_method="render") for ph in (0.25, 0.75)]
    depths = [-2.5 * np.log10(float(e.dimming[0])) for e in ecl]
    deep = max(depths)
    assert 1.0 < deep < 1.8            # Algol primary minimum at 477 nm
    assert min(depths) > 0.01          # the secondary eclipse is shallow (0.018 mag) but real
    # the dimmed magnitude feeds the rates: fewer photons, lower SNR
    k = int(np.argmax(depths))
    assert ecl[k].rate_cps[0] < 0.5 * out.rate_cps[0]
    assert ecl[k].mag_ab[0] > out.mag_ab[0] + 1.0
    # analytic path reports no dimming
    an = spectral_g2_snr(ALGOL, 60.0, spectrograph=spec, orbital_phase=0.0,
                         vis2_method="analytic")
    assert np.all(an.dimming == 1.0)
