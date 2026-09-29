"""Monte Carlo of a multiplexed g2 diameter measurement (hbtsim.montecarlo)."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import erf

from hbtsim.montecarlo import (MCConfig, box_estimate, expected_counts, fit_ud,
                               matched_filter_estimate, observing_blocks, run_mc,
                               simulate_histograms)
from hbtsim.snr import Spectrograph

SMALL = MCConfig(mag_ab=3.0, spectrograph=Spectrograph(lambda_min_nm=450.0, lambda_max_nm=480.0,
                                                       n_channels=12, throughput=0.6),
                 t_total_h=1.0, zenith_angles_deg=(45.0, 52.5, 60.0))


def test_observing_blocks_zenith_and_projection():
    b = observing_blocks(MCConfig())
    assert np.allclose(b.zenith_deg, (45.0, 52.5, 60.0))
    assert abs(b.hour_angle_h[0]) < 0.05                 # Sirius B transits at z = 45 from Teide
    assert np.all(b.b_proj_m <= 1750.0 + 1e-6)
    assert b.b_proj_m[0] == pytest.approx(1750.0, rel=1e-3)   # E-W baseline at transit
    assert b.t_s.sum() == pytest.approx(10 * 3600.0)
    with pytest.raises(ValueError):
        observing_blocks(replace(MCConfig(), zenith_angles_deg=(30.0,)))


def test_expected_counts_normalisation():
    e = expected_counts(SMALL)
    assert np.allclose(e.kernel().sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(e.bkg_per_bin, e.background**2 * e.t_s[None, :] * e.bin_s)
    assert np.allclose(e.signal_scale, e.p2 * e.tau_c[:, None] * e.rates**2 * e.t_s[None, :])


def test_fit_ud_noiseless_exact():
    e = expected_counts(SMALL)
    th, s = fit_ud(e.vis2_true, e.analytic_sigma_vis2(), e.b_proj_m, e.nm, 0.9 * SMALL.theta_true_mas)
    assert th == pytest.approx(SMALL.theta_true_mas, rel=1e-6) and s > 0


def test_mc_matched_filter_unbiased_with_analytic_scatter():
    rng = np.random.default_rng(3)
    e = expected_counts(SMALL)
    est = np.array([matched_filter_estimate(simulate_histograms(rng, e), e)[0] for _ in range(300)])
    s_an = e.analytic_sigma_vis2()
    assert np.all(np.abs(est.mean(axis=0) - e.vis2_true) < 4 * s_an / np.sqrt(300))
    assert np.median(est.std(axis=0, ddof=1) / s_an) == pytest.approx(1.0, abs=0.1)


def test_mc_uncorrected_box_is_biased_by_the_capture_fraction():
    rng = np.random.default_rng(5)
    e = expected_counts(SMALL)
    s = float(np.median(e.sigma_pair))
    est = np.array([box_estimate(simulate_histograms(rng, e), e, s, correct_capture=False)[0]
                    for _ in range(200)])
    ratio = np.median(est.mean(axis=0) / e.vis2_true)
    assert ratio == pytest.approx(erf(1 / np.sqrt(2)), abs=0.03)


@pytest.mark.slow
def test_run_mc_pulls():
    r = run_mc(SMALL, n_real=150, estimators=("matched", "box_opt"), seed=1)
    m = r.stats["matched"]
    assert abs(m.bias_frac) < 4 * m.precision_frac / np.sqrt(150)
    assert 0.85 < m.pull_std < 1.15
    assert r.stats["box_opt"].theta_std > m.theta_std * 0.98
