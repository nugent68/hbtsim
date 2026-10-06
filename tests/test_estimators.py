"""Estimator identities: the design study's formulas against the matched filter."""

from dataclasses import replace

import numpy as np
import pytest

from hbtsim.estimators import (box_capture_fraction, box_snr, classic_hbt_snr,
                               implied_dt_res, matched_filter_equivalents,
                               optimal_box_half_width, photon_level_snr)
from hbtsim.snr import Observation, g2_snr

OBS = dict(wavelength_nm=475.0, filter_width_nm=0.15, t_int_s=3600.0, coherence_broadening=False)


@pytest.fixture(scope="module")
def det(eonsii_mcp):
    return replace(eonsii_mcp, dark_cps_per_pixel=0.0, dead_time_ns=0.0)


@pytest.mark.parametrize("pol", ["unpolarized", "pbs"])
def test_photon_level_equals_matched_filter_when_mapped(pol, eonsii_tel, det):
    obs = Observation(**OBS, polarization_mode=pol)
    v2 = 1e-4                      # signal term negligible against the accidentals
    r = g2_snr(v2, 8.44, obs, telescope1=eonsii_tel, detector1=det)
    p2 = 0.5 if pol == "unpolarized" else 1.0
    eq = matched_filter_equivalents(r.sigma_pair_s, p2)
    s = photon_level_snr(v2, r.rate1_cps, r.rate2_cps, r.tau_c_s, obs.t_int_s, eq["dt_res_s"], eq["eta"])
    s *= np.sqrt(r.n_streams)
    assert s == pytest.approx(r.snr, rel=1e-5)


def test_classic_hbt_equals_matched_filter(eonsii_tel, det):
    obs = Observation(**OBS)
    r = g2_snr(0.6, 8.44, obs, telescope1=eonsii_tel, detector1=det)
    eq = matched_filter_equivalents(r.sigma_pair_s, 0.5)
    n_nu_det = r.rate1_cps * r.tau_c_s          # detected rate per Hz (tau_c = 1/dnu)
    s = classic_hbt_snr(1.0, 1.0, 1.0, n_nu_det, 0.6, 0.5, eq["b_el_hz"], obs.t_int_s)
    assert s == pytest.approx(r.snr, rel=1e-9)


def test_optimal_box_is_0943_of_matched_filter(eonsii_tel, det):
    obs = Observation(**OBS)
    r = g2_snr(0.6, 8.44, obs, telescope1=eonsii_tel, detector1=det)
    a = optimal_box_half_width(r.sigma_pair_s)
    assert a / r.sigma_pair_s == pytest.approx(1.40, rel=0.03)
    b = box_snr(0.6, 8.44, obs, eonsii_tel, det, a)
    assert b.snr / b.snr_matched == pytest.approx(0.943, abs=0.005)


def test_full_capture_box_overstates(eonsii_tel, det):
    obs = Observation(**OBS)
    r = g2_snr(0.6, 8.44, obs, telescope1=eonsii_tel, detector1=det)
    a = 0.25 * r.sigma_pair_s
    full = box_snr(0.6, 8.44, obs, eonsii_tel, det, a, capture="full")
    honest = box_snr(0.6, 8.44, obs, eonsii_tel, det, a)
    assert full.snr > r.snr > honest.snr
    assert honest.capture == pytest.approx(box_capture_fraction(a, r.sigma_pair_s))


def test_implied_dt_res():
    s = 27.4e-12
    assert implied_dt_res(1.0, 1.0, s) == pytest.approx(np.sqrt(np.pi) * s)
    # a paper 16x faster with eta twice ours needs dt_res 4x... (2^2/16)
    assert implied_dt_res(1.0, 16.0, s, eta_ratio=2.0) == pytest.approx(np.sqrt(np.pi) * s * 4 / 16)
