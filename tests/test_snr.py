"""Sanity checks of the g2 SNR calculator (hbtsim.snr)."""

import numpy as np
import pytest

from hbtsim.params import BETA_AUR, C_LIGHT
from hbtsim.snr import (C2PU, SPAD_LAMBDA, Detector, Observation, Telescope,
                        coherence_time_s, g2_snr, photon_flux, stellar_rate,
                        system_ab_mag)

# An idealized detector isolates the photon-budget scalings from dead time
# and dark counts.
IDEAL = Detector(name="ideal", pde_table_nm=((300.0, 1.0), (1000.0, 1.0)),
                 jitter_fwhm_ps=120.0, dead_time_ns=0.0,
                 dark_cps_per_pixel=0.0)
OBS = Observation(wavelength_nm=400.0, filter_width_nm=10.0, t_int_s=3600.0)


def test_photon_flux_zeropoint():
    """AB mag 0 at 550 nm is the classic ~1000 photons / cm^2 / s / Angstrom."""
    flux = photon_flux(0.0, 550.0, 0.1)  # 0.1 nm = 1 Angstrom, per m^2
    assert flux == pytest.approx(1000.0e4, rel=0.02)


def test_coherence_time():
    assert coherence_time_s(400.0, 10.0) == pytest.approx(
        (400e-9) ** 2 / (C_LIGHT * 10e-9), rel=1e-12)


def test_pde_interpolation():
    assert SPAD_LAMBDA.pde(400.0) == pytest.approx(0.22)
    assert SPAD_LAMBDA.pde(800.0) == pytest.approx(0.14)
    assert SPAD_LAMBDA.pde(510.0) == pytest.approx(0.495, abs=0.01)


def test_dead_time_saturation():
    det = Detector(name="d", pde_table_nm=((300.0, 1.0), (1000.0, 1.0)),
                   jitter_fwhm_ps=120.0, dead_time_ns=10.0,
                   dark_cps_per_pixel=0.0)
    assert det.detected_rate(1e3) == pytest.approx(1e3, rel=1e-4)
    assert det.detected_rate(1e12) < 1.0 / 10e-9  # cannot exceed 1/dead time
    # spreading over n pixels raises the ceiling n-fold: at the same
    # per-pixel load, n pixels detect n times more
    det32 = Detector(name="d32", pde_table_nm=det.pde_table_nm,
                     jitter_fwhm_ps=120.0, dead_time_ns=10.0,
                     dark_cps_per_pixel=0.0, n_pixels=32)
    assert det32.detected_rate(32 * 1e12) == pytest.approx(
        32 * det.detected_rate(1e12), rel=1e-9)


def test_snr_scales_with_area_and_time():
    base = g2_snr(0.5, 2.0, OBS, telescope1=Telescope(1.0), detector1=IDEAL)
    big = g2_snr(0.5, 2.0, OBS, telescope1=Telescope(2.0), detector1=IDEAL)
    assert big.snr == pytest.approx(4.0 * base.snr, rel=1e-6)  # SNR ~ area

    obs4 = Observation(wavelength_nm=400.0, filter_width_nm=10.0,
                       t_int_s=4 * 3600.0)
    longer = g2_snr(0.5, 2.0, obs4, telescope1=Telescope(1.0), detector1=IDEAL)
    assert longer.snr == pytest.approx(2.0 * base.snr, rel=1e-6)  # SNR ~ sqrt(T)


def test_snr_independent_of_filter_width_in_ideal_limit():
    """R ~ dlambda and tau_c ~ 1/dlambda cancel when dead time and dark
    counts are negligible."""
    snrs = []
    for dl in (1.0, 10.0, 30.0):
        obs = Observation(wavelength_nm=400.0, filter_width_nm=dl,
                          t_int_s=3600.0)
        snrs.append(g2_snr(0.5, 2.0, obs, telescope1=Telescope(1.0),
                           detector1=IDEAL).snr)
    assert np.allclose(snrs, snrs[0], rtol=1e-9)


def test_snr_proportional_to_vis2():
    a = g2_snr(0.2, 2.0, OBS, telescope1=C2PU, detector1=SPAD_LAMBDA)
    b = g2_snr(0.6, 2.0, OBS, telescope1=C2PU, detector1=SPAD_LAMBDA)
    assert b.snr == pytest.approx(3.0 * a.snr, rel=1e-9)


def test_system_ab_mag_hits_anchors():
    """At the anchor band wavelengths the model reproduces the observed
    magnitudes exactly (the offset is defined there)."""
    anchors = dict(BETA_AUR.mag_anchors)
    assert system_ab_mag(BETA_AUR, 477.0) == pytest.approx(anchors["g"], abs=1e-9)
    assert system_ab_mag(BETA_AUR, 763.0) == pytest.approx(anchors["i"], abs=1e-9)
    # between/beyond the bands the magnitude stays in a sane range
    assert 1.5 < system_ab_mag(BETA_AUR, 400.0) < 2.5
    assert 1.8 < system_ab_mag(BETA_AUR, 800.0) < 2.5


def test_dark_counts_only_add_noise():
    dark = Detector(name="dark", pde_table_nm=IDEAL.pde_table_nm,
                    jitter_fwhm_ps=120.0, dead_time_ns=0.0,
                    dark_cps_per_pixel=1e6)
    clean = g2_snr(0.5, 2.0, OBS, telescope1=Telescope(1.0), detector1=IDEAL)
    noisy = g2_snr(0.5, 2.0, OBS, telescope1=Telescope(1.0), detector1=dark)
    assert noisy.n_signal == pytest.approx(clean.n_signal, rel=1e-12)
    assert noisy.snr < clean.snr
