"""Sanity checks of the g2 SNR calculator (hbtsim.snr)."""

import numpy as np
import pytest

import warnings
from dataclasses import replace

from hbtsim.params import ALGOL, BETA_AUR, C_LIGHT, SPICA
from hbtsim.snr import (C2PU, DISPERSED_BACKEND, KECK, SPAD_LAMBDA, SPAD_LAMBDA_NG,
                        Detector, Observation, Spectrograph, Telescope,
                        coherence_time_s, g2_snr, photon_flux, spectral_g2_snr,
                        stellar_rate, system_ab_mag)

# An idealized detector isolates the photon-budget scalings from dead time
# and dark counts.
IDEAL = Detector(name="ideal", pde_table_nm=((300.0, 1.0), (1000.0, 1.0)),
                 jitter_fwhm_ps=120.0, dead_time_ns=0.0,
                 dark_cps_per_pixel=0.0)
OBS = Observation(wavelength_nm=400.0, filter_width_nm=10.0, t_int_s=3600.0,
                  coherence_broadening=False)


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
    det32 = replace(det, name="d32", n_pixels=32)
    assert det32.detected_rate(32 * 1e12) == pytest.approx(
        32 * det.detected_rate(1e12), rel=1e-9)


def test_snr_scales_with_area_and_time():
    base = g2_snr(0.5, 2.0, OBS, telescope1=Telescope(1.0), detector1=IDEAL)
    big = g2_snr(0.5, 2.0, OBS, telescope1=Telescope(2.0), detector1=IDEAL)
    assert big.snr == pytest.approx(4.0 * base.snr, rel=1e-6)  # SNR ~ area

    obs4 = replace(OBS, t_int_s=4 * 3600.0)
    longer = g2_snr(0.5, 2.0, obs4, telescope1=Telescope(1.0), detector1=IDEAL)
    assert longer.snr == pytest.approx(2.0 * base.snr, rel=1e-6)  # SNR ~ sqrt(T)


def test_snr_independent_of_filter_width_in_ideal_limit():
    """R ~ dlambda and tau_c ~ 1/dlambda cancel when dead time and dark
    counts are negligible."""
    snrs = []
    for dl in (1.0, 10.0, 30.0):
        obs = replace(OBS, filter_width_nm=dl)
        snrs.append(g2_snr(0.5, 2.0, obs, telescope1=Telescope(1.0),
                           detector1=IDEAL).snr)
    assert np.allclose(snrs, snrs[0], rtol=1e-9)


def test_first_principles_g2_normalization():
    """Hand-computed photon budget, no reuse of the function's outputs:
    m_AB = 2 at 400 nm, 10 nm rectangular band, one 1 m telescope pair
    with throughput 0.3, backend 0.9, PDE 1, 120 ps FWHM jitters, no
    dead time / dark counts, |V|^2 = 0.5, 1 h, unpolarized."""
    lam, dlam, mag, T = 400e-9, 10e-9, 2.0, 3600.0
    f_nu = 3.631e-23 * 10 ** (-0.4 * mag)                    # W m^-2 Hz^-1
    h = 6.62607015e-34
    dnu = C_LIGHT * dlam / lam**2
    flux = f_nu / (h * C_LIGHT / lam) * dnu                  # photons m^-2 s^-1
    area = np.pi * 0.5**2
    R = flux * area * 0.3 * 0.9                              # cps, PDE 1
    tau_c = lam**2 / (C_LIGHT * dlam)
    sigma = 120e-12 / (2 * np.sqrt(2 * np.log(2)))
    sigma_pair = np.sqrt(2) * sigma
    n_sig = 0.5 * 0.5 * tau_c * R * R * T
    n_bkg = R * R * T * 2 * np.sqrt(np.pi) * sigma_pair
    expect = n_sig / np.sqrt(n_bkg)
    got = g2_snr(0.5, mag, OBS, telescope1=Telescope(1.0, 0.3), detector1=IDEAL)
    assert got.snr == pytest.approx(expect, rel=1e-9)
    assert got.rate1_cps == pytest.approx(R, rel=1e-9)
    assert got.n_signal == pytest.approx(n_sig, rel=1e-9)


def test_polarization_modes():
    """A polarizing beamsplitter gains sqrt(2) in g2 SNR at the same
    photon budget; a single polarizer gains nothing; the per-stream
    rates halve in both."""
    base = g2_snr(0.5, 2.0, OBS, telescope1=Telescope(1.0), detector1=IDEAL)
    pbs = g2_snr(0.5, 2.0, replace(OBS, polarization_mode="pbs"),
                 telescope1=Telescope(1.0), detector1=IDEAL)
    one = g2_snr(0.5, 2.0, replace(OBS, polarization_mode="single_pol"),
                 telescope1=Telescope(1.0), detector1=IDEAL)
    assert pbs.snr == pytest.approx(np.sqrt(2.0) * base.snr, rel=1e-12)
    assert one.snr == pytest.approx(base.snr, rel=1e-12)
    assert pbs.rate1_cps == pytest.approx(0.5 * base.rate1_cps)
    assert pbs.n_streams == 2 and one.n_streams == 1
    with pytest.raises(ValueError, match="polarization_mode"):
        g2_snr(0.5, 2.0, replace(OBS, polarization_mode="circular"))


def test_coherence_broadening():
    """sigma_c = 0.376 tau_c widens the pair kernel: negligible at 10 nm,
    ~1% at 0.1 nm / 950 nm (tau_c = 30 ps vs sigma_pair = 72 ps)."""
    wide = replace(OBS, coherence_broadening=True)
    a = g2_snr(0.5, 2.0, OBS, detector1=IDEAL)
    b = g2_snr(0.5, 2.0, wide, detector1=IDEAL)
    assert b.snr == pytest.approx(a.snr, rel=1e-4)
    red = Observation(wavelength_nm=950.0, filter_width_nm=0.1, t_int_s=3600.0)
    c = g2_snr(0.5, 2.0, replace(red, coherence_broadening=False), detector1=IDEAL)
    d = g2_snr(0.5, 2.0, red, detector1=IDEAL)
    tau_c = coherence_time_s(950.0, 0.1)
    sp = np.sqrt(2) * IDEAL.jitter_sigma_s
    # SNR ~ 1/sqrt(window) ~ sigma_pair^{-1/2}
    expect = (sp**2 / (sp**2 + (0.376 * tau_c)**2)) ** 0.25
    assert d.snr / c.snr == pytest.approx(expect, rel=1e-3)
    assert 0.98 < d.snr / c.snr < 0.995


def test_array_capable_observation():
    nm = np.array([450.0, 600.0, 800.0])
    obs = Observation(wavelength_nm=nm, filter_width_nm=np.full(3, 1.0), t_int_s=100.0)
    r = g2_snr(np.array([0.1, 0.2, 0.3]), 2.0, obs)
    assert r.snr.shape == (3,)
    for k in range(3):
        one = g2_snr(float(0.1 * (k + 1)), 2.0,
                     Observation(wavelength_nm=float(nm[k]), filter_width_nm=1.0,
                                 t_int_s=100.0))
        assert r.snr[k] == pytest.approx(one.snr, rel=1e-12)


def test_constant_resolving_power_spectrograph():
    spec = Spectrograph.from_resolving_power(5000.0)
    assert spec.n_channels == 4325
    assert np.allclose(spec.channel_centers_nm / spec.channel_widths_nm, 5000.0)
    assert spec.channel_edges_nm[0] == 400.0
    assert spec.channel_edges_nm[-1] == pytest.approx(950.0, abs=0.2)
    assert 0.079 < spec.channel_widths_nm.min() < 0.081
    assert 0.18 < spec.channel_widths_nm.max() < 0.20
    with pytest.raises(ValueError, match="not uniform"):
        spec.channel_width_nm
    uni = Spectrograph(n_channels=320)
    assert uni.is_uniform and uni.channel_width_nm == pytest.approx(550.0 / 320)
    assert np.allclose(uni.channel_widths_nm, 550.0 / 320)


def test_readout_ceiling_and_dead_time_warning():
    """Spica on a 10 m telescope: ~1e10 detected cps over 320 channels,
    far beyond the SPAD Lambda's time-tag link (1e8): the rates are
    scaled down and flagged; a correlator readout is not, but the
    per-pixel load exceeds 1 and warns."""
    spec = Spectrograph(n_channels=320)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tt = spectral_g2_snr(SPICA, 85.0, spectrograph=spec, telescope1=KECK,
                             detector1=SPAD_LAMBDA, vis2_method="analytic")
    assert tt.readout_limited and tt.readout_scale < 1e-1
    assert tt.total_rate_cps[0] <= 1.01 * SPAD_LAMBDA.max_total_cps
    with pytest.warns(UserWarning, match="dead-time"):
        cr = spectral_g2_snr(SPICA, 85.0, spectrograph=spec, telescope1=KECK,
                             detector1=SPAD_LAMBDA_NG, vis2_method="analytic")
    assert not cr.readout_limited and cr.dead_time_load_max > 1.0
    assert cr.total_rate_cps[0] > 1e9
    assert cr.snr_total > tt.snr_total
    # spreading the light over pixels lowers the load
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp = spectral_g2_snr(SPICA, 85.0, spectrograph=spec, telescope1=KECK,
                             detector1=SPAD_LAMBDA_NG, vis2_method="analytic",
                             n_pixels_per_channel=16)
    assert sp.dead_time_load_max == pytest.approx(cr.dead_time_load_max / 16, rel=1e-9)
    assert sp.snr_total > cr.snr_total


def test_backend_throughput_applied():
    """Dispersed channels see telescope x spectrograph throughput (0.15
    by default), narrow-band filters 0.3 x 0.9."""
    spec = Spectrograph(lambda_min_nm=500.0, lambda_max_nm=600.0, n_channels=4)
    a = spectral_g2_snr(BETA_AUR, 50.0, spectrograph=spec, vis2_method="analytic",
                        detector1=IDEAL, pupils=None)
    b = spectral_g2_snr(BETA_AUR, 50.0, spectrograph=replace(spec, throughput=1.0),
                        vis2_method="analytic", detector1=IDEAL, pupils=None)
    assert a.snr_total == pytest.approx(DISPERSED_BACKEND.throughput * b.snr_total,
                                        rel=1e-9)
    assert Observation(wavelength_nm=500.0).backend_throughput == 0.9


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
    # between/beyond the bands the magnitude stays in a sane range; outside
    # the anchors it is an extrapolation and says so
    with pytest.warns(UserWarning, match="extrapolated"):
        assert 1.5 < system_ab_mag(BETA_AUR, 400.0) < 2.5
    with pytest.warns(UserWarning, match="extrapolated"):
        assert 1.8 < system_ab_mag(BETA_AUR, 800.0) < 2.5
    # array call matches scalar calls
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nm = np.array([420.0, 600.0, 900.0])
        arr = system_ab_mag(ALGOL, nm)
        assert np.allclose(arr, [system_ab_mag(ALGOL, float(l)) for l in nm])


def test_binary_vis2_analytic_limits():
    from hbtsim.hbt import binary_vis2_analytic

    rho = BETA_AUR.angular_semimajor_mas
    assert binary_vis2_analytic(0.0, 500.0, BETA_AUR, rho)[0] == pytest.approx(1.0)
    v2 = binary_vis2_analytic(np.arange(10.0, 160.0, 10.0), 500.0, BETA_AUR, rho)
    assert np.all((v2 >= 0.0) & (v2 <= 1.0))


def test_spectral_snr_quadrature_sum_and_channels():
    from hbtsim.snr import Spectrograph, spectral_g2_snr

    spec = Spectrograph(lambda_min_nm=400.0, lambda_max_nm=950.0, n_channels=32)
    res = spectral_g2_snr(BETA_AUR, 50.0, spectrograph=spec, t_int_s=3600.0,
                          vis2_method="analytic", pupils=None,
                          enforce_readout=False)
    assert res.snr_total == pytest.approx(np.sqrt(np.sum(res.snr**2)), rel=1e-12)
    assert res.channel_nm.size == 32
    assert res.channel_nm[0] == pytest.approx(400.0 + 550.0 / 32 / 2)
    assert spec.channel_width_nm == pytest.approx(550.0 / 32)

    # one channel cross-checked against a manual single-filter calculation
    from dataclasses import replace

    from hbtsim.hbt import binary_vis2_analytic
    from hbtsim.orbit import sky_positions
    from hbtsim.snr import SPAD_LAMBDA, g2_snr, system_ab_mag

    k = 10
    lam = float(res.channel_nm[k])
    rho = float(np.asarray(sky_positions(0.0, BETA_AUR).rho))
    vis2 = float(binary_vis2_analytic(50.0, lam, BETA_AUR, rho)[0])
    obs = Observation(wavelength_nm=lam, filter_width_nm=spec.channel_width_nm,
                      t_int_s=3600.0, backend_throughput=spec.throughput)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mag = system_ab_mag(BETA_AUR, lam)
    manual = g2_snr(vis2, mag, obs, detector1=replace(SPAD_LAMBDA, n_pixels=1))
    assert res.snr[k] == pytest.approx(manual.snr, rel=1e-12)


def test_spectral_multiplexing_gain():
    """With more channels over the same band the total SNR grows roughly as
    sqrt(n) (exact only for a flat spectrum/PDE/|V|^2, so allow slack)."""
    from hbtsim.snr import Spectrograph, spectral_g2_snr

    r40 = spectral_g2_snr(BETA_AUR, 50.0, vis2_method="analytic",
                          spectrograph=Spectrograph(n_channels=40))
    r320 = spectral_g2_snr(BETA_AUR, 50.0, vis2_method="analytic",
                           spectrograph=Spectrograph(n_channels=320))
    gain = r320.snr_total / r40.snr_total
    assert gain == pytest.approx(np.sqrt(320 / 40), rel=0.15)


def test_dark_counts_only_add_noise():
    dark = replace(IDEAL, name="dark", dark_cps_per_pixel=1e6)
    clean = g2_snr(0.5, 2.0, OBS, telescope1=Telescope(1.0), detector1=IDEAL)
    noisy = g2_snr(0.5, 2.0, OBS, telescope1=Telescope(1.0), detector1=dark)
    assert noisy.n_signal == pytest.approx(clean.n_signal, rel=1e-12)
    assert noisy.snr < clean.snr
