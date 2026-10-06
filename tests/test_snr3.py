"""Tests of the triple-correlation photon budget (hbtsim.snr3)."""

import numpy as np
import pytest

import warnings
from dataclasses import replace

from hbtsim.bispectrum import Station, Triangle, equilateral_triangle
from hbtsim.params import C_LIGHT
from hbtsim.snr import Detector, Observation, Spectrograph, Telescope, g2_snr
from hbtsim.snr3 import (POL_FACTOR_TRIPLE, array_g3_snr, binned_closure_phase_snr,
                         g3_snr, spectral_g3_snr, time_to_cos_phi,
                         time_to_precision, triple_window_s2)

IDEAL_DET = Detector(name="ideal", pde_table_nm=((300.0, 1.0), (1000.0, 1.0)),
                     jitter_fwhm_ps=120.0, dead_time_ns=0.0,
                     dark_cps_per_pixel=0.0, readout="correlator")
OBS = Observation(wavelength_nm=600.0, filter_width_nm=1.0, t_int_s=3600.0,
                  coherence_broadening=False)


def _tri(diam=1.0, jitter_ps=120.0, dead_ns=0.0, side=85.0):
    det = replace(IDEAL_DET, name="d", jitter_fwhm_ps=jitter_ps,
                  dead_time_ns=dead_ns)
    return equilateral_triangle(side, Telescope(diam, 0.3), det)


def test_triple_window_equal_jitters():
    """Equal jitters: A_2D = 4 pi sqrt(3) sigma^2."""
    d = IDEAL_DET
    s = d.jitter_sigma_s
    assert triple_window_s2(d, d, d) == pytest.approx(
        4.0 * np.pi * np.sqrt(3.0) * s**2, rel=1e-12)


def test_snr3_scalings():
    base = g3_snr(0.1, 2.0, OBS, _tri(diam=1.0))
    # ~ sqrt(A1 A2 A3): all areas x2 (diameter x sqrt(2)) -> SNR x 2^{3/2}
    big = g3_snr(0.1, 2.0, OBS, _tri(diam=np.sqrt(2.0)))
    assert big.snr == pytest.approx(2.0**1.5 * base.snr, rel=1e-9)
    # ~ sqrt(T)
    obs4 = replace(OBS, t_int_s=4 * 3600.0)
    assert g3_snr(0.1, 2.0, obs4, _tri()).snr == pytest.approx(
        2.0 * base.snr, rel=1e-9)
    # ~ 1/sigma_jitter (note: g2 scales only as 1/sqrt(sigma))
    fast = g3_snr(0.1, 2.0, OBS, _tri(jitter_ps=60.0))
    assert fast.snr == pytest.approx(2.0 * base.snr, rel=1e-9)
    # ~ triple_amp and cos_phi_c linearly
    assert g3_snr(0.2, 2.0, OBS, _tri()).snr == pytest.approx(
        2.0 * base.snr, rel=1e-9)
    assert g3_snr(0.1, 2.0, OBS, _tri(), cos_phi_c=0.5).snr == pytest.approx(
        0.5 * base.snr, rel=1e-9)
    # a polarizing beamsplitter: x2 over unpolarized; a single polarizer
    # (half the light discarded): x sqrt(2)
    pbs = g3_snr(0.1, 2.0, replace(OBS, polarization_mode="pbs"), _tri())
    assert pbs.snr == pytest.approx(2.0 * base.snr, rel=1e-12)
    assert pbs.rates_cps[0] == pytest.approx(0.5 * base.rates_cps[0])
    one = g3_snr(0.1, 2.0, replace(OBS, polarization_mode="single_pol"), _tri())
    assert one.snr == pytest.approx(np.sqrt(2.0) * base.snr, rel=1e-12)


def test_snr3_bandwidth_scaling_unsaturated():
    """At fixed source, R ~ dlam and tau_c ~ 1/dlam: N_sig ~ dlam, noise
    ~ dlam^{3/2} -> SNR3 ~ dlam^{-1/2} (narrower is better, unlike g2)."""
    tri = _tri()
    snrs = []
    for dl in (0.5, 2.0):
        obs = replace(OBS, filter_width_nm=dl)
        snrs.append(g3_snr(0.1, 2.0, obs, tri).snr)
    assert snrs[0] / snrs[1] == pytest.approx(2.0, rel=1e-6)


def test_first_principles_snr3_normalization():
    """Hand-computed from the module-docstring formula with explicit
    numbers, reusing none of g3_snr's outputs: m_AB = 2 at 600 nm, 1 nm
    band, three 1 m telescopes (throughput 0.3, backend 0.9, PDE 1),
    120 ps FWHM jitters, |g12 g23 g31| = 0.1, 1 h, unpolarized."""
    lam, dlam, mag, T = 600e-9, 1e-9, 2.0, 3600.0
    h = 6.62607015e-34
    f_nu = 3.631e-23 * 10 ** (-0.4 * mag)
    flux = f_nu / (h * C_LIGHT / lam) * (C_LIGHT * dlam / lam**2)
    R = flux * np.pi * 0.25 * 0.3 * 0.9
    tau_c = lam**2 / (C_LIGHT * dlam)
    s = 120e-12 / (2 * np.sqrt(2 * np.log(2)))
    A2d = 4 * np.pi * np.sqrt(3.0) * s**2
    n_sig = 0.25 * 2.0 * 0.1 * tau_c**2 * R**3 * T
    n_bkg = R**3 * T * A2d
    expect = n_sig / np.sqrt(n_bkg)
    got = g3_snr(0.1, mag, OBS, _tri(diam=1.0))
    assert got.snr == pytest.approx(expect, rel=1e-9)
    assert got.rates_cps[0] == pytest.approx(R, rel=1e-9)
    assert got.window_s2 == pytest.approx(A2d, rel=1e-12)


def test_zmija_hess_anchor():
    """Order-of-magnitude anchor against Zmija et al. 2025, Table 2:
    H.E.S.S. (3 x 100 m^2, tau_e = 5 ns, 10 nm, one channel) needs
    ~1100-2400 yr to reach Delta cos phi_c <= 0.1 on m_B ~ 2 stars.
    Their figure scales their MEASURED sensitivity; here a 5 ns FWHM
    Gaussian response, ~2 GHz detected per telescope ("of order GHz"),
    |g12 g23 g31| ~ 0.7.  Agreement within an order of magnitude checks
    the tau_c^2 / lag-plane normalization; a tau_c-vs-tau_c^2 slip would
    be off by 1e5."""
    det = replace(IDEAL_DET, jitter_fwhm_ps=5000.0)
    obs = Observation(wavelength_nm=440.0, filter_width_nm=10.0, t_int_s=3600.0,
                      coherence_broadening=False)
    d = 2 * np.sqrt(100.0 / np.pi)
    tri = equilateral_triangle(100.0, Telescope(d, 1.0), det)
    # choose the magnitude that gives ~2 GHz detected per telescope
    from hbtsim.snr import stellar_rate
    mag = 2.0
    rate = stellar_rate(mag, tri.stations[0].telescope, det, obs)
    scale = 2e9 / rate
    mag_eff = mag - 2.5 * np.log10(scale)
    r = g3_snr(0.7, mag_eff, obs, tri)
    years = 3600.0 * (10.0 / r.snr) ** 2 / (365.25 * 86400.0)
    assert 1e2 < years < 1e5


def test_coherence_broadening_triple():
    tri = _tri()
    red = Observation(wavelength_nm=950.0, filter_width_nm=0.1, t_int_s=3600.0)
    a = g3_snr(0.1, 2.0, replace(red, coherence_broadening=False), tri)
    b = g3_snr(0.1, 2.0, red, tri)
    assert 0.9 < b.snr / a.snr < 0.995
    assert b.window_s2 > a.window_s2
    # the closed form: Sigma + sigma_c^2 [[1, -1/2], [-1/2, 1]]
    from hbtsim.snr import COHERENCE_SIGMA_FACTOR, coherence_time_s
    s2 = IDEAL_DET.jitter_sigma_s**2
    c2 = (COHERENCE_SIGMA_FACTOR * coherence_time_s(950.0, 0.1))**2
    det_sigma = (2 * s2 + c2) ** 2 - (s2 + 0.5 * c2) ** 2
    assert b.window_s2 == pytest.approx(4 * np.pi * np.sqrt(det_sigma), rel=1e-12)


def test_ridge_ratio(spica, vlt_ut):
    """Pair ridges exceed the triple term by ~(p2/2p3)|g|^2 A_2D /
    (2 sqrt(pi) sigma tau_c |ggg|): hundreds at 0.1 nm."""
    obs = Observation(wavelength_nm=600.0, filter_width_nm=0.1, t_int_s=3600.0,
                      coherence_broadening=False)
    r = g3_snr(0.1, 2.0, obs, _tri(), pair_vis2=np.array([0.3, 0.3, 0.3]))
    s = IDEAL_DET.jitter_sigma_s
    expect = 3 * (0.5 / 0.5) * 0.3 * r.window_s2 / (2 * np.sqrt(np.pi) * np.sqrt(2) * s
                                                    * r.tau_c_s * 0.1)
    assert r.ridge_ratio == pytest.approx(expect, rel=1e-9)
    assert 50 < r.ridge_ratio < 2000
    spec = Spectrograph(lambda_min_nm=450.0, lambda_max_nm=900.0, n_channels=8)
    res = spectral_g3_snr(spica, vlt_ut.triangles()[0], spectrograph=spec)
    assert res.ridge_ratio.shape == (8,)
    assert np.all(res.required_kernel_accuracy(0.1) == 0.1 / res.ridge_ratio)


def test_spectral_g3_quadrature_and_methods(beta_aur, algol, maunakea_tri):
    spec = Spectrograph(lambda_min_nm=450.0, lambda_max_nm=900.0,
                        n_channels=8)
    ana = spectral_g3_snr(beta_aur, maunakea_tri, spectrograph=spec,
                          vis_method="analytic")
    assert ana.snr_total == pytest.approx(np.sqrt(np.sum(ana.snr**2)),
                                          rel=1e-12)
    rnd = spectral_g3_snr(beta_aur, maunakea_tri, spectrograph=spec,
                          vis_method="render")
    assert np.allclose(rnd.triple_amp, ana.triple_amp, atol=1e-3)
    assert rnd.snr_total == pytest.approx(ana.snr_total, rel=0.01)
    # in eclipse: analytic refuses, render works
    with pytest.raises(ValueError, match="eclipse"):
        spectral_g3_snr(algol, maunakea_tri, spectrograph=spec,
                        orbital_phase=0.25, vis_method="analytic")


def test_vlt_array_geometry(vlt_ut):
    """The published UT station coordinates reproduce the six pairwise
    separations (46.6, 56.5, 62.4, 89.3, 102.4, 130.2 m) and give four
    triangles."""
    lengths = sorted(float(np.hypot(*b)) for _, _, b in vlt_ut.pairs())
    assert lengths == pytest.approx([46.6, 56.5, 62.4, 89.3, 102.4, 130.2],
                                    abs=0.2)
    tris = vlt_ut.triangles()
    assert len(tris) == 4
    for tri in tris:
        assert np.allclose(tri.baseline_vectors().sum(axis=0), 0.0)


def test_array_g3_quadrature_combination(spica, vlt_ut):
    spec = Spectrograph(lambda_min_nm=450.0, lambda_max_nm=900.0,
                        n_channels=6)
    res = array_g3_snr(spica, vlt_ut, spectrograph=spec)
    assert len(res.per_triangle) == 4
    assert res.snr_total == pytest.approx(
        np.sqrt(sum(r.snr_total**2 for r in res.per_triangle)), rel=1e-12)
    assert res.snr_total > max(r.snr_total for r in res.per_triangle)


def test_time_to_precision_inversions_and_ordering(spica, maunakea_tri):
    spec = Spectrograph(lambda_min_nm=450.0, lambda_max_nm=900.0, n_channels=16)
    kw = dict(spectrograph=spec, enforce_readout=False)
    times = {s: time_to_precision(spica, 0.1, triangle=maunakea_tri,
                                  statistic=s, R_bin=50.0, **kw)
             for s in ("total", "amplitude", "binned", "channel")}
    # inversions
    ref = spectral_g3_snr(spica, maunakea_tri, t_int_s=times["total"], **kw)
    assert ref.snr_total == pytest.approx(10.0, rel=1e-6)
    ref = spectral_g3_snr(spica, maunakea_tri, t_int_s=times["amplitude"], **kw)
    assert ref.snr_amplitude == pytest.approx(10.0, rel=1e-6)
    ref = spectral_g3_snr(spica, maunakea_tri, t_int_s=times["binned"], **kw)
    assert np.median(binned_closure_phase_snr(ref, 50.0)[1]) == pytest.approx(10.0, rel=1e-6)
    ref = spectral_g3_snr(spica, maunakea_tri, t_int_s=times["channel"], **kw)
    assert np.median(ref.snr) == pytest.approx(10.0, rel=1e-6)
    # ordering: a global amplitude is the easiest, a single channel the hardest
    assert times["total"] <= times["amplitude"] <= times["binned"] <= times["channel"]
    # deprecated wrappers still answer
    with pytest.warns(DeprecationWarning):
        t_old = time_to_cos_phi(spica, maunakea_tri, target_dcos=0.1, **kw)
    assert t_old == pytest.approx(times["total"])
    with pytest.raises(ValueError, match="exactly one"):
        time_to_precision(spica, 0.1, spectrograph=spec)


def test_readout_and_polarization_in_spectral_g3(spica, vlt_ut, maunakea_tri):
    spec = Spectrograph(lambda_min_nm=450.0, lambda_max_nm=900.0, n_channels=6)
    a = spectral_g3_snr(spica, vlt_ut.triangles()[0], spectrograph=spec)
    b = spectral_g3_snr(spica, vlt_ut.triangles()[0], spectrograph=spec,
                        polarization_mode="pbs")
    assert not a.readout_limited      # next-gen correlator stations
    # six 75 nm channels: each pixel is saturated (1/tau_dead = 1e8 cps)
    assert a.dead_time_load_max > 10.0
    assert a.total_rate_cps[0] > 5e8
    # PBS: x2 in the ideal case, more here since it halves the dead-time load
    assert b.snr_total > 2.0 * a.snr_total
    # Maunakea's SPAD Lambda stations are time-tag limited
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = spectral_g3_snr(spica, maunakea_tri, spectrograph=spec)
        m0 = spectral_g3_snr(spica, maunakea_tri, spectrograph=spec,
                             enforce_readout=False)
    assert m.readout_limited and m.snr_total < 0.1 * m0.snr_total


def test_binned_statistic_is_quadrature_over_triangles(spica):
    """Two triangles of unequal sensitivity measure the same closure-phase
    bins: the combined binned SNR is the quadrature sum per bin (the old
    code summed linearly and divided by sqrt(N))."""
    from hbtsim.snr3 import _statistic_snr
    spec = Spectrograph(lambda_min_nm=450.0, lambda_max_nm=900.0, n_channels=16)
    kw = dict(spectrograph=spec, enforce_readout=False)
    r1 = spectral_g3_snr(spica, _tri(diam=1.0), **kw)
    r2 = spectral_g3_snr(spica, _tri(diam=2.0), **kw)
    b1 = binned_closure_phase_snr(r1, 50.0)[1]
    b2 = binned_closure_phase_snr(r2, 50.0)[1]
    assert not np.allclose(b1, b2)
    got = _statistic_snr([r1, r2], "binned", 50.0, "median")
    assert got == pytest.approx(float(np.median(np.sqrt(b1**2 + b2**2))), rel=1e-12)
    old = float(np.median((b1 + b2) / np.sqrt(2.0)))
    assert got > old
