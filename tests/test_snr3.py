"""Tests of the triple-correlation photon budget (hbtsim.snr3)."""

import numpy as np
import pytest

from hbtsim.bispectrum import (MAUNAKEA_SUBARU_KECK, Station, Triangle,
                               equilateral_triangle)
from hbtsim.params import ALGOL, BETA_AUR
from hbtsim.snr import Detector, Observation, Spectrograph, Telescope, g2_snr
from hbtsim.snr3 import (POL_FACTOR_TRIPLE, g3_snr, spectral_g3_snr,
                         time_to_cos_phi, triple_window_s2)

IDEAL_DET = Detector(name="ideal", pde_table_nm=((300.0, 1.0), (1000.0, 1.0)),
                     jitter_fwhm_ps=120.0, dead_time_ns=0.0,
                     dark_cps_per_pixel=0.0)
OBS = Observation(wavelength_nm=600.0, filter_width_nm=1.0, t_int_s=3600.0)


def _tri(diam=1.0, jitter_ps=120.0, dead_ns=0.0, side=85.0):
    det = Detector(name="d", pde_table_nm=IDEAL_DET.pde_table_nm,
                   jitter_fwhm_ps=jitter_ps, dead_time_ns=dead_ns,
                   dark_cps_per_pixel=0.0)
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
    obs4 = Observation(wavelength_nm=600.0, filter_width_nm=1.0,
                       t_int_s=4 * 3600.0)
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
    # polarized light: x4 over unpolarized
    pol = g3_snr(0.1, 2.0, OBS, _tri(), pol_factor_triple=1.0)
    assert pol.snr == pytest.approx(4.0 * base.snr, rel=1e-12)


def test_snr3_bandwidth_scaling_unsaturated():
    """At fixed source, R ~ dlam and tau_c ~ 1/dlam: N_sig ~ dlam, noise
    ~ dlam^{3/2} -> SNR3 ~ dlam^{-1/2} (narrower is better, unlike g2)."""
    tri = _tri()
    snrs = []
    for dl in (0.5, 2.0):
        obs = Observation(wavelength_nm=600.0, filter_width_nm=dl,
                          t_int_s=3600.0)
        snrs.append(g3_snr(0.1, 2.0, obs, tri).snr)
    assert snrs[0] / snrs[1] == pytest.approx(2.0, rel=1e-6)


def test_snr3_reduces_to_ndds_scaling():
    """SNR3 / [|ggg| (R)^{3/2} tau_c^2 sqrt(T) / sqrt(A_2D)] is constant:
    the Nunez & Domiciano de Souza eq. 8 shape."""
    tri = _tri()
    vals = []
    for mag in (1.0, 3.0):
        r = g3_snr(0.1, mag, OBS, tri)
        pred = (POL_FACTOR_TRIPLE * 2.0 * 0.1 * r.tau_c_s**2
                * np.prod(r.rates_cps) * OBS.t_int_s
                / np.sqrt(np.prod(r.rates_cps) * OBS.t_int_s * r.window_s2))
        vals.append(r.snr / pred)
    assert vals[0] == pytest.approx(vals[1], rel=1e-12)
    assert vals[0] == pytest.approx(1.0, rel=1e-12)


def test_spectral_g3_quadrature_and_methods():
    spec = Spectrograph(lambda_min_nm=450.0, lambda_max_nm=900.0,
                        n_channels=8)
    ana = spectral_g3_snr(BETA_AUR, MAUNAKEA_SUBARU_KECK, spectrograph=spec,
                          vis_method="analytic")
    assert ana.snr_total == pytest.approx(np.sqrt(np.sum(ana.snr**2)),
                                          rel=1e-12)
    fft = spectral_g3_snr(BETA_AUR, MAUNAKEA_SUBARU_KECK, spectrograph=spec,
                          vis_method="fft")
    assert np.allclose(fft.triple_amp, ana.triple_amp, atol=2e-3)
    assert fft.snr_total == pytest.approx(ana.snr_total, rel=0.05)
    # in eclipse: analytic refuses, fft works
    with pytest.raises(ValueError, match="eclipse"):
        spectral_g3_snr(ALGOL, MAUNAKEA_SUBARU_KECK, spectrograph=spec,
                        orbital_phase=0.25, vis_method="analytic")


def test_vlt_array_geometry():
    """The published UT station coordinates reproduce the six pairwise
    separations (46.6, 56.5, 62.4, 89.3, 102.4, 130.2 m) and give four
    triangles."""
    from hbtsim.bispectrum import VLT_UT

    lengths = sorted(float(np.hypot(*b)) for _, _, b in VLT_UT.pairs())
    assert lengths == pytest.approx([46.6, 56.5, 62.4, 89.3, 102.4, 130.2],
                                    abs=0.2)
    tris = VLT_UT.triangles()
    assert len(tris) == 4
    for tri in tris:
        assert np.allclose(tri.baseline_vectors().sum(axis=0), 0.0)


def test_array_g3_quadrature_combination():
    from hbtsim.bispectrum import VLT_UT
    from hbtsim.params import SPICA
    from hbtsim.snr3 import array_g3_snr

    spec = Spectrograph(lambda_min_nm=450.0, lambda_max_nm=900.0,
                        n_channels=6)
    res = array_g3_snr(SPICA, VLT_UT, spectrograph=spec)
    assert len(res.per_triangle) == 4
    assert res.snr_total == pytest.approx(
        np.sqrt(sum(r.snr_total**2 for r in res.per_triangle)), rel=1e-12)
    assert res.snr_total > max(r.snr_total for r in res.per_triangle)


def test_time_to_cos_phi_inversion():
    spec = Spectrograph(lambda_min_nm=450.0, lambda_max_nm=900.0,
                        n_channels=8)
    t = time_to_cos_phi(BETA_AUR, MAUNAKEA_SUBARU_KECK, target_dcos=0.1,
                        spectrograph=spec)
    ref = spectral_g3_snr(BETA_AUR, MAUNAKEA_SUBARU_KECK, spectrograph=spec,
                          t_int_s=t)
    assert ref.snr_total == pytest.approx(10.0, rel=1e-6)
