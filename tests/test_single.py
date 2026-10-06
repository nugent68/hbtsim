"""Single-star path (hbtsim.single) and the shared g2 budget."""

import os
from dataclasses import replace

import numpy as np
import pytest

from hbtsim.limbdark import visibility_ld_disk
from hbtsim.params import MAS
from hbtsim.single import (attach_newera_single, prepare_single, single_star_vis2,
                           spectral_g2_snr_single, ud_diameter_per_channel, ud_dtheta_dvis2,
                           ud_vis2)
from hbtsim.snr import Observation, Spectrograph, g2_snr

NM = np.array([410.0, 450.0, 500.0, 540.0])
SPEC = Spectrograph(lambda_min_nm=400.0, lambda_max_nm=550.0, n_channels=30, throughput=0.6)
NEWERA = "data/newera"
has_newera = pytest.mark.skipif(not os.path.isdir(NEWERA), reason="NewEra tables not present")


def test_linear_law_matches_analytic(vega):
    b = np.array([5.0, 10.0])
    v2 = single_star_vis2(vega, b, NM)
    x = np.pi * vega.theta_ld_mas * MAS * b[None, :] / (NM[:, None] * 1e-9)
    u = np.array([vega.star.ld_coeff(l) for l in NM])[:, None]
    assert np.allclose(v2, visibility_ld_disk(x, u) ** 2, rtol=1e-10)
    assert vega.drawn_diameter_mas == vega.theta_ld_mas       # no profile: no r_outer


@pytest.mark.parametrize("pupils", [None, (4.0, 4.0)])
def test_ud_round_trip(pupils):
    th = np.array([3.0, 3.1, 3.2, 3.3])
    v2 = ud_vis2(th, 12.0, NM, pupils)[:, 0]
    assert np.allclose(ud_diameter_per_channel(v2, 12.0, NM, pupils), th, rtol=1e-9)


def test_point_inversion_of_smeared_data_is_biased():
    """At D/B ~ 0.4 (4 m pupils, 10 m baseline) smearing must be inverted."""
    th = np.full(NM.size, 3.3)
    v2 = ud_vis2(th, 10.0, NM, (4.0, 4.0))[:, 0]
    naive = ud_diameter_per_channel(v2, 10.0, NM, None)
    assert np.all(np.abs(naive / th - 1) > 5e-3)
    assert np.allclose(ud_diameter_per_channel(v2, 10.0, NM, (4.0, 4.0)), th, rtol=1e-9)


def test_dtheta_dvis2_negative_on_first_lobe():
    d = ud_dtheta_dvis2(np.full(NM.size, 3.3), 12.0, NM, (4.0, 4.0))
    assert np.all(d < 0)


def test_single_budget_matches_g2_snr(vega, eonsii_tel, eonsii_spad):
    r = spectral_g2_snr_single(vega, 12.0, SPEC, telescope1=eonsii_tel,
                               detector1=eonsii_spad, pupils=None, enforce_readout=False)
    nm, w = SPEC.channel_centers_nm, SPEC.channel_widths_nm
    obs = Observation(wavelength_nm=nm, filter_width_nm=w, t_int_s=3600.0,
                      backend_throughput=SPEC.throughput)
    ref = g2_snr(single_star_vis2(vega, 12.0, nm)[:, 0], vega.ab_mag(nm), obs,
                 telescope1=eonsii_tel, detector1=replace(eonsii_spad, n_pixels=1))
    assert np.allclose(r.snr, ref.snr, rtol=1e-12)


def test_channel_mask_readout(vega, eonsii_tel, eonsii_spad):
    kw = dict(telescope1=eonsii_tel, detector1=eonsii_spad, pupils=None)
    full = spectral_g2_snr_single(vega, 12.0, SPEC, **kw)
    mask = np.zeros(SPEC.n_channels, bool)
    mask[:3] = True
    sub = spectral_g2_snr_single(vega, 12.0, SPEC, channel_mask=mask, **kw)
    assert full.readout_limited and sub.readout_scale > full.readout_scale
    assert np.all(sub.snr[~mask] == 0.0) and np.all(sub.snr[mask] > full.snr[mask])
    assert sub.total_rate_cps[0] < full.total_rate_cps[0]


@has_newera
def test_newera_sirius_vega(sirius_a, vega):
    s, rep = attach_newera_single(sirius_a, NEWERA)
    assert "interpolated" in rep
    assert 1.004 <= s.star.radius_scale <= 1.006
    assert s.drawn_diameter_mas == pytest.approx(sirius_a.theta_ld_mas * s.star.radius_scale)
    assert abs(s.v_check()) < 0.15
    v, _ = attach_newera_single(vega, NEWERA)
    p = prepare_single(v, SPEC)
    assert np.allclose(p.star.flux_table.wavelength_nm, SPEC.channel_centers_nm)
    # Balmer cores look larger than the neighbouring continuum
    nm = np.array([420.0, 434.17, 445.0])
    v2 = single_star_vis2(p, 15.0, nm, (4.0, 4.0))[:, 0]
    th = ud_diameter_per_channel(v2, 15.0, nm, (4.0, 4.0))
    assert th[1] > th[0] and th[1] > th[2]
