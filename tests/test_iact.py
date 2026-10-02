"""IACT arrays (hbtsim.iact) and the oblate-star / disk visibilities (hbtsim.single)."""

import numpy as np
import pytest

from hbtsim.iact import (MAGIC_LST1, MAGIC_SII, VERITAS, VERITAS_ANCHOR, VERITAS_TELESCOPE, VEGA_TO_AB_B,
                         binary_vis2_fn, classic_snr, pair_track, single_vis2_fn, veritas_calibrated, vis2_sigma)
from hbtsim.params import MAS, SPICA
from hbtsim.single import GAMMA_CAS, composite_vis, ellipse_vis, gaussian_disk_vis, single_star_vis2
from hbtsim.limbdark import star_disk_visibility


def test_veritas_and_magic_baselines():
    lens = sorted(float(np.hypot(*b)) for _, _, b in VERITAS.pairs())
    assert np.allclose(lens, [81.5, 99.4, 99.4, 108.8, 126.4, 172.5], atol=0.2)
    m = sorted(round(float(np.hypot(*b)), 1) for _, _, b in MAGIC_LST1.pairs())
    assert m == [86.0, 100.0, 100.0]


def test_ellipse_reduces_to_circle_and_scales_affinely():
    b = np.array([[50.0, 0.0], [0.0, 80.0], [60.0, 60.0]])
    circ = single_star_vis2(GAMMA_CAS, np.hypot(b[:, 0], b[:, 1]), 425.0)
    assert np.allclose(ellipse_vis(b, 425.0, GAMMA_CAS.star, 0.532, 1.0, 30.0) ** 2, circ)
    # along the minor axis the ellipse looks like a circle of diameter theta/r
    r, pa = 1.28, 116.0
    e_min = np.array([np.cos(np.radians(pa)), -np.sin(np.radians(pa))])
    bm = 90.0 * e_min[None, :]
    x = np.pi * (0.55 / r) * MAS * 90.0 / 425e-9
    ref = star_disk_visibility(GAMMA_CAS.star, np.array([[x]]), np.array([425.0]))
    assert np.allclose(ellipse_vis(bm, 425.0, GAMMA_CAS.star, 0.55, r, pa), ref)


def test_gaussian_and_composite():
    b = np.array([[30.0, 0.0]])
    fwhm = 2.9
    expect = np.exp(-(np.pi * fwhm * MAS * 30.0 / 425e-9) ** 2 / (4 * np.log(2)))
    assert np.allclose(gaussian_disk_vis(b, 425.0, fwhm), expect)
    v0 = composite_vis(b, 425.0, GAMMA_CAS.star, 0.532, disk_fraction=0.0)
    v1 = composite_vis(b, 425.0, GAMMA_CAS.star, 0.532, disk_fraction=1.0)
    assert np.allclose(v0, ellipse_vis(b, 425.0, GAMMA_CAS.star, 0.532)) and np.allclose(v1, expect)
    vh = composite_vis(b, 425.0, GAMMA_CAS.star, 0.532, disk_fraction=0.2)
    assert np.all((vh < v0) & (vh > v1))


def test_pair_track_shapes_and_phase():
    tr = pair_track(VERITAS, SPICA.dec_deg, 416.0, binary_vis2_fn(SPICA, pupils=False),
                    block_minutes=17.0, phase0=0.1, period_days=SPICA.period_days)
    assert tr.vis2.shape == (6, tr.hour_angle_h.size) and tr.baseline_len_m.max() <= 172.7
    assert np.all((tr.vis2 >= 0) & (tr.vis2 <= 1.0 + 1e-9))
    assert tr.phase[1] - tr.phase[0] == pytest.approx(tr.block_s / 86400.0 / SPICA.period_days)
    assert abs(tr.position_angle_deg).max() <= 180.0
    t2 = pair_track(MAGIC_LST1, GAMMA_CAS.dec_deg, 425.0,
                    single_vis2_fn(lambda bv, nm: composite_vis(bv, nm, GAMMA_CAS.star, 0.532)), block_minutes=30.0)
    assert t2.vis2.shape[0] == 3 and np.all(t2.phase == 0)


def test_sensitivity_calibration_and_reach():
    vb = veritas_calibrated()
    s = vis2_sigma(VERITAS_ANCHOR["mag_b"] + VEGA_TO_AB_B, VERITAS_TELESCOPE.area_m2, VERITAS_TELESCOPE.area_m2,
                   vb, VERITAS_ANCHOR["t_s"])
    assert s == pytest.approx(VERITAS_ANCHOR["sigma_vis2"], rel=1e-9)
    assert 0.05 < vb.q < 0.3
    # MAGIC says ~4 B mag is realistic: S/N at |V|^2 = 1 in 10 h should be tens
    snr = classic_snr(1.0, 4.0 + VEGA_TO_AB_B, 236.0, 236.0, MAGIC_SII, 10 * 3600.0)
    assert 10 < snr < 200
