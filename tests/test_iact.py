"""IACT arrays (hbtsim.iact) and the oblate-star / disk visibilities (hbtsim.single)."""

import numpy as np
import pytest

from hbtsim.iact import (binary_vis2_fn, calibrated, classic_snr, pair_track, single_vis2_fn,
                         track_sigma, vis2_sigma)
from hbtsim.params import MAS
from hbtsim.single import composite_vis, ellipse_vis, gaussian_disk_vis, single_star_vis2
from hbtsim.limbdark import star_disk_visibility


def test_veritas_and_magic_baselines(veritas, magic_lst1):
    lens = sorted(float(np.hypot(*b)) for _, _, b in veritas.pairs())
    assert np.allclose(lens, [81.5, 99.4, 99.4, 108.8, 126.4, 172.5], atol=0.2)
    # surveyed: MAGIC-I-II 85.3 m; LST-1 53.7 m from MAGIC-II and 105.9 m from MAGIC-I
    m = sorted(round(float(np.hypot(*b)), 1) for _, _, b in magic_lst1.pairs())
    assert m == [53.7, 85.3, 105.9]


def test_ctao_north_arrays(catalog):
    lst4, a27, alpha = (catalog.load_array(n) for n in ("ctao_n_lst4", "ctao_n_2027", "ctao_n_alpha"))
    assert [len(x.stations) for x in (lst4, a27, alpha)] == [4, 5, 13]
    assert len(alpha.pairs()) == 78
    L = sorted(float(np.hypot(*b)) for _, _, b in alpha.pairs())
    assert 68 < L[0] < 69 and 560 < L[-1] < 561
    assert sum(s.telescope.diameter_m == 23.0 for s in alpha.stations) == 4
    assert {s.telescope.diameter_m for s in a27.stations} == {23.0, 11.5}
    # LST-1 is the same station in every ORM array
    lst1 = next(s for s in alpha.stations if s.name == "LST-1")
    for n in ("magic_lst1", "magic_lst4", "ctao_n_lst4"):
        s = next(s for s in catalog.load_array(n).stations if s.name == "LST-1")
        assert (s.east_m, s.north_m) == (lst1.east_m, lst1.north_m)


def test_ellipse_reduces_to_circle_and_scales_affinely(gamma_cas):
    b = np.array([[50.0, 0.0], [0.0, 80.0], [60.0, 60.0]])
    circ = single_star_vis2(gamma_cas, np.hypot(b[:, 0], b[:, 1]), 425.0)
    assert np.allclose(ellipse_vis(b, 425.0, gamma_cas.star, 0.532, 1.0, 30.0) ** 2, circ)
    # along the minor axis the ellipse looks like a circle of diameter theta/r
    r, pa = 1.28, 116.0
    e_min = np.array([np.cos(np.radians(pa)), -np.sin(np.radians(pa))])
    bm = 90.0 * e_min[None, :]
    x = np.pi * (0.55 / r) * MAS * 90.0 / 425e-9
    ref = star_disk_visibility(gamma_cas.star, np.array([[x]]), np.array([425.0]))
    assert np.allclose(ellipse_vis(bm, 425.0, gamma_cas.star, 0.55, r, pa), ref)


def test_gaussian_and_composite(gamma_cas):
    b = np.array([[30.0, 0.0]])
    fwhm = 2.9
    expect = np.exp(-(np.pi * fwhm * MAS * 30.0 / 425e-9) ** 2 / (4 * np.log(2)))
    assert np.allclose(gaussian_disk_vis(b, 425.0, fwhm), expect)
    v0 = composite_vis(b, 425.0, gamma_cas.star, 0.532, disk_fraction=0.0)
    v1 = composite_vis(b, 425.0, gamma_cas.star, 0.532, disk_fraction=1.0)
    assert np.allclose(v0, ellipse_vis(b, 425.0, gamma_cas.star, 0.532)) and np.allclose(v1, expect)
    vh = composite_vis(b, 425.0, gamma_cas.star, 0.532, disk_fraction=0.2)
    assert np.all((vh < v0) & (vh > v1))


def test_pair_track_shapes_and_phase(veritas, magic_lst1, spica, gamma_cas):
    tr = pair_track(veritas, spica.dec_deg, 416.0, binary_vis2_fn(spica, pupils=False),
                    block_minutes=17.0, phase0=0.1, period_days=spica.period_days)
    assert tr.vis2.shape == (6, tr.hour_angle_h.size) and tr.baseline_len_m.max() <= 172.7
    assert np.all((tr.vis2 >= 0) & (tr.vis2 <= 1.0 + 1e-9))
    assert tr.phase[1] - tr.phase[0] == pytest.approx(tr.block_s / 86400.0 / spica.period_days)
    assert abs(tr.position_angle_deg).max() <= 180.0
    t2 = pair_track(magic_lst1, gamma_cas.dec_deg, 425.0,
                    single_vis2_fn(lambda bv, nm: composite_vis(bv, nm, gamma_cas.star, 0.532)),
                    block_minutes=30.0)
    assert t2.vis2.shape[0] == 3 and np.all(t2.phase == 0)


def test_sensitivity_calibration_and_reach(catalog, veritas_sii, veritas_tel, magic_sii):
    anchor = veritas_sii.anchor
    vb = calibrated(veritas_sii, veritas_tel.area_m2)
    s = vis2_sigma(anchor.mag_ab, veritas_tel.area_m2, veritas_tel.area_m2, vb, anchor.t_s)
    assert s == pytest.approx(anchor.sigma_vis2, rel=1e-9)
    assert 0.05 < vb.q < 0.3
    # MAGIC says ~4 B mag is realistic: S/N at |V|^2 = 1 in 10 h should be tens
    vega_to_ab_b = catalog.bands()["B"]["vega_to_ab"]
    snr = classic_snr(1.0, 4.0 + vega_to_ab_b, 236.0, 236.0, magic_sii, 10 * 3600.0, q=0.304, alpha=0.295)
    assert 10 < snr < 200
    with pytest.raises(ValueError):            # per-station backend without the stations
        classic_snr(1.0, 4.0, 236.0, 236.0, magic_sii, 3600.0)


def test_per_station_backends(catalog, magic_lst1, gamma_cas):
    """MAGIC + LST-1 and the CTAO-North arrays: alpha and q per station,
    the published sensitivity ratios, and the Stage II channel gain."""
    from hbtsim.iact import pair_snr
    magic_sii = catalog.load_backend("magic_sii")
    assert magic_sii.per_station and magic_sii.alpha is None and magic_sii.q is None
    m1, m2, l1 = magic_lst1.stations
    assert magic_sii.station_alpha_q(m1) == (0.295, 0.304)          # Abe et al. 2024, Table 4
    assert magic_sii.station_alpha_q(l1) == (0.40, 0.64)            # ctao_pmt, ctao_lst_23m
    mag = float(gamma_cas.ab_mag(425.0))
    s_mm = pair_snr(1.0, mag, m1, m2, magic_sii, 3600.0)
    s_ml = pair_snr(1.0, mag, m1, l1, magic_sii, 3600.0)
    assert s_ml / s_mm == pytest.approx(np.sqrt(370 * 0.40 * 0.64 / (236 * 0.295 * 0.304)), rel=1e-9)
    # track_sigma draws the same numbers through the track's stations
    tr = pair_track(magic_lst1, gamma_cas.dec_deg, 425.0,
                    single_vis2_fn(lambda bv, nm: composite_vis(bv, nm, gamma_cas.star, 0.532)), block_minutes=30.0)
    sig = track_sigma(tr, mag, magic_sii, 3600.0)
    assert sig[0] == pytest.approx(1 / s_mm) and sig[1] == pytest.approx(1 / s_ml) and sig[2] == pytest.approx(1 / s_ml)
    # Abe et al. 2025: MAGIC + LST-1 ~3.6x MAGIC alone, MAGIC + 4 LSTs ~11x (S/N added in quadrature over pairs)
    def total(array):
        return np.sqrt(sum(pair_snr(1.0, mag, array.stations[i], array.stations[j], magic_sii, 3600.0) ** 2
                           for i, j, _ in array.pairs()))
    assert 2.8 < total(magic_lst1) / s_mm < 3.6
    assert 9 < total(catalog.load_array("magic_lst4")) / s_mm < 13
    # CTAO Stage I / II on an LST-MST pair: alpha from the detector or the backend, q per telescope, sqrt(10) channels
    s1, s2 = catalog.load_backend("ctao_sii_stage1"), catalog.load_backend("ctao_sii_stage2")
    alpha = catalog.load_array("ctao_n_alpha")
    lst, mst = alpha.stations[0], alpha.stations[4]
    assert (lst.telescope.area_m2, mst.telescope.area_m2) == (370.0, 88.0)
    assert s1.station_alpha_q(lst) == (0.40, 0.64) and s1.station_alpha_q(mst) == (0.40, 0.73)
    assert s2.station_alpha_q(lst) == (0.45, 0.64) and s2.n_channels == 10
    r = pair_snr(1.0, mag, lst, mst, s2, 3600.0) / pair_snr(1.0, mag, lst, mst, s1, 3600.0)
    assert r == pytest.approx(0.45 / 0.40 * np.sqrt(2e9 / 4e8 * 10) * 1.15 / 1.05 * 1.0 / 0.91, rel=1e-9)
