"""EON-SII triangle, per-epoch grids, the auto vis method, the phase
statistic, shared geometry across backends and multi-night campaigns."""

import warnings

import numpy as np
import pytest

from hbtsim.bispectrum import equilateral_array, pair_array
from hbtsim.geometry import hour_angle_window
from hbtsim.orbit import positions_at
from hbtsim.params import GridConfig
from hbtsim.snr import Spectrograph
from hbtsim.snr3 import (Backend, auto_block_minutes, campaign_g3_snr,
                         campaign_nights_to_precision, geometry_samples,
                         resolve_block_method, spectral_g3_snr, track_g3_snr)

SPEC = Spectrograph(lambda_min_nm=420.0, lambda_max_nm=520.0, n_channels=5, throughput=0.6)
DV_CONJ = 0.971          # delta Vel conjunction (rho ~ 0.21 mas)
DV_QUAD = 0.2            # well away from both conjunctions
EONSII_NAMES = ("EON-1", "EON-2", "EON-3")


def _triangle(catalog, side_m, **generator_params):
    """The EON-SII equilateral triangle at Paranal with generator overrides."""
    return catalog.load_array("eonsii_triangle_paranal", side_m=side_m, **generator_params)


def test_eonsii_pair_site(catalog, teide, paranal, eonsii_tel, eonsii_mcp):
    assert catalog.load_array("eonsii_pair_teide", baseline_m=100.0).site == teide
    other = pair_array(100.0, eonsii_tel, eonsii_mcp, site=paranal, names=EONSII_NAMES[:2])
    assert other.site == paranal


def test_deltavel_windows(delta_vel, paranal, teide):
    h0, h1 = hour_angle_window(delta_vel.dec_deg, paranal.latitude_deg)
    assert h1 == pytest.approx(4.82, abs=0.01) and h0 == pytest.approx(-h1)
    assert hour_angle_window(delta_vel.dec_deg, teide.latitude_deg) == (0, 0)


def test_eonsii_triangle_geometry(catalog, paranal, teide, eonsii_tel, eonsii_mcp, eonsii_spad):
    arr = _triangle(catalog, 20.0, pa_deg=30.0)
    assert arr.site == paranal
    (tri,) = arr.triangles()
    assert np.allclose(tri.baseline_lengths(), 20.0)
    assert np.allclose(tri.baseline_vectors().sum(axis=0), 0.0)
    assert all(st.detector == eonsii_mcp for st in arr.stations)
    min_spacing = catalog.raw("array", "eonsii_triangle_paranal")["generator"]["min_spacing_m"]
    other = equilateral_array(10.0, eonsii_tel, eonsii_spad, site=teide, names=EONSII_NAMES,
                              min_spacing_m=min_spacing)
    assert other.site == teide
    with pytest.raises(ValueError):
        _triangle(catalog, min_spacing - 0.1)


def test_fit_epoch_small_at_conjunction(delta_vel):
    pos = positions_at(delta_vel, DV_CONJ)
    g = GridConfig().fit_epoch(delta_vel, pos)
    assert 256 <= g.n <= 512
    assert g.n < GridConfig().fit_orbit(delta_vel).n / 2
    assert g.pixel_scale_mas == GridConfig().pixel_scale_mas
    assert g.half_extent_mas > max(abs(float(pos.x1)), abs(float(pos.x2)))


def test_resolve_block_method(delta_vel):
    assert resolve_block_method(delta_vel, positions_at(delta_vel, DV_CONJ), "auto") == "render"
    assert resolve_block_method(delta_vel, positions_at(delta_vel, DV_QUAD), "auto") == "analytic"
    assert resolve_block_method(delta_vel, positions_at(delta_vel, DV_CONJ), "analytic") == "analytic"


def test_phase_statistic_complements_amplitude(catalog, delta_vel):
    (tri,) = _triangle(catalog, 20.0).triangles()
    r = spectral_g3_snr(delta_vel, tri, spectrograph=SPEC, orbital_phase=DV_QUAD)
    assert r.snr_amplitude**2 + r.snr_phase**2 == pytest.approx(r.snr_total**2, rel=1e-9)


def test_samples_injection_identical(catalog, delta_vel):
    (tri,) = _triangle(catalog, 20.0).triangles()
    direct = spectral_g3_snr(delta_vel, tri, spectrograph=SPEC, orbital_phase=DV_QUAD)
    smp = geometry_samples(delta_vel, tri, SPEC, DV_QUAD)
    shared = spectral_g3_snr(delta_vel, tri, spectrograph=SPEC, orbital_phase=DV_QUAD, samples=smp)
    assert np.array_equal(direct.snr, shared.snr)
    assert shared.snr_phase == direct.snr_phase
    with pytest.raises(ValueError):
        spectral_g3_snr(delta_vel, tri, spectrograph=Spectrograph(n_channels=7), samples=smp)


def _quiet_track(*a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return track_g3_snr(*a, **k)


def test_backends_share_geometry_and_match_single_runs(catalog, delta_vel, eonsii_mcp, eonsii_spad):
    arr = _triangle(catalog, 20.0)
    backends = (Backend("mcp", SPEC, eonsii_mcp), Backend("spad-pbs", SPEC, eonsii_spad, "pbs"))
    both = _quiet_track(delta_vel, arr, SPEC, block_minutes=60.0, phase0=DV_QUAD, backends=backends)
    one = _quiet_track(delta_vel, _triangle(catalog, 20.0, detector=eonsii_spad), SPEC,
                       block_minutes=60.0, phase0=DV_QUAD, polarization_mode="pbs")
    assert both["spad-pbs"].snr_phase == pytest.approx(one.snr_phase, rel=1e-12)
    assert both["spad-pbs"].snr_amplitude == pytest.approx(one.snr_amplitude, rel=1e-12)
    assert both["spad-pbs"].snr_total > both["mcp"].snr_total


def test_auto_block_minutes_bounds_drift(catalog, delta_vel, paranal):
    arr = _triangle(catalog, 40.0)
    win = hour_angle_window(delta_vel.dec_deg, paranal.latitude_deg)
    m = auto_block_minutes(delta_vel, arr, SPEC.lambda_min_nm, win, DV_QUAD)
    assert 5.0 <= m <= 60.0
    tr = _quiet_track(delta_vel, arr, SPEC, block_minutes="auto", phase0=DV_QUAD)
    assert tr.block_minutes == pytest.approx(m, rel=0.05)
    assert max(b.drift_cycles_max for b in tr.blocks) <= 0.125 * 1.25


def test_campaign_quadrature_phase0_and_cache(tmp_path, catalog, delta_vel, paranal, eonsii_mcp):
    arr = _triangle(catalog, 20.0)
    backends = (Backend("mcp", SPEC, eonsii_mcp),)
    phases = (0.15, 0.25)
    kw = dict(phases_mid=phases, block_minutes=60.0, vis_method="analytic")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = campaign_g3_snr(delta_vel, arr, backends, cache_dir=tmp_path, **kw)
        again = campaign_g3_snr(delta_vel, arr, backends, cache_dir=tmp_path, **kw)
    assert len(list(tmp_path.glob("night_*.json"))) == 2
    per = c.per_night("mcp", "phase")
    assert c.snr("mcp", "phase") == pytest.approx(np.sqrt(np.sum(per**2)))
    assert again.snr("mcp", "phase") == c.snr("mcp", "phase")
    h0, h1 = hour_angle_window(delta_vel.dec_deg, paranal.latitude_deg)
    half = 0.5 * (h1 - h0) / 24.0 / delta_vel.period_days
    assert c.nights[0].phase0 == pytest.approx(phases[0] - half)
    n = campaign_nights_to_precision(c, "mcp", 0.1, "phase")
    assert n == pytest.approx(2 * (10.0 / c.snr("mcp", "phase"))**2)
    with pytest.raises(ValueError):
        campaign_g3_snr(delta_vel, arr, backends)


def test_campaign_n_nights_cadence(spica):
    from hbtsim.snr3 import _night_phases
    ph = _night_phases(spica, None, 3, 0.9, 1.0)
    assert np.allclose(ph, np.mod(0.9 + np.arange(3) / spica.period_days, 1.0))


def test_analytic_and_epoch_render_agree_at_eclipse_guard(catalog, delta_vel):
    """Just outside the eclipse guard both paths are valid and must agree;
    the render runs on the cheap per-epoch grid."""
    from hbtsim.params import in_eclipse
    ph = max(p for p in np.arange(0.93, 0.971, 0.0005)
             if not in_eclipse(delta_vel, positions_at(delta_vel, p)))
    (tri,) = _triangle(catalog, 20.0).triangles()
    a = geometry_samples(delta_vel, tri, SPEC, ph, "analytic")
    r = geometry_samples(delta_vel, tri, SPEC, ph, "render", grid="epoch")
    assert r.grid.n <= 512 and r.vis_method == "render"
    assert np.allclose(r.ts.triple_amp, a.ts.triple_amp, rtol=1e-4)
    assert np.allclose(r.ts.cos_phi_c, a.ts.cos_phi_c, atol=1e-4)
    assert np.allclose(r.ts.vis2_pairs, a.ts.vis2_pairs, rtol=1e-4)
