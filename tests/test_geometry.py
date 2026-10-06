"""Sky orientation (Omega), sites, uv projection and the uv track."""

import warnings

import numpy as np
import pytest

from hbtsim.bispectrum import Array, Station, Triangle, closure_phase
from hbtsim.catalog import Catalog
from hbtsim.geometry import (Site, altitude_rad, drift_loss, enu_to_uvw, fringe_drift_cycles,
                             hour_angle_blocks, hour_angle_window)
from hbtsim.orbit import sky_positions
from hbtsim.params import MAS
from hbtsim.snr import Spectrograph
from hbtsim.snr3 import nights_to_precision, track_g3_snr

from dataclasses import replace

CAT = Catalog(env=False)
SYSTEM_NAMES = ("betaaur", "algol", "spica", "deltavel")


# ---------------------------------------------------------------------------
# Omega
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("system", [CAT.load_target(k) for k in SYSTEM_NAMES], ids=SYSTEM_NAMES)
def test_ascending_node_at_omega(system):
    """The secondary crosses the sky plane going behind the primary at
    position angle Omega, and PA increases with time for i < 90 deg."""
    psi = np.linspace(0, 2 * np.pi, 7201)
    p = sky_positions(psi, system)
    front = np.asarray(p.front2)
    k = np.where(front[:-1] & ~front[1:])[0][0]      # front -> behind
    pa = p.position_angle_deg
    assert pa[k] == pytest.approx(system.node_pa_deg, abs=0.05)
    dpa = np.diff(np.unwrap(np.radians(pa)))
    direct = system.inclination_deg < 90.0
    assert (dpa[k + 5] > 0) == direct


def test_omega_invariants(algol):
    """rho, front2 (hence eclipses and lightcurves) are Omega-independent;
    positions are rotated/reflected only."""
    psi = np.linspace(0, 2 * np.pi, 361)
    base = sky_positions(psi, algol)
    for om in (None, 0.0, 123.4):
        alt = sky_positions(psi, replace(algol, node_pa_deg=om))
        assert np.allclose(alt.rho, base.rho)
        assert np.array_equal(alt.front2, base.front2)
        assert np.allclose(np.hypot(alt.x2 - alt.x1, alt.y2 - alt.y1), base.rho)


def test_sky_rotation_equals_array_rotation(algol, maunakea_tri):
    """Rotating the sky by +delta (Omega -> Omega + delta) is the same
    closure phase as rotating the array by -delta (counterclockwise in
    (E, N))."""
    delta = 37.0
    sys_rot = replace(algol, node_pa_deg=algol.node_pa_deg + delta)
    d = np.radians(delta)

    def rot(st):
        e, n = st.east_m, st.north_m
        return replace(st, east_m=e * np.cos(d) - n * np.sin(d),
                       north_m=e * np.sin(d) + n * np.cos(d))

    tri_rot = Triangle(tuple(rot(s) for s in maunakea_tri.stations))
    a = closure_phase(sys_rot, maunakea_tri, 600.0, 0.1)
    b = closure_phase(algol, tri_rot, 600.0, 0.1)
    assert np.allclose(a.gammas, b.gammas, atol=1e-10)
    assert a.phi_c == pytest.approx(b.phi_c, abs=1e-10)


def test_position_angle_property(spica):
    psi = np.array([0.0, 1.0])
    p = sky_positions(psi, spica)
    pa = p.position_angle_deg
    e = p.x2 - p.x1
    n = p.y2 - p.y1
    assert np.allclose(np.sin(np.radians(pa)) * p.rho, e)
    assert np.allclose(np.cos(np.radians(pa)) * p.rho, n)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
def test_zenith_identity():
    enu = np.array([[145.8, 43.3, 0.0], [10.0, -20.0, 5.0]])
    phi = np.radians(19.826)
    uvw = enu_to_uvw(enu, 0.0, phi, phi)
    assert np.allclose(uvw, enu, atol=1e-12)


def test_hand_computed_projection():
    """H = 4 h (60 deg), dec = -11 deg, lat = -24.6 deg, an E-W baseline:
    u = E cos H, v = E sin dec sin H, w = -E cos dec sin H."""
    E = 100.0
    H, d, phi = np.radians(60.0), np.radians(-11.0), np.radians(-24.6)
    u, v, w = enu_to_uvw(np.array([E, 0.0, 0.0]), H, d, phi)
    assert u == pytest.approx(E * np.cos(H))
    assert v == pytest.approx(E * np.sin(d) * np.sin(H))
    assert w == pytest.approx(-E * np.cos(d) * np.sin(H))
    # projected length never exceeds the physical length
    for HH in np.linspace(-np.pi, np.pi, 25):
        uvw = enu_to_uvw(np.array([50.0, 80.0, 3.0]), HH, d, phi)
        assert np.hypot(uvw[0], uvw[1]) <= np.linalg.norm([50.0, 80.0, 3.0]) + 1e-9


def test_vlt_at_spica_transit(spica, algol, vlt_ut, maunakea_tri, paranal, maunakea):
    """UT1-UT4 (130.2 m on the ground) is foreshortened to ~129.4 m at
    Spica's transit (dec -11 from lat -24.6: the source is 13.5 deg
    from the zenith)."""
    proj = vlt_ut.projected(0.0, spica.dec_deg)
    l14 = [np.hypot(*b) for i, j, b in proj.pairs() if (i, j) == (0, 3)][0]
    assert l14 == pytest.approx(129.4, abs=0.2)
    assert proj.site == paranal
    tri = maunakea_tri.projected(2.0, algol.dec_deg)
    assert tri.site == maunakea
    assert tri.baseline_lengths().max() < maunakea_tri.baseline_lengths().max()
    with pytest.raises(ValueError, match="Site"):
        Triangle(maunakea_tri.stations).projected(0.0, 10.0)


def test_hour_angle_windows(spica, algol, paranal, maunakea):
    lo, hi = hour_angle_window(spica.dec_deg, paranal.latitude_deg)
    assert hi == pytest.approx(4.13, abs=0.02) and lo == -hi
    assert hour_angle_window(algol.dec_deg, paranal.latitude_deg) == (0.0, 0.0)
    assert hour_angle_window(algol.dec_deg, maunakea.latitude_deg)[1] > 4.0
    assert hour_angle_window(-80.0, paranal.latitude_deg, min_alt_deg=5.0) == (-12.0, 12.0)
    assert altitude_rad(0.0, np.radians(spica.dec_deg), paranal.latitude_rad) == \
        pytest.approx(np.radians(90.0 - abs(spica.dec_deg - paranal.latitude_deg)))
    mids = hour_angle_blocks(-2.0, 2.0, 60.0)
    assert np.allclose(mids, [-1.5, -0.5, 0.5, 1.5])
    assert hour_angle_blocks(1.0, 1.0, 30.0).size == 0


def test_fringe_drift_and_loss():
    sep = np.array([3.0, 0.0]) * MAS
    b0 = np.array([[100.0, 0.0]])
    b1 = np.array([[100.0 + 4.0, 0.0]])   # 4 m along the separation
    cyc = fringe_drift_cycles(b0, b1, sep, 400e-9)
    assert cyc[0] == pytest.approx(4.0 * 3.0 * MAS / 400e-9)
    assert drift_loss(0.0) == 1.0
    assert drift_loss(1.0) == pytest.approx(0.0, abs=1e-12)
    assert 0.98 < drift_loss(0.1) < 0.99


def test_vlt_60min_block_flagged_10min_not(spica, algol, vlt_ut):
    spec = Spectrograph(lambda_min_nm=400.0, lambda_max_nm=900.0, n_channels=6)
    with pytest.warns(UserWarning, match="drift"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*(extrapolated|dead-time).*")
            t60 = track_g3_snr(spica, vlt_ut, spec, block_minutes=60.0)
    assert t60.drift_flagged and t60.n_blocks == 9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings("error", message=".*drift.*")
        t10 = track_g3_snr(spica, vlt_ut, spec, block_minutes=10.0)
    assert not t10.drift_flagged
    assert t10.snr_amplitude > t60.snr_amplitude     # less sinc loss
    assert t10.t_total_s == pytest.approx(t60.t_total_s)
    assert 0 < t10.snr_amplitude <= t10.snr_total
    n = nights_to_precision(t10, 0.1)
    assert n == pytest.approx((10.0 / t10.snr_amplitude) ** 2)
    # a source below the horizon: empty track
    empty = track_g3_snr(algol, vlt_ut, spec)
    assert empty.n_blocks == 0 and nights_to_precision(empty) == np.inf


def test_track_uses_triangle_or_array(algol, maunakea_tri):
    spec = Spectrograph(lambda_min_nm=450.0, lambda_max_nm=900.0, n_channels=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tri = track_g3_snr(algol, maunakea_tri, spec, block_minutes=20.0,
                           hour_angle_window_h=(-1.0, 0.0), vis_method="render")
    assert tri.n_blocks == 3
    assert all(len(b.per_triangle) == 1 for b in tri.blocks)
    # phases advance with time
    ph = [b.orbital_phase for b in tri.blocks]
    assert np.all(np.diff(ph) > 0)
    assert ph[-1] - ph[0] == pytest.approx(40.0 / 60.0 / 24.0 / algol.period_days)
