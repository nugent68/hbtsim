"""Finite-aperture averaging (hbtsim.aperture) against closed forms, and
its plumbing through the spectral, bispectrum and SNR paths."""

import numpy as np
import pytest

from hbtsim.aperture import (PupilQuadrature, TripleQuadrature, airy_amplitude,
                             circle_overlap_area, disk_quadrature,
                             fringe_period_m, fringe_smearing_factor,
                             point_quadrature, pupil_pair_quadrature,
                             pupil_pair_quadrature_for, quadrature_order,
                             resolve_pupils)
from hbtsim.bispectrum import closure_phase, spectral_triple
from hbtsim.orbit import SkyPositions, sky_positions
from hbtsim.params import MAS, GridConfig
from hbtsim.snr import Spectrograph, spectral_g2_snr
from hbtsim.spectral import spectral_vis2

LAM = 400e-9


def _pos(system, psi):
    return SkyPositions(*(np.asarray(v) for v in sky_positions(psi, system)))


def test_circle_overlap_limits():
    assert circle_overlap_area(1.0, 1.0, 0.0) == pytest.approx(np.pi)
    assert circle_overlap_area(1.0, 2.0, 0.5) == pytest.approx(np.pi)   # inside
    assert circle_overlap_area(1.0, 1.0, 2.0) == 0.0
    assert circle_overlap_area(1.0, 1.0, 3.0) == 0.0
    # half-overlap identity: two unit circles at d = 1 share ~1.228 units
    assert circle_overlap_area(1.0, 1.0, 1.0) == pytest.approx(1.2283697, rel=1e-6)


@pytest.mark.parametrize("d1,d2", [(10.0, 10.0), (8.2, 10.0), (1.0, 1.0),
                                   (4.0, 4.0), (3.4, 4.0)])
@pytest.mark.parametrize("d_over_p", [0.1, 0.3, 0.5, 0.6, 0.8, 1.2, 1.74, 2.5])
def test_pair_quadrature_vs_airy(d1, d2, d_over_p):
    """A pure fringe of period P smeared by pupils D1, D2 keeps the
    contrast A(pi D1/P) A(pi D2/P).  Beyond the validated D/P = 0.8 the
    fringe-aware rule raises its order (and warns)."""
    P = max(d1, d2) / d_over_p
    rho = LAM / P
    if d_over_p > 0.8:
        with pytest.warns(UserWarning, match="D/P"):
            q = pupil_pair_quadrature_for(d1, d2, P)
        assert q.n_points > 60
    else:
        q = pupil_pair_quadrature_for(d1, d2, P)
        assert q.n_points == 60
    assert q.weights.sum() == pytest.approx(1.0)
    B = np.array([[37.0, 11.0]])
    pts = q.points(B)[0]
    v2 = 0.5 * (1.0 + np.cos(2 * np.pi * pts[:, 0] * rho / LAM))
    exact = 0.5 * (1.0 + np.cos(2 * np.pi * B[0, 0] * rho / LAM)
                   * fringe_smearing_factor(d1, d2, rho, LAM))
    assert abs(q.reduce(v2) - exact) < 3e-4


def test_quadrature_order_and_refusal():
    assert quadrature_order(0.5) == (5, 12)
    assert quadrature_order(0.8) == (5, 12)
    assert quadrature_order(1.6) == (10, 24)
    assert quadrature_order(1.74, base=(2, 7), validated=0.7) == (5, 18)
    with pytest.raises(ValueError, match="D/P"):
        quadrature_order(4.0)
    with pytest.raises(ValueError, match="D/P"):
        pupil_pair_quadrature_for(8.2, 8.2, 2.0)
    # no period -> base rule, no warning
    assert pupil_pair_quadrature_for(8.2, 8.2, None).n_points == 60
    assert resolve_pupils(True, (8.2, 8.2), None).n_points == 60
    # fringe period helper: shortest period over the band
    assert fringe_period_m(1.0, [400.0, 800.0]) == pytest.approx(400e-9 / MAS)
    assert np.isinf(fringe_period_m(0.0, 400.0))


@pytest.mark.parametrize("d_over_p", [0.5, 1.0, 1.5])
def test_triple_quadrature_scaled_vs_closed_form(d_over_p):
    """Two point sources on 4 m pupils: the fringe-aware triple rule
    matches the closed form beyond the base validation range."""
    d = (4.0, 4.0, 4.0)
    B = np.array([[30.0, 0.0], [-15.0, 26.0], [-15.0, -26.0]])
    P = 4.0 / d_over_p
    rho = LAM / P
    th = [np.array([0.3, 0.1]) * rho, np.array([-0.6, -0.2]) * rho]
    f = [1.0, 0.4]
    if d_over_p > 0.7:
        with pytest.warns(UserWarning, match="D/P"):
            tq = TripleQuadrature.for_fringe(*d, P / np.linalg.norm(th[0] - th[1]) * rho)
    else:
        tq = TripleQuadrature.for_fringe(*d, P / np.linalg.norm(th[0] - th[1]) * rho)
    pts = tq.flat_points(B)
    gam = sum(fs * np.exp(-2j * np.pi * (pts @ ts) / LAM)
              for fs, ts in zip(f, th)) / sum(f)
    bis, v2 = tq.reduce(gam)
    A = lambda D, dth: airy_amplitude(np.pi * D * np.linalg.norm(dth) / LAM)
    tot = 0.0
    for s_ in range(2):
        for t in range(2):
            for u in range(2):
                ph = np.exp(-2j * np.pi * (B[0] @ th[s_] + B[1] @ th[t] + B[2] @ th[u]) / LAM)
                tot += (f[s_] * f[t] * f[u] * ph * A(d[0], th[u] - th[s_])
                        * A(d[1], th[s_] - th[t]) * A(d[2], th[t] - th[u]))
    tot /= sum(f) ** 3
    assert abs(bis - tot) < 1e-4


def test_pair_quadrature_constant_and_point():
    q = pupil_pair_quadrature(10.0, 8.2)
    assert q.reduce(np.full(q.n_points, 0.37)) == pytest.approx(0.37)
    p = point_quadrature()
    assert p.n_points == 1 and p.reduce(np.array([0.5])) == 0.5
    assert resolve_pupils(None, (1, 1)) is None
    assert resolve_pupils(True, (1.0, 2.0)).diameters_m == (1.0, 2.0)
    assert resolve_pupils(q, None) is q


def test_disk_quadrature_moments():
    pts, w = disk_quadrature(2.0)      # unit radius
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(w @ pts, 0.0, atol=1e-12)
    # <r^2> over a uniform disk of radius R is R^2/2
    assert w @ (pts**2).sum(axis=1) == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("sep_mas", [2.0, 4.0, 6.0])
def test_triple_quadrature_vs_closed_form(sep_mas, maunakea_tri):
    """Two point sources: the three-pupil average has a closed form,
    each pupil contributing one Airy amplitude of the source
    separation it 'sees'."""
    d = (8.2, 10.0, 10.0)
    tq = TripleQuadrature.from_diameters(*d)
    B = maunakea_tri.baseline_vectors()
    rho = sep_mas * MAS
    th = [np.array([0.3, 0.1]) * rho, np.array([-0.6, -0.2]) * rho]
    f = [1.0, 0.4]
    pts = tq.flat_points(B)
    gam = sum(fs * np.exp(-2j * np.pi * (pts @ ts) / LAM)
              for fs, ts in zip(f, th)) / sum(f)
    bis, v2 = tq.reduce(gam)

    A = lambda D, dth: airy_amplitude(np.pi * D * np.linalg.norm(dth) / LAM)
    tot = 0.0
    for s in range(2):
        for t in range(2):
            for u in range(2):
                ph = np.exp(-2j * np.pi * (B[0] @ th[s] + B[1] @ th[t] + B[2] @ th[u]) / LAM)
                tot += (f[s] * f[t] * f[u] * ph * A(d[0], th[u] - th[s])
                        * A(d[1], th[s] - th[t]) * A(d[2], th[t] - th[u]))
    tot /= sum(f) ** 3
    p12 = sum(f[s] * f[t] * np.exp(-2j * np.pi * (B[0] @ (th[s] - th[t])) / LAM)
              * A(d[0], th[s] - th[t]) * A(d[1], th[s] - th[t])
              for s in range(2) for t in range(2)) / sum(f) ** 2
    assert abs(bis - tot) < 1e-5
    assert abs(v2[0] - p12.real) < 1e-5
    # batched reduce (extra leading axis) agrees
    bis2, v22 = tq.reduce(np.stack([gam, gam]))
    assert np.allclose(bis2, bis) and np.allclose(v22[1], v2)


def test_smearing_off_reproduces_point_sampling(beta_aur, maunakea_tri):
    pos = _pos(beta_aur, 0.3)
    nm = np.array([450.0, 800.0])
    b = np.array([30.0, 85.0])
    a = np.asarray(spectral_vis2(pos, b, nm, beta_aur, GridConfig()))
    q = point_quadrature()
    c = np.asarray(spectral_vis2(pos, b, nm, beta_aur, GridConfig(), pupils=q))
    assert np.allclose(a, c, atol=1e-7)
    ts0 = spectral_triple(pos, maunakea_tri, nm, beta_aur, pupils=None)
    assert np.allclose(ts0.bispectrum, ts0.gammas.prod(axis=1))
    assert not ts0.smeared


def test_single_small_disk_barely_smeared(beta_aur):
    """For a source whose structure scale (1/theta) is much larger than
    the pupil, aperture averaging changes |V|^2 by < 1e-3."""
    pos = _pos(beta_aur, 0.0)
    nm = np.array([800.0])
    b = np.array([20.0, 50.0])
    a = np.asarray(spectral_vis2(pos, b, nm, beta_aur, GridConfig()))
    c = np.asarray(spectral_vis2(pos, b, nm, beta_aur, GridConfig(), pupils=(1.0, 1.0)))
    assert np.abs(a - c).max() < 1e-3


def test_keck_smears_beta_aur_fringe(beta_aur):
    """10 m pupils on the 85 m Keck baseline at 400 nm: the Beta Aur
    fringe (period ~25 m at quadrature) loses ~1/3 of its contrast
    (Airy^2 at D/P ~ 0.4 -> 0.66)."""
    pos = _pos(beta_aur, 0.0)
    rho = float(pos.rho) * MAS
    factor = fringe_smearing_factor(10.0, 10.0, rho, LAM)
    assert 0.6 < factor < 0.72
    b = np.linspace(70.0, 100.0, 61)
    nm = np.array([400.0])
    point = np.asarray(spectral_vis2(pos, b, nm, beta_aur, GridConfig()))[0]
    smear = np.asarray(spectral_vis2(pos, b, nm, beta_aur, GridConfig(),
                                     pupils=(10.0, 10.0)))[0]
    # peak-to-trough over the window: the fringe term is attenuated by
    # `factor`, but the slowly varying disk envelopes (which the pupils
    # hardly touch) also contribute, so the measured ratio sits between
    # the Airy prediction and one
    contrast = lambda v: v.max() - v.min()
    ratio = contrast(smear) / contrast(point)
    assert factor < ratio < 0.85


def test_analytic_and_render_agree_with_smearing(beta_aur, keck, spad_lambda):
    spec = Spectrograph(lambda_min_nm=410.0, lambda_max_nm=940.0, n_channels=8)
    a = spectral_g2_snr(beta_aur, 85.0, spectrograph=spec, telescope1=keck,
                        detector1=spad_lambda, vis2_method="analytic")
    r = spectral_g2_snr(beta_aur, 85.0, spectrograph=spec, telescope1=keck,
                        detector1=spad_lambda, vis2_method="render")
    assert np.allclose(a.vis2, r.vis2, atol=1e-3)
    p = spectral_g2_snr(beta_aur, 85.0, spectrograph=spec, telescope1=keck,
                        detector1=spad_lambda, vis2_method="analytic", pupils=None)
    assert not np.allclose(a.vis2, p.vis2, atol=1e-3)


def test_closure_phase_smeared_fields(beta_aur, maunakea_tri):
    # 800 nm: the Maunakea gammas of Beta Aur are not near a null there
    # (at 500 nm |gamma| ~ 0.02 on two arms and the phase is ill-conditioned)
    a = closure_phase(beta_aur, maunakea_tri, 800.0, 0.1)
    s = closure_phase(beta_aur, maunakea_tri, 800.0, 0.1, pupils=True)
    r = closure_phase(beta_aur, maunakea_tri, 800.0, 0.1, pupils=True,
                      method="render")
    assert not a.smeared and s.smeared
    assert np.allclose(a.gammas, s.gammas)          # point gammas kept
    assert np.allclose(a.vis2_pairs, np.abs(a.gammas) ** 2)
    assert s.vis2_pairs.shape == (3,)
    assert abs(s.bispectrum - r.bispectrum) < 1e-3
    assert abs(np.angle(np.exp(1j * (s.phi_c - r.phi_c)))) < np.radians(0.15)


def test_vlt_delta_vel_smearing_is_large(delta_vel, vlt_ut):
    """delta Vel near maximum separation on the VLT at its correct
    distance (25.1 pc): 8.2 m pupils on a fringe period of ~4.7 m at
    400 nm (D/P ~ 1.7) wipe the fringe out (contrast < 5 %), and the
    three-pupil bispectrum collapses -- the largest single correction
    found in the review.  The fringe-aware quadrature warns."""
    psi = np.linspace(0, 2 * np.pi, 721)
    rho = sky_positions(psi, delta_vel).rho
    phase = psi[np.argmax(rho)] / (2 * np.pi)
    factor = fringe_smearing_factor(8.2, 8.2, rho.max() * MAS, LAM)
    assert factor < 0.05
    # pair fringe contrast over 40-60 m (ten periods) through the pupils
    from hbtsim.bispectrum import binary_vis_complex_analytic
    from hbtsim.hbt import baseline_vectors_along_pa, binary_vis2_analytic
    from hbtsim.orbit import positions_at
    pos = positions_at(delta_vel, phase)
    b = np.linspace(40.0, 60.0, 201)
    point = np.squeeze(binary_vis2_analytic(b, 400.0, delta_vel, float(pos.rho)))
    with pytest.warns(UserWarning, match="D/P"):
        q = pupil_pair_quadrature_for(8.2, 8.2, LAM / (rho.max() * MAS))
    pts = q.points(baseline_vectors_along_pa(b, float(pos.pa)))
    v = binary_vis_complex_analytic(pts.reshape(-1, 2), 400.0, delta_vel, pos)
    smear = q.reduce(np.abs(v).reshape(1, b.size, -1) ** 2)[0]
    # fringe amplitude = rms residual about the smooth disk envelope
    fringe = lambda x: np.std(x - np.polyval(np.polyfit(b, x, 3), b))
    assert fringe(smear) < 0.1 * fringe(point)
    # the three-pupil path warns and is finite
    tri = vlt_ut.triangles()[0]
    with pytest.warns(UserWarning, match="D/P"):
        s = closure_phase(delta_vel, tri, 400.0, phase, pupils=True)
    assert np.isfinite(s.triple_amp)
