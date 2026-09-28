"""Eccentric-orbit (Kepler) machinery and the delta Velorum system."""

import numpy as np
import pytest

from hbtsim.bispectrum import VLT_UT
from hbtsim.orbit import SkyPositions, sky_positions, solve_kepler
from hbtsim.params import BETA_AUR, DELTA_VEL, SPICA
from hbtsim.snr import Spectrograph
from hbtsim.snr3 import array_g3_snr


def test_kepler_solver():
    rng_M = np.linspace(0.0, 2 * np.pi, 97)
    for e in (0.1, 0.29, 0.7):
        E = solve_kepler(rng_M, e)
        assert np.allclose(E - e * np.sin(E), rng_M, atol=1e-12)


def test_circular_path_unchanged():
    """e = 0 systems must follow the exact pre-Kepler code path: psi = 0 at
    quadrature, eclipse phasing preserved."""
    psi = np.linspace(0, 2 * np.pi, 73)
    pos = sky_positions(psi, BETA_AUR)
    a = BETA_AUR.angular_semimajor_mas
    assert pos.rho[0] == pytest.approx(a, rel=1e-12)
    assert np.allclose(np.hypot(pos.x2 - pos.x1, pos.y2 - pos.y1), pos.rho)
    assert bool(sky_positions(np.pi / 2, BETA_AUR).front2)


def test_eccentric_geometry():
    """The projected separation is bounded by the instantaneous 3D
    separation a(1-e)..a(1+e), exceeds the circular radius a somewhere
    (eccentricity visible on the sky), and nearly vanishes at the
    grazing conjunctions.  Note rho_max < a(1+e) here because at
    i = 89 deg the apoapsis direction (u = omega + 180 deg) points
    close to the line of sight and is foreshortened."""
    psi = np.linspace(0, 2 * np.pi, 20001)
    pos = sky_positions(psi, DELTA_VEL)
    a = DELTA_VEL.angular_semimajor_mas
    e = DELTA_VEL.eccentricity
    assert pos.rho.max() <= a * (1 + e) + 1e-9
    assert pos.rho.max() > a               # eccentricity visible
    assert pos.rho.min() < 0.02 * a        # near-edge-on conjunctions
    # periastron at psi = 0: the projected separation cannot exceed the
    # 3D periastron distance a(1-e)
    p0 = sky_positions(0.0, DELTA_VEL)
    assert float(p0.rho) <= a * (1 - e) + 1e-9


def test_deltavel_eclipses_and_scale():
    """At the Merand et al. 2011 orbital parallax (39.8 mas, 25.1 pc):
    a = 16.6 mas, disks 1.10 / 0.93 mas."""
    assert DELTA_VEL.distance_pc == pytest.approx(1000.0 / 39.8, rel=2e-3)
    assert DELTA_VEL.angular_semimajor_mas == pytest.approx(16.56, abs=0.1)
    th = [2 * DELTA_VEL.angular_radius_mas(s)
          for s in (DELTA_VEL.primary, DELTA_VEL.secondary)]
    assert th[0] == pytest.approx(1.100, abs=0.01)
    assert th[1] == pytest.approx(0.934, abs=0.01)
    # grazing but real eclipses: minimum projected separation below the
    # sum of the radii
    psi = np.linspace(0, 2 * np.pi, 100001)
    rho_min = sky_positions(psi, DELTA_VEL).rho.min()
    assert rho_min < (th[0] + th[1]) / 2.0


def test_deltavel_distance_matches_photometry():
    """The check that would have caught the 80.6 pc error: the blackbody
    model at the catalogue distance must reproduce the g-band anchor
    without an anchor offset of more than 0.1 mag."""
    from hbtsim.snr import model_ab_mag
    g_model = float(model_ab_mag(DELTA_VEL, 477.0))
    g_anchor = dict(DELTA_VEL.mag_anchors)["g"]
    assert abs(g_model - g_anchor) < 0.1


def test_deltavel_vlt_fringes_are_smeared_out():
    """At 25.1 pc the fringe period at 400 nm (4.7 m) is well below the
    8.2 m UT pupils: D/P ~ 1.7, contrast retained < 5 %, the three-pupil
    bispectrum collapses.  delta Vel is a target for 1-4 m telescopes,
    not for the VLT (see docs)."""
    from hbtsim.aperture import fringe_smearing_factor
    from hbtsim.bispectrum import closure_phase
    from hbtsim.params import MAS
    psi = np.linspace(0, 2 * np.pi, 721)
    rho = sky_positions(psi, DELTA_VEL).rho
    phase = psi[np.argmax(rho)] / (2 * np.pi)
    factor = fringe_smearing_factor(8.2, 8.2, rho.max() * MAS, 400e-9)
    assert factor < 0.05
    # the pair fringe along the separation axis over ten fringe periods:
    # its peak-to-trough contrast through 8.2 m pupils is < 10 % of the
    # point-sampled one
    from hbtsim.aperture import pupil_pair_quadrature_for
    from hbtsim.bispectrum import binary_vis_complex_analytic
    from hbtsim.hbt import baseline_vectors_along_pa, binary_vis2_analytic
    from hbtsim.orbit import positions_at
    pos = positions_at(DELTA_VEL, phase)
    b = np.linspace(40.0, 60.0, 201)
    point = np.squeeze(binary_vis2_analytic(b, 400.0, DELTA_VEL, float(pos.rho)))
    with pytest.warns(UserWarning, match="D/P"):
        q = pupil_pair_quadrature_for(8.2, 8.2, 400e-9 / (rho.max() * MAS))
    pts = q.points(baseline_vectors_along_pa(b, float(pos.pa)))
    v = binary_vis_complex_analytic(pts.reshape(-1, 2), 400.0, DELTA_VEL, pos)
    smear = q.reduce(np.abs(v).reshape(1, b.size, -1) ** 2)[0]
    # fringe amplitude = rms residual about the smooth disk envelope
    fringe = lambda x: np.std(x - np.polyval(np.polyfit(b, x, 3), b))
    assert fringe(smear) < 0.1 * fringe(point)
    # the three-pupil bispectrum path warns too
    tri = VLT_UT.triangles()[0]
    with pytest.warns(UserWarning, match="D/P"):
        closure_phase(DELTA_VEL, tri, 400.0, phase, pupils=True)
    # and the eclipses are still real (grazing) at the corrected scale
    th = [2 * DELTA_VEL.angular_radius_mas(s) for s in (DELTA_VEL.primary, DELTA_VEL.secondary)]
    assert rho.min() < (th[0] + th[1]) / 2.0
