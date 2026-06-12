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
    assert pos.rho.min() < 0.1             # near-edge-on conjunctions
    # periastron at psi = 0: the projected separation cannot exceed the
    # 3D periastron distance a(1-e)
    p0 = sky_positions(0.0, DELTA_VEL)
    assert float(p0.rho) <= a * (1 - e) + 1e-9


def test_deltavel_eclipses_and_scale():
    assert DELTA_VEL.angular_semimajor_mas == pytest.approx(5.16, abs=0.05)
    th = [2 * DELTA_VEL.angular_radius_mas(s)
          for s in (DELTA_VEL.primary, DELTA_VEL.secondary)]
    assert th[0] == pytest.approx(0.343, abs=0.01)
    assert th[1] == pytest.approx(0.291, abs=0.01)
    # grazing but real eclipses: minimum projected separation below the
    # sum of the radii
    psi = np.linspace(0, 2 * np.pi, 100001)
    rho_min = sky_positions(psi, DELTA_VEL).rho.min()
    assert rho_min < (th[0] + th[1]) / 2.0


def test_deltavel_is_a_strong_vlt_target():
    """Unresolved disks keep the triple amplitude high: delta Vel's
    combined VLT sensitivity lands within a factor ~2 of Spica despite
    being 1.2 mag fainter."""
    spec = Spectrograph(n_channels=300)
    d = array_g3_snr(DELTA_VEL, VLT_UT, spectrograph=spec, orbital_phase=0.5)
    s = array_g3_snr(SPICA, VLT_UT, spectrograph=spec)
    assert d.snr_total > 0.3 * s.snr_total
    assert max(r.triple_amp.max() for r in d.per_triangle) > 0.5
