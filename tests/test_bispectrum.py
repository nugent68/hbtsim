"""Tests of complex visibilities, the triple product, and closure phase."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hbtsim import hbt
from hbtsim.bispectrum import (MAUNAKEA_SUBARU_KECK, BispectrumResult,
                               Station, Triangle,
                               binary_vis_complex_analytic, closure_phase,
                               equilateral_triangle, spectral_bispectrum)
from hbtsim.orbit import SkyPositions, sky_positions
from hbtsim.params import ALGOL, BETA_AUR, GridConfig
from hbtsim.render import linear_rows_jnp, render_image, render_kernel
from hbtsim.snr import KECK

GRID = GridConfig()
TRI = MAUNAKEA_SUBARU_KECK


def _pos(system, psi):
    return SkyPositions(*(np.asarray(v) for v in sky_positions(psi, system)))


def test_triangle_geometry():
    """Site coordinates reproduce the nominal 152/85/226 m distances and
    the baseline vectors close exactly."""
    lengths = TRI.baseline_lengths()
    assert lengths == pytest.approx([152.1, 84.9, 225.9], abs=0.5)
    assert np.allclose(TRI.baseline_vectors().sum(axis=0), 0.0, atol=1e-12)
    eq = equilateral_triangle(85.0)
    assert eq.baseline_lengths() == pytest.approx([85.0] * 3, rel=1e-12)


def test_rendered_complex_vis_matches_analytic():
    """Modulus to 1e-3 and phases to 0.15 deg out of eclipse, both
    systems.  Renderer-limited (soft-rim midpoint sampling): the exact
    DFT itself is good to 1e-6 / 1e-5 rad (tests/test_dft_core.py); the
    Phase 4 renderer tightens this to 0.01 deg."""
    for system in (BETA_AUR, ALGOL):
        for phase in (0.0, 0.1):
            for lam in (450.0, 800.0):
                rnd = closure_phase(system, TRI, lam, phase, method="render")
                ana = closure_phase(system, TRI, lam, phase, method="analytic")
                assert np.allclose(np.abs(rnd.gammas), np.abs(ana.gammas),
                                   atol=1e-3)
                dphase = np.abs(np.angle(rnd.gammas * np.conj(ana.gammas)))
                assert np.degrees(dphase).max() < 0.15
                dphic = abs(np.angle(np.exp(1j * (rnd.phi_c - ana.phi_c))))
                assert np.degrees(dphic) < 0.15


def test_analytic_vectorized_over_wavelength():
    pos = _pos(ALGOL, 0.05)
    bv = TRI.baseline_vectors()
    nm = np.array([420.0, 610.0, 880.0])
    batch = binary_vis_complex_analytic(bv, nm, ALGOL, pos)
    assert batch.shape == (3, 3)
    for k, lam in enumerate(nm):
        assert np.allclose(batch[k], binary_vis_complex_analytic(bv, float(lam),
                                                                 ALGOL, pos))


def test_closure_phase_translation_invariance():
    """Shifting the whole image moves every gamma phase but leaves the
    closure phase unchanged (vector baselines close)."""
    pos = _pos(ALGOL, 0.0)
    shift = 0.4  # mas
    pos_shifted = SkyPositions(x1=pos.x1 + shift, y1=pos.y1 - shift,
                               x2=pos.x2 + shift, y2=pos.y2 - shift,
                               front2=pos.front2, rho=pos.rho, pa=pos.pa)
    bvecs = TRI.baseline_vectors()
    g0 = binary_vis_complex_analytic(bvecs, 600.0, ALGOL, pos)
    g1 = binary_vis_complex_analytic(bvecs, 600.0, ALGOL, pos_shifted)
    # individual phases move...
    assert np.degrees(np.abs(np.angle(g1 * np.conj(g0)))).max() > 5.0
    # ...the closure phase does not
    phi0 = np.angle(g0.prod())
    phi1 = np.angle(g1.prod())
    assert abs(np.angle(np.exp(1j * (phi1 - phi0)))) < 1e-10


def test_station_relabeling_conjugates_only():
    """Reversing the station order conjugates the bispectrum (cos phi_c
    invariant)."""
    rev = Triangle(tuple(reversed(TRI.stations)))
    a = closure_phase(ALGOL, TRI, 700.0, 0.0)
    b = closure_phase(ALGOL, rev, 700.0, 0.0)
    assert b.phi_c == pytest.approx(-a.phi_c, abs=1e-9)
    assert b.cos_phi_c == pytest.approx(a.cos_phi_c, abs=1e-12)


def test_point_source_and_single_disk():
    """A tiny centered single 'star' has gamma ~ 1 and phi_c ~ 0; a single
    centered LD disk has a REAL bispectrum (phases 0 or pi)."""
    bv = TRI.baseline_vectors()
    # tiny disk = effectively unresolved point source
    rows = (linear_rows_jnp(0.3, GRID.n_mu), linear_rows_jnp(0.0, GRID.n_mu))
    img = render_kernel(0.0, 0.0, 300.0, 300.0, False,
                        3.0, 1.0, 1.0, 0.0, *rows, GRID.n)
    g = np.asarray(hbt.vis_of_baselines(img, bv, 500e-9, GRID))
    # a 3 px (0.03 mas radius) disk on the 226 m arm already has
    # |V| = 1 - x^2/8 ~ 0.979 -- "unresolved" is approximate
    assert np.abs(g).min() > 0.97
    assert abs(np.angle(g.prod())) < 1e-4

    # resolved centered disk: bispectrum real (sign from the lobes)
    img = render_kernel(0.0, 0.0, 300.0, 300.0, False,
                        50.0, 1.0, 1.0, 0.0, *rows, GRID.n)
    g = np.asarray(hbt.vis_of_baselines(img, bv, 500e-9, GRID))
    assert abs(np.sin(np.angle(g.prod()))) < 1e-4


def test_spectral_bispectrum_matches_closure_phase_and_chunks():
    pos = _pos(BETA_AUR, 0.0)
    nm = np.array([450.0, 600.0, 800.0])
    gam = np.asarray(spectral_bispectrum(pos, TRI, nm, BETA_AUR, GRID,
                                         chunk_size=2))
    gam1 = np.asarray(spectral_bispectrum(pos, TRI, nm, BETA_AUR, GRID,
                                          chunk_size=1))
    assert np.allclose(gam, gam1, atol=1e-7)
    for k, lam in enumerate(nm):
        ref = closure_phase(BETA_AUR, TRI, float(lam), 0.0, method="render")
        assert np.allclose(gam[k], ref.gammas, atol=1e-6)


def test_spectral_bispectrum_through_eclipse():
    """Mid primary eclipse of Algol: the analytic path refuses, the
    rendered path returns finite, bounded, smooth-in-lambda gammas."""
    with pytest.raises(ValueError, match="eclipse"):
        closure_phase(ALGOL, TRI, 600.0, 0.25, method="analytic")
    pos = _pos(ALGOL, np.pi / 2)
    nm = np.linspace(450.0, 900.0, 10)
    gam = np.asarray(spectral_bispectrum(pos, TRI, nm, ALGOL, GRID,
                                         chunk_size=4))
    assert np.all(np.isfinite(gam))
    assert np.abs(gam).max() <= 1.0 + 1e-5
    bis = gam[:, 0] * gam[:, 1] * gam[:, 2]
    # smooth in lambda: no jumps larger than ~half the dynamic range
    assert np.abs(np.diff(np.abs(bis))).max() < 0.5 * np.ptp(np.abs(bis)) + 1e-9
