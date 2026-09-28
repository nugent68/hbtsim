"""Input guards: eclipse test shared by the analytic paths, grid-extent
check in the renderer, Kepler-solver convergence."""

import numpy as np
import pytest

from hbtsim import hbt
from hbtsim.bispectrum import (MAUNAKEA_SUBARU_KECK, binary_vis_complex_analytic,
                               closure_phase)
from hbtsim.orbit import SkyPositions, sky_positions, solve_kepler
from hbtsim.params import (ALGOL, BETA_AUR, DELTA_VEL, ECLIPSE_MARGIN,
                           GridConfig, in_eclipse, in_eclipse_rho,
                           require_out_of_eclipse)
from hbtsim.render import check_extent, render_image
from hbtsim.spectral import spectral_vis2


def _pos(system, psi):
    return SkyPositions(*(np.asarray(v) for v in sky_positions(psi, system)))


def test_in_eclipse_helpers():
    psi = np.linspace(0, 2 * np.pi, 721)
    pos = sky_positions(psi, ALGOL)
    ecl = in_eclipse(ALGOL, pos)
    assert ecl.shape == psi.shape
    assert ecl[np.argmin(np.abs(psi - np.pi / 2))]
    assert not ecl[0]
    # margin: exactly at the sum of the radii is "near eclipse"
    s = ALGOL.sum_of_radii_mas
    assert in_eclipse_rho(ALGOL, s * 1.0)
    assert not in_eclipse_rho(ALGOL, s * ECLIPSE_MARGIN * 1.001)
    with pytest.raises(ValueError, match="eclipse"):
        require_out_of_eclipse(ALGOL, float(pos.rho.min()))
    require_out_of_eclipse(ALGOL, float(pos.rho.max()))


def test_every_analytic_path_refuses_eclipse():
    ecl = _pos(ALGOL, np.pi / 2)
    with pytest.raises(ValueError, match="eclipse"):
        hbt.binary_vis2_analytic(np.array([50.0]), 500.0, ALGOL, float(ecl.rho))
    with pytest.raises(ValueError, match="eclipse"):
        binary_vis_complex_analytic(MAUNAKEA_SUBARU_KECK.baseline_vectors(),
                                    500.0, ALGOL, ecl)
    with pytest.raises(ValueError, match="eclipse"):
        closure_phase(ALGOL, MAUNAKEA_SUBARU_KECK, 500.0, 0.25, method="analytic")
    # the rendered path works there
    r = closure_phase(ALGOL, MAUNAKEA_SUBARU_KECK, 500.0, 0.25, method="render")
    assert np.all(np.isfinite(r.gammas))


def test_deprecated_fft_method_alias():
    with pytest.warns(DeprecationWarning):
        r = closure_phase(BETA_AUR, MAUNAKEA_SUBARU_KECK, 500.0, 0.1, method="fft")
    assert np.all(np.isfinite(r.gammas))
    with pytest.raises(ValueError, match="unknown method"):
        closure_phase(BETA_AUR, MAUNAKEA_SUBARU_KECK, 500.0, 0.1, method="nope")


def test_extent_guard():
    """delta Vel at apoapsis (16.6 mas orbit at 25.1 pc) fits the grid
    for_system builds (2048 px) but not the default 1024-pixel grid
    (5.1 mas half-width) nor a 256-pixel one, and every entry point
    must refuse rather than clip the disk."""
    psi = np.linspace(0, 2 * np.pi, 2001)
    pos_all = sky_positions(psi, DELTA_VEL)
    fit = GridConfig().for_system(DELTA_VEL)
    assert fit.n >= 2048
    check_extent(pos_all, DELTA_VEL, fit)
    with pytest.raises(ValueError, match="grid"):
        check_extent(pos_all, DELTA_VEL, GridConfig())
    k = int(np.argmax(np.abs(pos_all.x2)))
    pos = SkyPositions(*(np.asarray(v)[k] for v in pos_all))
    small = GridConfig(n=256)
    with pytest.raises(ValueError, match="grid"):
        check_extent(pos, DELTA_VEL, small)
    with pytest.raises(ValueError, match="grid"):
        render_image(pos, DELTA_VEL, 500.0, small)
    with pytest.raises(ValueError, match="grid"):
        spectral_vis2(pos, [50.0], [500.0], DELTA_VEL, small)
    # and renders fine on the fitted grid
    img = render_image(pos, DELTA_VEL, 500.0, fit)
    assert float(img.sum()) > 0.0


def test_kepler_solver_guards():
    with pytest.raises(ValueError):
        solve_kepler(np.array([0.3]), 1.2)
    with pytest.raises(RuntimeError, match="converge"):
        solve_kepler(np.linspace(0, 2 * np.pi, 50), 0.9, n_iter=1)
    E = solve_kepler(np.linspace(0, 2 * np.pi, 50), 0.9)
    assert np.allclose(E - 0.9 * np.sin(E), np.linspace(0, 2 * np.pi, 50), atol=1e-12)
