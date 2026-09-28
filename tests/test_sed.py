"""Model-atmosphere hooks (hbtsim.sed, Star.flux_table / ld_profile) and
the supersampled tabulated-limb-darkening renderer."""

import os
import warnings
from dataclasses import replace

import numpy as np
import pytest

from hbtsim import hbt
from hbtsim.bispectrum import MAUNAKEA_SUBARU_KECK, closure_phase
from hbtsim.limbdark import visibility_ld_disk, visibility_profile
from hbtsim.orbit import SkyPositions, sky_positions
from hbtsim.params import (ALGOL, BETA_AUR, DELTA_VEL, LD_ALGOL_B, MAS, FluxTable,
                           GridConfig, LDProfile, Star, linear_ld_rows)
from hbtsim.render import linear_rows_jnp, render_image, render_kernel, spectral_weights
from hbtsim.sed import (bin_to_step, linear_ld_profile, load_star_tables,
                        planck_flux_table, save_star_tables, with_tables)
from hbtsim.snr import model_ab_mag, system_ab_mag

LAM = np.linspace(380.0, 1000.0, 311)


def _pos(system, psi):
    return SkyPositions(*(np.asarray(v) for v in sky_positions(psi, system)))


def _tabled(system, n_mu=64):
    """The same system with Planck flux tables and the linear law
    tabulated: every number must reproduce the default path."""
    def conv(star):
        return with_tables(star, planck_flux_table(star.teff, LAM),
                           linear_ld_profile(star.ld_table_nm, LAM, n_mu))
    return replace(system, primary=conv(system.primary),
                   secondary=conv(system.secondary))


def test_star_hooks_reproduce_defaults():
    s = ALGOL.secondary
    t = with_tables(s, planck_flux_table(s.teff, LAM),
                    linear_ld_profile(s.ld_table_nm, LAM, 257))
    lam = np.array([420.0, 556.0, 800.0])   # on the table grid
    assert np.allclose(t.surface_flux(lam), s.surface_flux(lam), rtol=1e-6)
    assert np.allclose(t.disk_flux_factor(lam), s.disk_flux_factor(lam), rtol=1e-12)
    assert np.allclose(t.central_intensity(lam), s.central_intensity(lam), rtol=1e-6)
    mu = np.linspace(0, 1, 33)
    assert np.allclose(t.ld_rows(lam, mu), s.ld_rows(lam, mu), atol=1e-12)
    assert t.ld_mode == "table" and s.ld_mode == "linear"


def test_planck_tables_reproduce_render_weights_and_magnitudes():
    sysm = _tabled(BETA_AUR)
    cw0 = spectral_weights(LAM[::10], BETA_AUR)
    cw1 = spectral_weights(LAM[::10], sysm)
    assert np.allclose(cw0.w1, cw1.w1, rtol=1e-5)
    assert np.allclose(cw0.i1, cw1.i1, atol=1e-6)
    assert model_ab_mag(sysm, 500.0) == pytest.approx(model_ab_mag(BETA_AUR, 500.0), abs=1e-6)
    # with tables the anchors are only a check: no offset is applied, and
    # the blackbody's ~0.4 mag miss is reported
    with pytest.warns(UserWarning, match="anchor"):
        m = system_ab_mag(sysm, 477.0)
    assert m == pytest.approx(model_ab_mag(BETA_AUR, 477.0), abs=1e-5)
    assert abs(m - dict(BETA_AUR.mag_anchors)["g"]) > 0.2


def test_linear_table_reproduces_linear_kernel():
    pos = _pos(ALGOL, 0.3)
    grid = GridConfig()
    a = np.asarray(render_image(pos, ALGOL, 450.0, grid))
    b = np.asarray(render_image(pos, _tabled(ALGOL, 257), 450.0, grid))
    assert np.allclose(a, b, atol=2e-6)
    # a coarse profile table on the kernel's finer mu grid is still exact
    # for a linear law
    c = np.asarray(render_image(pos, _tabled(ALGOL, 9), 450.0, grid))
    assert np.allclose(a, c, atol=2e-6)


def test_tabulated_ld_disk_vs_numeric_hankel():
    """A single disk with a NON-linear profile (quadratic law with a
    steep limb) rendered with supersampling matches the numeric Hankel
    transform of the same profile to 1e-4 in |V|^2."""
    grid = GridConfig(supersample=4)
    mu = np.linspace(0.0, 1.0, 129)
    prof = 1.0 - 0.6 * (1.0 - mu) - 0.3 * (1.0 - mu) ** 2
    prof_j = np.asarray(np.interp(grid.mu_grid, mu, prof), dtype=np.float32)
    R = 50.0
    img = render_kernel(0.0, 0.0, 300.0, 300.0, False, R, 1.0, 1.0, 0.0,
                        prof_j, linear_rows_jnp(0.0, grid.n_mu), grid.n, 4)
    theta_d = 2 * R * grid.pixel_scale_rad
    lam = 400e-9
    B = np.linspace(5.0, 0.95 * 1.22 * lam / theta_d, 60)
    v2 = np.asarray(hbt.vis2_along_pa(img, B, lam, 0.0, grid))
    x = np.pi * theta_d * B / lam
    ref = visibility_profile(x, mu, prof) ** 2
    assert np.abs(v2 - ref).max() < 1e-4
    # and the numeric transform itself reproduces the linear series
    u = 0.5
    assert np.allclose(visibility_profile(x, mu, 1 - u * (1 - mu)),
                       visibility_ld_disk(x, u), atol=2e-5)


def test_supersampled_renderer_accuracy():
    """R = 52 px, u = 0.5, supersample 4: flux to 3e-5 and |V|^2 to 3e-5
    (pixel-window corrected), an order of magnitude better than the
    plain kernel."""
    for s, tol_f, tol_v in ((1, 1e-3, 3e-4), (4, 3e-5, 3e-5)):
        grid = GridConfig(supersample=s)
        R, u = 52.0, 0.5
        img = render_kernel(0.0, 0.0, 300.0, 300.0, False, R, 1.0, 1.0, 0.0,
                            linear_rows_jnp(u, grid.n_mu),
                            linear_rows_jnp(0.0, grid.n_mu), grid.n, s)
        flux_err = float(img.sum()) / (np.pi * R**2 * (1 - u / 3)) - 1.0
        assert abs(flux_err) < tol_f
        theta_d = 2 * R * grid.pixel_scale_rad
        lam = 400e-9
        B = np.linspace(5.0, 0.95 * 1.22 * lam / theta_d, 60)
        v2 = np.asarray(hbt.vis2_along_pa(img, B, lam, 0.0, grid))
        x = np.pi * theta_d * B / lam
        assert np.abs(v2 - visibility_ld_disk(x, u) ** 2).max() < tol_v


def test_pixel_window_correction_in_reference():
    grid = GridConfig(supersample=2)
    img = render_kernel(0.0, 0.0, 300.0, 300.0, False, 30.0, 1.0, 1.0, 0.0,
                        linear_rows_jnp(0.3, grid.n_mu),
                        linear_rows_jnp(0.0, grid.n_mu), grid.n, 2)
    u = np.array([200.0, -150.0]) / 400e-9
    v = np.array([40.0, 90.0]) / 400e-9
    a = np.asarray(hbt.vis_points(img, u, v, grid))
    b = hbt.vis_points_np(img, u, v, grid)
    assert np.abs(a - b).max() < 2e-6
    # the correction is the sinc of the pixel window, > 1e-3 here
    plain = hbt.vis_points_np(img, u, v, replace(grid, supersample=1))
    assert np.abs(np.abs(a) / np.abs(plain) - 1).min() > 1e-4


def test_algol_closure_phase_on_accurate_grid():
    """Rendered vs analytic closure phase to 0.02 deg and |gamma| to 3e-5
    on the system-adapted supersampled grid (0.15 deg / 1e-3 on the
    plain default grid)."""
    grid = GridConfig(supersample=4).for_system(ALGOL)
    assert grid.n == 1024 and grid.pixel_scale_mas < 0.01
    for phase in (0.05, 0.1):
        for lam in (450.0, 800.0):
            r = closure_phase(ALGOL, MAUNAKEA_SUBARU_KECK, lam, phase,
                              method="render", grid=grid)
            a = closure_phase(ALGOL, MAUNAKEA_SUBARU_KECK, lam, phase)
            assert np.abs(np.abs(r.gammas) - np.abs(a.gammas)).max() < 3e-5
            dphic = abs(np.angle(np.exp(1j * (r.phi_c - a.phi_c))))
            assert np.degrees(dphic) < 0.02


def test_for_system_and_delta_vel_apoapsis():
    grid = GridConfig().for_system(DELTA_VEL)
    r_min = min(DELTA_VEL.angular_radius_mas(DELTA_VEL.primary),
                DELTA_VEL.angular_radius_mas(DELTA_VEL.secondary))
    assert r_min / grid.pixel_scale_mas >= 50.0
    assert grid.n in (2048, 4096)
    psi = np.linspace(0, 2 * np.pi, 2001)
    pos_all = sky_positions(psi, DELTA_VEL)
    k = int(np.argmax(pos_all.rho))
    img = render_image(pos_all.take(k), DELTA_VEL, 500.0, grid)
    assert float(img.sum()) > 0
    with pytest.raises(ValueError, match="n_max"):
        GridConfig().for_system(DELTA_VEL, min_radius_px=200.0)
    # a system that already fits keeps the default scale and size
    g2 = GridConfig().for_system(ALGOL, min_radius_px=10.0)
    assert g2 == GridConfig()


def test_bin_and_roundtrip(tmp_path):
    lam = np.linspace(400.0, 401.0, 1001)
    vals = np.stack([lam, 2 * lam], axis=1)
    c, b = bin_to_step(lam, vals, 400.0, 401.0, 0.1)
    assert c.size == 10 and np.allclose(b[:, 1], 2 * b[:, 0])
    assert np.allclose(b[:, 0], c, atol=0.06)
    tab = dict(wavelength_nm=LAM, flux=np.pi * np.ones_like(LAM),
               mu=np.linspace(0, 1, 5), intensity=np.ones((LAM.size, 5)),
               path="synthetic")
    f = tmp_path / "t.npz"
    save_star_tables(str(f), tab)
    ft, ld = load_star_tables(str(f))
    assert isinstance(ft, FluxTable) and isinstance(ld, LDProfile)
    assert ft(500.0) == pytest.approx(np.pi)
    assert ld.rows(500.0).shape == (1, 5)
    assert ld.source == "synthetic"


def test_analytic_paths_with_tables():
    """The analytic binary visibility with tabulated (linear-equivalent)
    profiles and Planck tables equals the linear/blackbody path."""
    sysm = _tabled(ALGOL, 129)
    pos = _pos(ALGOL, 0.0)
    B = np.arange(10.0, 150.0, 10.0)
    for lam in (450.0, 800.0):
        a = hbt.binary_vis2_analytic(B, lam, ALGOL, float(pos.rho))
        b = hbt.binary_vis2_analytic(B, lam, sysm, float(pos.rho))
        assert np.allclose(a, b, atol=3e-5)
    ca = closure_phase(ALGOL, MAUNAKEA_SUBARU_KECK, 600.0, 0.1)
    cb = closure_phase(sysm, MAUNAKEA_SUBARU_KECK, 600.0, 0.1)
    assert np.allclose(ca.gammas, cb.gammas, atol=3e-5)


@pytest.mark.skipif(not os.path.exists("data/newera_example.npz"),
                    reason="prepared NewEra table not present")
def test_newera_table_sanity():
    ft, ld = load_star_tables("data/newera_example.npz")
    assert ld.mu[0] >= 0.0 and ld.mu[-1] == pytest.approx(1.0)
    assert np.allclose(ld.intensity[:, -1], 1.0)
    assert np.all(ld.intensity >= -1e-6)
    # a 6000 K model: the flux is within a factor 2 of pi B_lambda
    bb = planck_flux_table(6000.0, ft.wavelength_nm)
    ratio = ft.flux / bb.flux
    assert 0.3 < np.median(ratio) < 2.0


def test_anchoring_disabled_with_sed_tables():
    """With flux tables on both stars the synthetic lightcurve keeps its
    own zero point (as snr.system_ab_mag does); the anchor only warns
    when missed by more than ANCHOR_CHECK_MAG."""
    import warnings
    from hbtsim.photometry import anchored_mags
    sys_t = _tabled(ALGOL)
    assert sys_t.has_sed_tables and not ALGOL.has_sed_tables
    m = np.array([2.10, 2.30, 2.05])
    g = dict(ALGOL.mag_anchors)["g"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = anchored_mags(m, "g", sys_t, reference_mag=g + 0.05)
    assert np.array_equal(out, m)
    with pytest.warns(UserWarning, match="not anchoring"):
        out = anchored_mags(m, "g", sys_t, reference_mag=g + 0.5)
    assert np.array_equal(out, m)
    # the blackbody system is still anchored
    anc = anchored_mags(m, "g", ALGOL, reference_mag=2.05)
    assert anc[2] == pytest.approx(g)
