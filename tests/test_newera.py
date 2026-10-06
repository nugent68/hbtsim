"""NewEra model-atmosphere hooks: spherical radius convention, native-mu
rendering, channel averaging, the vectorized Hankel transform and the
(T_eff, log g) grid interpolation.  Synthetic tables throughout; the
tests on real tables are gated on data/newera/."""

import glob
import os
import warnings
from dataclasses import replace

import numpy as np
import pytest

from hbtsim.limbdark import (DiskVisibilityCache, hankel_nodes, visibility_ld_disk,
                             visibility_profile, visibility_profile_batch)
from hbtsim.orbit import positions_at
from hbtsim.params import MAS, FluxTable, GridConfig, LDProfile, Star, planck
from hbtsim.render import render_image
from hbtsim.sed import (NewEraGrid, band_average_flux, band_average_profile,
                        limb_edge, planck_flux_table, prepare_system,
                        rebin_to_channels, save_star_tables, spherical_extension,
                        with_newera, with_tables)
from hbtsim.snr import Spectrograph, model_ab_mag, spectral_g2_snr, system_ab_mag

LAM = np.linspace(380.0, 1000.0, 3101)          # 0.2 nm
NEWERA_DIR = "data/newera"


def spherical_profile(lam, mu_edge=0.07, u_cont=0.6, r_outer=None):
    """A NewEra-like profile: an exact linear-law disk of radius R_edge =
    R_outer sqrt(1 - mu_edge^2) (I linear in the INNER mu' = sqrt(1 -
    (r/R_edge)^2)), tabulated on the outer-boundary mu grid, with a sharp
    fall to ~0 over the tangent rays outside it and a slightly less
    darkened core at 486 nm."""
    mu = np.concatenate([np.linspace(0.02, mu_edge, 12), np.linspace(mu_edge + 0.005, 1.0, 40)])
    r_out = 1.0 / np.sqrt(1.0 - mu_edge**2) if r_outer is None else r_outer
    r_over_redge = np.sqrt(np.clip(1.0 - mu**2, 0.0, 1.0)) * r_out     # r / R_edge
    mu_inner = np.sqrt(np.clip(1.0 - r_over_redge**2, 0.0, 1.0))
    rows = np.empty((lam.size, mu.size))
    for i, l in enumerate(lam):
        u = u_cont - 0.35 * np.exp(-0.5 * ((l - 486.0) / 0.4) ** 2)
        inner = 1.0 - u * (1.0 - mu_inner)
        tail = (1.0 - u) * np.clip((mu - 0.02) / (mu_edge - 0.02), 0, 1) ** 4 * 1e-3
        rows[i] = np.where(r_over_redge <= 1.0, inner, tail)
    return LDProfile(mu=mu, wavelength_nm=lam, intensity=rows, source="synthetic",
                     r_outer=r_out, mu_edge=mu_edge)


def line_flux(lam, teff, depth=0.6, centre=486.0, width=0.4):
    f = np.pi * planck(lam * 1e-9, teff)
    return f * (1.0 - depth * np.exp(-0.5 * ((lam - centre) / width) ** 2))


# ---------------------------------------------------------------------------
# radius convention
# ---------------------------------------------------------------------------
def test_limb_edge_and_extension():
    prof = spherical_profile(LAM[::100], mu_edge=0.07)
    assert limb_edge(prof.mu, prof.intensity[0]) == pytest.approx(0.07, abs=0.004)
    ext = spherical_extension(prof, wavelength_nm=500.0)
    assert ext["r_outer_over_edge"] == pytest.approx(1.0 / np.sqrt(1 - 0.07**2), rel=2e-4)
    flat = LDProfile(mu=np.linspace(0, 1, 10), wavelength_nm=LAM[:2],
                     intensity=np.ones((2, 10)))
    assert limb_edge(flat.mu, flat.intensity[0]) == 0.0


def test_drawn_radius_and_first_null_shift(beta_aur):
    """radius_rsun is the tau = 1 radius: the disk is drawn out to
    R_outer = r_outer R, and the first null of |V|^2 moves inward by the
    same factor."""
    lam = np.array([450.0, 500.0])
    prof = spherical_profile(lam, mu_edge=0.10)
    star = with_tables(beta_aur.primary, planck_flux_table(9350.0, lam), prof)
    assert star.radius_scale == pytest.approx(prof.r_outer)
    assert replace(star, radius_ref="outer").radius_scale == 1.0
    sysm = replace(beta_aur, primary=star)
    assert sysm.drawn_radius_mas(star) == pytest.approx(
        sysm.angular_radius_mas(star) * prof.r_outer)
    assert sysm.sum_of_radii_mas > beta_aur.sum_of_radii_mas
    # the tabulated profile's own V(x): the null of the inner (linear-law)
    # disk of radius R_edge = R_outer/r_outer sits at x_null(u) * r_outer
    # in units of the drawn radius
    x = np.linspace(2.5, 5.0, 20001)
    v = visibility_profile_batch(x, prof.mu, prof.intensity[:1])[0]
    x_null = x[np.argmin(np.abs(v))]
    vl = visibility_ld_disk(x, 0.6)
    x_null_lin = x[np.argmin(np.abs(vl))]
    assert x_null == pytest.approx(x_null_lin * prof.r_outer, rel=1e-3)


def test_native_mu_kernel_flux_matches_disk_factor(algol):
    """The renderer draws the table's own piecewise-linear profile out
    to the outer boundary: the rendered flux equals pi r_drawn^2 dff."""
    lam = np.array([500.0])
    prof = spherical_profile(lam, mu_edge=0.12)
    star = with_tables(algol.primary, planck_flux_table(12550.0, lam), prof)
    sysm = replace(algol, primary=star, secondary=with_tables(
        algol.secondary, planck_flux_table(4900.0, lam), prof))
    grid = GridConfig(supersample=4).for_system(sysm)
    pos = positions_at(sysm, 0.0)
    img = np.asarray(render_image(pos, sysm, 500.0, grid), dtype=float)
    from hbtsim.spectral import reference_flux
    ref = reference_flux(sysm, lam, grid)[0]
    assert img.sum() == pytest.approx(ref, rel=3e-4)
    # the I -> 0 tail and the mu = 0 padding: with a wide tail (mu_edge =
    # 0.5, i.e. the outer 13 % of the drawn radius) pixels inside the drawn
    # rim but outside the limb are dark, and the flux still matches
    wide = spherical_profile(lam, mu_edge=0.5)
    star_w = with_tables(algol.primary, planck_flux_table(12550.0, lam), wide)
    sysw = replace(sysm, primary=star_w)
    gridw = GridConfig(supersample=4).for_system(sysw)
    posw = positions_at(sysw, 0.0)
    imgw = np.asarray(render_image(posw, sysw, 500.0, gridw), dtype=float)
    assert imgw.sum() == pytest.approx(reference_flux(sysw, lam, gridw)[0], rel=3e-4)
    r_px = sysw.drawn_radius_mas(star_w) / gridw.pixel_scale_mas
    c = (gridw.n - 1) / 2.0
    xc = c + float(posw.x1) / gridw.pixel_scale_mas
    yc = c + float(posw.y1) / gridw.pixel_scale_mas
    rim = imgw[int(round(yc)), int(round(xc + 0.95 * r_px))]
    centre = imgw[int(round(yc)), int(round(xc))]
    inner = imgw[int(round(yc)), int(round(xc + 0.80 * r_px))]
    assert rim < 1e-2 * centre and inner > 0.3 * centre


def test_on_grid_zero_below_mu_min():
    prof = spherical_profile(LAM[:3], mu_edge=0.1)
    row = prof.on_grid(np.array([0.0, 0.01, 0.5, 1.0]), LAM[0])[0]
    assert row[0] == 0.0 and row[1] == 0.0 and row[3] == pytest.approx(1.0)
    assert prof.nodes[0] == 0.0 and prof.rows_on_nodes(LAM[0])[0, 0] == 0.0


# ---------------------------------------------------------------------------
# Hankel transform
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["quadratic", "spherical"])
def test_batch_hankel_matches_reference(kind):
    mu = np.linspace(0.0, 1.0, 65) if kind == "quadratic" else spherical_profile(LAM[:2]).mu
    if kind == "quadratic":
        rows = np.stack([1 - 0.4 * (1 - mu) - 0.25 * (1 - mu) ** 2,
                         1 - 0.6 * (1 - mu) - 0.10 * (1 - mu) ** 2])
    else:
        rows = spherical_profile(LAM[:2]).intensity
    x = np.linspace(0.0, 15.0, 301)
    ref = np.stack([visibility_profile(x, mu, r, n_r=20000) for r in rows])
    got = visibility_profile_batch(x, mu, rows)
    # the 20000-point trapezoid reference is itself ~5e-5 off at a sharp
    # edge; the Gauss-Legendre rule on the native segments is the exact one
    assert np.abs(got - ref).max() < (2e-5 if kind == "quadratic" else 1e-4)
    nodes, w, mat = hankel_nodes(mu)
    assert np.allclose(mat.sum(axis=1)[nodes > mu.min()], 1.0)
    # linear law: exact 1 - u/3 normalization
    lin = 1 - 0.5 * (1 - np.linspace(0, 1, 9))
    n2, w2, m2 = hankel_nodes(np.linspace(0, 1, 9))
    assert np.sum(w2 * n2 * (m2 @ lin)) == pytest.approx((1 - 0.5 / 3) / 2, rel=1e-12)


def test_disk_visibility_cache():
    prof = spherical_profile(LAM[::400])
    cache = DiskVisibilityCache()
    x = np.random.default_rng(1).uniform(0.0, 12.0, size=(prof.wavelength_nm.size, 5000))
    got = cache.lookup(prof, prof.wavelength_nm, x)
    ref = visibility_profile_batch(x, prof.mu, prof.rows(prof.wavelength_nm))
    assert np.abs(got - ref).max() < 1e-4
    assert len(cache._store) == 1
    cache.lookup(prof, prof.wavelength_nm, x)          # hit
    cache.lookup(prof, prof.wavelength_nm, x * 1.2)    # x_max 14.4 -> same 16-unit table
    assert len(cache._store) == 1


@pytest.mark.slow
def test_tabulated_analytic_path_is_fast(beta_aur, spec_r5000):
    import time
    from hbtsim.hbt import binary_vis2_analytic
    lam = spec_r5000.channel_centers_nm
    prof = spherical_profile(lam)
    sysm = replace(beta_aur,
                   primary=with_tables(beta_aur.primary, planck_flux_table(9350.0, lam), prof),
                   secondary=with_tables(beta_aur.secondary, planck_flux_table(9200.0, lam), prof))
    pos = positions_at(sysm, 0.0)
    b = np.linspace(20.0, 200.0, 591)
    t0 = time.perf_counter()
    v = binary_vis2_analytic(b, lam, sysm, float(pos.rho))
    dt = time.perf_counter() - t0
    assert v.shape == (lam.size, b.size) and np.all(np.isfinite(v))
    assert dt < 20.0


# ---------------------------------------------------------------------------
# channel averaging
# ---------------------------------------------------------------------------
def test_band_average_flux_exact_for_linear_table():
    lam = np.linspace(400.0, 500.0, 1001)
    ft = FluxTable(wavelength_nm=lam, flux=2.0 + 0.01 * lam)
    edges = np.array([400.0, 410.0, 420.5, 450.0])
    got = band_average_flux(ft, edges)
    exact = 2.0 + 0.01 * 0.5 * (edges[:-1] + edges[1:])
    assert np.allclose(got, exact, rtol=1e-9)
    # a channel narrower than the table step: the centre value
    narrow = band_average_flux(ft, np.array([440.0, 440.05]))
    assert narrow[0] == pytest.approx(2.0 + 0.01 * 440.025, rel=1e-9)


def test_rebin_recovers_channel_means_and_changes_rates(beta_aur, c2pu, spad_lambda):
    lam = LAM
    flux = line_flux(lam, 9350.0)
    prof = spherical_profile(lam)
    star = with_tables(beta_aur.primary, FluxTable(lam, flux), prof)
    spec = Spectrograph(lambda_min_nm=480.0, lambda_max_nm=492.0, n_channels=6)
    edges = spec.channel_edges_nm
    rb = rebin_to_channels(star, edges)
    assert np.allclose(rb.flux_table.wavelength_nm, spec.channel_centers_nm)
    # channel 3 (486-488 nm) contains the line: its mean is below the
    # centre-sampled continuum and above the line-core value
    k = 3
    assert rb.flux_table.flux[k] < star.flux_table(spec.channel_centers_nm[k] + 0.9)
    assert rb.flux_table.flux[k] > star.flux_table(486.0)
    # the profile row of that channel is intensity-weighted (the core is
    # less darkened than the continuum): between the two
    rows = band_average_profile(prof, star.flux_table, edges)
    core = prof.rows(486.0)[0]
    cont = prof.rows(489.0)[0]
    j = np.argmin(np.abs(prof.mu - 0.3))
    assert min(core[j], cont[j]) <= rows[k, j] <= max(core[j], cont[j])
    # through the SNR entry point the rates differ from centre sampling
    sysm = replace(beta_aur, primary=star,
                   secondary=with_tables(beta_aur.secondary, FluxTable(lam, line_flux(lam, 9200.0)), prof))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = spectral_g2_snr(sysm, 40.0, spectrograph=spec, vis2_method="analytic",
                            telescope1=c2pu, detector1=spad_lambda)
        m_centre = system_ab_mag(sysm, spec.channel_centers_nm)
    assert not np.allclose(r.mag_ab, m_centre)
    assert prepare_system(beta_aur, spec) is beta_aur


# ---------------------------------------------------------------------------
# the (T_eff, log g) grid
# ---------------------------------------------------------------------------
def _write_corner(path, teff, logg, lam, mu_edge):
    prof = spherical_profile(lam, mu_edge=mu_edge, u_cont=0.5 + (teff - 9000.0) / 4000.0)
    save_star_tables(path, dict(wavelength_nm=lam, flux=line_flux(lam, teff, depth=0.3 + logg / 20),
                                mu=prof.mu, intensity=prof.intensity, path=str(path),
                                r_outer_over_tau1=prof.r_outer, mu_tau1=mu_edge,
                                teff=teff, logg=logg))


@pytest.fixture
def synthetic_grid(tmp_path):
    lam = LAM[::10]
    for t in (9200.0, 9400.0):
        for g in (3.5, 4.0):
            _write_corner(tmp_path / f"newera_lte{int(t):05d}-{g:.2f}-0.0_380-1000nm_2nm.npz",
                          t, g, lam, mu_edge=0.06 + 0.02 * (4.0 - g))
    return NewEraGrid.scan(str(tmp_path)), lam


def test_grid_scan_and_corners(synthetic_grid):
    grid, lam = synthetic_grid
    assert grid.coverage() == dict(teff=[9200.0, 9400.0], logg=[3.5, 4.0], n=4)
    t = grid.interpolate(9200.0, 4.0)
    ft, ld = grid.load((9200.0, 4.0, 0.0))
    assert np.allclose(t.flux_table.flux, ft.flux, rtol=1e-12)
    assert t.corners == ((9200.0, 4.0, 1.0),)
    assert not t.extrapolated
    assert ld.r_outer == pytest.approx(1.0 / np.sqrt(1 - 0.06**2))
    assert t.ld_profile.r_outer == pytest.approx(ld.r_outer)
    # exact at the corner: the profile re-expressed on the union grid
    # integrates to the same disk factor
    from hbtsim.params import integrate_profile_times_mu
    d0 = 2 * integrate_profile_times_mu(ld.mu, ld.intensity[5])
    d1 = 2 * integrate_profile_times_mu(t.ld_profile.mu, t.ld_profile.intensity[5])
    assert d1 == pytest.approx(d0, rel=2e-3)


def test_grid_bilinear_weights_and_edge(synthetic_grid):
    grid, lam = synthetic_grid
    t = grid.interpolate(9300.0, 3.75)
    w = dict(((c[0], c[1]), c[2]) for c in t.corners)
    assert all(w[k] == pytest.approx(0.25) for k in w) and len(w) == 4
    logf = sum(w[(tt, g)] * np.log10(grid.load((tt, g, 0.0))[0].flux) for tt, g in w)
    assert np.allclose(np.log10(t.flux_table.flux), logf, atol=1e-12)
    # the edge of the interpolated model: r_outer = the largest corner
    r_out = max(grid.load(k)[1].r_outer for k in grid.tables)
    assert t.ld_profile.r_outer == pytest.approx(r_out)
    # the interpolated limb (half-intensity drop) sits at r' = 1 within 1e-3
    ext = spherical_extension(t.ld_profile, t.flux_table)
    rp_edge = np.sqrt(1 - ext["mu_edge"]**2) * t.ld_profile.r_outer
    assert rp_edge == pytest.approx(1.0, abs=2e-3)
    assert t.ld_profile.intensity[:, -1].max() == pytest.approx(1.0)


def test_grid_refuses_and_clamps(synthetic_grid):
    grid, _ = synthetic_grid
    with pytest.raises(ValueError, match="outside the NewEra grid"):
        grid.interpolate(9800.0, 4.0)
    with pytest.warns(UserWarning, match="clamped"):
        t = grid.interpolate(9800.0, 4.2, allow_extrapolation=True)
    assert t.extrapolated and t.clamped == {"teff": (9800.0, 9400.0), "logg": (4.2, 4.0)}
    assert t.corners == ((9400.0, 4.0, 1.0),)
    with pytest.raises(ValueError, match="M/H"):
        grid.interpolate(9300.0, 3.8, z=0.5)
    # clamping is bounded: a star far beyond the edge is never clamped
    with pytest.raises(ValueError, match="not clamped"):
        grid.interpolate(20900.0, 4.15, allow_extrapolation=True)
    with pytest.raises(ValueError, match="not clamped"):
        grid.interpolate(9300.0, 5.0, allow_extrapolation=True)


def test_with_newera_report(synthetic_grid, beta_aur, algol, delta_vel):
    grid, lam = synthetic_grid
    sysm, report = with_newera(beta_aur, grid)
    assert sysm.has_sed_tables
    assert all("interpolated" in v for v in report.values())
    # log g from mass and radius: 3.93 / 3.98 -> inside the grid
    assert 3.5 < beta_aur.primary.log_g < 4.0
    # a star outside the grid keeps its defaults and is reported
    sysm2, rep2 = with_newera(algol, grid)
    assert sysm2.secondary.flux_table is None and "no NewEra coverage" in rep2[algol.secondary.name]
    assert sysm2.primary.flux_table is None
    # Algol A (12 550 K) is > 1000 K beyond this grid's 9400 K edge: never clamped
    sysm3, rep3 = with_newera(algol, grid, which=("primary",), allow_extrapolation=True)
    assert sysm3.primary.flux_table is None and "not clamped" in rep3[algol.primary.name]
    # delta Vel Ab (9830 K) is 430 K beyond: clamped with a warning when allowed
    with pytest.warns(UserWarning, match="clamped"):
        sysm4, rep4 = with_newera(delta_vel, grid, allow_extrapolation=True)
    assert sysm4.secondary.flux_table is not None and "clamped" in rep4[delta_vel.secondary.name]


# ---------------------------------------------------------------------------
# real tables (gated)
# ---------------------------------------------------------------------------
needs_data = pytest.mark.skipif(not glob.glob(os.path.join(NEWERA_DIR, "newera_lte*.npz")),
                                reason="no NewEra tables in data/newera")


@needs_data
def test_real_grid_covers_beta_aur_and_delta_vel(beta_aur, delta_vel, algol):
    grid = NewEraGrid.scan(NEWERA_DIR)
    for sysm in (beta_aur, delta_vel):
        s2, rep = with_newera(sysm, grid)
        assert s2.has_sed_tables, rep
        for star in (s2.primary, s2.secondary):
            assert 1.002 < star.ld_profile.r_outer < 1.010
            assert star.ld_profile.mu_edge > 0.05
        # the model photometry against the observed anchors: Beta Aur lands
        # on its g anchor to 0.02 mag; delta Vel (Merand 2011 radii, Teffs
        # and distance) comes out 0.24 mag BRIGHTER than the A-only V ~ 2.0
        # -- a real tension in the literature parameters, flagged, not hidden
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = model_ab_mag(s2, np.array([477.0, 763.0]))
        anchors = dict(sysm.mag_anchors)
        tol = 0.2 if sysm == beta_aur else 0.3
        assert abs(m[0] - anchors[477.0]) < tol and abs(m[1] - anchors[763.0]) < tol
    s3, rep3 = with_newera(algol, grid)
    assert s3.primary.flux_table is None and s3.secondary.flux_table is None
    with pytest.warns(UserWarning, match="clamped"):
        s4, rep4 = with_newera(algol, grid, allow_extrapolation=True)
    assert s4.primary.flux_table is not None and s4.secondary.flux_table is None


@needs_data
def test_real_tables_edge_definitions_agree():
    for path in sorted(glob.glob(os.path.join(NEWERA_DIR, "newera_lte*.npz")))[:4]:
        d = np.load(path)
        if "mu_tau1" not in d:
            continue
        r_tau = 1 / np.sqrt(1 - float(d["mu_tau1"])**2)
        r_drop = float(d["r_outer_over_edge"])
        assert abs(r_tau - r_drop) < 1e-4


def test_rebin_is_memoized(beta_aur):
    """Repeated channel averaging of the same tables returns the same
    objects (so the disk-visibility cache keyed on them keeps hitting)."""
    lam = LAM
    prof = spherical_profile(lam)
    star = with_tables(beta_aur.primary, FluxTable(lam, line_flux(lam, 9350.0)), prof)
    spec = Spectrograph(lambda_min_nm=480.0, lambda_max_nm=492.0, n_channels=6)
    a = rebin_to_channels(star, spec.channel_edges_nm)
    b = rebin_to_channels(star, spec.channel_edges_nm)
    assert a.ld_profile is b.ld_profile and a.flux_table is b.flux_table
    c = rebin_to_channels(star, Spectrograph(lambda_min_nm=480.0, lambda_max_nm=492.0, n_channels=7).channel_edges_nm)
    assert c.ld_profile is not a.ld_profile


def test_bracket_on_node_ignores_zero_weight_corners(tmp_path):
    """A star on a grid node in log g must not demand an upper-log g
    neighbour it gives zero weight (Vega at log g 4.0 once 4.5 tables
    exist for other temperatures)."""
    lam = LAM[::10]
    for t, g in ((9400.0, 3.5), (9400.0, 4.0), (9600.0, 3.5), (9600.0, 4.0), (9800.0, 4.5)):
        _write_corner(tmp_path / f"newera_lte{int(t):05d}-{g:.2f}-0.0_x.npz", t, g, lam, 0.07)
    grid = NewEraGrid.scan(str(tmp_path))
    t = grid.interpolate(9550.0, 4.0)
    assert {(c[0], c[1]) for c in t.corners} == {(9400.0, 4.0), (9600.0, 4.0)}
    assert sum(c[2] for c in t.corners) == pytest.approx(1.0)
    # T_eff on a node needs no T neighbour either
    t2 = grid.interpolate(9600.0, 3.75)
    assert {(c[0], c[1]) for c in t2.corners} == {(9600.0, 3.5), (9600.0, 4.0)}
    # a genuinely missing weighted corner still raises
    with pytest.raises(ValueError, match="missing corner"):
        grid.interpolate(9700.0, 4.25)


@needs_data
def test_real_grid_covers_sirius_and_vega():
    grid = NewEraGrid.scan(NEWERA_DIR)
    if (9800.0, 4.5, 0.0) not in grid.tables:
        pytest.skip("log g 4.5 corners not binned")
    t = grid.interpolate(9940.0, 4.33)
    assert {(c[0], c[1]) for c in t.corners} == {(9800.0, 4.0), (9800.0, 4.5), (10000.0, 4.0), (10000.0, 4.5)}
    assert 1.002 < t.ld_profile.r_outer < 1.01
    v = grid.interpolate(9550.0, 4.0)
    assert {(c[0], c[1]) for c in v.corners} == {(9400.0, 4.0), (9600.0, 4.0)}
