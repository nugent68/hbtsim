"""The literature numbers behind the shipped catalog, pinned so that an
edit to a JSON file that changes the science shows up here (the paper and
docs/three_telescope_feasibility.md quote these)."""

import numpy as np
import pytest

from hbtsim.catalog import Catalog
from hbtsim.params import MAS

cat = Catalog(env=False)


def test_binary_orbits():
    b = cat.load_target("betaaur")
    assert (b.period_days, b.inclination_deg, b.distance_pc) == (3.96004, 76.8, 24.87)
    assert b.semimajor_au == 0.08186 and b.angular_semimajor_mas == pytest.approx(3.29, abs=0.005)
    assert b.node_pa_deg == 295.15 and b.mag_anchors == ((477.0, 1.80), (763.0, 2.10))
    a = cat.load_target("algol")
    assert (a.period_days, a.inclination_deg, a.distance_pc, a.semimajor_au) == (2.867328, 98.70, 28.82, 0.0620)
    assert (a.primary.teff, a.secondary.teff) == (12550.0, 4900.0) and a.node_pa_deg == 43.43
    s = cat.load_target("spica")
    assert (s.period_days, s.inclination_deg, s.distance_pc, s.semimajor_au) == (4.0145, 63.1, 76.6, 0.1311)
    assert s.eccentricity == 0.108 and s.arg_periastron_deg == 232.0 and s.node_pa_deg == 131.6 and s.dec_deg == pytest.approx(-11.161319)
    d = cat.load_target("deltavel")
    assert (d.period_days, d.eccentricity, d.arg_periastron_deg, d.inclination_deg) == (45.1503, 0.290, 109.7, 89.0)
    assert (d.distance_pc, d.semimajor_au, d.node_pa_deg) == (25.13, 0.4156, 65.0)
    assert "CHECK" in " ".join(cat.raw("target", "deltavel")["notes"])


def test_single_stars():
    sa = cat.load_target("sirius_a")
    assert (sa.theta_ld_mas, sa.distance_pc, sa.star.teff, sa.star.logg) == (6.039, 2.637, 9940.0, 4.33)
    assert sa.star.radius_rsun == pytest.approx(0.5 * 6.039 * MAS * 2.637 * 3.0856775814913673e16 / 6.957e8)
    v = cat.load_target("vega")
    assert (v.theta_ld_mas, v.v_mag, v.dec_deg) == (3.329, 0.03, 38.7837)
    h = cat.load_target("hd17652")
    assert h.mag_anchors == ((551.0, 4.456 + 0.02), (1630.0, 2.256 + 1.39), (2190.0, 2.139 + 1.85))
    g = cat.load_target("gammacas")
    assert g.ellipse.axis_ratio == 1.28 and g.ellipse.pa_deg == 116.0
    assert g.ellipse.theta_major_mas == pytest.approx(0.43 * 1.28) and g.ellipse.theta_minor_mas == pytest.approx(0.43)
    assert g.mag_anchors == ((445.0, 2.29 - 0.09), (551.0, 2.39 + 0.02))


def test_hardware():
    assert cat.load_telescope("keck_10m").diameter_m == 10.0
    e = cat.load_telescope("eonsii_4m")
    assert (e.diameter_m, e.throughput, e.collecting_area_m2) == (4.0, 0.64, 9.0)
    s = cat.load_detector("spad_lambda")
    assert (s.jitter_fwhm_ps, s.dead_time_ns, s.dark_cps_per_pixel, s.max_total_cps) == (120.0, 10.0, 250.0, 1.4e8)
    assert s.pde(520.0) == 0.50 and s.readout == "timetag"
    ng = cat.load_detector("spad_lambda_ng")
    assert ng.readout == "correlator" and ng.max_total_cps is None and ng.pde_table_nm == s.pde_table_nm
    m = cat.load_detector("eonsii_mcp_pmt")
    from hbtsim.snr import FWHM_TO_SIGMA
    assert m.jitter_fwhm_ps * FWHM_TO_SIGMA * np.sqrt(2.0) == pytest.approx(27.4)   # pair sigma
    kk = cat.load_detector("kk_ideal")
    assert kk.jitter_fwhm_ps * FWHM_TO_SIGMA == pytest.approx(16.6)
    sp = cat.load_spectrograph("spad_lambda_320")
    assert (sp.lambda_min_nm, sp.lambda_max_nm, sp.n_channels, sp.throughput) == (400.0, 950.0, 320, 0.5)
    r = cat.load_spectrograph("r5000_400_950")
    assert r.resolving_power == 5000.0 and r.n_channels == 4325
    assert cat.load_spectrograph("eonsii_r7500").n_channels == 2389
    assert cat.load_spectrograph("eonsii_1000ch").channel_width_nm == pytest.approx(0.15)


def test_arrays_and_sites():
    tri = cat.load_triangle("maunakea_subaru_keck")
    assert sorted(np.round(tri.baseline_lengths(), 1)) == [84.9, 152.1, 225.9]
    (_, _, b), = cat.load_array("keck_pair").pairs()
    assert np.hypot(*b) == pytest.approx(84.9, abs=0.05)
    vlt = cat.load_array("vlt_ut")
    lengths = sorted(np.hypot(*b) for _, _, b in vlt.pairs())
    assert lengths[0] == pytest.approx(46.6, abs=0.1) and lengths[-1] == pytest.approx(130.2, abs=0.1)
    ver = cat.load_array("veritas")
    got = sorted(float(np.hypot(*b)) for _, _, b in ver.pairs())
    assert got == pytest.approx([81.5, 99.4, 99.4, 108.8, 126.4, 172.5], abs=0.15)   # positions fitted to the published baselines
    assert cat.load_site("paranal").latitude_deg == pytest.approx(-24.6272)
    assert cat.load_site("teide").elevation_m == 2390.0
    cal = cat.raw("site", "calern")
    assert "TO BE CONFIRMED" in " ".join(cal["notes"])


def test_iact_backends():
    v = cat.load_backend("veritas_sii")
    assert (v.lambda_nm, v.dlambda_nm, v.alpha, v.b_el_hz) == (416.0, 13.0, 0.30, 125e6)
    assert v.anchor.star == "eps Ori" and v.anchor.mag_ab == pytest.approx(1.50 - 0.09)
    assert (v.anchor.sigma_vis2, v.anchor.t_s) == (0.016, 4.25 * 3600.0)
    m = cat.load_backend("magic_sii")
    assert (m.lambda_nm, m.dlambda_nm, m.alpha, m.q, m.noise_factor, m.sigma_spec) == (425.0, 26.0, 0.295, 0.304, 1.15, 0.87)
    assert m.anchor is None


def test_lpqi_pathfinder():
    """The LPQI-Pathfinder entries (docs/lpqi_pathfinder.md)."""
    pf = cat.load_array("lpqi_pathfinder")
    (_, _, b), = pf.pairs()
    assert np.hypot(*b) == pytest.approx(531.8, abs=0.1)                # surveyed NOT-TNG baseline (project quotes 550 m)
    assert np.degrees(np.arctan2(b[0], b[1])) % 360 == pytest.approx(226.7, abs=0.1)
    assert (pf.stations[0].telescope.diameter_m, pf.stations[1].telescope.diameter_m) == (2.56, 3.58)
    assert pf.site.latitude_deg == pytest.approx(28.75728)
    orm = cat.load_array("lpqi_orm")
    lengths = sorted(float(np.hypot(*b)) for _, _, b in orm.pairs())
    assert lengths[0] == pytest.approx(409, abs=2) and lengths[-1] == pytest.approx(1518, abs=5)   # TNG-GTC, GTC-INT
    assert {s.telescope.diameter_m for s in orm.stations} == {2.56, 3.58, 10.4, 4.2, 2.54}
    d = cat.load_detector("lpqi_spad64_i2cass")
    assert d.pde(550.0) == pytest.approx(0.026) and d.jitter_fwhm_ps == 500.0 and d.n_pixels == 25
    assert d.max_total_cps == 6.7e6 and d.readout == "timetag"
    n = cat.load_detector("lpqi_spad64_i2cass_nextgen")           # F. Prada, 2026-10-07
    assert n.pde(425.0) == 0.5 and n.pde(550.0) == 0.5 and n.pde(650.0) == pytest.approx(0.2)
    assert n.pde(656.28) == pytest.approx(0.194, abs=0.002) and n.jitter_fwhm_ps == 10.0 and n.n_pixels == 4   # 10 ps: F. Prada 2026-10-09
    assert (n.dead_time_ns, n.dark_cps_per_pixel, n.max_total_cps, n.max_cps_per_pixel) == (10.0, 1.68, None, 6.7e6)
    assert n.jitter_fwhm_ps == 10.0 and n.pde(500.0) == 0.5 and n.pde(950.0) == pytest.approx(0.025)   # F. Prada 2026-10-09
    r = cat.load_spectrograph("lpqi_r10000")
    assert 8600 < r.n_channels < 8700 and r.channel_centers_nm[0] < 401 and r.channel_centers_nm[-1] > 949
    assert cat.load_backend("lpqi_r10000_nextgen").detector.max_cps_per_pixel == 6.7e6
    m = cat.load_detector("lpqi_spad1_mpd")
    assert (m.pde(550.0), m.jitter_fwhm_ps, m.dead_time_ns, m.dark_cps_per_pixel, m.n_pixels) == (0.3, 35.0, 55.0, 50.0, 1)
    assert m.max_total_cps == 12e6 and m.readout == "timetag"      # F. Prada 2026-10-09
    mb = cat.load_detector("lpqi_spad1_mpd_blue")
    assert mb.jitter_fwhm_ps == 275.0 and mb.pde(425.0) == 0.3 and mb.dead_time_ns == 55.0
    for key, lam in (("halpha", 656.28), ("hbeta", 486.13), ("500", 500.0), ("550", 550.0), ("425", 425.0)):
        f = cat.load_spectrograph(f"filter_lpqi_{key}_1nm")
        assert f.n_channels == 1 and f.channel_width_nm == pytest.approx(1.0)
        assert float(f.channel_centers_nm[0]) == pytest.approx(lam)
        assert cat.load_backend(f"lpqi_{key}").detector == d
        assert cat.load_backend(f"lpqi_nextgen_{key}").detector == n
        assert cat.load_backend(f"lpqi_mpd_{key}").detector == (mb if key == "425" else m)
    assert "surveyed" in " ".join(cat.raw("array", "lpqi_pathfinder")["notes"])        # Caporali & Barbieri + NOT GPS


def test_lpqi_hot_star_targets():
    """The northern O/B targets of g2_singles_lpqi_pathfinder: theta_LD = 2R/d
    from the quoted radii and distances (no interferometric diameters)."""
    expect = {"lamori_a": (0.238, 9.934, 3.7), "betcep": (0.320, 70.561, 3.23), "etaori": (0.201, -2.397, 3.42),
              "epsper": (0.363, 40.010, 2.88), "delori_aa": (0.319, -0.299, 2.23), "gampeg": (0.349, 15.184, 2.84),
              "zettau": (0.365, 21.1425, 3.01)}
    for name, (theta, dec, v) in expect.items():
        t = cat.load_target(name)
        assert t.theta_ld_mas == pytest.approx(theta, abs=0.001) and t.dec_deg == pytest.approx(dec) and t.v_mag == v
        assert t.star.teff > 15000 and t.mag_anchors[0][0] == 551.0
        raw = cat.raw("target", name)
        assert any("theta_ld_mas" in a for a in raw["assumptions"])
    camp = cat.load_campaign("g2_singles_lpqi_pathfinder")
    assert len(camp.targets) == 8 and len(camp.backends) == 10 and camp.option("options.diameter_precision") == 0.05


def test_etaori_binary_target():
    """eta Ori Aa as a binary (the orbit movie): Kepler semi-major axis from
    the masses, the angular scales at the Hipparcos distance, the eclipses
    at quadrature +/- 0.25 and the flagged assumptions."""
    from hbtsim.orbit import positions_at
    from hbtsim.params import in_eclipse
    s = cat.load_target("etaori_ab")
    assert s.period_days == pytest.approx(7.98763) and s.inclination_deg == pytest.approx(87.62)
    assert s.semimajor_au == pytest.approx(((s.primary.mass_msun + s.secondary.mass_msun) * (s.period_days / 365.25) ** 2) ** (1 / 3), rel=1e-3)
    assert s.angular_semimajor_mas == pytest.approx(0.724, abs=0.002)
    assert 2 * s.angular_radius_mas(s.primary) == pytest.approx(0.201, abs=0.002)
    assert 2 * s.angular_radius_mas(s.secondary) == pytest.approx(0.149, abs=0.002)
    assert not in_eclipse(s, positions_at(s, 0.0)) and in_eclipse(s, positions_at(s, 0.25)) and in_eclipse(s, positions_at(s, 0.75))
    raw = cat.raw("target", "etaori_ab")
    assert any("node_pa_deg" in a for a in raw["assumptions"]) and any("distance_pc" in a for a in raw["assumptions"])
    s90 = cat.load_target("etaori_ab_node90")
    assert s90.node_pa_deg == 90.0 and s90.semimajor_au == s.semimajor_au and s90.mag_anchors == s.mag_anchors
    assert cat.load_campaign("movie_etaori_lpqi_orbit_node90").target.node_pa_deg == 90.0
    for nm in ("movie_etaori_lpqi_2orbits", "movie_etaori_lpqi_2orbits_node90"):
        c2 = cat.load_campaign(nm)
        assert c2.option("options.nights") == 16 and c2.runner == "nightmovie"
    assert cat.load_campaign("movie_etaori_lpqi_2orbits_node90").target.node_pa_deg == 90.0
    cg = cat.load_campaign("movie_etaori_lpqi_2orbits_gap_node90")
    assert cg.option("options.gap") == {"after_night": 8, "days": 323} and cg.option("options.nights") == 16
    camp = cat.load_campaign("movie_etaori_lpqi_orbit")
    assert camp.runner == "nightmovie" and camp.option("options.third_light_fraction") == pytest.approx(0.214)
    assert len(camp.backends) == 1 and camp.backends[0].name.startswith("LPQI 1 nm H-alpha")


def test_iotaori_binary_target_and_red_filter():
    """iota Ori Aa (the Pathfinder's best geometric-distance binary) and the
    proposed 650 nm red continuum filter: angular scales at the orbit's
    412 pc, the eccentric separation range, and the campaign's plan."""
    from hbtsim.orbit import positions_at
    from hbtsim.params import in_eclipse
    s = cat.load_target("iotaori_aa")
    assert s.period_days == pytest.approx(29.13376) and s.eccentricity == pytest.approx(0.7452)
    assert s.semimajor_au == pytest.approx(132.32 * 0.00465047, rel=1e-3)
    assert s.distance_pc == 378.0                       # Gaia DR3 B + C mean (A has no parallax)
    assert s.angular_semimajor_mas == pytest.approx(1.628, abs=0.003)
    assert 2 * s.angular_radius_mas(s.primary) == pytest.approx(0.224, abs=0.002)
    assert 2 * s.angular_radius_mas(s.secondary) == pytest.approx(0.122, abs=0.002)
    import numpy as np
    rho = np.array([float(positions_at(s, p).rho) for p in np.linspace(0, 1, 200, endpoint=False)])
    assert 0.15 < rho.min() < 0.30 and 1.9 < rho.max() < 2.2
    assert not any(in_eclipse(s, positions_at(s, p)) for p in np.linspace(0, 1, 200, endpoint=False))
    f = cat.load_spectrograph("filter_lpqi_650_1nm")
    assert f.channel_centers_nm[0] == pytest.approx(650.0) and f.channel_widths_nm[0] == pytest.approx(1.0)
    b = cat.load_backend("lpqi_mpd_650")
    assert b.detector.name == cat.load_detector("lpqi_spad1_mpd").name
    camp = cat.load_campaign("movie_iotaori_lpqi_orbit")
    assert camp.option("options.night_step_days") == pytest.approx(29.13376 / 8, abs=1e-3)
    assert camp.option("options.third_light_fraction") == pytest.approx(0.07) and camp.option("options.nights") == 8
    refs = camp.option("options.reference_distances")
    assert [r["pc"] for r in refs] == [378.0, 359.0, 384.0, 714.0] and "lo_pc" not in refs[1]
    raw = cat.raw("target", "iotaori_aa")
    assert any("Pablo" in a for a in raw["assumptions"]) and any("node_pa_deg" in a for a in raw["assumptions"])


def test_isi():
    """The relocated ISI at Palomar with MPD PDM 100 um SPADs: site,
    telescope, detector (the operating cap), filters, the reconfigurable
    pair and triangle, the cool targets and Capella."""
    import numpy as np
    s = cat.load_site("palomar")
    assert (s.latitude_deg, s.elevation_m) == (33.356, 1712.0) and cat.load_site("mt_wilson").latitude_deg == 34.2259
    t = cat.load_telescope("isi_1p65m")
    assert t.diameter_m == 1.65 and t.throughput == 0.5
    d = cat.load_detector("mpd_pdm_100um")
    assert (d.pde(400.0), d.pde(550.0), d.pde(650.0)) == (0.24, 0.49, 0.37)          # PDM datasheet, Feb 2018
    assert 0.05 < d.pde(900.0) < 0.2 and d.pde(1100.0) == d.pde(1000.0) < 0.05
    assert (d.jitter_fwhm_ps, d.dead_time_ns, d.dark_cps_per_pixel, d.max_total_cps, d.readout) == (50.0, 77.0, 50.0, 4e6, "timetag")
    for nm in (650.0, 900.0):
        f = cat.load_spectrograph(f"filter_isi_{nm:.0f}_1nm")
        assert f.channel_centers_nm[0] == pytest.approx(nm) and f.channel_widths_nm[0] == pytest.approx(1.0)
        assert cat.load_backend(f"isi_pdm_{nm:.0f}").detector.max_total_cps == 4e6
    a = cat.load_array("isi_pair")
    (_, _, b), = a.pairs()
    assert np.hypot(*b) == pytest.approx(100.0) and b[0] == pytest.approx(100.0, abs=1e-6)       # East-West
    assert a.stations[0].detector.name == d.name
    (_, _, b12), = cat.load_array("isi_pair", baseline_m=12.0).pairs()
    assert np.hypot(*b12) == pytest.approx(12.0)
    tri = cat.load_triangle("isi_triangle")
    assert np.allclose(tri.baseline_lengths(), 100.0)
    line = cat.load_array("isi_line", spacings_m=[14.0, 30.0])                 # three trucks in a row: a, b, a + b
    assert sorted(round(float(np.hypot(*b)), 6) for _, _, b in line.pairs()) == [14.0, 30.0, 44.0]
    assert all(abs(st.north_m) < 1e-9 for st in line.stations)                  # East-West
    with pytest.raises(Exception, match="minimum"):
        cat.load_array("isi_line", spacings_m=[1.0, 30.0])
    assert cat.load_campaign("movie_epstau_isi_line").option("instrument.array") == "isi_line"
    for name, theta, dec in (("betelgeuse", 42.3, 7.4071), ("aldebaran", 20.58, 16.5093), ("arcturus", 21.05, 19.1824),
                             ("mu_cep", 14.1, 58.78), ("eps_tau", 2.59, 19.1804)):
        st = cat.load_target(name)
        assert st.theta_ld_mas == theta and st.dec_deg == pytest.approx(dec) and st.mag_anchors[-1][0] == 2190.0
        assert float(st.ab_mag(900.0)) < float(st.ab_mag(551.0))                        # red stars: brighter in the red
    assert float(cat.load_target("betelgeuse").star.ld_coeff(900.0)) == pytest.approx(0.60)
    u = [v for _, v in cat.raw("ld_table", "m_giant_rough")["table_nm"]]
    assert all(x > y for x, y in zip(u, u[1:]))
    c = cat.load_target("capella")
    assert c.period_days == 104.02128 and c.angular_semimajor_mas == pytest.approx(56.44, abs=0.05)
    assert 2 * c.angular_radius_mas(c.primary) == pytest.approx(8.47, abs=0.02) and c.node_pa_deg == 40.522
    for name in ("g2_giants_isi", "g2_binaries_isi", "movie_betelgeuse_isi", "movie_altair_isi_oblate", "movie_regulus_isi_oblate",
                 "movie_betelgeuse_isi_chromatic", "movie_spica_isi_orbit", "movie_mira_isi", "movie_capella_isi_orbit",
                 "movie_betaaur_isi_orbit"):
        camp = cat.load_campaign(name)
        assert camp.option("instrument.array").startswith("isi_") and all("PDM" in b.name for b in camp.backends)
    assert len(cat.load_campaign("g2_giants_isi").option("options.baseline_schedule")) == 15
    assert len(cat.load_campaign("movie_betelgeuse_isi").option("options.schedule")) == 9


def test_first_week_targets():
    """The first-week science targets of the ISI: the oblate rotators'
    ellipses (Monnier 2007, McAlister 2005), Mira's size and TiO-band
    factor, Betelgeuse's chromatic diameter, Spica's eccentric orbit and
    beta Aur's orbital parallax."""
    a = cat.load_target("altair")
    assert a.theta_ld_mas == 3.296 and (a.ellipse.theta_major_mas, a.ellipse.axis_ratio, a.ellipse.pa_deg) == (3.67, 1.24, 28.2)
    assert a.dec_deg == pytest.approx(8.8683) and a.distance_pc == 5.13 and a.theta_factor(650.0) == 1.0
    r = cat.load_target("regulus")
    assert r.theta_ld_mas == 1.436 and (r.ellipse.theta_major_mas, r.ellipse.axis_ratio, r.ellipse.pa_deg) == (1.65, 1.32, 175.5)
    assert abs(a.ellipse.theta_major_mas / a.ellipse.axis_ratio ** 0.5 - a.theta_ld_mas) < 0.002       # equal-area disks
    assert abs(r.ellipse.theta_major_mas / r.ellipse.axis_ratio ** 0.5 - r.theta_ld_mas) < 0.002
    m = cat.load_target("mira")
    assert m.theta_ld_mas == 32.0 and m.dec_deg == pytest.approx(-2.9776) and m.distance_pc == 92.0
    assert float(m.ab_mag(900.0)) < float(m.ab_mag(551.0)) and m.theta_factor(650.0) == 1.4 and m.theta_factor(900.0) == 1.0
    b = cat.load_target("betelgeuse")
    assert (b.theta_factor(650.0), b.theta_factor(900.0), b.theta_factor(1650.0), b.theta_factor(2200.0)) == (1.12, 1.02, 1.0, 1.0)
    assert b.theta_factor(775.0) == pytest.approx(1.07)                                                # interpolated
    s = cat.load_target("spica")
    assert (s.eccentricity, s.arg_periastron_deg, s.node_pa_deg) == (0.108, 232.0, 131.6)
    camp = cat.load_campaign("movie_spica_isi_orbit")
    assert camp.option("options.reference_periastron")["omega_deg"] == 90.0 and camp.option("options.fit_periastron_step_deg") == 2.0
    assert cat.load_campaign("movie_capella_isi_orbit").option("options.pupils") is True
    assert cat.load_campaign("movie_altair_isi_oblate").option("options.oblateness") is True
    ba = cat.load_target("betaaur")
    assert ba.angular_semimajor_mas == pytest.approx(3.292, abs=0.002) and 2 * ba.angular_radius_mas(ba.primary) == pytest.approx(1.033, abs=0.002)
    bc = cat.load_campaign("movie_betaaur_isi_orbit")
    assert [(e["side_m"], e["nights"]) for e in bc.option("options.schedule")] == [(30.0, 3), (60.0, 4)]
    assert bc.option("options.fit_theta_range") == [0.85, 1.15] and bc.option("options.fit_scale_step") == 0.002
    assert [r["pc"] for r in bc.option("options.reference_distances")] == [24.87, 24.3]
