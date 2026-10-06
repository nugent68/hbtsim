"""Phase-5 hygiene: helpers, anchoring, station coordinates, movie
pieces, and the corrected Beta Aur semi-major axis."""

import numpy as np
import pytest

from hbtsim.movie import RGB_DISPLAY_NM, precompute_frames, render_display_rgb, stretch_rgb
from hbtsim.orbit import max_separation_phase, positions_at, sky_positions
from hbtsim.params import AU, DAY, PARSEC, MAS, GridConfig, MovieConfig, Star
from hbtsim.photometry import anchored_mags, apparent_ab_mag, max_light_mag
from hbtsim.render import render_image
from hbtsim.g2spec import _bin_channels

from dataclasses import replace

G_SUN = 1.32712440018e20   # m^3 s^-2 (GM_sun)


def test_positions_at_and_take(algol, beta_aur):
    p = positions_at(algol, 0.3)
    assert np.ndim(p.rho) == 0
    arr = sky_positions(np.array([0.0, 2 * np.pi * 0.3]), algol)
    q = arr.take(1)
    for a, b in zip(p, q):
        assert np.allclose(a, b)
    assert 0.0 <= max_separation_phase(algol) < 1.0
    assert sky_positions(2 * np.pi * max_separation_phase(beta_aur), beta_aur).rho \
        == pytest.approx(beta_aur.angular_semimajor_mas, rel=1e-6)


def test_omega_ignored_for_circular_orbits(algol):
    a = sky_positions(np.linspace(0, 6, 7), algol)
    b = sky_positions(np.linspace(0, 6, 7), replace(algol, arg_periastron_deg=77.0))
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_beta_aur_semimajor_from_kepler(beta_aur):
    """a^3 = G(M1 + M2) P^2 / 4 pi^2 with the S07 masses and period."""
    m = (beta_aur.primary.mass_msun + beta_aur.secondary.mass_msun) * G_SUN
    P = beta_aur.period_days * DAY
    a = (m * P**2 / (4 * np.pi**2)) ** (1.0 / 3.0) / AU
    assert beta_aur.semimajor_au == pytest.approx(a, rel=2e-4)
    assert beta_aur.angular_semimajor_mas == pytest.approx(3.29, abs=0.01)


def test_ld_table_is_required(catalog):
    with pytest.raises(TypeError):
        Star("x", 1.0, 1.0, 6000.0)
    ld_algol_a = catalog.load_ld_table("claret11_algol_a")
    assert Star("x", 1.0, 1.0, 6000.0, ld_table_nm=ld_algol_a).ld_coeff(500.0) > 0


def test_anchoring_independent_of_phase_window(algol):
    """With a max-light reference the anchored zero point does not depend
    on which epochs the curve samples (the legacy min() did)."""
    grid = GridConfig()
    ref = max_light_mag(algol, 477.0, grid)
    anchors = dict(algol.mag_anchors)

    def curve(phases):
        f = [float(render_image(positions_at(algol, ph), algol, 477.0, grid).sum())
             for ph in phases]
        return apparent_ab_mag(np.array(f), 477.0, algol, grid)

    m_ecl = anchored_mags(curve([0.2, 0.25, 0.3]), 477.0, algol, ref)   # all in eclipse
    m_out = anchored_mags(curve([0.0, 0.05, 0.5]), 477.0, algol, ref)
    assert m_out.min() == pytest.approx(anchors[477.0], abs=2e-3)
    assert m_ecl.min() > anchors[477.0] + 0.2        # partial phases stay dimmed
    assert m_ecl.max() > anchors[477.0] + 1.0        # mid-eclipse stays deep
    legacy = anchored_mags(curve([0.2, 0.25, 0.3]), 477.0, algol)
    assert legacy.min() == pytest.approx(anchors[477.0])   # the old rectification


def test_maunakea_station_coordinates_from_site(maunakea, maunakea_tri):
    """The ENU station offsets agree with the quoted latitudes and
    longitudes (Subaru 19d49m32s N 155d28m34s W; Keck I 19.8259465 N
    155.474719 W; Keck II 19.8265606 N 155.474234 W) on a local sphere."""
    R = 6371e3 + maunakea.elevation_m
    lat0, lon0 = 19 + 49 / 60 + 32 / 3600, -(155 + 28 / 60 + 34 / 3600)
    coords = {"Keck I": (19.8259465, -155.474719), "Keck II": (19.8265606, -155.474234)}
    for st in maunakea_tri.stations[1:]:
        lat, lon = coords[st.name]
        east = np.radians(lon - lon0) * R * np.cos(np.radians(lat0))
        north = np.radians(lat - lat0) * R
        assert st.east_m == pytest.approx(east, abs=3.0)
        assert st.north_m == pytest.approx(north, abs=3.0)


def test_rgb_composite_matches_batched_frames(algol):
    cfg = replace(MovieConfig(), n_frames=4)
    grid = GridConfig()
    fd = precompute_frames(algol, grid, cfg, verbose=False)
    for k in (0, 2):
        ref = render_display_rgb(positions_at(algol, fd.phase[k]), algol, grid)
        assert np.allclose(fd.disp_imgs[k], ref, atol=1e-5)
    rgb = stretch_rgb(fd.disp_imgs)
    assert rgb.shape == fd.disp_imgs.shape and rgb.max() <= 1.0 and rgb.min() >= 0.0
    # the hot primary is bluer than the cool secondary in the composite
    assert len(RGB_DISPLAY_NM) == 3


def test_movie_aperture_option(beta_aur):
    cfg = replace(MovieConfig(), n_frames=3, aperture_m=10.0)
    fd = precompute_frames(beta_aur, GridConfig(), cfg, verbose=False)
    fd0 = precompute_frames(beta_aur, GridConfig(), replace(cfg, aperture_m=None),
                            verbose=False)
    assert fd.g2_fine.shape == fd0.g2_fine.shape
    assert not np.allclose(fd.g2_fine, fd0.g2_fine, atol=1e-3)
    assert np.allclose(fd0.g2_fine[:, :, 0], 2.0, atol=1e-5)   # point sampling: g2(0) = 2
    assert np.all(fd.g2_fine[:, :, 0] < 2.0)   # pupil-averaged around B = 0: below 2


def test_bin_channels():
    nm = np.linspace(400.0, 500.0, 11)
    noisy = np.tile(np.arange(11.0), (2, 1))
    sigma = np.ones((2, 11))
    sigma[:, 1] = 1e-3   # one very precise channel dominates its bin
    c, y, e = _bin_channels(noisy, sigma, nm, 5)
    assert c.shape == (2,) and y.shape == (2, 2)
    assert y[0, 0] == pytest.approx(1.0, abs=1e-4)
    assert e[0, 0] == pytest.approx(1e-3, rel=1e-3)
    assert y[0, 1] == pytest.approx(7.0)


def test_g2spec_cadence(delta_vel, c2pu, spad_lambda):
    from hbtsim.g2spec import precompute
    from hbtsim.snr import Spectrograph
    d = precompute(system=delta_vel, spectrograph=Spectrograph(n_channels=4),
                   telescope=c2pu, detector=spad_lambda,
                   cadence_hours=240.0, verbose=False)
    assert d["hours"].size == 5            # 1084 h period / 240 h
    assert d["vis2"].shape == (5, 4)
