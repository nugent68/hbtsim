"""Algol (beta Per A-B) and per-star limb-darkening tests.

Algol pairs a hot B8V dwarf (12550 K, weak limb darkening) with a cool
K0IV subgiant (4900 K, strong limb darkening), so it exercises the
per-star LD path that Beta Aurigae's near-twins cannot.
"""

import jax
import numpy as np
import pytest

from hbtsim import hbt
from hbtsim.limbdark import disk_flux_factor, visibility_ld_disk
from hbtsim.orbit import SkyPositions, sky_positions
from hbtsim.params import ALGOL, MAS, GridConfig
from hbtsim.render import linear_rows_jnp, render_image, render_kernel
from hbtsim.snr import system_ab_mag

GRID = GridConfig()


def _pos(psi):
    return SkyPositions(*(np.asarray(v) for v in sky_positions(psi, ALGOL)))


# ---------------------------------------------------------------------------
# Per-star limb darkening
# ---------------------------------------------------------------------------
def test_per_star_ld_applied():
    """Two well-separated equal disks with u1 = 0.2, u2 = 0.8: each disk's
    flux must carry its OWN (1 - u/3) factor."""
    r = 40.0
    img = np.asarray(render_kernel(-200.0, 0.0, 200.0, 0.0, False,
                                   r, r, 1.0, 1.0, linear_rows_jnp(0.2, GRID.n_mu),
                                   linear_rows_jnp(0.8, GRID.n_mu), GRID.n))
    half = GRID.n // 2
    flux1 = img[:, :half].sum()
    flux2 = img[:, half:].sum()
    assert flux1 == pytest.approx(np.pi * r**2 * disk_flux_factor(0.2), rel=2e-3)
    assert flux2 == pytest.approx(np.pi * r**2 * disk_flux_factor(0.8), rel=2e-3)


def test_secondary_visibility_uses_its_own_u():
    """A lone Algol-B-like disk (primary switched off, w1 = 0) must match
    the analytic LD visibility with u_B, and NOT with u_A."""
    lam_nm = 400.0
    u_a = ALGOL.primary.ld_coeff(lam_nm)
    u_b = ALGOL.secondary.ld_coeff(lam_nm)
    th_b = 2 * ALGOL.angular_radius_mas(ALGOL.secondary) * MAS
    r_px = ALGOL.angular_radius_mas(ALGOL.secondary) / GRID.pixel_scale_mas

    img = render_kernel(300.0, 300.0, 0.0, 0.0, True,
                        1.0, r_px, 0.0, 1.0, linear_rows_jnp(u_a, GRID.n_mu),
                        linear_rows_jnp(u_b, GRID.n_mu), GRID.n)
    B = np.linspace(5.0, 150.0, 100)
    v2 = np.asarray(hbt.vis2_along_pa(img, B, lam_nm * 1e-9, 0.0, GRID))
    x = np.pi * th_b * B / (lam_nm * 1e-9)
    assert np.allclose(v2, visibility_ld_disk(x, u_b) ** 2, atol=1e-3)
    assert not np.allclose(v2, visibility_ld_disk(x, u_a) ** 2, atol=1e-3)


# ---------------------------------------------------------------------------
# Geometry (vs Baron et al. 2012 measured values)
# ---------------------------------------------------------------------------
def test_algol_geometry():
    assert ALGOL.angular_semimajor_mas == pytest.approx(2.151, abs=0.02)
    assert 2 * ALGOL.angular_radius_mas(ALGOL.primary) == pytest.approx(0.881, abs=0.01)
    assert 2 * ALGOL.angular_radius_mas(ALGOL.secondary) == pytest.approx(1.123, abs=0.01)

    psi = np.linspace(0, 2 * np.pi, 1441)
    pos = sky_positions(psi, ALGOL)
    rho_min = ALGOL.angular_semimajor_mas * abs(np.cos(np.radians(ALGOL.inclination_deg)))
    assert pos.rho.min() == pytest.approx(rho_min, rel=1e-3)
    assert rho_min == pytest.approx(0.325, abs=0.01)
    # the cool secondary transits the hot primary at psi = pi/2
    assert bool(sky_positions(np.pi / 2, ALGOL).front2)


# ---------------------------------------------------------------------------
# Eclipse depths (headline validation)
# ---------------------------------------------------------------------------
def _depths(lam_nm):
    flux = {}
    for label, psi in (("max", 0.0), ("primary", np.pi / 2),
                       ("secondary", 3 * np.pi / 2)):
        flux[label] = float(render_image(_pos(psi), ALGOL, lam_nm, GRID).sum())
    d1 = -2.5 * np.log10(flux["primary"] / flux["max"])
    d2 = -2.5 * np.log10(flux["secondary"] / flux["max"])
    return d1, d2


def test_algol_eclipse_depths():
    """Published V-band primary depth is 1.27 mag for the unresolved triple;
    removing Algol C's ~10% third light gives ~1.5 mag for A-B alone, and
    g (bluer) is slightly deeper.  The spherical blackbody model (no
    ellipsoidal variation or reflection) should land in a generous band
    around that.  The secondary eclipse (A occults the cool B) is shallow
    and much deeper in i than in g, since B contributes ~10x more light in
    the red."""
    d1_g, d2_g = _depths(477.0)
    d1_i, d2_i = _depths(763.0)
    assert 1.3 < d1_g < 1.9
    assert 0.9 < d1_i < 1.5
    assert d1_g > d1_i                  # bluer = deeper primary
    assert 0.003 < d2_g < 0.1
    assert d2_i > d2_g                  # secondary eclipse is red
    # flat out of eclipse (spherical model: no ellipsoidal variation);
    # Algol's eclipses are wide -- first contact (rho < r_A + r_B) is at
    # phase ~0.17, so sample phases 0.03-0.16
    psi = 2 * np.pi * np.linspace(0.03, 0.16, 8)
    f = [float(render_image(_pos(p), ALGOL, 477.0, GRID).sum()) for p in psi]
    assert np.std(-2.5 * np.log10(np.asarray(f) / f[0])) < 1e-3


# ---------------------------------------------------------------------------
# FFT vs analytic with per-star LD
# ---------------------------------------------------------------------------
def test_algol_vis2_matches_analytic_at_quadrature():
    pos = _pos(0.0)
    for lam_nm in (400.0, 800.0):
        img = render_image(pos, ALGOL, lam_nm, GRID)
        B = np.arange(0.0, 150.0, 0.5)
        v2 = np.asarray(hbt.vis2_along_pa(img, B, lam_nm * 1e-9,
                                          float(pos.pa), GRID))
        ana = hbt.binary_vis2_analytic(B, lam_nm, ALGOL, float(pos.rho))
        assert np.allclose(v2, ana, atol=1e-3)


def test_algol_anchors():
    anchors = dict(ALGOL.mag_anchors)
    assert system_ab_mag(ALGOL, 477.0) == pytest.approx(anchors["g"], abs=1e-9)
    assert system_ab_mag(ALGOL, 763.0) == pytest.approx(anchors["i"], abs=1e-9)
