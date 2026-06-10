"""Sanity checks of the numerical pipeline against analytic results.

Run with:  python -m pytest tests/ -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from hbtsim import hbt
from hbtsim.limbdark import disk_flux_factor, visibility_ld_disk
from hbtsim.orbit import SkyPositions, sky_positions
from hbtsim.params import BETA_AUR, MAS, GridConfig, MovieConfig
from hbtsim.render import _render_kernel, render_image

SYSTEM = BETA_AUR
GRID = GridConfig()
THETA1_MAS = 2 * SYSTEM.angular_radius_mas(SYSTEM.primary)   # angular diameter


def single_star_image(radius_mas: float, u: float) -> jnp.ndarray:
    """One star at the grid center, the other switched off (w2 = 0)."""
    r_px = radius_mas / GRID.pixel_scale_mas
    return _render_kernel(0.0, 0.0, 300.0, 300.0, False,
                          r_px, 1.0, 1.0, 0.0, u, GRID.n)


# ---------------------------------------------------------------------------
# Orbit geometry
# ---------------------------------------------------------------------------
def test_max_separation_at_phase_zero():
    psi = np.linspace(0, 2 * np.pi, 1441)
    pos = sky_positions(psi, SYSTEM)
    assert pos.rho[0] == pytest.approx(SYSTEM.angular_semimajor_mas, rel=1e-12)
    assert pos.rho[0] >= pos.rho.max() - 1e-12
    # minimum projected separation a*cos(i) at conjunction
    rho_min = SYSTEM.angular_semimajor_mas * np.cos(np.radians(SYSTEM.inclination_deg))
    assert pos.rho.min() == pytest.approx(rho_min, rel=1e-6)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def test_single_star_flux_matches_analytic():
    """Disk-integrated flux = pi R^2 (1 - u/3) for the linear LD law."""
    for u in (0.0, 0.3, 0.52):
        img = single_star_image(0.5, u)
        r_px = 0.5 / GRID.pixel_scale_mas
        expected = np.pi * r_px**2 * disk_flux_factor(u)
        assert float(img.sum()) == pytest.approx(expected, rel=2e-3)


# ---------------------------------------------------------------------------
# g2 / visibility
# ---------------------------------------------------------------------------
def test_g2_zero_baseline_is_two():
    for psi in (0.0, np.pi / 2, 1.0):
        pos = SkyPositions(*(np.asarray(v) for v in sky_positions(psi, SYSTEM)))
        for lam_nm in (400.0, 800.0):
            img = render_image(pos, SYSTEM, lam_nm, GRID)
            v2 = hbt.vis2_map(img, GRID.pad)
            g2 = hbt.g2_of_baseline(v2, np.array([0.0]), lam_nm * 1e-9,
                                    float(pos.pa), GRID)
            assert float(g2[0]) == pytest.approx(2.0, abs=1e-6)


def test_uniform_disk_matches_airy():
    """u = 0: |V|^2 = |2 J1(x)/x|^2, first null at B = 1.22 lambda / theta."""
    from scipy.special import j1

    lam = 400e-9
    img = single_star_image(THETA1_MAS / 2, 0.0)
    v2map = hbt.vis2_map(img, GRID.pad)
    B = np.linspace(5.0, 150.0, 200)
    v2 = np.asarray(hbt.vis2_of_baseline(v2map, B, lam, 0.0, GRID))
    x = np.pi * THETA1_MAS * MAS * B / lam
    airy = (2 * j1(x) / x) ** 2
    assert np.allclose(v2, airy, atol=2e-3)

    b_null_expected = 1.22 * lam / (THETA1_MAS * MAS)  # ~97.7 m
    fine = np.linspace(80, 115, 701)
    v2f = np.asarray(hbt.vis2_of_baseline(v2map, fine, lam, 0.0, GRID))
    assert fine[np.argmin(v2f)] == pytest.approx(b_null_expected, rel=0.01)


def test_limb_darkened_disk_matches_analytic():
    """Linear LD law against the analytic Bessel-series visibility."""
    lam = 400e-9
    u = SYSTEM.ld_coeff(400.0)
    img = single_star_image(THETA1_MAS / 2, u)
    v2map = hbt.vis2_map(img, GRID.pad)
    B = np.linspace(5.0, 150.0, 200)
    v2 = np.asarray(hbt.vis2_of_baseline(v2map, B, lam, 0.0, GRID))
    x = np.pi * THETA1_MAS * MAS * B / lam
    v_ana = visibility_ld_disk(x, u)
    assert np.allclose(v2, v_ana**2, atol=2e-3)


def test_binary_vis2_matches_analytic_at_quadrature():
    """Numerical |V|^2 along the separation axis matches the analytic binary
    formula (Rai, Basak & Saha 2021, eq. 5; hbt.binary_vis2_analytic),
    whose fringe period is lambda/rho: 25.0 m @400 nm, 50.0 m @800 nm."""
    pos = SkyPositions(*(np.asarray(v) for v in sky_positions(0.0, SYSTEM)))

    for lam_nm in (400.0, 800.0):
        lam = lam_nm * 1e-9
        img = render_image(pos, SYSTEM, lam_nm, GRID)
        v2map = hbt.vis2_map(img, GRID.pad)
        B = np.arange(0.0, 150.0, 0.5)
        v2 = np.asarray(hbt.vis2_of_baseline(v2map, B, lam, float(pos.pa), GRID))
        v2_ana = hbt.binary_vis2_analytic(B, lam_nm, SYSTEM, float(pos.rho))
        # agreement to better than 1% of the zero-baseline amplitude checks
        # both the fringe period (lambda/rho) and the disk envelopes
        assert np.allclose(v2, v2_ana, atol=5e-3)


def test_interpolation_matches_direct_dft():
    """FFT + bilinear interpolation vs direct DFT at the 15 requested baselines."""
    cfg = MovieConfig()
    pos = SkyPositions(*(np.asarray(v) for v in sky_positions(0.3, SYSTEM)))
    lam = 400e-9
    img = np.asarray(render_image(pos, SYSTEM, 400.0, GRID), dtype=float)
    v2map = hbt.vis2_map(jnp.asarray(img), GRID.pad)
    b = np.asarray(cfg.baselines_m)
    v2_interp = np.asarray(hbt.vis2_of_baseline(v2map, b, lam, float(pos.pa), GRID))
    v2_exact = hbt.vis2_direct(img, b, lam, float(pos.pa), GRID)
    # bilinear interpolation on the ~1 m FFT grid is good to a few 1e-3
    assert np.allclose(v2_interp, v2_exact, atol=5e-3)


# ---------------------------------------------------------------------------
# Lightcurve
# ---------------------------------------------------------------------------
def _fluxes(psi_arr, lam_nm):
    out = np.empty(len(psi_arr))
    pos_all = sky_positions(np.asarray(psi_arr), SYSTEM)
    for k in range(len(psi_arr)):
        pos = SkyPositions(*(np.asarray(v)[k] for v in pos_all))
        out[k] = float(render_image(pos, SYSTEM, lam_nm, GRID).sum())
    return out


def test_lightcurve_flat_out_of_eclipse_and_depths():
    psi = 2 * np.pi * np.linspace(0, 1, 120, endpoint=False)
    flux = _fluxes(psi, 477.0)
    mag = -2.5 * np.log10(flux / flux[0])

    # flat away from the eclipses (|sin psi| small <-> near quadrature)
    quad = np.abs(np.sin(psi)) < 0.5
    assert np.std(mag[quad]) < 1e-3

    # eclipses at phase 0.25 and 0.75
    k1 = np.argmin(np.abs(psi - np.pi / 2))
    k2 = np.argmin(np.abs(psi - 3 * np.pi / 2))
    depth1, depth2 = mag[k1], mag[k2]
    assert 0.06 < depth1 < 0.11   # published primary depth ~0.09 mag
    assert 0.0 < depth2 <= depth1  # secondary minimum is shallower


def test_apparent_magnitudes_reasonable():
    """Synthetic AB magnitudes near the observed system brightness (V ~ 1.9;
    blackbody approximation allows a few tenths of a magnitude offset), and
    anchoring pins maximum light to the observed band magnitudes while
    preserving the eclipse depth."""
    from hbtsim.photometry import anchored_mags, apparent_ab_mag

    anchors = dict(SYSTEM.mag_anchors)
    for band, lam_nm in (("g", 477.0), ("i", 763.0)):
        synth = np.empty(2)
        for k, psi in enumerate((0.0, np.pi / 2)):  # max light, mid-eclipse
            pos = SkyPositions(*(np.asarray(v) for v in sky_positions(psi, SYSTEM)))
            flux = float(render_image(pos, SYSTEM, lam_nm, GRID).sum())
            synth[k] = float(apparent_ab_mag(flux, lam_nm, SYSTEM, GRID))
        # blackbody zero point is within ~0.6 mag of the observed anchor
        assert abs(synth[0] - anchors[band]) < 0.6
        anchored = anchored_mags(synth, band, SYSTEM)
        assert anchored[0] == pytest.approx(anchors[band], abs=1e-12)
        # anchoring is a constant shift: eclipse depth unchanged
        assert (anchored[1] - anchored[0]) == pytest.approx(synth[1] - synth[0],
                                                            abs=1e-9)


def test_uniform_disk_eclipse_depth_matches_circle_overlap():
    """With u = 0 and equal surface brightness, the blocked flux is exactly
    the lens-shaped overlap area of the two disks."""
    r1 = SYSTEM.angular_radius_mas(SYSTEM.primary) / GRID.pixel_scale_mas
    r2 = SYSTEM.angular_radius_mas(SYSTEM.secondary) / GRID.pixel_scale_mas
    pos = sky_positions(np.pi / 2, SYSTEM)  # mid-eclipse
    d = float(pos.rho) / GRID.pixel_scale_mas

    img_ecl = _render_kernel(float(pos.x1) / GRID.pixel_scale_mas,
                             float(pos.y1) / GRID.pixel_scale_mas,
                             float(pos.x2) / GRID.pixel_scale_mas,
                             float(pos.y2) / GRID.pixel_scale_mas,
                             bool(pos.front2), r1, r2, 1.0, 1.0, 0.0, GRID.n)
    full = np.pi * (r1**2 + r2**2)

    # analytic lens area
    a1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
    a2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
    a3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    overlap = a1 + a2 - a3

    blocked = full - float(img_ecl.sum())
    assert blocked == pytest.approx(overlap, rel=2e-3)
