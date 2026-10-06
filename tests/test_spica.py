"""Spica (alpha Vir) sanity checks: geometry vs the Narrabri
intensity-interferometer measurement (Herbison-Evans et al. 1971) and
the closure-phase feasibility ordering the system was added for."""

import numpy as np
import pytest

from hbtsim.orbit import sky_positions
from hbtsim.snr3 import spectral_g3_snr


def test_spica_geometry(spica):
    # Narrabri measured theta_A = 0.90 +/- 0.04 mas [HE71]
    assert 2 * spica.angular_radius_mas(spica.primary) == pytest.approx(
        0.91, abs=0.04)
    # angular semi-major axis ~ 1.7 mas at the Hipparcos distance
    assert spica.angular_semimajor_mas == pytest.approx(1.71, abs=0.03)
    # non-eclipsing at i = 63.1 deg: minimum projected separation exceeds
    # the sum of the angular radii
    psi = np.linspace(0, 2 * np.pi, 721)
    rho_min = sky_positions(psi, spica).rho.min()
    sum_r = (spica.angular_radius_mas(spica.primary)
             + spica.angular_radius_mas(spica.secondary))
    assert rho_min > sum_r


def test_spica_anchors_and_color(spica):
    from hbtsim.snr import system_ab_mag
    anchors = dict(spica.mag_anchors)
    assert system_ab_mag(spica, 477.0) == pytest.approx(anchors[477.0], abs=1e-9)
    # hot photosphere: brighter in g than in i
    assert anchors[763.0] > anchors[477.0]


def test_spica_is_the_better_g3_target(spica, algol, maunakea_tri, spec_r5000):
    """The reason Spica is in the registry: in the narrow-channel
    (unsaturated) regime where brightness enters as R^{3/2}, its
    bispectrum sensitivity on the Maunakea triangle beats Algol's by
    >5x in SNR (>25x in time).  At wide channels both targets are
    dead-time saturated and the gap shrinks -- so test the regime the
    recommendation is actually about."""
    kw = dict(spectrograph=spec_r5000, enforce_readout=False)
    s = spectral_g3_snr(spica, maunakea_tri, **kw)
    a = spectral_g3_snr(algol, maunakea_tri, **kw)
    assert s.snr_total > 5.0 * a.snr_total
