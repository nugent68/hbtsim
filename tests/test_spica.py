"""Spica (alpha Vir) sanity checks: geometry vs the Narrabri
intensity-interferometer measurement (Herbison-Evans et al. 1971) and
the closure-phase feasibility ordering the system was added for."""

import numpy as np
import pytest

from hbtsim.bispectrum import MAUNAKEA_SUBARU_KECK
from hbtsim.orbit import sky_positions
from hbtsim.params import ALGOL, SPICA
from hbtsim.snr import Spectrograph
from hbtsim.snr3 import spectral_g3_snr


def test_spica_geometry():
    # Narrabri measured theta_A = 0.90 +/- 0.04 mas [HE71]
    assert 2 * SPICA.angular_radius_mas(SPICA.primary) == pytest.approx(
        0.91, abs=0.04)
    # angular semi-major axis ~ 1.7 mas at the Hipparcos distance
    assert SPICA.angular_semimajor_mas == pytest.approx(1.71, abs=0.03)
    # non-eclipsing at i = 63.1 deg: minimum projected separation exceeds
    # the sum of the angular radii
    psi = np.linspace(0, 2 * np.pi, 721)
    rho_min = sky_positions(psi, SPICA).rho.min()
    sum_r = (SPICA.angular_radius_mas(SPICA.primary)
             + SPICA.angular_radius_mas(SPICA.secondary))
    assert rho_min > sum_r


def test_spica_anchors_and_color():
    from hbtsim.snr import system_ab_mag
    anchors = dict(SPICA.mag_anchors)
    assert system_ab_mag(SPICA, 477.0) == pytest.approx(anchors["g"], abs=1e-9)
    # hot photosphere: brighter in g than in i
    assert anchors["i"] > anchors["g"]


def test_spica_is_the_better_g3_target():
    """The reason Spica is in the registry: in the narrow-channel
    (unsaturated) regime where brightness enters as R^{3/2}, its
    bispectrum sensitivity on the Maunakea triangle beats Algol's by
    >5x in SNR (>25x in time).  At wide channels both targets are
    dead-time saturated and the gap shrinks -- so test the regime the
    recommendation is actually about."""
    spec = Spectrograph.from_resolving_power(5000.0)
    kw = dict(spectrograph=spec, enforce_readout=False)
    s = spectral_g3_snr(SPICA, MAUNAKEA_SUBARU_KECK, **kw)
    a = spectral_g3_snr(ALGOL, MAUNAKEA_SUBARU_KECK, **kw)
    assert s.snr_total > 5.0 * a.snr_total
