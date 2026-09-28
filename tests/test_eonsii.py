"""EON-SII presets (arXiv:2608.17444) and the pair geometry."""

import numpy as np
import pytest

from hbtsim.bispectrum import eonsii_pair
from hbtsim.geometry import TEIDE
from hbtsim.limbdark import visibility_ld_disk
from hbtsim.params import MAS
from hbtsim.snr import (EON_SII_TELESCOPE, EONSII_MCP_PMT, EONSII_SPAD, EONSII_SPECTROGRAPH,
                        EONSII_SPECTROGRAPH_R7500, Observation, g2_snr, incident_rate,
                        pair_sigma_s, spectral_g2_snr)


def test_eonsii_presets():
    assert EON_SII_TELESCOPE.area_m2 == pytest.approx(9.0)
    assert EON_SII_TELESCOPE.diameter_m == 4.0            # the smearing pupil
    obs = Observation(wavelength_nm=475.0, filter_width_nm=0.15, coherence_broadening=False)
    assert pair_sigma_s(EONSII_MCP_PMT, EONSII_MCP_PMT, obs) == pytest.approx(27.4e-12, rel=1e-3)
    assert EONSII_SPAD.jitter_fwhm_ps == 20.0
    assert EONSII_SPECTROGRAPH.n_channels == 1000
    assert EONSII_SPECTROGRAPH.channel_width_nm == pytest.approx(0.15)
    assert EONSII_SPECTROGRAPH.throughput == 0.6
    assert 2300 < EONSII_SPECTROGRAPH_R7500.n_channels < 2500
    assert EONSII_SPECTROGRAPH_R7500.resolving_power == 7500.0


def test_eonsii_pair_geometry():
    arr = eonsii_pair(120.0, pa_deg=90.0)
    assert arr.site is TEIDE and len(arr.stations) == 2
    (i, j, b), = arr.pairs()
    assert np.hypot(*b) == pytest.approx(120.0) and b[0] == pytest.approx(120.0)
    assert arr.triangles() == []
    assert all(s.telescope is EON_SII_TELESCOPE for s in arr.stations)
    from dataclasses import replace
    three = eonsii_pair(100.0, third=replace(arr.stations[1], name="T3", east_m=50.0, north_m=80.0))
    assert len(three.triangles()) == 1


def test_sirius_b_rates_match_the_paper():
    """Sirius B (V = 8.44): the paper quotes ~5-10 kHz per spectral channel
    and up to ~7 MHz per telescope on its MCP-PMT case."""
    obs = Observation(wavelength_nm=EONSII_SPECTROGRAPH.channel_centers_nm,
                      filter_width_nm=EONSII_SPECTROGRAPH.channel_widths_nm,
                      backend_throughput=EONSII_SPECTROGRAPH.throughput)
    rate = incident_rate(8.44, EON_SII_TELESCOPE, EONSII_MCP_PMT, obs)
    assert 3e3 < np.median(rate) < 1.5e4
    assert 2e6 < rate.sum() < 1.5e7


def test_eonsii_g2_on_a_binary_runs():
    from hbtsim.params import BETA_AUR
    r = spectral_g2_snr(BETA_AUR, 60.0, spectrograph=EONSII_SPECTROGRAPH,
                        telescope1=EON_SII_TELESCOPE, detector1=EONSII_SPAD,
                        vis2_method="analytic")
    assert r.snr_total > 0 and r.smeared and r.channel_nm.size == 1000
    # a V ~ 1.9 star on 9 m^2 x 1000 channels exceeds even the 1 GHz link:
    # the rates are scaled to it
    assert r.readout_limited and r.total_rate_cps[0] <= 1.01e9
