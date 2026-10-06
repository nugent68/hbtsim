"""EON-SII presets (arXiv:2608.17444) and the pair geometry."""

from dataclasses import replace

import numpy as np
import pytest

from hbtsim.bispectrum import Array
from hbtsim.snr import Observation, incident_rate, pair_sigma_s, spectral_g2_snr


def test_eonsii_presets(eonsii_tel, eonsii_mcp, eonsii_spad, eonsii_spec, eonsii_spec_r7500):
    assert eonsii_tel.area_m2 == pytest.approx(9.0)
    assert eonsii_tel.diameter_m == 4.0            # the smearing pupil
    obs = Observation(wavelength_nm=475.0, filter_width_nm=0.15, coherence_broadening=False)
    assert pair_sigma_s(eonsii_mcp, eonsii_mcp, obs) == pytest.approx(27.4e-12, rel=1e-3)
    assert eonsii_spad.jitter_fwhm_ps == 20.0
    assert eonsii_spec.n_channels == 1000
    assert eonsii_spec.channel_width_nm == pytest.approx(0.15)
    assert eonsii_spec.throughput == 0.6
    assert 2300 < eonsii_spec_r7500.n_channels < 2500
    assert eonsii_spec_r7500.resolving_power == 7500.0


def test_eonsii_pair_geometry(catalog, teide, eonsii_tel):
    arr = catalog.load_array("eonsii_pair_teide", baseline_m=120.0, pa_deg=90.0)
    assert arr.site == teide and len(arr.stations) == 2
    (i, j, b), = arr.pairs()
    assert np.hypot(*b) == pytest.approx(120.0) and b[0] == pytest.approx(120.0)
    assert arr.triangles() == []
    assert all(s.telescope == eonsii_tel for s in arr.stations)
    pair = catalog.load_array("eonsii_pair_teide", baseline_m=100.0)
    third = replace(arr.stations[1], name="T3", east_m=50.0, north_m=80.0)
    three = Array(pair.stations + (third,), pair.site)
    assert len(three.triangles()) == 1


def test_sirius_b_rates_match_the_paper(eonsii_tel, eonsii_mcp, eonsii_spec):
    """Sirius B (V = 8.44): the paper quotes ~5-10 kHz per spectral channel
    and up to ~7 MHz per telescope on its MCP-PMT case."""
    obs = Observation(wavelength_nm=eonsii_spec.channel_centers_nm,
                      filter_width_nm=eonsii_spec.channel_widths_nm,
                      backend_throughput=eonsii_spec.throughput)
    rate = incident_rate(8.44, eonsii_tel, eonsii_mcp, obs)
    assert 3e3 < np.median(rate) < 1.5e4
    assert 2e6 < rate.sum() < 1.5e7


def test_eonsii_g2_on_a_binary_runs(beta_aur, eonsii_spec, eonsii_tel, eonsii_spad):
    r = spectral_g2_snr(beta_aur, 60.0, spectrograph=eonsii_spec,
                        telescope1=eonsii_tel, detector1=eonsii_spad,
                        vis2_method="analytic")
    assert r.snr_total > 0 and r.smeared and r.channel_nm.size == 1000
    # a V ~ 1.9 star on 9 m^2 x 1000 channels exceeds even the 1 GHz link:
    # the rates are scaled to it
    assert r.readout_limited and r.total_rate_cps[0] <= 1.01e9
