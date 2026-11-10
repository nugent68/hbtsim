"""A readout ceiling attenuates only the telescope whose detector needs it:
the pair S/N with the ceiling equals the S/N with that telescope's
throughput scaled by its own factor and the partner untouched (the
earlier shared attenuation dimmed both by the stricter factor)."""

from dataclasses import replace

import numpy as np
import pytest

from hbtsim.snr import Observation, Telescope, incident_rate, spectral_g2_snr, system_ab_mag


def _pair():
    return Telescope(diameter_m=1.0, throughput=0.3), Telescope(diameter_m=10.0, throughput=0.3)


def test_spectral_readout_attenuates_only_the_saturated_telescope(beta_aur, spec320, spad_lambda):
    small, big = _pair()
    det = replace(spad_lambda, readout="timetag", max_total_cps=None, max_cps_per_pixel=None)
    nm = spec320.channel_centers_nm
    obs = Observation(wavelength_nm=nm, filter_width_nm=spec320.channel_widths_nm,
                      backend_throughput=spec320.throughput)
    mags = system_ab_mag(beta_aur, nm)
    inc_small = float(np.sum(incident_rate(mags, small, det, obs)))
    inc_big = float(np.sum(incident_rate(mags, big, det, obs)))
    ceiling = 0.5 * inc_big
    assert inc_small < ceiling < inc_big
    kw = dict(spectrograph=spec320, telescope1=small, telescope2=big, vis2_method="analytic")
    lim = spectral_g2_snr(beta_aur, 50.0, detector1=replace(det, max_total_cps=ceiling), **kw)
    assert lim.readout_limited
    s = lim.readout_scale
    assert 0 < s < 1
    ref = spectral_g2_snr(beta_aur, 50.0, detector1=det, enforce_readout=False,
                          **{**kw, "telescope2": replace(big, throughput=big.throughput * s)})
    assert lim.snr_total == pytest.approx(ref.snr_total, rel=1e-6)
    # the earlier shared attenuation dimmed the small telescope too (about a
    # factor s lower in S/N, less where dark counts and dead time intervene)
    shared = spectral_g2_snr(beta_aur, 50.0, detector1=det, enforce_readout=False,
                             **{**kw, "telescope1": replace(small, throughput=small.throughput * s),
                                "telescope2": replace(big, throughput=big.throughput * s)})
    assert s * ref.snr_total < shared.snr_total < 0.95 * ref.snr_total


def test_nightmovie_budget_attenuates_only_the_saturated_telescope(vega, spad_lambda):
    from hbtsim.runners.nightmovie import _budget
    small, big = _pair()
    det = replace(spad_lambda, readout="timetag", max_total_cps=None, max_cps_per_pixel=None)
    free = _budget(vega, 650.0, 1.0, 1800.0, small, big, det, "unpolarized", 0.9)
    r_small, r_big = free["r1"], free["r2"]
    ceiling = 0.5 * r_big
    assert r_small < ceiling
    lim = _budget(vega, 650.0, 1.0, 1800.0, small, big, replace(det, max_total_cps=ceiling), "unpolarized", 0.9)
    s = lim["readout_scale"]
    assert 0 < s < 1
    ref = _budget(vega, 650.0, 1.0, 1800.0, small, replace(big, throughput=big.throughput * s), det,
                  "unpolarized", 0.9)
    assert lim["sigma_v2"] == pytest.approx(ref["sigma_v2"], rel=1e-6)
