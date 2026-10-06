"""Chromatic diameters (hbtsim.chromatic)."""

import os

import numpy as np
import pytest

from hbtsim.catalog import Catalog
from hbtsim.chromatic import (BALMER_VAC_NM, asimov_significance, chromatic_signature,
                              continuum_fit, line_masks, select_channels_for_link)
from hbtsim.single import attach_newera_single
from hbtsim.snr import Spectrograph

CAT = Catalog(env=False)
NM = CAT.load_spectrograph("eonsii_1000ch").channel_centers_nm
has_newera = pytest.mark.skipif(not os.path.isdir("data/newera"), reason="NewEra tables not present")


def test_line_masks_disjoint_and_complete():
    m = line_masks(NM)
    assert set(BALMER_VAC_NM) <= set(m)
    parts = [m["continuum"]] + [m[k][p] for k in BALMER_VAC_NM for p in ("core", "wing")]
    assert np.all(np.sum(parts, axis=0) == 1)
    for k, lam in BALMER_VAC_NM.items():
        assert m[k]["core"].any() and np.all(np.abs(NM[m[k]["core"]] - lam) < 0.12)


def test_continuum_fit_removes_smooth_slopes():
    sig = np.full(NM.size, 1e-3)
    y = 3.0 + 0.02 * (NM / 475 - 1) - 0.05 * (NM / 475 - 1) ** 2
    assert asimov_significance(NM, y, sig, deg=2) < 1e-6
    assert np.allclose(continuum_fit(NM, y, sig, np.ones(NM.size, bool), 2), y)
    bump = y + 0.01 * (np.abs(NM - 486.27) < 0.12)
    assert asimov_significance(NM, bump, sig, deg=2) > 5


def test_select_channels_fits_link_and_matches_brute_force():
    rng = np.random.default_rng(0)
    n = 12
    rate = rng.uniform(1, 3, n)
    info = np.where(np.arange(n) < 4, 0.0, rng.uniform(0, 5, n))
    cont = np.arange(n) < 4
    sigma = rng.uniform(1, 2, n)
    link = 9.0
    tag = select_channels_for_link(rate, info, link, cont, sigma, ref_frac=0.25, min_ref=1)
    assert rate[tag].sum() <= link + 1e-12 and (tag & cont).sum() >= 1
    # the greedy fill of the remaining budget is within the knapsack optimum's reach
    free = link - rate[tag & cont].sum()
    sig_idx = np.flatnonzero(~cont)
    best = 0.0
    for bits in range(1 << sig_idx.size):
        sel = [sig_idx[i] for i in range(sig_idx.size) if bits >> i & 1]
        if rate[sel].sum() <= free:
            best = max(best, info[sel].sum())
    assert info[tag & ~cont].sum() >= 0.7 * best


def test_significance_scales_as_sqrt_time(vega, eonsii_tel, eonsii_spad_correlator, teide):
    spec = Spectrograph(lambda_min_nm=480.0, lambda_max_nm=492.0, n_channels=80, throughput=0.6)
    # the former readout="correlator": every channel on a correlator detector
    kw = dict(telescope=eonsii_tel, detector=eonsii_spad_correlator, site=teide,
              channel_selection="all", deg=1)
    a = chromatic_signature(vega, 18.0, spec, t_int_s=3600.0, **kw)
    b = chromatic_signature(vega, 18.0, spec, t_int_s=4 * 3600.0, **kw)
    assert b.significance == pytest.approx(2.0 * a.significance, rel=1e-9)
    with pytest.raises(ValueError):
        chromatic_signature(vega, 18.0, spec, **{**kw, "channel_selection": "bogus"})


@has_newera
def test_sirius_subset_beats_link_and_signal_sign(sirius_a, eonsii_tel, eonsii_spad, eonsii_spec,
                                                  teide):
    s, _ = attach_newera_single(sirius_a, "data/newera")
    # the time-tag SPAD: "all" channels share the link (the former readout="link")
    kw = dict(telescope=eonsii_tel, detector=eonsii_spad, site=teide)
    link = chromatic_signature(s, 10.0, eonsii_spec, channel_selection="all", **kw)
    sub = chromatic_signature(s, 10.0, eonsii_spec, channel_selection="subset", **kw)
    assert link.readout_scale < 0.05 and sub.readout_scale > 0.9
    assert sub.significance > 3 * link.significance
    for k in BALMER_VAC_NM:
        assert 2.5 < link.line_signal_pct[k] < 6.0     # cores look 2.5-6 % larger


def test_reference_channels_shared_between_line_windows():
    n = 10
    rate = np.ones(n)
    cont = np.ones(n, bool)
    sigma = np.arange(n, dtype=float)          # group A (0-4) holds all the best channels
    groups = [np.arange(n) < 5, np.arange(n) >= 5]
    tag = select_channels_for_link(rate, np.zeros(n), 4.0, cont, sigma, ref_frac=1.0,
                                   min_ref=4, ref_groups=groups)
    assert tag.sum() == 4 and tag[:5].sum() == 2 and tag[5:].sum() == 2
