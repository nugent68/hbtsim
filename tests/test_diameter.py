"""Angular-scale precision (hbtsim.diameter) and the Kim & Kaiser noise formula."""

import numpy as np
import pytest

from hbtsim.diameter import kim_kaiser_sigma_vis2, scale_precision, scale_precision_scan
from hbtsim.snr import C_LIGHT, Observation, Spectrograph, g2_snr, incident_rate


def test_kim_kaiser_formula_is_the_matched_filter(hd17652, kk_filters, kk_tel, kk_det):
    spec = kk_filters["H"]
    nm, w = spec.channel_centers_nm, spec.channel_widths_nm
    obs = Observation(wavelength_nm=nm, filter_width_nm=w, t_int_s=7200.0, backend_throughput=1.0)
    mag = hd17652.ab_mag(nm)
    r = g2_snr(np.array([0.5]), mag, obs, telescope1=kk_tel, detector1=kk_det)
    rate = float(np.asarray(incident_rate(mag, kk_tel, kk_det, obs))[0])
    dnu = C_LIGHT * w[0] * 1e-9 / (nm[0] * 1e-9) ** 2
    ours = 0.5 / float(np.asarray(r.snr)[0])
    assert ours == pytest.approx(kim_kaiser_sigma_vis2(rate / dnu, 7200.0, kk_det.jitter_sigma_s), rel=1e-6)


def test_scale_precision_scales_with_time_and_prefers_first_lobe(hd17652, hd360, kk_filters,
                                                                  kk_tel, kk_det):
    hw = dict(telescope=kk_tel, detector=kk_det)
    b = np.arange(40.0, 301.0, 20.0)
    _, sig, best = scale_precision_scan(hd17652, b, kk_filters["H"], t_int_s=7200.0, **hw)
    assert np.all(np.isfinite(sig)) and 40.0 <= best.baseline_m <= 200.0
    assert 0.3 < best.vis2[0] < 0.9                      # first lobe, not the null
    r4 = scale_precision(hd17652, best.baseline_m, kk_filters["H"], t_int_s=4 * 7200.0, **hw)
    assert r4.sigma_s == pytest.approx(best.sigma_s / 2.0, rel=1e-9)
    faint = scale_precision(hd360, best.baseline_m, kk_filters["H"], t_int_s=7200.0, **hw)
    assert faint.sigma_s > best.sigma_s


def test_multiplexing_adds_information(hd17652, kk_filters, kk_tel, kk_det):
    hw = dict(telescope=kk_tel, detector=kk_det)
    one = scale_precision(hd17652, 100.0, kk_filters["V"], t_int_s=7200.0, **hw)
    many = scale_precision(hd17652, 100.0, Spectrograph(lambda_min_nm=507.0, lambda_max_nm=595.0,
                                                         n_channels=50, throughput=1.0),
                           t_int_s=7200.0, **hw)
    assert many.sigma_s < one.sigma_s
    mask = np.zeros(50, bool); mask[:10] = True
    sub = scale_precision(hd17652, 100.0, many.snr.spectrograph, t_int_s=7200.0, channel_mask=mask, **hw)
    assert np.all(sub.fisher[~mask] == 0.0) and sub.sigma_s > many.sigma_s
