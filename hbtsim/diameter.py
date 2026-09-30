"""Precision of a single star's angular scale from multiplexed g2 --
the observable of Kim & Kaiser (2026, PASP 138, 044202: intensity
interferometry as a validation of the red-clump surface-brightness-colour
relation), on hbtsim's photon budget.

Their scale parameter s stretches a model intensity profile,
I(theta; s) = I0(theta / s), and its Cramer-Rao bound is
sigma_s^-2 = sum_channels (d|V|^2/ds)^2 / sigma^2(|V|^2).  Their noise,
sigma(|V|^2)^-1 = (dGamma/dnu) (T/sigma_t)^(1/2) (128 pi)^(-1/4) in the
jitter-dominated regime, is hbtsim's matched filter (snr.g2_snr) with
p2 = 1/2 for unpolarized light and a Gaussian pair kernel of width
sqrt(2) sigma_t -- kim_kaiser_sigma_vis2 reproduces it, and the tests
check the two agree.  What hbtsim adds: spectral multiplexing with real
detectors, band-averaged NewEra limb darkening, pupil smearing, dead
time, link ceilings, and the radius convention of spherical models.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .single import SingleStar, prepare_single, single_star_vis2, spectral_g2_snr_single
from .snr import Detector, Spectrograph, SpectralSNRResult, Telescope

# Kim & Kaiser's fiducial instrument: two 4 m telescopes, 0.3 overall
# throughput (atmosphere, optics, detector), sigma_t = 16.6 ps per detector
# (their formula uses sigma_t; the "42.4 ps FWHM" they quote alongside would
# be 18.0 ps, so we adopt the 16.6 ps that enters their numbers), no dark
# counts, no readout ceiling
KK_SIGMA_T_PS = 16.6
KK_TELESCOPE = Telescope(diameter_m=4.0, throughput=0.3)
KK_DETECTOR = Detector(name="Kim & Kaiser fiducial (sigma_t 16.6 ps, throughput in the telescope)",
                       pde_table_nm=((300.0, 1.0), (2600.0, 1.0)),
                       jitter_fwhm_ps=KK_SIGMA_T_PS * 2.0 * np.sqrt(2.0 * np.log(2.0)),
                       dead_time_ns=0.0, dark_cps_per_pixel=0.0, readout="correlator",
                       max_total_cps=None)


def single_filter(name: str, centre_nm: float, fwhm_nm: float) -> Spectrograph:
    """One broad filter as a one-channel 'spectrograph' (throughput 1: the
    0.3 of KK_TELESCOPE already covers the whole chain)."""
    return Spectrograph(lambda_min_nm=centre_nm - fwhm_nm / 2, lambda_max_nm=centre_nm + fwhm_nm / 2,
                        n_channels=1, throughput=1.0, name=name)


# Johnson-Cousins / 2MASS-like passbands as top-hat filters (centre, FWHM)
BANDS = {"B": single_filter("B", 445.0, 94.0), "V": single_filter("V", 551.0, 88.0),
         "R": single_filter("R", 658.0, 138.0), "I": single_filter("I", 806.0, 149.0),
         "H": single_filter("H", 1630.0, 300.0), "K": single_filter("K", 2190.0, 390.0)}


def kim_kaiser_sigma_vis2(dgamma_dnu, t_s: float, sigma_t_s: float) -> float:
    """Their Eq. 8: sigma(|V|^2) for a photon spectral rate dGamma/dnu
    [photons/s/Hz] per telescope, integration T and per-detector jitter
    sigma_t (jitter-dominated regime; unpolarized light implicit)."""
    return 1.0 / (dgamma_dnu * np.sqrt(t_s / sigma_t_s) * (128.0 * np.pi) ** -0.25)


@dataclass(frozen=True)
class ScaleResult:
    target: str
    baseline_m: float
    t_int_s: float
    sigma_s: float                 # fractional precision on the angular scale
    nm: np.ndarray
    vis2: np.ndarray
    dvis2_ds: np.ndarray           # d|V|^2 / d ln s per channel
    sigma_vis2: np.ndarray         # per channel in t_int_s (inf where untagged)
    fisher: np.ndarray             # per-channel information (dvis2_ds / sigma)^2
    snr: SpectralSNRResult

    @property
    def n_channels(self) -> int:
        return int(self.nm.size)


def vis2_scale_derivative(target: SingleStar, baseline_m: float, nm, pupils=None,
                          rel: float = 1e-3) -> np.ndarray:
    """d|V|^2 / d ln s at s = 1 (the scale multiplies the drawn diameter),
    by central finite difference; (n_lambda,) at one baseline."""
    up = single_star_vis2(replace(target, theta_ld_mas=target.theta_ld_mas * (1 + rel)),
                          baseline_m, nm, pupils)[:, 0]
    dn = single_star_vis2(replace(target, theta_ld_mas=target.theta_ld_mas * (1 - rel)),
                          baseline_m, nm, pupils)[:, 0]
    return (up - dn) / (2.0 * rel)


def scale_precision(target: SingleStar, baseline_m: float, spectrograph: Spectrograph, *,
                    telescope: Telescope = KK_TELESCOPE, detector: Detector = KK_DETECTOR,
                    polarization_mode: str = "unpolarized", t_int_s: float = 7200.0,
                    pupils=True, enforce_readout: bool = True, channel_mask=None) -> ScaleResult:
    """Cramer-Rao precision of the angular scale from every channel of the
    spectrograph at one baseline, on hbtsim's photon budget."""
    tgt = prepare_single(target, spectrograph)
    nm = spectrograph.channel_centers_nm
    pup = (telescope.diameter_m, telescope.diameter_m) if pupils is True else pupils
    res = spectral_g2_snr_single(tgt, baseline_m, spectrograph, t_int_s=t_int_s,
                                 telescope1=telescope, detector1=detector,
                                 polarization_mode=polarization_mode, pupils=pup,
                                 channel_mask=channel_mask, enforce_readout=enforce_readout)
    d = vis2_scale_derivative(tgt, baseline_m, nm, pup)
    with np.errstate(divide="ignore", invalid="ignore"):
        sig = np.where(res.snr > 0, res.vis2 / res.snr, np.inf)
        fisher = np.where(np.isfinite(sig), (d / sig) ** 2, 0.0)
    total = float(np.sum(fisher))
    return ScaleResult(target=target.name, baseline_m=float(baseline_m), t_int_s=t_int_s,
                       sigma_s=(np.inf if total <= 0 else total ** -0.5), nm=nm, vis2=res.vis2,
                       dvis2_ds=d, sigma_vis2=sig, fisher=fisher, snr=res)


def scale_precision_scan(target: SingleStar, baselines_m, spectrograph: Spectrograph, **kw):
    """(baselines, sigma_s per baseline, best ScaleResult)."""
    b = np.asarray(baselines_m, dtype=float)
    results = [scale_precision(target, float(x), spectrograph, **kw) for x in b]
    sig = np.array([r.sigma_s for r in results])
    return b, sig, results[int(np.argmin(sig))]
