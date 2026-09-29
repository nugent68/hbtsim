"""Alternative g2 estimators, for comparison with the matched filter of
hbtsim.snr and with the EON-SII design study (Schweizer et al. 2026,
arXiv:2608.17444).

hbtsim's pair S/N (snr.g2_snr) is the Gaussian matched filter on the
coincidence-lag histogram,

    S/N_mf = p2 |V|^2 tau_c R1 R2 sqrt(T) / sqrt(b1 b2 2 sqrt(pi) sigma_pair).

The design study quotes two other forms:

  (Eq. 3, classic HBT)   S/N = A alpha q n_nu |V|^2 (sigma_spec/F) sqrt(b_el T / 2)
  (Eq. 5, photon level)  S/N = eta R1 R2 tau_c |V|^2 T
                               / sqrt(eta R1 R2 tau_c |V|^2 T + 2 R1 R2 dt_res T)

Both reduce EXACTLY to the matched filter when eta = p2 (1/2 for
unpolarized light), dt_res = sqrt(pi) sigma_pair and b_el =
1/(sqrt(pi) sigma_pair) (matched_filter_equivalents).  A coincidence
window ("box") of half-width a that honestly counts only the fraction
erf(a / (sqrt 2 sigma)) of the signal it captures peaks at a ~ 1.40
sigma, at 0.943 of the matched filter; a box that assumes it captures
everything ("full" capture) overstates the S/N and, when used as an
estimator, is biased low by exactly that fraction.  The Monte Carlo in
hbtsim.montecarlo tests these statements on simulated histograms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erf

from .snr import (Detector, Observation, Telescope, g2_snr,
                  polarization_streams)


def photon_level_snr(vis2, rate1_cps, rate2_cps, tau_c_s, t_int_s, dt_res_s, eta=0.5):
    """The design study's Eq. 5 (photon-level estimator)."""
    sig = eta * np.asarray(rate1_cps) * np.asarray(rate2_cps) * tau_c_s * np.asarray(vis2) * t_int_s
    bkg = 2.0 * np.asarray(rate1_cps) * np.asarray(rate2_cps) * dt_res_s * t_int_s
    return sig / np.sqrt(sig + bkg)


def classic_hbt_snr(area_m2, alpha, q, n_nu, vis2, sigma_spec_over_f, b_el_hz, t_int_s):
    """The design study's Eq. 3 (classic Hanbury Brown-Twiss form);
    n_nu in photons m^-2 s^-1 Hz^-1 at the telescope."""
    return (area_m2 * alpha * q * n_nu * np.asarray(vis2) * sigma_spec_over_f
            * np.sqrt(b_el_hz * t_int_s / 2.0))


def box_capture_fraction(half_width_s, sigma_pair_s):
    """Fraction of a Gaussian-kernel excess inside |tau| <= half_width."""
    return erf(np.asarray(half_width_s) / (np.sqrt(2.0) * sigma_pair_s))


def optimal_box_half_width(sigma_pair_s) -> float:
    """Half-width maximizing erf(a/(sqrt2 s))/sqrt(2a): ~1.40 sigma."""
    # optimize in units of sigma (the default xatol would swamp picoseconds)
    r = minimize_scalar(lambda u: -erf(u / np.sqrt(2.0)) / np.sqrt(2.0 * u),
                        bounds=(0.05, 6.0), method="bounded", options={"xatol": 1e-8})
    return float(r.x) * float(sigma_pair_s)


def matched_filter_equivalents(sigma_pair_s, p2=0.5) -> dict:
    """Parameters for which Eq. 3 / Eq. 5 equal the matched filter
    (background-dominated, no dark counts)."""
    return dict(dt_res_s=np.sqrt(np.pi) * sigma_pair_s, eta=p2,
                b_el_hz=1.0 / (np.sqrt(np.pi) * sigma_pair_s))


def implied_dt_res(t_paper_h, t_ours_h, sigma_pair_s, eta_ratio=1.0) -> float:
    """The dt_res that would make Eq. 5 reproduce t_paper when our matched
    filter needs t_ours (times scale as 1/S/N^2; S/N as eta/sqrt(dt)):
    dt_paper = sqrt(pi) sigma (eta_paper/eta_ours)^2 (t_paper/t_ours)."""
    return float(np.sqrt(np.pi) * sigma_pair_s * eta_ratio**2 * t_paper_h / t_ours_h)


@dataclass(frozen=True)
class BoxSNR:
    snr: object
    snr_matched: object       # the matched filter on the same photons
    capture: object           # fraction of the signal inside the box
    half_width_s: float
    sigma_pair_s: object


def box_snr(vis2, mag_ab, obs: Observation, telescope1: Telescope, detector1: Detector,
            half_width_s: float, *, capture: str = "gaussian",
            telescope2: Telescope | None = None, detector2: Detector | None = None) -> BoxSNR:
    """S/N of a coincidence box |tau| <= half_width on the same photon
    budget as snr.g2_snr.  capture="gaussian" counts the erf fraction
    of the excess that falls in the box (the honest estimate);
    capture="full" assumes all of it (what a literal Eq. 5 with a narrow
    dt_res does)."""
    res = g2_snr(vis2, mag_ab, obs, telescope1=telescope1, telescope2=telescope2,
                 detector1=detector1, detector2=detector2)
    n_streams, _, _, _ = polarization_streams(obs.polarization_mode)
    sigma = np.asarray(res.sigma_pair_s)
    frac = np.ones_like(sigma) if capture == "full" else box_capture_fraction(half_width_s, sigma)
    # background per unit lag, from the matched-filter window the result used
    bkg_density = np.asarray(res.n_background) / (2.0 * np.sqrt(np.pi) * sigma)
    snr = np.sqrt(n_streams) * np.asarray(res.n_signal) * frac / np.sqrt(bkg_density * 2.0 * half_width_s)
    f = (lambda a: float(a) if np.ndim(a) == 0 else a)
    return BoxSNR(snr=f(snr), snr_matched=res.snr, capture=f(frac),
                  half_width_s=float(half_width_s), sigma_pair_s=res.sigma_pair_s)
