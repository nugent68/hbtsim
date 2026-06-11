"""Signal-to-noise of the triple correlation (bispectrum / closure phase).

Extends the matched-filter photon budget of hbtsim.snr to three
telescopes.  Per spectral channel:

  signal:  N_sig = pol3 . 2|g12 g23 g31| cos(phi_c) . tau_c^2 . R1 R2 R3 . T

  -- the lag-plane (tau1, tau2) integral of the triple term for a
  rectangular passband is exactly tau_c^2 = 1/dnu^2 (Parseval on the
  cubed unit-area spectrum), paralleling the pair case's tau_c.
  pol3 = 1/4 for unpolarized light: two independent modes each carry
  I/2, and the triple term scales as 2 (1/2)^3 (the pair terms carry
  the familiar 1/2).

  noise:   accidental triples at density b1 b2 b3 per unit lag^2,
  smeared by the detectors' jitters into a correlated 2D Gaussian with
  covariance Sigma = [[s1^2+s2^2, -s2^2], [-s2^2, s2^2+s3^2]] (telescope
  2 enters both lags with opposite signs); the matched-filter effective
  area is A_2D = 4 pi sqrt(det Sigma), det Sigma = s1^2 s2^2 + s2^2 s3^2
  + s3^2 s1^2 (equal jitters: 4 pi sqrt(3) s^2).

  SNR3 = N_sig / sqrt(b1 b2 b3 . T . A_2D)

With R = n A alpha dnu this reduces to SNR3 ~ |ggg| (n A alpha)^{3/2}
(1/sigma) sqrt(T/dnu) -- Nunez & Domiciano de Souza 2015 eq. 8 -- and to
Zmija et al. 2025 eq. 12, SNR3 ~ (1/tau_e) sqrt(A^3 T / dlam).  The key
contrasts with the pair SNR: three-telescope sensitivity scales as
1/sigma_jitter (not 1/sqrt(sigma)), and as 1/sqrt(dlam) at fixed source
(not bandwidth-independent) -- narrow channels and heavy spectral
multiplexing are the levers.  Estimating cos(phi_c) to precision
d(cos phi_c) requires SNR3(cos = 1) >= 1/d.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .bispectrum import (Triangle, binary_vis_complex_analytic, closure_phase,
                         spectral_bispectrum)
from .orbit import SkyPositions, sky_positions
from .params import BinarySystem, GridConfig
from .snr import (Observation, Spectrograph, coherence_time_s, stellar_rate,
                  system_ab_mag)

POL_FACTOR_TRIPLE = 0.25  # unpolarized; 1.0 for fully polarized light


def triple_window_s2(det1, det2, det3) -> float:
    """Matched-filter effective area on the (tau1, tau2) lag plane,
    4 pi sqrt(det Sigma) [s^2]."""
    s1, s2, s3 = (d.jitter_sigma_s for d in (det1, det2, det3))
    det_sigma = s1**2 * s2**2 + s2**2 * s3**2 + s3**2 * s1**2
    return 4.0 * np.pi * np.sqrt(det_sigma)


@dataclass(frozen=True)
class SNR3Result:
    snr: float               # of the triple term (signed by cos phi_c)
    rates_cps: tuple         # detected stellar rate per station
    tau_c_s: float
    window_s2: float
    n_signal: float
    n_background: float
    triple_amp: float        # |g12 g23 g31|
    cos_phi_c: float
    obs: Observation


def g3_snr(triple_amp: float, mag_ab: float, obs: Observation,
           triangle: Triangle, *, cos_phi_c: float = 1.0,
           pol_factor_triple: float = POL_FACTOR_TRIPLE) -> SNR3Result:
    """SNR of the bispectrum term for one spectral channel (see module
    docstring).  Each station's light goes to one detector pixel."""
    dets = [replace(s.detector, n_pixels=1) for s in triangle.stations]
    rates = tuple(stellar_rate(mag_ab, s.telescope, d, obs)
                  for s, d in zip(triangle.stations, dets))
    bg = [r + d.dark_cps + obs.sky_cps for r, d in zip(rates, dets)]

    tau_c = coherence_time_s(obs.wavelength_nm, obs.filter_width_nm)
    window = triple_window_s2(*dets)

    n_sig = (pol_factor_triple * 2.0 * triple_amp * cos_phi_c
             * tau_c**2 * rates[0] * rates[1] * rates[2] * obs.t_int_s)
    n_bkg = bg[0] * bg[1] * bg[2] * obs.t_int_s * window
    return SNR3Result(snr=n_sig / np.sqrt(n_bkg), rates_cps=rates,
                      tau_c_s=tau_c, window_s2=window, n_signal=n_sig,
                      n_background=n_bkg, triple_amp=triple_amp,
                      cos_phi_c=cos_phi_c, obs=obs)


@dataclass(frozen=True)
class SpectralSNR3Result:
    snr_total: float         # quadrature sum at |cos phi_c| = 1 per channel
    triangle: Triangle
    spectrograph: Spectrograph
    channel_nm: np.ndarray
    snr: np.ndarray          # per channel, at cos = 1 (sensitivity)
    triple_amp: np.ndarray   # per channel
    cos_phi_c: np.ndarray    # model closure-phase cosine per channel
    rates_cps: np.ndarray    # (n_channels, 3)
    mag_ab: np.ndarray
    vis_method: str = ""


def spectral_g3_snr(system: BinarySystem, triangle: Triangle,
                    spectrograph: Spectrograph = Spectrograph(),
                    t_int_s: float = 3600.0, orbital_phase: float = 0.0,
                    vis_method: str = "analytic",
                    grid: GridConfig = GridConfig(),
                    chunk_size: int | None = None,
                    pol_factor_triple: float = POL_FACTOR_TRIPLE,
                    sky_cps_per_channel: float = 0.0) -> SpectralSNR3Result:
    """Per-channel bispectrum sensitivity with the source dispersed over
    the array, and the quadrature total (the sensitivity to a global
    rescaling of the model's per-channel cos phi_c pattern)."""
    nm = spectrograph.channel_centers_nm
    pos = SkyPositions(*(np.asarray(v) for v in
                         sky_positions(2 * np.pi * orbital_phase, system)))
    bvecs = triangle.baseline_vectors()

    if vis_method == "analytic":
        sum_r = (system.angular_radius_mas(system.primary)
                 + system.angular_radius_mas(system.secondary))
        if float(pos.rho) < 1.05 * sum_r:
            raise ValueError("in (or near) eclipse: use vis_method='fft'")
        gam = np.array([binary_vis_complex_analytic(bvecs, float(l), system, pos)
                        for l in nm])  # (n_lambda, 3)
    elif vis_method == "fft":
        gam = np.asarray(spectral_bispectrum(pos, triangle, nm, system, grid,
                                             chunk_size=chunk_size))
    else:
        raise ValueError(f"unknown vis_method {vis_method!r}")

    bis = gam[:, 0] * gam[:, 1] * gam[:, 2]
    triple_amp = np.abs(bis)
    cosphi = np.cos(np.angle(bis))

    snr = np.empty(nm.size)
    rates = np.empty((nm.size, 3))
    mags = np.empty(nm.size)
    for k, lam_nm in enumerate(nm):
        mags[k] = system_ab_mag(system, float(lam_nm))
        obs = Observation(wavelength_nm=float(lam_nm),
                          filter_width_nm=spectrograph.channel_width_nm,
                          t_int_s=t_int_s, sky_cps=sky_cps_per_channel)
        r = g3_snr(float(triple_amp[k]), mags[k], obs, triangle,
                   cos_phi_c=1.0, pol_factor_triple=pol_factor_triple)
        snr[k] = r.snr
        rates[k] = r.rates_cps
    return SpectralSNR3Result(snr_total=float(np.sqrt(np.sum(snr**2))),
                              triangle=triangle, spectrograph=spectrograph,
                              channel_nm=nm, snr=snr, triple_amp=triple_amp,
                              cos_phi_c=cosphi, rates_cps=rates, mag_ab=mags,
                              vis_method=vis_method)


def time_to_cos_phi(system: BinarySystem, triangle: Triangle,
                    target_dcos: float = 0.1,
                    spectrograph: Spectrograph = Spectrograph(),
                    orbital_phase: float = 0.0,
                    vis_method: str = "analytic", **kw) -> float:
    """Integration time [s] for the multiplexed bispectrum to reach
    sigma(cos phi_c) <= target_dcos (SNR3 proportional to sqrt(T))."""
    ref = spectral_g3_snr(system, triangle, spectrograph=spectrograph,
                          t_int_s=3600.0, orbital_phase=orbital_phase,
                          vis_method=vis_method, **kw)
    needed = 1.0 / target_dcos
    return 3600.0 * (needed / ref.snr_total) ** 2
