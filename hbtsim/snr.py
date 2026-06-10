"""Signal-to-noise of a g2(B) measurement with a pair of telescopes.

Photon-counting intensity interferometry (see Guerin et al. 2025,
arXiv:2503.22446; Rai, Basak & Saha 2021, arXiv:2105.09532).  Two
telescopes feed single-photon detectors whose time tags are
cross-correlated.  For unpolarized thermal light the coincidence
histogram shows a bump of contrast

    g2(tau) - 1 = (1/2) |V(B)|^2 tau_c x kernel(tau),

where tau_c = lambda^2 / (c dlambda) is the coherence time of a
rectangular passband of width dlambda and the kernel is the (normalized)
pair time-response: a Gaussian of width sigma_pair =
sqrt(sigma_1^2 + sigma_2^2) set by the detectors' timing jitter, since
tau_c (tens of fs) << jitter (tens of ps).

Over an integration time T the excess (signal) coincidences are

    N_sig = (1/2) |V|^2 tau_c R1 R2 T,

with R_i the detected stellar count rates.  The accidental-coincidence
density is rho = b1 b2 T per unit time lag, where b_i = R_i + dark + sky
includes uncorrelated counts.  Weighting the histogram with the known
Gaussian kernel (matched filter) gives

    SNR = N_sig / sqrt(rho * 2 sqrt(pi) sigma_pair)
        = (1/2) |V|^2 tau_c R1 R2 sqrt(T)
          / sqrt(b1 b2 * 2 sqrt(pi) sigma_pair).

Notes:
  * SNR is nearly independent of the filter width: R_i ~ dlambda while
    tau_c ~ 1/dlambda (until dead time or sky/dark counts matter).
  * Detector dead time is applied per pixel (non-paralyzable,
    r -> r / (1 + r tau_dead)); spreading the light over n_pixels of an
    array detector raises the saturation ceiling.
  * pol_factor = 1/2 for unpolarized light; use 1 for a polarized setup
    (with the corresponding flux loss applied via throughput).

All defaults describe the C2PU pair (Centre Pedagogique Planete Univers,
Calern plateau): two 1 m telescopes on a 15 m baseline, with Pi Imaging
SPAD Lambda detectors (datasheet: background/SPADlambdadatasheet.pdf).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .params import (AB_ZERO_FNU, C_LIGHT, H_PLANCK, MAS, BinarySystem,
                     GridConfig, planck)

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Telescope:
    """A light collector; throughput covers atmosphere, optics and coupling
    losses up to (but not including) the detector PDE."""
    diameter_m: float
    throughput: float = 0.3

    @property
    def area_m2(self) -> float:
        return np.pi * (self.diameter_m / 2.0) ** 2


@dataclass(frozen=True)
class Detector:
    name: str
    # photon detection efficiency vs wavelength [nm], linearly interpolated
    pde_table_nm: tuple
    jitter_fwhm_ps: float
    dead_time_ns: float
    dark_cps_per_pixel: float
    n_pixels: int = 1  # pixels the stellar light is spread over

    def pde(self, wavelength_nm: float) -> float:
        lam, p = zip(*self.pde_table_nm)
        return float(np.interp(wavelength_nm, lam, p))

    @property
    def jitter_sigma_s(self) -> float:
        return self.jitter_fwhm_ps * 1e-12 * FWHM_TO_SIGMA

    @property
    def dark_cps(self) -> float:
        return self.dark_cps_per_pixel * self.n_pixels

    def detected_rate(self, incident_cps: float) -> float:
        """Non-paralyzable dead time applied per pixel."""
        r = incident_cps / self.n_pixels
        r_det = r / (1.0 + r * self.dead_time_ns * 1e-9)
        return r_det * self.n_pixels


# Pi Imaging SPAD Lambda (datasheet v2.3, 01.2026).  PDE read from the
# "Photon detection probability" curve (peak 50% at 520 nm); median DCR
# 250 cps/pixel; dead time 10 ns; timing jitter 120 ps FWHM typical.
SPAD_LAMBDA = Detector(
    name="Pi Imaging SPAD Lambda",
    pde_table_nm=((400.0, 0.22), (450.0, 0.40), (500.0, 0.49), (520.0, 0.50),
                  (550.0, 0.48), (600.0, 0.44), (650.0, 0.36), (700.0, 0.28),
                  (750.0, 0.20), (800.0, 0.14), (850.0, 0.09), (900.0, 0.06),
                  (950.0, 0.04)),
    jitter_fwhm_ps=120.0,
    dead_time_ns=10.0,
    dark_cps_per_pixel=250.0,
    n_pixels=1,
)

C2PU = Telescope(diameter_m=1.0, throughput=0.3)


@dataclass(frozen=True)
class Observation:
    wavelength_nm: float
    filter_width_nm: float = 10.0
    t_int_s: float = 3600.0
    pol_factor: float = 0.5  # unpolarized light
    sky_cps: float = 0.0     # detected sky background per telescope


@dataclass(frozen=True)
class Spectrograph:
    """Light dispersed along the detector's linear array: each pixel is an
    independent spectral channel that measures its own g2.  Channel SNRs add
    in quadrature, a ~sqrt(n_channels) multiplexing gain over a single
    filter of the same total band."""
    lambda_min_nm: float = 400.0   # SPAD Lambda sensitivity range
    lambda_max_nm: float = 950.0
    n_channels: int = 320          # SPAD Lambda: 320 x 1 pixels

    @property
    def channel_width_nm(self) -> float:
        return (self.lambda_max_nm - self.lambda_min_nm) / self.n_channels

    @property
    def channel_centers_nm(self) -> np.ndarray:
        return (self.lambda_min_nm
                + (np.arange(self.n_channels) + 0.5) * self.channel_width_nm)


# ---------------------------------------------------------------------------
# Photon budget
# ---------------------------------------------------------------------------
def coherence_time_s(wavelength_nm: float, filter_width_nm: float) -> float:
    """tau_c = lambda^2 / (c dlambda) for a rectangular passband."""
    lam = wavelength_nm * 1e-9
    return lam**2 / (C_LIGHT * filter_width_nm * 1e-9)


def photon_flux(mag_ab: float, wavelength_nm: float,
                filter_width_nm: float) -> float:
    """Source photon flux through the filter [photons / m^2 / s]."""
    lam = wavelength_nm * 1e-9
    f_nu = AB_ZERO_FNU * 10.0 ** (-0.4 * mag_ab)
    nu = C_LIGHT / lam
    dnu = C_LIGHT * filter_width_nm * 1e-9 / lam**2
    return f_nu / (H_PLANCK * nu) * dnu


def stellar_rate(mag_ab: float, telescope: Telescope, detector: Detector,
                 obs: Observation) -> float:
    """Detected stellar count rate [cps], including dead time."""
    incident = (photon_flux(mag_ab, obs.wavelength_nm, obs.filter_width_nm)
                * telescope.area_m2 * telescope.throughput
                * detector.pde(obs.wavelength_nm))
    return detector.detected_rate(incident)


# ---------------------------------------------------------------------------
# SNR
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SNRResult:
    snr: float
    rate1_cps: float
    rate2_cps: float
    tau_c_s: float
    sigma_pair_s: float
    n_signal: float        # excess (signal) coincidences in T
    n_background: float    # accidental coincidences in the matched window
    vis2: float
    obs: Observation


def vis2_noise(mag_ab: float, obs: Observation,
               telescope1: Telescope = C2PU,
               telescope2: Telescope | None = None,
               detector1: Detector = SPAD_LAMBDA,
               detector2: Detector | None = None) -> float:
    """1-sigma uncertainty of a |V|^2 measurement over obs.t_int_s (the
    noise-equivalent squared visibility): sigma = sqrt(N_bkg) /
    (pol_factor tau_c R1 R2 T), i.e. SNR = |V|^2 / vis2_noise."""
    r = g2_snr(1.0, mag_ab, obs, telescope1=telescope1, telescope2=telescope2,
               detector1=detector1, detector2=detector2)
    return 1.0 / r.snr


def g2_snr(vis2: float, mag_ab: float, obs: Observation,
           telescope1: Telescope = C2PU, telescope2: Telescope | None = None,
           detector1: Detector = SPAD_LAMBDA,
           detector2: Detector | None = None) -> SNRResult:
    """SNR of the g2 bump for one baseline (see module docstring)."""
    telescope2 = telescope1 if telescope2 is None else telescope2
    detector2 = detector1 if detector2 is None else detector2

    r1 = stellar_rate(mag_ab, telescope1, detector1, obs)
    r2 = stellar_rate(mag_ab, telescope2, detector2, obs)
    b1 = r1 + detector1.dark_cps + obs.sky_cps
    b2 = r2 + detector2.dark_cps + obs.sky_cps

    tau_c = coherence_time_s(obs.wavelength_nm, obs.filter_width_nm)
    sigma_pair = np.hypot(detector1.jitter_sigma_s, detector2.jitter_sigma_s)

    n_sig = obs.pol_factor * vis2 * tau_c * r1 * r2 * obs.t_int_s
    eff_window = 2.0 * np.sqrt(np.pi) * sigma_pair  # matched-filter width
    n_bkg = b1 * b2 * obs.t_int_s * eff_window
    return SNRResult(snr=n_sig / np.sqrt(n_bkg), rate1_cps=r1, rate2_cps=r2,
                     tau_c_s=tau_c, sigma_pair_s=sigma_pair, n_signal=n_sig,
                     n_background=n_bkg, vis2=vis2, obs=obs)


@dataclass(frozen=True)
class SpectralSNRResult:
    snr_total: float
    spectrograph: Spectrograph
    baseline_m: float
    channel_nm: np.ndarray
    snr: np.ndarray          # per channel
    rate_cps: np.ndarray     # detected stellar rate per channel per telescope
    vis2: np.ndarray         # per channel
    mag_ab: np.ndarray       # per channel
    vis2_method: str = ""


def spectral_g2_snr(system: BinarySystem, baseline_m: float,
                    spectrograph: Spectrograph = Spectrograph(),
                    t_int_s: float = 3600.0,
                    telescope1: Telescope = C2PU,
                    telescope2: Telescope | None = None,
                    detector1: Detector = SPAD_LAMBDA,
                    detector2: Detector | None = None,
                    pol_factor: float = 0.5, sky_cps_per_channel: float = 0.0,
                    orbital_phase: float = 0.0,
                    vis2_method: str = "fft",
                    grid: GridConfig = GridConfig(),
                    chunk_size: int | None = None) -> SpectralSNRResult:
    """Total g2 SNR with the source spectrum dispersed over the array.

    Each channel (= one pixel per telescope, so dead time and dark counts
    are per channel) measures g2 independently at its own wavelength, with
    the baseline along the projected separation axis at the requested
    orbital phase.  SNR_total = sqrt(sum SNR_i^2).

    vis2_method:
      "fft"      -- batched FFT pipeline (hbtsim.spectral.spectral_vis2):
                    valid at all phases including eclipses; fast on GPU,
                    ~1 s/channel on CPU.
      "analytic" -- hbt.binary_vis2_analytic: instant, agrees with the FFT
                    to <0.5%, but only valid OUT of eclipse (raises during
                    one).
    """
    from dataclasses import replace

    from .hbt import binary_vis2_analytic
    from .orbit import SkyPositions, sky_positions

    telescope2 = telescope1 if telescope2 is None else telescope2
    detector2 = detector1 if detector2 is None else detector2
    # one pixel per channel
    det1 = replace(detector1, n_pixels=1)
    det2 = replace(detector2, n_pixels=1)

    pos = SkyPositions(*(np.asarray(v) for v in
                         sky_positions(2.0 * np.pi * orbital_phase, system)))
    rho_mas = float(pos.rho)
    nm = spectrograph.channel_centers_nm

    if vis2_method == "fft":
        from .spectral import spectral_vis2
        vis2 = np.asarray(spectral_vis2(pos, [baseline_m], nm, system, grid,
                                        chunk_size=chunk_size))[:, 0].astype(float)
    elif vis2_method == "analytic":
        sum_radii = (system.angular_radius_mas(system.primary)
                     + system.angular_radius_mas(system.secondary))
        if rho_mas < 1.05 * sum_radii:
            raise ValueError(
                f"orbital phase {orbital_phase} is in (or near) eclipse "
                f"(rho = {rho_mas:.3f} mas, disks overlap below "
                f"{1.05 * sum_radii:.3f} mas): the analytic binary visibility "
                f"is invalid there; use vis2_method='fft'")
        vis2 = np.array([float(binary_vis2_analytic(baseline_m, lam_nm,
                                                    system, rho_mas)[0])
                         for lam_nm in nm])
    else:
        raise ValueError(f"unknown vis2_method {vis2_method!r} "
                         f"(expected 'fft' or 'analytic')")

    snr = np.empty(nm.size)
    rate = np.empty(nm.size)
    mag = np.empty(nm.size)
    for k, lam_nm in enumerate(nm):
        mag[k] = system_ab_mag(system, lam_nm)
        obs = Observation(wavelength_nm=lam_nm,
                          filter_width_nm=spectrograph.channel_width_nm,
                          t_int_s=t_int_s, pol_factor=pol_factor,
                          sky_cps=sky_cps_per_channel)
        res = g2_snr(float(vis2[k]), mag[k], obs, telescope1=telescope1,
                     telescope2=telescope2, detector1=det1, detector2=det2)
        snr[k] = res.snr
        rate[k] = res.rate1_cps
    return SpectralSNRResult(snr_total=float(np.sqrt(np.sum(snr**2))),
                             spectrograph=spectrograph, baseline_m=baseline_m,
                             channel_nm=nm, snr=snr, rate_cps=rate, vis2=vis2,
                             mag_ab=mag, vis2_method=vis2_method)


# ---------------------------------------------------------------------------
# Source model: out-of-eclipse magnitude of the binary at any wavelength
# ---------------------------------------------------------------------------
def system_ab_mag(system: BinarySystem, wavelength_nm: float) -> float:
    """Apparent AB magnitude of the (uneclipsed) binary at one wavelength:
    blackbody disks f_nu = sum_s B_lambda(T_s) pi theta_s^2 (1 - u/3) lam^2/c,
    corrected by the observed anchor offsets (params.BinarySystem.mag_anchors)
    interpolated linearly in wavelength -- the same blackbody zero-point fix
    applied to the lightcurves (see photometry.py)."""
    lam = wavelength_nm * 1e-9
    u = system.ld_coeff(wavelength_nm)
    f_nu = 0.0
    for star in (system.primary, system.secondary):
        theta_r = system.angular_radius_mas(star) * MAS
        f_nu += (planck(lam, star.teff) * np.pi * theta_r**2
                 * (1.0 - u / 3.0) * lam**2 / C_LIGHT)
    m_synth = -2.5 * np.log10(f_nu / AB_ZERO_FNU)

    # anchor offsets at the photometric bands, interpolated in wavelength
    band_lam = {"g": 477.0, "i": 763.0}
    lams, offsets = [], []
    for band, m_obs in system.mag_anchors:
        lam_b = band_lam[band]
        u_b = system.ld_coeff(lam_b)
        f_b = sum(planck(lam_b * 1e-9, s.teff) * np.pi
                  * (system.angular_radius_mas(s) * MAS) ** 2
                  * (1.0 - u_b / 3.0) * (lam_b * 1e-9) ** 2 / C_LIGHT
                  for s in (system.primary, system.secondary))
        lams.append(lam_b)
        offsets.append(m_obs - (-2.5 * np.log10(f_b / AB_ZERO_FNU)))
    order = np.argsort(lams)
    offset = float(np.interp(wavelength_nm, np.array(lams)[order],
                             np.array(offsets)[order]))
    return m_synth + offset
