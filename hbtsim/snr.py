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
sqrt(sigma_1^2 + sigma_2^2 + sigma_c^2) set by the detectors' timing
jitter and, marginally, by the coherence time itself (sigma_c = 0.376
tau_c, the Gaussian of the same FWHM as the sinc^2 coherence function of
a rectangular band; a few per cent at 0.1 nm channels in the red).

Over an integration time T the excess (signal) coincidences are

    N_sig = p2 |V|^2 tau_c R1 R2 T,

with R_i the detected stellar count rates and p2 the polarization
factor.  The accidental-coincidence density is rho = b1 b2 T per unit
time lag, where b_i = R_i + dark + sky includes uncorrelated counts.
Weighting the histogram with the known Gaussian kernel (matched filter)
gives

    SNR = N_sig / sqrt(rho * 2 sqrt(pi) sigma_pair)
        = p2 |V|^2 tau_c R1 R2 sqrt(T) / sqrt(b1 b2 * 2 sqrt(pi) sigma_pair).

Polarization (Observation.polarization_mode):
  * "unpolarized" -- one stream per telescope carrying both modes:
    p2 = 1/2 (the two modes are mutually incoherent).
  * "pbs" -- a polarizing beamsplitter feeds two detectors per telescope,
    each with half the rate and p2 = 1 within its stream; the two
    stream correlations add in quadrature: sqrt(2) better than
    unpolarized at the same photon budget AND half the per-pixel load
    (for the triple correlation the gain is a factor 2, hbtsim.snr3).
  * "single_pol" -- one polarizer, half the light thrown away: p2 = 1
    on half the rate, no net gain over unpolarized.

Throughput is split into Telescope.throughput (atmosphere + telescope
optics to the backend entrance, 0.3) and a Backend (0.9 for a
narrow-band filter, 0.5 for a cross-dispersed spectrograph with its
coupling optics: a dispersed channel therefore sees 0.15 overall, not
0.3).  Detector PDE is applied separately.

Readout.  The count rates of a bright star dispersed over thousands of
channels reach 1e10-1e11 detected photons per second per telescope --
orders of magnitude beyond any time-tag link.  Detector.readout is
"timetag" (with Detector.max_total_cps the link ceiling; the spectral
functions scale the rates down to it like a neutral-density filter and
set readout_limited) or "correlator" (on-detector/FPGA correlation,
no link ceiling, the next-generation design).  Dead time is applied
per pixel (non-paralyzable, r -> r / (1 + r tau_dead)); the model is
only trustworthy for r tau_dead <~ 1, which dead_time_load reports.

Telescopes and detectors are passed explicitly; the named ones (the C2PU
1 m pair with Pi Imaging SPAD Lambda detectors, Keck, Subaru, EON-SII, ...)
are hbtsim/configs/{telescopes,detectors}/*.json, hbtsim.catalog.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np

from .params import (AB_ZERO_FNU, ANCHOR_CHECK_MAG, C_LIGHT, H_PLANCK, MAS,
                     BinarySystem, GridConfig, planck)

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
# Gaussian of the same FWHM as sinc^2(pi dnu tau): FWHM = 0.886 tau_c
COHERENCE_SIGMA_FACTOR = 0.886 * FWHM_TO_SIGMA   # 0.376

POLARIZATION_MODES = {
    # mode: (streams per telescope, flux fraction per stream, p2, p3)
    "unpolarized": (1, 1.0, 0.5, 0.25),
    "pbs": (2, 0.5, 1.0, 1.0),
    "single_pol": (1, 0.5, 1.0, 1.0),
}


def polarization_streams(mode: str) -> tuple:
    try:
        return POLARIZATION_MODES[mode]
    except KeyError:
        raise ValueError(f"unknown polarization_mode {mode!r}; expected one "
                         f"of {sorted(POLARIZATION_MODES)}") from None


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Telescope:
    """A light collector; throughput covers atmosphere, telescope optics
    and coupling losses up to the backend entrance (not the backend
    itself, nor the detector PDE).  diameter_m is the pupil that smears
    the fringe (hbtsim.aperture); collecting_area_m2, when given, is the
    photon-collecting area (segmented or obstructed mirrors)."""
    diameter_m: float
    throughput: float = 0.3
    collecting_area_m2: float | None = None
    name: str = ""

    @property
    def area_m2(self) -> float:
        if self.collecting_area_m2 is not None:
            return float(self.collecting_area_m2)
        return np.pi * (self.diameter_m / 2.0) ** 2


# Throughput of the optics between the telescope focus and the detector:
# a narrow-band filter, or a cross-dispersed spectrograph (the defaults of
# Observation.backend_throughput and Spectrograph.throughput).
FILTER_THROUGHPUT = 0.9
DISPERSED_THROUGHPUT = 0.5


@dataclass(frozen=True)
class Detector:
    name: str
    # photon detection efficiency vs wavelength [nm], linearly interpolated
    pde_table_nm: tuple
    jitter_fwhm_ps: float
    dead_time_ns: float
    dark_cps_per_pixel: float
    n_pixels: int = 1          # pixels the stellar light is spread over
    readout: str = "timetag"   # "timetag" | "correlator"
    max_total_cps: float | None = None   # time-tag link ceiling per detector
    _pde_lam: np.ndarray = field(init=False, repr=False, compare=False)
    _pde_val: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.readout not in ("timetag", "correlator"):
            raise ValueError(f"readout must be 'timetag' or 'correlator', "
                             f"not {self.readout!r}")
        lam, p = zip(*self.pde_table_nm)
        object.__setattr__(self, "_pde_lam", np.asarray(lam, dtype=float))
        object.__setattr__(self, "_pde_val", np.asarray(p, dtype=float))

    def pde(self, wavelength_nm):
        """PDE at a wavelength or array of wavelengths [nm]."""
        out = np.interp(wavelength_nm, self._pde_lam, self._pde_val)
        return float(out) if np.ndim(out) == 0 else out

    @property
    def jitter_sigma_s(self) -> float:
        return self.jitter_fwhm_ps * 1e-12 * FWHM_TO_SIGMA

    @property
    def dark_cps(self) -> float:
        return self.dark_cps_per_pixel * self.n_pixels

    def dead_time_load(self, incident_cps):
        """r tau_dead per pixel: the model is valid for values <~ 1."""
        return np.asarray(incident_cps, dtype=float) / self.n_pixels * self.dead_time_ns * 1e-9

    def detected_rate(self, incident_cps):
        """Non-paralyzable dead time applied per pixel (array-capable)."""
        r = np.asarray(incident_cps, dtype=float) / self.n_pixels
        r_det = r / (1.0 + r * self.dead_time_ns * 1e-9)
        out = r_det * self.n_pixels
        return float(out) if np.ndim(out) == 0 else out


# The detector and telescope presets (SPAD Lambda, C2PU, Keck, Subaru, EON-SII,
# ...) are hbtsim/configs/{detectors,telescopes}/*.json (hbtsim.catalog).


@dataclass(frozen=True)
class Observation:
    """One channel (or an array of channels: wavelength_nm and
    filter_width_nm may be arrays of equal shape)."""
    wavelength_nm: object
    filter_width_nm: object = 10.0     # rectangular passband FULL width
    t_int_s: float = 3600.0
    sky_cps: float = 0.0               # detected sky background per telescope
    polarization_mode: str = "unpolarized"
    backend_throughput: float = FILTER_THROUGHPUT
    coherence_broadening: bool = True


@dataclass(frozen=True)
class Spectrograph:
    """Light dispersed along the detector's linear array: each pixel is an
    independent spectral channel that measures its own g2.  Channel SNRs
    add in quadrature, a ~sqrt(n_channels) multiplexing gain over a
    single filter of the same total band.

    Channels are uniform in wavelength (the default: the SPAD Lambda's
    320 pixels over 400-950 nm, 1.72 nm each) or, via
    from_resolving_power, geometric with a constant lambda/dlambda = R.
    throughput is the backend's (DISPERSED_THROUGHPUT, 0.5)."""
    lambda_min_nm: float = 400.0   # SPAD Lambda sensitivity range
    lambda_max_nm: float = 950.0
    n_channels: int = 320          # SPAD Lambda: 320 x 1 pixels
    resolving_power: float | None = None
    throughput: float = DISPERSED_THROUGHPUT
    name: str = ""
    # wavelength frame of the channel grid: model tables are in vacuum
    # (NewEra); an "air" grid is converted before channel averaging
    # (sed.prepare_system; 0.14 nm at 500 nm, ~1.4 channels at R = 5000)
    frame: str = "vacuum"

    @classmethod
    def from_resolving_power(cls, R: float, lambda_min_nm: float = 400.0,
                             lambda_max_nm: float = 950.0,
                             throughput: float = DISPERSED_THROUGHPUT,
                             name: str = "") -> "Spectrograph":
        """Geometric channel edges e_k = lambda_min q^k with
        q = (2R + 1)/(2R - 1), so every channel has centre/width = R
        exactly; the last edge lands at or just beyond lambda_max."""
        q = (2.0 * R + 1.0) / (2.0 * R - 1.0)
        n = int(np.ceil(np.log(lambda_max_nm / lambda_min_nm) / np.log(q) - 1e-9))
        return cls(lambda_min_nm=lambda_min_nm, lambda_max_nm=lambda_min_nm * q**n,
                   n_channels=n, resolving_power=float(R), throughput=throughput,
                   name=name or f"R = {R:g} spectrograph")

    @property
    def channel_edges_nm(self) -> np.ndarray:
        if self.resolving_power is None:
            return np.linspace(self.lambda_min_nm, self.lambda_max_nm,
                               self.n_channels + 1)
        return self.lambda_min_nm * np.exp(
            np.arange(self.n_channels + 1)
            * np.log(self.lambda_max_nm / self.lambda_min_nm) / self.n_channels)

    @property
    def channel_centers_nm(self) -> np.ndarray:
        e = self.channel_edges_nm
        return 0.5 * (e[:-1] + e[1:])

    @property
    def channel_widths_nm(self) -> np.ndarray:
        return np.diff(self.channel_edges_nm)

    @property
    def channel_width_nm(self) -> float:
        """The common channel width; raises for a constant-R grid."""
        w = self.channel_widths_nm
        if not np.allclose(w, w[0], rtol=1e-9):
            raise ValueError("channel widths are not uniform (constant-R "
                             "spectrograph): use channel_widths_nm")
        return float(w[0])

    @property
    def is_uniform(self) -> bool:
        return self.resolving_power is None


# ---------------------------------------------------------------------------
# Photon budget
# ---------------------------------------------------------------------------
def coherence_time_s(wavelength_nm, filter_width_nm):
    """tau_c = lambda^2 / (c dlambda) for a rectangular passband."""
    lam = np.asarray(wavelength_nm, dtype=float) * 1e-9
    return lam**2 / (C_LIGHT * np.asarray(filter_width_nm, dtype=float) * 1e-9)


def photon_flux(mag_ab, wavelength_nm, filter_width_nm):
    """Source photon flux through the filter [photons / m^2 / s]."""
    lam = np.asarray(wavelength_nm, dtype=float) * 1e-9
    f_nu = AB_ZERO_FNU * 10.0 ** (-0.4 * np.asarray(mag_ab, dtype=float))
    nu = C_LIGHT / lam
    dnu = C_LIGHT * np.asarray(filter_width_nm, dtype=float) * 1e-9 / lam**2
    return f_nu / (H_PLANCK * nu) * dnu


def incident_rate(mag_ab, telescope: Telescope, detector: Detector,
                  obs: Observation):
    """Photon rate at the detector before dead time [cps], per stream:
    flux x area x telescope x backend throughput x PDE x polarization
    flux fraction."""
    _, frac, _, _ = polarization_streams(obs.polarization_mode)
    return (photon_flux(mag_ab, obs.wavelength_nm, obs.filter_width_nm)
            * telescope.area_m2 * telescope.throughput * obs.backend_throughput
            * detector.pde(obs.wavelength_nm) * frac)


def stellar_rate(mag_ab, telescope: Telescope, detector: Detector,
                 obs: Observation):
    """Detected stellar count rate per stream [cps], including dead time."""
    return detector.detected_rate(incident_rate(mag_ab, telescope, detector, obs))


def pair_sigma_s(detector1: Detector, detector2: Detector, obs: Observation):
    """Width of the pair kernel: detector jitters plus, optionally, the
    coherence-time broadening."""
    s2 = detector1.jitter_sigma_s**2 + detector2.jitter_sigma_s**2
    if obs.coherence_broadening:
        s2 = s2 + (COHERENCE_SIGMA_FACTOR
                   * coherence_time_s(obs.wavelength_nm, obs.filter_width_nm))**2
    return np.sqrt(s2)


# ---------------------------------------------------------------------------
# SNR
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SNRResult:
    snr: object            # total over polarization streams
    rate1_cps: object      # detected stellar rate per stream, telescope 1
    rate2_cps: object
    tau_c_s: object
    sigma_pair_s: object
    n_signal: object       # excess (signal) coincidences in T, per stream
    n_background: object   # accidental coincidences in the matched window
    vis2: object
    obs: Observation
    n_streams: int = 1
    dead_time_load: object = 0.0   # max over the two telescopes


def g2_snr(vis2, mag_ab, obs: Observation, *,
           telescope1: Telescope, telescope2: Telescope | None = None,
           detector1: Detector, detector2: Detector | None = None) -> SNRResult:
    """SNR of the g2 bump for one baseline (see module docstring);
    array-capable over channels when obs carries arrays."""
    telescope2 = telescope1 if telescope2 is None else telescope2
    detector2 = detector1 if detector2 is None else detector2
    n_streams, frac, p2, _ = polarization_streams(obs.polarization_mode)

    inc1 = incident_rate(mag_ab, telescope1, detector1, obs)
    inc2 = incident_rate(mag_ab, telescope2, detector2, obs)
    r1 = detector1.detected_rate(inc1)
    r2 = detector2.detected_rate(inc2)
    # the (unpolarized) sky reaches each stream with the same flux
    # fraction as the star: half of it per pbs stream, half through a
    # single polarizer
    b1 = r1 + detector1.dark_cps + obs.sky_cps * frac
    b2 = r2 + detector2.dark_cps + obs.sky_cps * frac

    tau_c = coherence_time_s(obs.wavelength_nm, obs.filter_width_nm)
    sigma_pair = pair_sigma_s(detector1, detector2, obs)

    n_sig = p2 * np.asarray(vis2, dtype=float) * tau_c * r1 * r2 * obs.t_int_s
    eff_window = 2.0 * np.sqrt(np.pi) * sigma_pair  # matched-filter width
    n_bkg = b1 * b2 * obs.t_int_s * eff_window
    snr = np.sqrt(n_streams) * n_sig / np.sqrt(n_bkg)
    load = np.maximum(detector1.dead_time_load(inc1), detector2.dead_time_load(inc2))
    f = (lambda a: float(a) if np.ndim(a) == 0 else a)
    return SNRResult(snr=f(snr), rate1_cps=f(r1), rate2_cps=f(r2),
                     tau_c_s=f(tau_c), sigma_pair_s=f(sigma_pair),
                     n_signal=f(n_sig), n_background=f(n_bkg), vis2=vis2,
                     obs=obs, n_streams=n_streams, dead_time_load=f(load))


def vis2_noise(mag_ab, obs: Observation, *,
               telescope1: Telescope, telescope2: Telescope | None = None,
               detector1: Detector, detector2: Detector | None = None):
    """1-sigma uncertainty of a |V|^2 measurement over obs.t_int_s (the
    noise-equivalent squared visibility): SNR = |V|^2 / vis2_noise."""
    r = g2_snr(1.0, mag_ab, obs, telescope1=telescope1, telescope2=telescope2,
               detector1=detector1, detector2=detector2)
    return 1.0 / r.snr


def _check_dead_time(load, where: str) -> None:
    lmax = float(np.max(load))
    if lmax > 1.0:
        warnings.warn(f"{where}: per-pixel dead-time load r tau_dead reaches "
                      f"{lmax:.1f} (> 1): the non-paralyzable model is "
                      f"unreliable there; spread the light over more pixels "
                      f"or a polarizing beamsplitter", stacklevel=3)


def readout_scale(detector: Detector, total_incident_cps: float) -> float:
    """Factor (<= 1) by which the rates must be attenuated to fit the
    detector's time-tag link; 1 for a correlator readout."""
    if detector.readout == "correlator" or detector.max_total_cps is None:
        return 1.0
    return min(1.0, detector.max_total_cps / max(total_incident_cps, 1e-300))


@dataclass(frozen=True)
class SpectralSNRResult:
    snr_total: float
    spectrograph: Spectrograph
    baseline_m: float
    channel_nm: np.ndarray
    snr: np.ndarray          # per channel
    rate_cps: np.ndarray     # detected stellar rate per channel per stream, telescope 1
    vis2: np.ndarray         # per channel
    mag_ab: np.ndarray       # per channel
    vis2_method: str = ""
    channel_widths_nm: np.ndarray = None
    total_rate_cps: tuple = (0.0, 0.0)   # detected per telescope (all streams)
    readout_limited: bool = False
    readout_scale: float = 1.0           # attenuation applied to fit the link
    dead_time_load_max: float = 0.0
    polarization_mode: str = "unpolarized"
    smeared: bool = False
    dimming: np.ndarray = None       # rendered eclipse dimming per channel (1 = none)


def spectral_g2_snr(system: BinarySystem, baseline_m: float,
                    spectrograph: Spectrograph, *,
                    t_int_s: float = 3600.0,
                    telescope1: Telescope,
                    telescope2: Telescope | None = None,
                    detector1: Detector,
                    detector2: Detector | None = None,
                    polarization_mode: str = "unpolarized",
                    sky_cps_per_channel: float = 0.0,
                    orbital_phase: float = 0.0,
                    vis2_method: str = "render",
                    grid: GridConfig | None = None,
                    chunk_size: int | None = None,
                    pupils=True,
                    n_pixels_per_channel: int = 1,
                    coherence_broadening: bool = True,
                    enforce_readout: bool = True) -> SpectralSNRResult:
    """Total g2 SNR with the source spectrum dispersed over the array.

    Each channel (n_pixels_per_channel pixels per telescope and stream,
    so dead time and dark counts are per channel) measures g2
    independently at its own wavelength, with the baseline along the
    projected separation axis at the requested orbital phase.
    SNR_total = sqrt(sum SNR_i^2).

    vis2_method:
      "render"   -- batched render + exact DFT (hbtsim.spectral): valid at
                    all phases including eclipses (~10 ms/channel on CPU).
      "analytic" -- hbt.binary_vis2_analytic: instant, but only valid OUT
                    of eclipse (raises during one).
    ("fft" is accepted as a deprecated alias of "render".)

    pupils: True (default) averages |V|^2 over the two telescope
    apertures (hbtsim.aperture), which is what the correlator measures
    and what suppresses the binary fringe for pupils comparable to the
    fringe period; None samples |V|^2 at a point; or a (d1, d2) pair /
    PupilQuadrature.

    Throughput = telescope x spectrograph.throughput x PDE.  With
    enforce_readout the rates of a time-tag detector are scaled to its
    link ceiling (readout_limited / readout_scale report it); a
    per-pixel dead-time load above 1 raises a warning.
    """
    from .aperture import resolve_pupils
    from .hbt import baseline_vectors_along_pa, binary_vis2_analytic
    from .orbit import positions_at

    telescope2 = telescope1 if telescope2 is None else telescope2
    detector2 = detector1 if detector2 is None else detector2
    det1 = replace(detector1, n_pixels=n_pixels_per_channel)
    det2 = replace(detector2, n_pixels=n_pixels_per_channel)

    pos = positions_at(system, orbital_phase)
    nm = spectrograph.channel_centers_nm
    widths = spectrograph.channel_widths_nm
    if grid is None:
        grid = GridConfig().fit_orbit(system)
    # model tables: Doppler-shift for the epoch and average over the channels
    from .sed import prepare_system
    system = prepare_system(system, spectrograph, pos)

    if vis2_method == "fft":
        warnings.warn("vis2_method='fft' is now 'render'", DeprecationWarning,
                      stacklevel=2)
        vis2_method = "render"
    from .aperture import fringe_period_m
    period = fringe_period_m(float(pos.rho), nm)
    quad = resolve_pupils(pupils, (telescope1.diameter_m, telescope2.diameter_m), period)
    dimming = np.ones(nm.size)
    if vis2_method == "render":
        from .spectral import eclipse_dimming, spectral_vis2
        v2, flux = spectral_vis2(pos, [baseline_m], nm, system, grid,
                                 chunk_size=chunk_size, pupils=quad,
                                 fringe_period_m=period, return_flux=True)
        vis2 = np.asarray(v2)[:, 0].astype(float)
        dimming = eclipse_dimming(flux, system, nm, grid)
    elif vis2_method == "analytic":
        if quad is None:
            vis2 = binary_vis2_analytic(baseline_m, nm, system, float(pos.rho))[:, 0]
        else:
            from .bispectrum import binary_vis_complex_analytic
            pts = quad.points(baseline_vectors_along_pa([baseline_m], float(pos.pa)))
            v = binary_vis_complex_analytic(pts.reshape(-1, 2), nm, system, pos)
            vis2 = quad.reduce(np.abs(v) ** 2)
    else:
        raise ValueError(f"unknown vis2_method {vis2_method!r} "
                         f"(expected 'render' or 'analytic')")

    # out-of-eclipse model magnitude, dimmed by the rendered eclipse
    mag = np.asarray(system_ab_mag(system, nm)) - 2.5 * np.log10(dimming)
    return _g2_budget(vis2, mag, spectrograph, baseline_m, t_int_s=t_int_s,
                      telescope1=telescope1, telescope2=telescope2, det1=det1, det2=det2,
                      polarization_mode=polarization_mode,
                      sky_cps_per_channel=sky_cps_per_channel,
                      coherence_broadening=coherence_broadening,
                      enforce_readout=enforce_readout, vis2_method=vis2_method,
                      smeared=quad is not None, dimming=dimming, caller="spectral_g2_snr")


def _g2_budget(vis2, mag, spectrograph: Spectrograph, baseline_m: float, *,
               t_int_s: float, telescope1: Telescope, telescope2: Telescope,
               det1: Detector, det2: Detector, polarization_mode: str = "unpolarized",
               sky_cps_per_channel: float = 0.0, coherence_broadening: bool = True,
               enforce_readout: bool = True, vis2_method: str = "", smeared: bool = False,
               dimming=None, channel_mask=None, caller: str = "spectral_g2_snr"
               ) -> SpectralSNRResult:
    """The photon-budget half of a multiplexed g2 measurement, shared by the
    binary (spectral_g2_snr) and single-star (single.spectral_g2_snr_single)
    paths: per-channel |V|^2 and AB magnitude in, SNR per channel out.
    channel_mask (bool per channel) tags only the selected channels: the
    link ceiling is applied to their total rate alone and the others
    contribute nothing."""
    nm = spectrograph.channel_centers_nm
    widths = spectrograph.channel_widths_nm
    mag = np.asarray(mag, dtype=float)
    mask = np.ones(nm.size, dtype=bool) if channel_mask is None else np.asarray(channel_mask, bool)
    obs = Observation(wavelength_nm=nm, filter_width_nm=widths, t_int_s=t_int_s,
                      sky_cps=sky_cps_per_channel,
                      polarization_mode=polarization_mode,
                      backend_throughput=spectrograph.throughput,
                      coherence_broadening=coherence_broadening)
    n_streams, _, _, _ = polarization_streams(polarization_mode)

    # readout ceiling: total incident rate over all tagged channels and streams
    scale = 1.0
    if enforce_readout:
        tot = [float(np.sum(np.asarray(incident_rate(mag, t, d, obs))[mask])) * n_streams
               for t, d in ((telescope1, det1), (telescope2, det2))]
        scale = min(readout_scale(det1, tot[0]), readout_scale(det2, tot[1]))
    mag_eff = mag - 2.5 * np.log10(scale) if scale < 1.0 else mag

    res = g2_snr(vis2, mag_eff, obs, telescope1=telescope1, telescope2=telescope2,
                 detector1=det1, detector2=det2)
    _check_dead_time(res.dead_time_load, caller)
    snr = np.where(mask, np.asarray(res.snr), 0.0)
    total = (float(np.sum(np.asarray(res.rate1_cps)[mask])) * n_streams,
             float(np.sum(np.asarray(res.rate2_cps)[mask])) * n_streams)
    return SpectralSNRResult(
        snr_total=float(np.sqrt(np.sum(snr**2))),
        spectrograph=spectrograph, baseline_m=baseline_m, channel_nm=nm,
        snr=snr, rate_cps=np.asarray(res.rate1_cps), vis2=np.asarray(vis2),
        mag_ab=mag, vis2_method=vis2_method, channel_widths_nm=widths,
        total_rate_cps=total, readout_limited=scale < 1.0, readout_scale=scale,
        dead_time_load_max=float(np.max(res.dead_time_load)),
        polarization_mode=polarization_mode, smeared=smeared,
        dimming=np.ones(nm.size) if dimming is None else dimming)


# ---------------------------------------------------------------------------
# Source model: out-of-eclipse magnitude of the binary at any wavelength
# ---------------------------------------------------------------------------


def model_ab_mag(system: BinarySystem, wavelength_nm):
    """Synthetic AB magnitude of the uneclipsed binary from the stars'
    surface fluxes (model SED tables, or pi B_lambda(T_eff) blackbodies):
    f_nu = sum_s F_s(lambda) theta_s^2 lambda^2 / c, theta_s = R_s/d (the drawn, outer radius for a
    spherical model: F is the flux at its outer boundary)."""
    lam_nm = np.asarray(wavelength_nm, dtype=float)
    lam = lam_nm * 1e-9
    f_nu = 0.0
    for star in (system.primary, system.secondary):
        theta_r = system.drawn_radius_mas(star) * MAS
        f_nu = f_nu + star.surface_flux(lam_nm) * theta_r**2 * lam**2 / C_LIGHT
    out = -2.5 * np.log10(f_nu / AB_ZERO_FNU)
    if getattr(system, "a_v", 0.0):
        from .sed import cardelli_extinction
        out = out + cardelli_extinction(lam_nm, system.a_v)
    return float(out) if np.ndim(out) == 0 else out


_blackbody_ab_mag = model_ab_mag   # backward-compatible name


def _has_sed_tables(system: BinarySystem) -> bool:
    return system.has_sed_tables


@lru_cache(maxsize=None)
def _anchor_offsets(system: BinarySystem):
    """(log10 lambda_nm, offset) at the anchor wavelengths, sorted."""
    lams, offs = [], []
    for lam_b, m_obs in system.mag_anchors:
        lams.append(np.log10(float(lam_b)))
        offs.append(m_obs - float(model_ab_mag(system, lam_b)))
    order = np.argsort(lams)
    return np.asarray(lams)[order], np.asarray(offs)[order]




def system_ab_mag(system: BinarySystem, wavelength_nm):
    """Apparent AB magnitude of the (uneclipsed) binary at one wavelength
    or an array.

    With model-atmosphere flux tables on both stars (hbtsim.sed) the
    magnitude is the model's own, F_s(lambda) (R_s/d)^2, and the
    observed anchors (params.BinarySystem.mag_anchors) only serve as a
    check (a warning if the model misses one by more than 0.2 mag).
    Otherwise it is the anchored blackbody: the synthetic blackbody
    magnitude corrected by the anchor offsets interpolated, and
    extrapolated, linearly in log lambda; outside the anchor bands
    (477-763 nm) that is an extrapolation and a warning says so."""
    lam_nm = np.asarray(wavelength_nm, dtype=float)
    loglam, offs = _anchor_offsets(system)
    if offs.size == 0:
        return model_ab_mag(system, lam_nm)        # unanchored model
    if _has_sed_tables(system):
        worst = float(np.max(np.abs(offs)))
        if worst > ANCHOR_CHECK_MAG:
            warnings.warn(f"{system.name}: the SED tables miss the observed "
                          f"anchor magnitudes by up to {worst:.2f} mag "
                          f"(radii, distance or third light?)", stacklevel=2)
        return model_ab_mag(system, lam_nm)
    x = np.log10(lam_nm)
    if offs.size >= 2:
        slope = (offs[-1] - offs[0]) / (loglam[-1] - loglam[0])
        offset = np.where(x < loglam[0], offs[0] + slope * (x - loglam[0]),
                          np.where(x > loglam[-1], offs[-1] + slope * (x - loglam[-1]),
                                   np.interp(x, loglam, offs)))
    else:
        offset = np.full_like(x, offs[0])
    if np.any(x < loglam[0] - 1e-12) or np.any(x > loglam[-1] + 1e-12):
        warnings.warn(f"{system.name}: anchored-blackbody magnitude "
                      f"extrapolated outside the anchor bands "
                      f"({10**loglam[0]:.0f}-{10**loglam[-1]:.0f} nm); "
                      f"attach an SED table for accurate rates", stacklevel=2)
    out = model_ab_mag(system, lam_nm) + offset
    return float(out) if np.ndim(out) == 0 else out
