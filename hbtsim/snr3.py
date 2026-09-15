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
multiplexing are the levers.

Statistics.  The per-channel snr is the sensitivity at cos(phi_c) = 1.
Two ways of combining channels are reported:
  * snr_total     -- quadrature sum, the sensitivity to a common
                     amplitude if every channel had cos = 1 (an upper
                     bound that ignores the model's own signs);
  * snr_amplitude -- sqrt(sum (snr_ch cos phi_c,ch)^2), the sensitivity
                     to ONE global amplitude multiplying the model's
                     per-channel cos phi_c pattern (a template fit).
Neither is the precision of an individual channel's closure phase,
which is snr_ch alone (hundreds to thousands of times worse); the
paper's per-statistic framing is built on these in Phase 3.

Multi-hour observations: the (u, v) points rotate with hour angle, so
consecutive blocks see different baselines and different bispectra --
they cannot be averaged coherently.  track_g3_snr integrates block by
block along the uv track, applying the aperture smearing and the
orbital-phase advance, and combines the blocks as a template fit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np

from .bispectrum import Array, Triangle, spectral_triple
from .geometry import drift_loss, fringe_drift_cycles, hour_angle_blocks, hour_angle_window
from .orbit import SkyPositions, sky_positions
from .params import DAY, MAS, BinarySystem, GridConfig
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
    triple_amp: np.ndarray   # per channel (pupil-averaged if smeared)
    cos_phi_c: np.ndarray    # model closure-phase cosine per channel
    rates_cps: np.ndarray    # (n_channels, 3)
    mag_ab: np.ndarray
    vis_method: str = ""
    snr_amplitude: float = 0.0     # template-fit sensitivity, sqrt(sum (snr cos)^2)
    vis2_pairs: np.ndarray = None  # (n_channels, 3) pair-smeared |gamma_ij|^2
    smeared: bool = False
    t_int_s: float = 3600.0
    orbital_phase: float = 0.0


def spectral_g3_snr(system: BinarySystem, triangle: Triangle,
                    spectrograph: Spectrograph = Spectrograph(),
                    t_int_s: float = 3600.0, orbital_phase: float = 0.0,
                    vis_method: str = "analytic",
                    grid: GridConfig = GridConfig(),
                    chunk_size: int | None = None,
                    pol_factor_triple: float = POL_FACTOR_TRIPLE,
                    sky_cps_per_channel: float = 0.0,
                    pupils=True) -> SpectralSNR3Result:
    """Per-channel bispectrum sensitivity with the source dispersed over
    the array, plus the quadrature total and the template-amplitude
    sensitivity (module docstring).  pupils=True (default) uses the
    exact three-pupil average for the triangle's telescope diameters;
    None samples the bispectrum at a point.  vis_method "analytic"
    (out of eclipse) or "render" (any phase)."""
    if vis_method == "fft":
        warnings.warn("vis_method='fft' is now 'render'", DeprecationWarning,
                      stacklevel=2)
        vis_method = "render"
    nm = spectrograph.channel_centers_nm
    pos = SkyPositions(*(np.asarray(v) for v in
                         sky_positions(2 * np.pi * orbital_phase, system)))
    ts = spectral_triple(pos, triangle, nm, system, grid, method=vis_method,
                         pupils=pupils, chunk_size=chunk_size)
    triple_amp = ts.triple_amp
    cosphi = ts.cos_phi_c

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
    return SpectralSNR3Result(
        snr_total=float(np.sqrt(np.sum(snr**2))),
        triangle=triangle, spectrograph=spectrograph, channel_nm=nm, snr=snr,
        triple_amp=triple_amp, cos_phi_c=cosphi, rates_cps=rates, mag_ab=mags,
        vis_method=vis_method,
        snr_amplitude=float(np.sqrt(np.sum((snr * cosphi)**2))),
        vis2_pairs=ts.vis2_pairs, smeared=ts.smeared, t_int_s=t_int_s,
        orbital_phase=orbital_phase)


@dataclass(frozen=True)
class ArraySNR3Result:
    snr_total: float                 # quadrature over triangles and channels
    per_triangle: tuple              # SpectralSNR3Result per triangle
    triangle_names: tuple
    snr_amplitude: float = 0.0       # template fit over triangles and channels


def array_g3_snr(system: BinarySystem, array, **kw) -> ArraySNR3Result:
    """Bispectrum sensitivity of an N-telescope array: spectral_g3_snr on
    every triangle, combined in quadrature.  Each triangle's triple
    coincidences carry (largely) independent accidental noise even though
    triangles share telescopes, so quadrature is the right combination
    for detection sensitivity; only (N-1)(N-2)/2 of the C(N,3) closure
    PHASES are independent (for VLT: 3 of 4)."""
    tris = array.triangles()
    results = tuple(spectral_g3_snr(system, tri, **kw) for tri in tris)
    names = tuple(tri.name for tri in tris)
    total = float(np.sqrt(sum(r.snr_total**2 for r in results)))
    amp = float(np.sqrt(sum(r.snr_amplitude**2 for r in results)))
    return ArraySNR3Result(snr_total=total, per_triangle=results,
                           triangle_names=names, snr_amplitude=amp)


def array_time_to_cos_phi(system: BinarySystem, array,
                          target_dcos: float = 0.1, **kw) -> float:
    """Integration time [s] for the array-combined bispectrum sensitivity
    (snr_total) to reach sigma(cos phi_c) <= target_dcos."""
    ref = array_g3_snr(system, array, t_int_s=3600.0, **kw)
    return 3600.0 * (1.0 / target_dcos / ref.snr_total) ** 2


def time_to_cos_phi(system: BinarySystem, triangle: Triangle,
                    target_dcos: float = 0.1,
                    spectrograph: Spectrograph = Spectrograph(),
                    orbital_phase: float = 0.0,
                    vis_method: str = "analytic", **kw) -> float:
    """Integration time [s] for the multiplexed bispectrum (snr_total) to
    reach sigma(cos phi_c) <= target_dcos (SNR3 proportional to sqrt(T))."""
    ref = spectral_g3_snr(system, triangle, spectrograph=spectrograph,
                          t_int_s=3600.0, orbital_phase=orbital_phase,
                          vis_method=vis_method, **kw)
    needed = 1.0 / target_dcos
    return 3600.0 * (needed / ref.snr_total) ** 2


# ---------------------------------------------------------------------------
# Along the uv track
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrackBlock:
    hour_angle_h: float
    orbital_phase: float
    t_s: float
    snr_amplitude: float       # template-fit sensitivity of this block
    snr_total: float
    drift_cycles_max: float    # worst fringe drift over the block (lambda_min)
    drift_loss: float          # sinc contrast factor applied
    per_triangle: tuple        # SpectralSNR3Result per triangle


@dataclass(frozen=True)
class TrackResult:
    snr_amplitude: float       # template fit over blocks, triangles, channels
    snr_total: float           # quadrature (cos = 1) over the same
    t_total_s: float
    hour_angle_window_h: tuple
    blocks: tuple
    drift_flagged: bool

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)


def track_g3_snr(system: BinarySystem, array, spectrograph: Spectrograph = Spectrograph(),
                 *, block_minutes: float = 30.0,
                 hour_angle_window_h: tuple | None = None,
                 min_alt_deg: float = 30.0, phase0: float = 0.0,
                 vis_method: str = "analytic", pupils=True,
                 drift_warn_cycles: float = 0.125, **kw) -> TrackResult:
    """Bispectrum sensitivity accumulated over one night's uv track.

    The observable window (source above min_alt_deg from the array's
    site, or an explicit (H_start, H_end) in hours) is cut into blocks
    of block_minutes; each block is evaluated as a snapshot at its
    mid-hour-angle with the telescopes at their projected (u, v)
    positions, the orbital phase advanced from phase0, and the aperture
    smearing on.  Within a block the fringe drifts as the projected
    baselines rotate; the drift over the block (in cycles, at the
    shortest wavelength, worst baseline) attenuates the block by
    sinc(pi cycles) and is flagged when it exceeds drift_warn_cycles.
    Blocks are combined as a template fit (never coherently):
    snr_amplitude = sqrt(sum_blocks sum_triangles sum_channels
    (snr cos phi_c)^2).  With vis_method="analytic" a block inside an
    eclipse raises; use "render"."""
    if system.dec_deg is None or array.site is None:
        raise ValueError("track_g3_snr needs system.dec_deg and array.site")
    h0, h1 = (hour_angle_window(system.dec_deg, array.site.latitude_deg,
                                min_alt_deg)
              if hour_angle_window_h is None else hour_angle_window_h)
    mids = hour_angle_blocks(h0, h1, block_minutes)
    if mids.size == 0:
        return TrackResult(0.0, 0.0, 0.0, (h0, h1), (), False)
    block_h = (h1 - h0) / mids.size
    block_s = block_h * 3600.0
    lam_min = spectrograph.lambda_min_nm * 1e-9
    is_array = isinstance(array, Array)

    blocks = []
    amp2 = tot2 = 0.0
    flagged = False
    for H in mids:
        phase = phase0 + (H - h0) * 3600.0 / (system.period_days * DAY)
        pos = sky_positions(2 * np.pi * phase, system)
        sep = np.array([float(pos.x2 - pos.x1), float(pos.y2 - pos.y1)]) * MAS
        obj = array.projected(H, system.dec_deg)
        tris = obj.triangles() if is_array else [obj]
        b_start = (array.projected(H - block_h / 2, system.dec_deg))
        b_end = (array.projected(H + block_h / 2, system.dec_deg))
        t0 = b_start.triangles() if is_array else [b_start]
        t1 = b_end.triangles() if is_array else [b_end]
        drift = max(float(fringe_drift_cycles(a.baseline_vectors(),
                                              b.baseline_vectors(), sep,
                                              lam_min).max())
                    for a, b in zip(t0, t1))
        loss = float(drift_loss(drift))
        if drift > drift_warn_cycles:
            flagged = True
        res = tuple(spectral_g3_snr(system, tri, spectrograph=spectrograph,
                                    t_int_s=block_s, orbital_phase=phase,
                                    vis_method=vis_method, pupils=pupils, **kw)
                    for tri in tris)
        b_amp2 = loss**2 * sum(r.snr_amplitude**2 for r in res)
        b_tot2 = loss**2 * sum(r.snr_total**2 for r in res)
        amp2 += b_amp2
        tot2 += b_tot2
        blocks.append(TrackBlock(float(H), float(phase), block_s,
                                 float(np.sqrt(b_amp2)), float(np.sqrt(b_tot2)),
                                 drift, loss, res))
    if flagged:
        warnings.warn(f"fringe drift within a {block_minutes:.0f}-min block "
                      f"exceeds {drift_warn_cycles} cycles on some baseline "
                      f"(sinc loss applied); shorten block_minutes",
                      stacklevel=2)
    return TrackResult(float(np.sqrt(amp2)), float(np.sqrt(tot2)),
                       block_s * mids.size, (float(h0), float(h1)),
                       tuple(blocks), flagged)


def nights_to_precision(track: TrackResult, target_dcos: float = 0.1,
                        statistic: str = "amplitude") -> float:
    """Nights of the given track needed for the chosen combined statistic
    to reach a 1-sigma precision target_dcos on cos phi_c."""
    snr = {"amplitude": track.snr_amplitude,
           "total": track.snr_total}[statistic]
    if snr <= 0.0:
        return np.inf
    return (1.0 / target_dcos / snr) ** 2
