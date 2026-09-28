"""Signal-to-noise of the triple correlation (bispectrum / closure phase).

Extends the matched-filter photon budget of hbtsim.snr to three
telescopes.  Per spectral channel:

  signal:  N_sig = p3 . 2|g12 g23 g31| cos(phi_c) . tau_c^2 . R1 R2 R3 . T

  -- the lag-plane (tau1, tau2) integral of the triple term for a
  rectangular passband is exactly tau_c^2 = 1/dnu^2 (Parseval on the
  cubed unit-area spectrum), paralleling the pair case's tau_c.
  p3 = 1/4 for unpolarized light: two independent modes each carry
  I/2, and the triple term scales as 2 (1/2)^3 (the pair terms carry
  the familiar 1/2).  With a polarizing beamsplitter (two streams per
  telescope at half the rate, p3 = 1 each) the two stream triples add
  in quadrature: a factor 2 in SNR3 over unpolarized light at the same
  photon budget (hbtsim.snr.POLARIZATION_MODES).

  noise:   accidental triples at density b1 b2 b3 per unit lag^2,
  smeared by the detectors' jitters into a correlated 2D Gaussian with
  covariance Sigma = [[s1^2+s2^2, -s2^2], [-s2^2, s2^2+s3^2]] (telescope
  2 enters both lags with opposite signs), plus the coherence-time
  broadening sigma_c^2 [[1, -1/2], [-1/2, 1]]; the matched-filter
  effective area is A_2D = 4 pi sqrt(det Sigma) (equal jitters, no
  broadening: 4 pi sqrt(3) s^2).

  SNR3 = N_sig / sqrt(b1 b2 b3 . T . A_2D)

With R = n A alpha dnu this reduces to SNR3 ~ |ggg| (n A alpha)^{3/2}
(1/sigma) sqrt(T/dnu) -- Nunez & Domiciano de Souza 2015 eq. 8 -- and to
Zmija et al. 2025 eq. 12, SNR3 ~ (1/tau_e) sqrt(A^3 T / dlam).  The key
contrasts with the pair SNR: three-telescope sensitivity scales as
1/sigma_jitter (not 1/sqrt(sigma)), and as 1/sqrt(dlam) at fixed source
(not bandwidth-independent) -- narrow channels and heavy spectral
multiplexing are the levers.

Pair ridges.  The same triple histogram carries the pair correlations
as ridges (|g12|^2 along tau1 = 0 for every tau2, etc.) whose excess
inside the triple matched-filter window exceeds the triple term by

    ridge_ratio = sum_pairs (p2 / 2 p3) |g_ij|^2 A_2D
                  / (2 sqrt(pi) sigma_ij tau_c |g12 g23 g31|)

-- typically 50-750.  They are subtracted with the simultaneously
measured g2's, which is statistically cheap, but a fractional error
eps in the modeled pair kernel shape biases cos phi_c by eps x
ridge_ratio: required_kernel_accuracy = target / ridge_ratio is the
kernel-calibration requirement.

Statistics.  The per-channel snr is the sensitivity at cos(phi_c) = 1.
Three ways of combining channels are reported, and time_to_precision
inverts any of them:
  * "amplitude" -- snr_amplitude = sqrt(sum (snr_ch cos phi_c,ch)^2):
                   one global amplitude multiplying the model's
                   per-channel cos phi_c pattern (a template fit; this
                   is what "detecting the closure-phase signal" means);
  * "binned"    -- closure phases binned to resolving power R_bin
                   (channels within a bin combined in quadrature; the
                   median bin is quoted);
  * "channel"   -- one closure phase per channel (the median channel).
snr_total, the quadrature sum at cos = 1 everywhere, is kept as the
(unattainable) upper bound of "amplitude".

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
from .orbit import positions_at, sky_positions
from .params import DAY, MAS, BinarySystem, GridConfig
from .snr import (COHERENCE_SIGMA_FACTOR, Observation, Spectrograph, _check_dead_time,
                  coherence_time_s, incident_rate, polarization_streams,
                  readout_scale, system_ab_mag)

POL_FACTOR_TRIPLE = 0.25  # p3 for unpolarized light (see POLARIZATION_MODES)


def triple_window_s2(det1, det2, det3, obs: Observation | None = None):
    """Matched-filter effective area on the (tau1, tau2) lag plane,
    4 pi sqrt(det Sigma) [s^2], including the coherence broadening when
    obs is given with coherence_broadening=True (array-capable).

    The coherence term uses the Gaussian-equivalent covariance
    c2 [[1, -1/2], [-1/2, 1]]; the exact sinc^2 lag-plane covariance of
    a rectangular passband has a cross term 4/3 larger.  With
    sigma_c << sigma_jitter (0.1 nm channels: sigma_c ~ 0.3 ps against
    50 ps) the difference is < 1 % in the window."""
    s1, s2, s3 = (d.jitter_sigma_s for d in (det1, det2, det3))
    c2 = 0.0
    if obs is not None and obs.coherence_broadening:
        c2 = (COHERENCE_SIGMA_FACTOR
              * coherence_time_s(obs.wavelength_nm, obs.filter_width_nm))**2
    a = s1**2 + s2**2 + c2
    b = -(s2**2) - 0.5 * c2
    d = s2**2 + s3**2 + c2
    det_sigma = a * d - b * b
    return 4.0 * np.pi * np.sqrt(det_sigma)


@dataclass(frozen=True)
class SNR3Result:
    snr: object              # of the triple term (signed by cos phi_c), all streams
    rates_cps: tuple         # detected stellar rate per station per stream
    tau_c_s: object
    window_s2: object
    n_signal: object         # per stream
    n_background: object
    triple_amp: object       # |g12 g23 g31|
    cos_phi_c: object
    obs: Observation
    n_streams: int = 1
    ridge_ratio: object = None      # pair-ridge excess / triple excess
    dead_time_load: object = 0.0


def g3_snr(triple_amp, mag_ab, obs: Observation, triangle: Triangle, *,
           cos_phi_c=1.0, pair_vis2=None) -> SNR3Result:
    """SNR of the bispectrum term for one spectral channel (or arrays of
    channels).  Each station's light goes to one detector pixel per
    stream.  pair_vis2 (..., 3) = |g12|^2, |g23|^2, |g31|^2 enables the
    ridge_ratio."""
    dets = [replace(s.detector, n_pixels=1) for s in triangle.stations]
    n_streams, frac, p2, p3 = polarization_streams(obs.polarization_mode)
    inc = [incident_rate(mag_ab, s.telescope, d, obs)
           for s, d in zip(triangle.stations, dets)]
    rates = tuple(d.detected_rate(i) for d, i in zip(dets, inc))
    bg = [r + d.dark_cps + obs.sky_cps * frac for r, d in zip(rates, dets)]

    tau_c = coherence_time_s(obs.wavelength_nm, obs.filter_width_nm)
    window = triple_window_s2(*dets, obs)
    amp = np.asarray(triple_amp, dtype=float)

    n_sig = (p3 * 2.0 * amp * cos_phi_c
             * tau_c**2 * rates[0] * rates[1] * rates[2] * obs.t_int_s)
    n_bkg = bg[0] * bg[1] * bg[2] * obs.t_int_s * window
    snr = np.sqrt(n_streams) * n_sig / np.sqrt(n_bkg)

    ridge = None
    if pair_vis2 is not None:
        v2 = np.asarray(pair_vis2, dtype=float)
        pairs = ((0, 1), (1, 2), (2, 0))
        ridge = 0.0
        for k, (i, j) in enumerate(pairs):
            s_ij = np.sqrt(dets[i].jitter_sigma_s**2 + dets[j].jitter_sigma_s**2)
            ridge = ridge + (p2 / (2.0 * p3)) * v2[..., k] * window / (
                2.0 * np.sqrt(np.pi) * s_ij * tau_c * np.maximum(amp, 1e-300))
    load = np.max([d.dead_time_load(i) for d, i in zip(dets, inc)], axis=0)
    f = (lambda a: float(a) if np.ndim(a) == 0 else a)
    return SNR3Result(snr=f(snr), rates_cps=tuple(f(r) for r in rates),
                      tau_c_s=f(tau_c), window_s2=f(window), n_signal=f(n_sig),
                      n_background=f(n_bkg), triple_amp=triple_amp,
                      cos_phi_c=cos_phi_c, obs=obs, n_streams=n_streams,
                      ridge_ratio=None if ridge is None else f(ridge),
                      dead_time_load=f(load))


@dataclass(frozen=True)
class SpectralSNR3Result:
    snr_total: float         # quadrature sum at |cos phi_c| = 1 per channel
    triangle: Triangle
    spectrograph: Spectrograph
    channel_nm: np.ndarray
    snr: np.ndarray          # per channel, at cos = 1 (sensitivity)
    triple_amp: np.ndarray   # per channel (pupil-averaged if smeared)
    cos_phi_c: np.ndarray    # model closure-phase cosine per channel
    rates_cps: np.ndarray    # (n_channels, 3), per stream
    mag_ab: np.ndarray
    vis_method: str = ""
    snr_amplitude: float = 0.0     # template-fit sensitivity, sqrt(sum (snr cos)^2)
    vis2_pairs: np.ndarray = None  # (n_channels, 3) pair-smeared |gamma_ij|^2
    smeared: bool = False
    t_int_s: float = 3600.0
    orbital_phase: float = 0.0
    ridge_ratio: np.ndarray = None
    channel_widths_nm: np.ndarray = None
    total_rate_cps: tuple = ()     # detected, per station, all streams
    readout_limited: bool = False
    readout_scale: float = 1.0
    dead_time_load_max: float = 0.0
    polarization_mode: str = "unpolarized"
    dimming: np.ndarray = None       # rendered eclipse dimming per channel (1 = none)

    def required_kernel_accuracy(self, target_dcos: float = 0.1) -> np.ndarray:
        """Fractional accuracy of the pair-kernel model needed per
        channel to keep the ridge-subtraction bias on cos phi_c below
        target_dcos."""
        return target_dcos / self.ridge_ratio


def spectral_g3_snr(system: BinarySystem, triangle: Triangle,
                    spectrograph: Spectrograph = Spectrograph(),
                    t_int_s: float = 3600.0, orbital_phase: float = 0.0,
                    vis_method: str = "analytic",
                    grid: GridConfig | None = None,
                    chunk_size: int | None = None,
                    polarization_mode: str = "unpolarized",
                    sky_cps_per_channel: float = 0.0,
                    pupils=True, coherence_broadening: bool = True,
                    enforce_readout: bool = True) -> SpectralSNR3Result:
    """Per-channel bispectrum sensitivity with the source dispersed over
    the array, plus the combined statistics (module docstring).
    pupils=True (default) uses the exact three-pupil average for the
    triangle's telescope diameters; None samples the bispectrum at a
    point.  vis_method "analytic" (out of eclipse) or "render" (any
    phase).  Throughput = telescope x spectrograph.throughput x PDE;
    time-tag detectors are scaled to their link ceiling when
    enforce_readout."""
    if vis_method == "fft":
        warnings.warn("vis_method='fft' is now 'render'", DeprecationWarning,
                      stacklevel=2)
        vis_method = "render"
    nm = spectrograph.channel_centers_nm
    widths = spectrograph.channel_widths_nm
    pos = positions_at(system, orbital_phase)
    if grid is None:
        grid = GridConfig().fit_orbit(system)
    from .sed import prepare_system
    system = prepare_system(system, spectrograph, pos)
    ts = spectral_triple(pos, triangle, nm, system, grid, method=vis_method,
                         pupils=pupils, chunk_size=chunk_size)
    triple_amp = ts.triple_amp
    cosphi = ts.cos_phi_c

    dimming = np.ones(nm.size)
    if ts.flux is not None:
        from .spectral import eclipse_dimming
        dimming = eclipse_dimming(ts.flux, system, nm, grid)
    mags = np.asarray(system_ab_mag(system, nm)) - 2.5 * np.log10(dimming)
    obs = Observation(wavelength_nm=nm, filter_width_nm=widths, t_int_s=t_int_s,
                      sky_cps=sky_cps_per_channel,
                      polarization_mode=polarization_mode,
                      backend_throughput=spectrograph.throughput,
                      coherence_broadening=coherence_broadening)
    n_streams, _, _, _ = polarization_streams(polarization_mode)
    scale = 1.0
    if enforce_readout:
        dets = [replace(s.detector, n_pixels=1) for s in triangle.stations]
        tots = [float(np.sum(incident_rate(mags, s.telescope, d, obs))) * n_streams
                for s, d in zip(triangle.stations, dets)]
        scale = min(readout_scale(d, t) for d, t in zip(dets, tots))
    mag_eff = mags - 2.5 * np.log10(scale) if scale < 1.0 else mags

    r = g3_snr(triple_amp, mag_eff, obs, triangle, cos_phi_c=1.0,
               pair_vis2=ts.vis2_pairs)
    _check_dead_time(r.dead_time_load, "spectral_g3_snr")
    snr = np.asarray(r.snr)
    rates = np.stack(r.rates_cps, axis=-1)
    return SpectralSNR3Result(
        snr_total=float(np.sqrt(np.sum(snr**2))),
        triangle=triangle, spectrograph=spectrograph, channel_nm=nm, snr=snr,
        triple_amp=triple_amp, cos_phi_c=cosphi, rates_cps=rates, mag_ab=mags,
        vis_method=vis_method,
        snr_amplitude=float(np.sqrt(np.sum((snr * cosphi)**2))),
        vis2_pairs=ts.vis2_pairs, smeared=ts.smeared, t_int_s=t_int_s,
        orbital_phase=orbital_phase, ridge_ratio=np.asarray(r.ridge_ratio),
        channel_widths_nm=widths,
        total_rate_cps=tuple(float(np.sum(rates[:, k])) * n_streams
                             for k in range(3)),
        readout_limited=scale < 1.0, readout_scale=scale,
        dead_time_load_max=float(np.max(r.dead_time_load)),
        polarization_mode=polarization_mode, dimming=dimming)


def binned_closure_phase_snr(res: SpectralSNR3Result, R_bin: float = 100.0):
    """Closure-phase sensitivity per bin when the channels are binned to
    resolving power R_bin (geometric bins; channels combined in
    quadrature within a bin, assuming cos phi_c is constant across it).
    Returns (bin_centre_nm, snr_bin)."""
    edges = res.spectrograph.channel_edges_nm
    lo, hi = edges[0], edges[-1]
    q = (2.0 * R_bin + 1.0) / (2.0 * R_bin - 1.0)
    n = max(1, int(np.ceil(np.log(hi / lo) / np.log(q) - 1e-9)))
    bedges = lo * np.exp(np.arange(n + 1) * np.log(hi / lo) / n)
    idx = np.clip(np.searchsorted(bedges, res.channel_nm, side="right") - 1, 0, n - 1)
    s2 = np.bincount(idx, weights=res.snr**2, minlength=n)
    centres = 0.5 * (bedges[:-1] + bedges[1:])
    keep = np.bincount(idx, minlength=n) > 0     # bins narrower than a channel stay empty
    return centres[keep], np.sqrt(s2[keep])


def _statistic_snr(results, statistic: str, R_bin: float, aggregate: str) -> float:
    """Combined SNR of a statistic over a list of SpectralSNR3Result
    (one per triangle)."""
    agg = {"median": np.median, "max": np.max, "min": np.min}[aggregate]
    if statistic == "amplitude":
        return float(np.sqrt(sum(r.snr_amplitude**2 for r in results)))
    if statistic == "total":
        return float(np.sqrt(sum(r.snr_total**2 for r in results)))
    if statistic == "binned":
        per = [binned_closure_phase_snr(r, R_bin)[1] for r in results]
        # the triangles are independent measurements of the same
        # closure-phase bins: quadrature sum per bin, then the aggregate
        return float(agg(np.sqrt(np.sum(np.stack(per)**2, axis=0))))
    if statistic == "channel":
        per = np.stack([r.snr for r in results])
        # a closure phase per channel per triangle: the typical one
        return float(agg(per))
    raise ValueError(f"unknown statistic {statistic!r}")


def time_to_precision(system: BinarySystem, target_dcos: float = 0.1, *,
                      triangle: Triangle | None = None, array=None,
                      statistic: str = "amplitude", R_bin: float = 100.0,
                      aggregate: str = "median", t_ref_s: float = 3600.0,
                      spectrograph: Spectrograph = Spectrograph(),
                      orbital_phase: float = 0.0, vis_method: str = "analytic",
                      **kw) -> float:
    """Integration time [s] for the chosen statistic (see the module
    docstring: "amplitude", "binned", "channel", or the upper-bound
    "total") to reach a 1-sigma precision of target_dcos on cos phi_c,
    on one triangle or on every triangle of an array (independent
    accidental noise per triangle: quadrature; for "binned"/"channel"
    the typical bin/channel is quoted).  SNR3 grows as sqrt(T)."""
    if (triangle is None) == (array is None):
        raise ValueError("pass exactly one of triangle= or array=")
    tris = [triangle] if triangle is not None else array.triangles()
    results = [spectral_g3_snr(system, tri, spectrograph=spectrograph,
                               t_int_s=t_ref_s, orbital_phase=orbital_phase,
                               vis_method=vis_method, **kw) for tri in tris]
    snr = _statistic_snr(results, statistic, R_bin, aggregate)
    if snr <= 0.0:
        return np.inf
    return t_ref_s * (1.0 / target_dcos / snr) ** 2


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
    """Deprecated: time_to_precision(system, target, array=..., statistic="total")."""
    warnings.warn("array_time_to_cos_phi is deprecated; use time_to_precision"
                  "(..., array=array, statistic='total'|'amplitude')",
                  DeprecationWarning, stacklevel=2)
    return time_to_precision(system, target_dcos, array=array, statistic="total", **kw)


def time_to_cos_phi(system: BinarySystem, triangle: Triangle,
                    target_dcos: float = 0.1, **kw) -> float:
    """Deprecated: time_to_precision(system, target, triangle=..., statistic="total")."""
    warnings.warn("time_to_cos_phi is deprecated; use time_to_precision"
                  "(..., triangle=triangle, statistic='total'|'amplitude')",
                  DeprecationWarning, stacklevel=2)
    return time_to_precision(system, target_dcos, triangle=triangle,
                             statistic="total", **kw)


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

    def statistic_snr(self, statistic: str = "amplitude", R_bin: float = 100.0,
                      aggregate: str = "median") -> float:
        """Combined SNR of a statistic over the whole track: blocks add
        in quadrature (independent noise) with their drift losses."""
        if statistic in ("amplitude", "total"):
            return {"amplitude": self.snr_amplitude, "total": self.snr_total}[statistic]
        s2 = 0.0
        for b in self.blocks:
            s2 += b.drift_loss**2 * _statistic_snr(list(b.per_triangle), statistic,
                                                  R_bin, aggregate)**2
        return float(np.sqrt(s2))


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
    eclipse raises; use "render".  Extra keyword arguments go to
    spectral_g3_snr (polarization_mode, coherence_broadening, ...)."""
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
                        statistic: str = "amplitude", R_bin: float = 100.0,
                        aggregate: str = "median") -> float:
    """Nights of the given track needed for the chosen combined statistic
    to reach a 1-sigma precision target_dcos on cos phi_c."""
    snr = track.statistic_snr(statistic, R_bin, aggregate)
    if snr <= 0.0:
        return np.inf
    return (1.0 / target_dcos / snr) ** 2
