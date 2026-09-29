"""Monte Carlo of a spectrally multiplexed g2 diameter measurement.

Reproduces the kind of experiment in the EON-SII design study
(Schweizer et al. 2026, Sec. 6.2: Sirius B, 10 h at three zenith angles,
1000 channels, uniform-disk fit) with hbtsim's photon budget, to test
estimators against each other on simulated data rather than formulas:

  1. observing_blocks: hour angles / zenith angles on the site, the
     projected length of a ground baseline, time per block.
  2. expected_counts: per channel and block, detected rates (snr
     machinery: area, throughput, PDE, dead time, dark, polarization),
     coherence time, pair kernel width, true |V|^2 of the disk, and the
     expected coincidence-lag histogram (accidental floor + Gaussian
     excess p2 |V|^2 tau_c R1 R2 T).
  3. simulate_histograms: Poisson counts in bins of bin_ps.
  4. estimators: matched filter (Gaussian kernel, sideband background) and
     a coincidence box (capture-corrected or not) -> |V|^2 and its sigma.
  5. fit_ud: one-parameter uniform-disk (or linear-LD) fit of all channels
     and blocks -> theta and sigma(theta).
  6. run_mc: repeat, and report bias, scatter and pulls per estimator.

mode="gaussian" skips the histograms and draws |V|^2 directly with the
analytic matched-filter sigma, optionally with a correlation between
adjacent channels (the design study measured 5-7 %).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.special import j1

from .estimators import box_capture_fraction, optimal_box_half_width
from .geometry import TEIDE, Site, enu_to_uv
from .limbdark import visibility_ld_disk
from .params import MAS
from .snr import (EON_SII_TELESCOPE, EONSII_MCP_PMT, EONSII_SPECTROGRAPH, Detector,
                  Observation, Spectrograph, Telescope, coherence_time_s, incident_rate,
                  pair_sigma_s, polarization_streams)


@dataclass(frozen=True)
class MCConfig:
    theta_true_mas: float = 0.0285      # Sirius B (design study Table 2)
    mag_ab: float = 8.44
    ld_u: float = 0.0                   # truth: 0 = uniform disk
    dec_deg: float = -16.716
    site: Site = TEIDE
    ground_baseline_m: float = 1750.0
    ground_pa_deg: float = 90.0         # East-West
    zenith_angles_deg: tuple | None = (45.0, 52.5, 60.0)
    t_total_h: float = 10.0
    blocks_per_zenith: int = 1
    spectrograph: Spectrograph = EONSII_SPECTROGRAPH
    telescope: Telescope = EON_SII_TELESCOPE
    detector: Detector = EONSII_MCP_PMT
    polarization_mode: str = "unpolarized"
    extinction: str | None = None       # None | "izana": k(lambda) (X - 1) beyond the zenith budget
    channel_corr: float = 0.0           # adjacent-channel correlation (gaussian mode)
    bin_ps: float = 3.125
    lag_half_range_ps: float = 400.0
    sideband_sigma: float = 6.0
    # accidental level: "singles" = R1 R2 bin T from the singles counts (what a
    # correlator normalizes by; known to ~1e-4), "sideband" = mean of the
    # histogram beyond sideband_sigma (adds its own noise to every estimate)
    background: str = "singles"


@dataclass(frozen=True)
class Blocks:
    hour_angle_h: np.ndarray
    zenith_deg: np.ndarray
    airmass: np.ndarray
    t_s: np.ndarray
    b_proj_m: np.ndarray


def _hour_angle_for_zenith(z_deg, dec_deg, lat_deg):
    """Hour angles (h, <= 0: rising) at which the source sits at zenith
    angle z; raises when it never gets that high."""
    z_transit = abs(lat_deg - dec_deg)
    if np.any(np.asarray(z_deg) < z_transit - 0.1):
        raise ValueError(f"source at dec {dec_deg} never reaches zenith angle {z_deg} from "
                         f"lat {lat_deg} (transit at {z_transit:.2f} deg)")
    z = np.radians(np.maximum(np.asarray(z_deg, dtype=float), z_transit))
    d, p = np.radians(dec_deg), np.radians(lat_deg)
    c = (np.cos(z) - np.sin(p) * np.sin(d)) / (np.cos(p) * np.cos(d))
    return -np.degrees(np.arccos(np.clip(c, -1.0, 1.0))) / 15.0


def observing_blocks(cfg: MCConfig) -> Blocks:
    """Blocks at the requested zenith angles (equal time each; the first
    angle at or nearest transit), with the projected length of the ground
    baseline at each block's hour angle."""
    z = np.repeat(np.asarray(cfg.zenith_angles_deg, dtype=float), cfg.blocks_per_zenith)
    H = _hour_angle_for_zenith(z, cfg.dec_deg, cfg.site.latitude_deg)
    t = np.full(z.size, cfg.t_total_h * 3600.0 / z.size)
    pa = np.radians(cfg.ground_pa_deg)
    enu = np.array([cfg.ground_baseline_m * np.sin(pa), cfg.ground_baseline_m * np.cos(pa), 0.0])
    b = np.array([np.hypot(*enu_to_uv(enu, np.radians(15.0 * h), np.radians(cfg.dec_deg),
                                      cfg.site.latitude_rad)) for h in H])
    return Blocks(hour_angle_h=H, zenith_deg=z, airmass=1.0 / np.cos(np.radians(z)), t_s=t, b_proj_m=b)


def izana_extinction_mag(lam_nm, airmass):
    """Extinction beyond the zenith budget [mag]: Rayleigh at 2390 m
    (pressure 0.74 atm) plus 0.03 mag aerosol, times (X - 1)."""
    lam_um = np.asarray(lam_nm) * 1e-3
    tau_r = 0.0088 * lam_um**-4.05 * 0.74
    k = 1.086 * tau_r + 0.03
    return k * (np.asarray(airmass) - 1.0)


def ud_vis2(theta_mas, b_m, lam_nm, u=0.0):
    x = np.pi * theta_mas * MAS * np.asarray(b_m) / (np.asarray(lam_nm) * 1e-9)
    if u == 0.0:
        xs = np.where(x == 0, 1.0, x)
        return np.where(x == 0, 1.0, (2.0 * j1(xs) / xs) ** 2)
    return visibility_ld_disk(x, u) ** 2


@dataclass(frozen=True)
class Expected:
    nm: np.ndarray            # (n_ch,)
    rates: np.ndarray         # (n_ch, n_blk) detected per stream (both telescopes equal)
    background: np.ndarray    # (n_ch, n_blk) accidental rate R+dark per stream
    tau_c: np.ndarray         # (n_ch,)
    sigma_pair: np.ndarray    # (n_ch,)
    vis2_true: np.ndarray     # (n_ch, n_blk)
    t_s: np.ndarray           # (n_blk,)
    b_proj_m: np.ndarray      # (n_blk,)
    lags_s: np.ndarray        # (n_bins,) bin centres
    bin_s: float
    p2: float
    n_streams: int

    @property
    def signal_scale(self) -> np.ndarray:
        """Expected excess coincidences per unit |V|^2 (n_ch, n_blk), per stream."""
        return self.p2 * self.tau_c[:, None] * self.rates**2 * self.t_s[None, :]

    @property
    def bkg_per_bin(self) -> np.ndarray:
        return self.background**2 * self.t_s[None, :] * self.bin_s

    def kernel(self) -> np.ndarray:
        """Gaussian pair kernel integrated over each bin (n_ch, n_bins)."""
        s = self.sigma_pair[:, None]
        return (np.exp(-0.5 * (self.lags_s[None, :] / s) ** 2) / (np.sqrt(2 * np.pi) * s)) * self.bin_s

    def analytic_sigma_vis2(self) -> np.ndarray:
        """Matched-filter sigma(|V|^2) per channel and block (all streams)."""
        k = self.kernel()
        sumk2 = np.sum(k**2, axis=1)[:, None]
        return np.sqrt(self.bkg_per_bin / sumk2) / self.signal_scale / np.sqrt(self.n_streams)


def expected_counts(cfg: MCConfig, blocks: Blocks | None = None) -> Expected:
    blocks = observing_blocks(cfg) if blocks is None else blocks
    spec = cfg.spectrograph
    nm, w = spec.channel_centers_nm, spec.channel_widths_nm
    obs = Observation(wavelength_nm=nm, filter_width_nm=w, t_int_s=1.0,
                      polarization_mode=cfg.polarization_mode, backend_throughput=spec.throughput)
    det = replace(cfg.detector, n_pixels=1)
    n_streams, _, p2, _ = polarization_streams(cfg.polarization_mode)
    rates = np.empty((nm.size, blocks.t_s.size))
    for k in range(blocks.t_s.size):
        mag = np.full(nm.size, cfg.mag_ab, dtype=float)
        if cfg.extinction == "izana":
            mag = mag + izana_extinction_mag(nm, blocks.airmass[k])
        rates[:, k] = det.detected_rate(incident_rate(mag, cfg.telescope, det, obs))
    tau_c = coherence_time_s(nm, w)
    sig = np.asarray(pair_sigma_s(det, det, obs))
    n_half = int(round(cfg.lag_half_range_ps / cfg.bin_ps))
    lags = (np.arange(-n_half, n_half + 1)) * cfg.bin_ps * 1e-12
    vis2 = np.stack([ud_vis2(cfg.theta_true_mas, blocks.b_proj_m[k], nm, cfg.ld_u)
                     for k in range(blocks.t_s.size)], axis=1)
    return Expected(nm=nm, rates=rates, background=rates + det.dark_cps, tau_c=tau_c,
                    sigma_pair=sig, vis2_true=vis2, t_s=blocks.t_s, b_proj_m=blocks.b_proj_m,
                    lags_s=lags, bin_s=cfg.bin_ps * 1e-12, p2=p2, n_streams=n_streams)


def simulate_histograms(rng, exp: Expected, vis2=None) -> np.ndarray:
    """Poisson coincidence-lag histograms (n_streams, n_ch, n_blk, n_bins)."""
    vis2 = exp.vis2_true if vis2 is None else vis2
    mean = (exp.bkg_per_bin[..., None]
            + (exp.signal_scale * vis2)[..., None] * exp.kernel()[:, None, :])
    return rng.poisson(mean[None, ...], size=(exp.n_streams,) + mean.shape)


def _sideband(exp: Expected, sideband_sigma: float) -> np.ndarray:
    return np.abs(exp.lags_s)[None, :] > sideband_sigma * exp.sigma_pair[:, None]


def _background(counts, exp: Expected, background: str, sideband_sigma: float) -> np.ndarray:
    """Accidental coincidences per bin (n_streams, n_ch, n_blk)."""
    if background == "singles":
        return np.broadcast_to(exp.bkg_per_bin, counts.shape[:3])
    if background != "sideband":
        raise ValueError(f"background must be 'singles' or 'sideband', not {background!r}")
    sb = _sideband(exp, sideband_sigma)
    return np.array([[counts[s, c][:, sb[c]].mean(axis=1) for c in range(exp.nm.size)]
                     for s in range(counts.shape[0])])


def matched_filter_estimate(counts, exp: Expected, sideband_sigma: float = 6.0,
                            background: str = "singles"):
    """|V|^2 per channel and block from the Gaussian matched filter;
    returns (vis2_hat, sigma) combining the streams.  The streams are
    weighted by the model sigma (expected accidentals), never by the
    realization's own background, which would correlate weights with
    the estimates."""
    k = exp.kernel()                                    # (n_ch, n_bins)
    counts = np.asarray(counts, dtype=float)
    bkg = _background(counts, exp, background, sideband_sigma)
    exc = counts - bkg[..., None]
    amp = np.einsum("scbj,cj->scb", exc, k) / np.sum(k**2, axis=1)[None, :, None]
    v2 = amp / exp.signal_scale[None, ...]
    sigma = np.sqrt(exp.bkg_per_bin / np.sum(k**2, axis=1)[:, None]) / exp.signal_scale
    return v2.mean(axis=0), sigma / np.sqrt(counts.shape[0])


def box_estimate(counts, exp: Expected, half_width_s: float, correct_capture: bool = True,
                 sideband_sigma: float = 6.0, background: str = "singles"):
    """|V|^2 from the excess inside |tau| <= half_width (optionally divided
    by the Gaussian capture fraction); returns (vis2_hat, sigma), sigma
    from the model accidentals."""
    inside = np.abs(exp.lags_s) <= half_width_s
    n_in = int(inside.sum())
    counts = np.asarray(counts, dtype=float)
    bkg = _background(counts, exp, background, sideband_sigma)
    frac = box_capture_fraction(half_width_s, exp.sigma_pair) if correct_capture else np.ones(exp.nm.size)
    scale = frac[:, None] * exp.signal_scale                          # (n_ch, n_blk)
    exc = counts[..., inside].sum(axis=-1) - bkg * n_in               # (n_streams, n_ch, n_blk)
    sigma = np.sqrt(exp.bkg_per_bin * n_in) / scale
    return (exc / scale[None, ...]).mean(axis=0), sigma / np.sqrt(counts.shape[0])


def fit_ud(vis2_hat, sigma, b_proj_m, nm, theta0_mas, ld_u: float = 0.0, n_iter: int = 30):
    """Weighted least-squares diameter from all channels and blocks
    (1-D Gauss-Newton); returns (theta_mas, sigma_theta_mas).  V^2 is even
    in theta, so the fit runs on a signed theta and reports |theta|; at
    low S/N a realization can legitimately land near theta = 0 (V^2 ~ 1),
    where the derivative step is floored at 1e-3 theta0."""
    theta = float(theta0_mas)
    b = np.asarray(b_proj_m)[None, :]
    lam = np.asarray(nm)[:, None]
    w = 1.0 / np.asarray(sigma) ** 2

    def deriv(t):
        h = 1e-4 * max(abs(t), 1e-3 * theta0_mas)
        return (ud_vis2(t + h, b, lam, ld_u) - ud_vis2(t - h, b, lam, ld_u)) / (2 * h)

    for _ in range(n_iter):
        d = deriv(theta)
        step = np.sum(w * d * (vis2_hat - ud_vis2(theta, b, lam, ld_u))) / np.sum(w * d * d)
        step = float(np.clip(step, -0.5 * theta0_mas, 0.5 * theta0_mas))   # damp wild steps
        theta += step
        if abs(step) < 1e-10 * theta0_mas:
            break
    d = deriv(theta)
    return abs(theta), float(1.0 / np.sqrt(np.sum(w * d * d)))


@dataclass(frozen=True)
class EstimatorStats:
    theta_mean: float
    theta_std: float
    sigma_pred: float          # mean fitted sigma(theta)
    pull_std: float
    bias_frac: float           # (mean - truth)/truth
    precision_frac: float      # std/truth


@dataclass(frozen=True)
class MCResult:
    config: MCConfig
    n_real: int
    stats: dict = field(default_factory=dict)      # estimator -> EstimatorStats
    analytic_sigma_theta: float = 0.0              # from the analytic matched-filter sigma


def _correlated_noise(rng, sigma, rho):
    """Gaussian noise with adjacent-channel correlation rho along axis 0."""
    n = sigma.shape[0]
    if rho == 0.0:
        return rng.standard_normal(sigma.shape) * sigma
    cov = np.eye(n) + rho * (np.eye(n, k=1) + np.eye(n, k=-1))
    L = np.linalg.cholesky(cov)
    z = L @ rng.standard_normal(sigma.shape)
    return z * sigma


def run_mc(cfg: MCConfig, n_real: int = 200, estimators=("matched", "box_opt", "box_sigma_raw"),
           seed: int = 0, mode: str = "poisson") -> MCResult:
    """Repeat the simulated observation n_real times; fit theta with each
    estimator.  Estimators: 'matched'; 'box_opt' (1.40 sigma,
    capture-corrected); 'box_sigma' / 'box_sigma_raw' (half-width sigma,
    corrected / uncorrected); 'box_tdc' / 'box_tdc_raw' (half-width one
    3.125 ps bin).  mode='gaussian' draws |V|^2 directly (matched filter
    only), with cfg.channel_corr."""
    rng = np.random.default_rng(seed)
    exp = expected_counts(cfg)
    truth = cfg.theta_true_mas
    s_an = exp.analytic_sigma_vis2()
    _, sig_theta_an = fit_ud(exp.vis2_true, s_an, exp.b_proj_m, exp.nm, truth, cfg.ld_u)
    sig_ref = float(np.median(exp.sigma_pair))
    widths = {"box_opt": (optimal_box_half_width(sig_ref), True),
              "box_sigma": (sig_ref, True), "box_sigma_raw": (sig_ref, False),
              "box_tdc": (cfg.bin_ps * 1e-12, True), "box_tdc_raw": (cfg.bin_ps * 1e-12, False)}
    if mode == "gaussian":
        estimators = ("matched",)
    thetas = {e: [] for e in estimators}
    sigmas = {e: [] for e in estimators}
    for _ in range(n_real):
        if mode == "gaussian":
            v2 = exp.vis2_true + _correlated_noise(rng, s_an, cfg.channel_corr)
            th, st = fit_ud(v2, s_an, exp.b_proj_m, exp.nm, truth, cfg.ld_u)
            thetas["matched"].append(th); sigmas["matched"].append(st)
            continue
        counts = simulate_histograms(rng, exp)
        for e in estimators:
            if e == "matched":
                v2, s = matched_filter_estimate(counts, exp, cfg.sideband_sigma, cfg.background)
            else:
                hw, corr = widths[e]
                v2, s = box_estimate(counts, exp, hw, corr, cfg.sideband_sigma, cfg.background)
            th, st = fit_ud(v2, s, exp.b_proj_m, exp.nm, truth, cfg.ld_u)
            thetas[e].append(th); sigmas[e].append(st)
    stats = {}
    for e in estimators:
        t = np.array(thetas[e]); s = np.array(sigmas[e])
        stats[e] = EstimatorStats(theta_mean=float(t.mean()), theta_std=float(t.std(ddof=1)),
                                  sigma_pred=float(s.mean()),
                                  pull_std=float(np.std((t - truth) / s, ddof=1)),
                                  bias_frac=float(t.mean() / truth - 1.0),
                                  precision_frac=float(t.std(ddof=1) / truth))
    return MCResult(config=cfg, n_real=n_real, stats=stats, analytic_sigma_theta=sig_theta_an)
