"""Chromatic (line-core vs continuum) diameters of single A stars.

A Balmer-line core forms high in the atmosphere where the temperature
gradient is shallow, so it is less limb darkened than the continuum: at
fixed true diameter the uniform-disk-equivalent diameter theta_UD(lambda)
that a single-baseline |V|^2 returns is LARGER in the core.  The
measurement is differential across the spectrum: the absolute
diameter, the spherical-limb definition and the g2 calibration cancel;
what is tested is the model's centre-to-limb contrast between the
line- and continuum-forming layers.

Per channel: |V|^2 of the pupil-averaged NewEra disk (hbtsim.single),
inverted to theta_UD with the same smearing, and sigma(theta_UD) from
the multiplexed g2 budget.  The detection statistic is Asimov and
local to each line: the minimum chi^2 of a low-order polynomial in
lambda (degree `deg`, which absorbs the continuum chromaticity)
through the model theta_UD(lambda) over the line's core and wings plus
its neighbouring continuum (within LOCAL_CONT_NM of the wings);
significance = sqrt(chi^2_min), and the lines (independent layers)
add in quadrature.  A single polynomial across the whole band would
also count its own failure to follow three separate bumps.

Readout strategies (EON-SII's time-tag links carry ~1e9 cps; Sirius
delivers ~1e11): "link" tags every channel and attenuates to the link;
"subset" tags only the channels that fit at full rate -- continuum
references first (ref_frac of the link), then the channels with the
most signal per photon; "correlator" is an on-detector correlator with
no link ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .geometry import Site, hour_angle_window
from .single import (SingleStar, prepare_single, single_star_vis2, spectral_g2_snr_single,
                     ud_diameter_per_channel, ud_dtheta_dvis2)
from .snr import Detector, Observation, Spectrograph, Telescope, incident_rate, polarization_streams

BALMER_VAC_NM = {"Hdelta": 410.29, "Hgamma": 434.17, "Hbeta": 486.27}
CORE_HALF_NM = 0.12       # +-: the line core at R ~ 5000-7500
WING_HALF_NM = 8.0        # +-: beyond this the A-star Balmer wings are ~continuum
LOCAL_CONT_NM = 15.0      # continuum used for a line: up to this far beyond its wings


def line_masks(nm, lines=None, core_half_nm: float = CORE_HALF_NM,
               wing_half_nm: float = WING_HALF_NM) -> dict:
    """{line: {"core": mask, "wing": mask}, "continuum": mask} over the
    channel centres; wings exclude the core, the continuum is outside
    every window; all masks are disjoint."""
    nm = np.asarray(nm, dtype=float)
    lines = BALMER_VAC_NM if lines is None else lines
    out, any_win = {}, np.zeros(nm.size, bool)
    for name, lam0 in lines.items():
        d = np.abs(nm - lam0)
        if not np.any(d < wing_half_nm):
            continue
        core = d < core_half_nm
        wing = (d < wing_half_nm) & ~core & ~any_win
        core &= ~any_win
        out[name] = {"core": core, "wing": wing}
        any_win |= core | wing
    out["continuum"] = ~any_win
    return out


def local_continuum(nm, masks, line: str, lines=None,
                    local_cont_nm: float = LOCAL_CONT_NM) -> np.ndarray:
    """Continuum channels within local_cont_nm of the line's wings."""
    lam0 = (BALMER_VAC_NM if lines is None else lines)[line]
    d = np.abs(np.asarray(nm, dtype=float) - lam0)
    return masks["continuum"] & (d < WING_HALF_NM + local_cont_nm)


def _x(nm):
    nm = np.asarray(nm, dtype=float)
    mid = 0.5 * (nm.min() + nm.max())
    return nm / mid - 1.0


def continuum_fit(nm, y, sigma, mask, deg: int = 2) -> np.ndarray:
    """Weighted polynomial (degree deg in lambda) through the masked
    channels, evaluated on every channel."""
    x = _x(nm)
    m = np.asarray(mask, bool)
    c = np.polyfit(x[m], np.asarray(y)[m], deg, w=1.0 / np.asarray(sigma)[m])
    return np.polyval(c, x)


def asimov_significance(nm, theta, sigma, mask=None, deg: int = 2) -> float:
    """sqrt(min chi^2) of a degree-deg polynomial through theta(lambda) on
    the masked channels: how strongly the model rejects 'smooth'."""
    m = np.ones(len(nm), bool) if mask is None else np.asarray(mask, bool)
    if m.sum() <= deg + 1:
        return 0.0
    fit = continuum_fit(nm, theta, sigma, m, deg)
    r = (np.asarray(theta) - fit) / np.asarray(sigma)
    return float(np.sqrt(np.sum(r[m] ** 2)))


def select_channels_for_link(rate_cps, info, link_cps: float, continuum_mask, sigma,
                             ref_frac: float = 0.25, min_ref: int = 4,
                             ref_groups=None) -> np.ndarray:
    """Channels to tag so their summed rate fits the link at full rate:
    continuum references (smallest sigma first; round-robin over
    ref_groups, e.g. each line's local continuum, when given) up to
    ref_frac of the link (at least min_ref), then the rest greedily by
    info / rate."""
    rate = np.asarray(rate_cps, dtype=float)
    info = np.asarray(info, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cont = np.asarray(continuum_mask, bool)
    groups = [cont] if not ref_groups else [np.asarray(g, bool) & cont for g in ref_groups]
    queues = [list(np.flatnonzero(g)[np.argsort(sigma[g])]) for g in groups]
    tag = np.zeros(rate.size, bool)
    used = 0.0
    while any(queues):
        progressed = False
        for q in queues:
            while q and tag[q[0]]:
                q.pop(0)
            if not q:
                continue
            k = q[0]
            if (tag.sum() >= min_ref and used + rate[k] > ref_frac * link_cps) \
                    or used + rate[k] > link_cps:
                q.clear()
                continue
            tag[k] = True
            used += rate[k]
            q.pop(0)
            progressed = True
        if not progressed:
            break
    for k in np.argsort(-info / np.maximum(rate, 1e-300)):
        if tag[k] or info[k] <= 0.0:
            continue
        if used + rate[k] <= link_cps:
            tag[k] = True
            used += rate[k]
    return tag


@dataclass(frozen=True)
class ChromaticResult:
    target: str
    baseline_m: float
    channel_selection: str          # "all" | "subset"
    t_int_s: float
    nm: np.ndarray
    vis2: np.ndarray
    theta_ud: np.ndarray            # model, per channel [mas]
    sigma_theta: np.ndarray         # per channel in t_int_s [mas] (inf where untagged)
    continuum: np.ndarray           # smooth fit through the continuum channels [mas]
    tagged: np.ndarray              # channels read out
    readout_scale: float
    significance: float             # Asimov, lines in quadrature
    line_significance: dict = field(default_factory=dict)
    line_signal_pct: dict = field(default_factory=dict)   # core theta_UD / continuum - 1
    continuum_chromaticity_pct: float = 0.0               # continuum theta_UD across the band
    deg: int = 2
    polarization_mode: str = "unpolarized"

    def nights_to(self, n_sigma: float = 5.0) -> float:
        """Nights (of t_int_s each) to reach n_sigma."""
        return np.inf if self.significance <= 0 else (n_sigma / self.significance) ** 2


def night_seconds(target: SingleStar, site: Site, min_alt_deg: float = 30.0,
                  max_night_h: float = 8.0) -> float:
    """Time above min_alt_deg per night, capped at max_night_h."""
    h0, h1 = hour_angle_window(target.dec_deg, site.latitude_deg, min_alt_deg)
    return 3600.0 * min(h1 - h0, max_night_h)


def chromatic_signature(target: SingleStar, baseline_m: float,
                        spectrograph: Spectrograph, *, telescope: Telescope,
                        detector: Detector, site: Site, polarization_mode: str = "unpolarized",
                        t_int_s: float | None = None, channel_selection: str = "all",
                        deg: int = 1, pupils=True,
                        ref_frac: float = 0.25, lines=None) -> ChromaticResult:
    """The chromatic-diameter signature and its detectability in one
    night (t_int_s; default: the time above 30 deg at `site`, <= 8 h).

    channel_selection "all" tags every channel (a time-tag detector is
    attenuated to its link ceiling, detector.max_total_cps; a correlator
    detector has none), "subset" tags only the channels that fit the link
    at full rate: continuum references first (ref_frac of the link), then
    the channels with the most signal per photon."""
    if channel_selection not in ("all", "subset"):
        raise ValueError(f"channel_selection must be 'all' or 'subset', not {channel_selection!r}")
    t_int_s = night_seconds(target, site) if t_int_s is None else float(t_int_s)
    tgt = prepare_single(target, spectrograph)
    nm = spectrograph.channel_centers_nm
    pup = (telescope.diameter_m, telescope.diameter_m) if pupils is True else pupils
    v2 = single_star_vis2(tgt, baseline_m, nm, pup)[:, 0]
    th = ud_diameter_per_channel(v2, baseline_m, nm, pup)
    dth = np.abs(ud_dtheta_dvis2(th, baseline_m, nm, pup))
    masks = line_masks(nm, lines)
    cont = masks["continuum"]
    line_names = [k for k in masks if k != "continuum"]
    narrow = cont.sum() <= deg + 1
    if narrow:
        # a band inside one line's wings: every non-core channel anchors the fit
        cont = ~np.any([masks[k]["core"] for k in line_names], axis=0)
    kw = dict(spectrograph=spectrograph, t_int_s=t_int_s, telescope1=telescope,
              detector1=detector, polarization_mode=polarization_mode, pupils=pup)

    def sigma_of(res):
        with np.errstate(divide="ignore"):
            return np.where(res.snr > 0, dth * v2 / res.snr, np.inf)

    if channel_selection == "all":
        res = spectral_g2_snr_single(tgt, baseline_m, enforce_readout=True, **kw)
        tagged = np.ones(nm.size, bool)
    else:
        full = spectral_g2_snr_single(tgt, baseline_m, enforce_readout=False, **kw)
        s_full = sigma_of(full)
        info = np.zeros(nm.size)
        for name in line_names:
            win = masks[name]["core"] | masks[name]["wing"]
            loc = local_continuum(nm, masks, name, lines) if not narrow else cont
            if loc.sum() > deg + 1:
                smooth = continuum_fit(nm, th, s_full, loc, deg)
                info[win] = ((th[win] - smooth[win]) / s_full[win]) ** 2
        obs = Observation(wavelength_nm=nm, filter_width_nm=spectrograph.channel_widths_nm,
                          t_int_s=t_int_s, polarization_mode=polarization_mode,
                          backend_throughput=spectrograph.throughput)
        n_streams = polarization_streams(polarization_mode)[0]
        rate = np.asarray(incident_rate(tgt.ab_mag(nm), telescope, replace(detector, n_pixels=1),
                                        obs)) * n_streams
        link = detector.max_total_cps if detector.max_total_cps else np.inf
        groups = None if narrow else [local_continuum(nm, masks, n, lines) for n in line_names]
        tagged = select_channels_for_link(rate, info, link, cont, s_full, ref_frac,
                                          ref_groups=groups)
        res = spectral_g2_snr_single(tgt, baseline_m, enforce_readout=True,
                                     channel_mask=tagged, **kw)
    sig = sigma_of(res)
    sig_fit = np.where(np.isfinite(sig), sig, 1e30)
    fit_mask = tagged & cont
    d_glob = int(min(deg, max(0, int(fit_mask.sum()) - 2)))
    continuum = (continuum_fit(nm, th, sig_fit, fit_mask, d_glob)
                 if fit_mask.sum() > d_glob else np.full(nm.size, np.nan))
    per_line, signal = {}, {}
    d_eff = deg
    for name in line_names:
        m = masks[name]
        win = (m["core"] | m["wing"]) & tagged
        loc = (local_continuum(nm, masks, name, lines) if not narrow else cont) & tagged
        d_loc = int(min(deg, max(0, int(loc.sum()) - 2)))
        d_eff = min(d_eff, d_loc)
        per_line[name] = (asimov_significance(nm, th, sig_fit, win | loc, d_loc)
                          if win.any() and loc.sum() > d_loc else 0.0)
        loc_all = local_continuum(nm, masks, name, lines) if not narrow else cont
        if m["core"].any() and loc_all.sum() > 1:
            smooth = continuum_fit(nm, th, np.ones(nm.size), loc_all, min(deg, 1))
            signal[name] = 100.0 * (float(np.median(th[m["core"]]))
                                    / float(np.median(smooth[m["core"]])) - 1.0)
    total = float(np.sqrt(sum(v**2 for v in per_line.values())))
    smooth_all = continuum_fit(nm, th, np.ones(nm.size), cont, min(deg, 2))
    chroma = 100.0 * (smooth_all[-1] / smooth_all[0] - 1.0)
    return ChromaticResult(
        target=target.name, baseline_m=float(baseline_m), channel_selection=channel_selection,
        t_int_s=t_int_s,
        nm=nm, vis2=v2, theta_ud=th, sigma_theta=sig, continuum=continuum, tagged=tagged,
        readout_scale=res.readout_scale, significance=total, line_significance=per_line,
        line_signal_pct=signal, continuum_chromaticity_pct=float(chroma), deg=d_eff,
        polarization_mode=polarization_mode)


def optimal_baseline(target: SingleStar, spectrograph: Spectrograph, *, telescope: Telescope,
                     detector: Detector, site: Site, channel_selection: str = "all",
                     x_grid=np.arange(1.0, 3.61, 0.2), min_baseline_m: float = 0.0,
                     **kw) -> tuple:
    """(baseline, ChromaticResult) maximizing the total significance over
    first-lobe baselines x = pi theta B / lambda_mid in x_grid, never
    below min_baseline_m (e.g. the array generator's min_spacing_m)."""
    from .params import MAS
    lam_mid = 0.5 * (spectrograph.lambda_min_nm + spectrograph.lambda_max_nm) * 1e-9
    best = None
    for x in x_grid:
        b = max(float(x) * lam_mid / (np.pi * target.drawn_diameter_mas * MAS), min_baseline_m)
        r = chromatic_signature(target, b, spectrograph, telescope=telescope, detector=detector,
                                site=site, channel_selection=channel_selection, **kw)
        if best is None or r.significance > best[1].significance:
            best = (b, r)
    return best
