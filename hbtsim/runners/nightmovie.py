"""Runner `nightmovie`: one night of a single star on a two-telescope array
as a movie -- the uv track, the measurements walking along the disk's
|V|^2(B) curve through its null, and the coincidence histogram building up.

Campaign fields: target (a single star), instrument.array (two stations),
one backend (filter + detector), night.block_minutes (the frame step),
options:
  fps, dpi                      movie encoding (default 12, 110)
  theta_alternatives_pct        extra |V|^2 curves at theta x (1 + p/100) (default [-5, 5])
  seed                          the noise realization
  display_bin_blocks            blocks averaged per displayed |V|^2 point (default 6: 30 min of 5-min blocks)
  histogram                     {bin_ps, lag_half_range_ps} of the coincidence panel, which shows the
                                excess over the accidentals smoothed with the matched filter, in units
                                of its shot noise, so the g2 bump rises out of the noise over the night
  nights                        optional: a second act accumulates this many nights (each a new
                                noise realization of the same track): the binned points average
                                down, the g2 bump rises out of the matched-filter noise, and the
                                diameter's error band narrows
  pulsation                     optional {period_days, amplitude_frac, phase0, fold_nights,
                                phase_bins}: theta(t) = theta0 (1 + A sin(2 pi (t - t0)/P))
                                breathes along the night, and a second act folds
                                fold_nights nights on pulsation phase to show the
                                diameter curve emerging (beta Cep).
Outputs <name>.mp4 (frames as PNG when ffmpeg is missing or opts.figures is off),
and results.json with the per-block numbers.
"""

from __future__ import annotations

import shutil
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from hbtsim.bispectrum import Array
from hbtsim.geometry import hour_angle_blocks, hour_angle_window
from hbtsim.params import MAS
from hbtsim.single import prepare_single, single_star_vis2
from hbtsim.snr import (Observation, coherence_time_s, g2_snr, incident_rate, pair_sigma_s,
                        polarization_streams, readout_scale)


def _night(array, dec_deg, block_minutes, min_alt_deg):
    h0, h1 = hour_angle_window(dec_deg, array.site.latitude_deg, min_alt_deg)
    mids = hour_angle_blocks(h0, h1, block_minutes)
    if mids.size == 0:
        raise SystemExit("the target never rises above the altitude limit from this site")
    block_s = (h1 - h0) / mids.size * 3600.0
    bvec = np.array([array.projected(float(H), dec_deg).pairs()[0][2] for H in mids])
    alt = np.degrees(np.arcsin(np.sin(np.radians(array.site.latitude_deg)) * np.sin(np.radians(dec_deg))
                               + np.cos(np.radians(array.site.latitude_deg)) * np.cos(np.radians(dec_deg))
                               * np.cos(np.radians(mids * 15.0))))
    return mids, block_s, bvec, alt


def _budget(target, lam, w, block_s, t1, t2, det, pol, throughput):
    """(sigma of |V|^2 per block, rate per telescope, n_sig per unit |V|^2, b1 b2,
    tau_c, sigma_pair) for one block at the star's magnitude."""
    mag = float(target.ab_mag(lam))
    obs = Observation(wavelength_nm=lam, filter_width_nm=w, t_int_s=block_s, polarization_mode=pol,
                      backend_throughput=throughput)
    n_streams = polarization_streams(pol)[0]
    r1 = float(incident_rate(mag, t1, det, obs)) * n_streams
    r2 = float(incident_rate(mag, t2, det, obs)) * n_streams
    scale = min(readout_scale(det, r1), readout_scale(det, r2))
    mag_eff = mag - 2.5 * np.log10(scale) if scale < 1.0 else mag
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = g2_snr(1.0, mag_eff, obs, telescope1=t1, telescope2=t2, detector1=det)
    sigma_v2 = 1.0 / float(r.snr)
    b1 = float(r.rate1_cps) + det.dark_cps
    b2 = float(r.rate2_cps) + det.dark_cps
    return dict(sigma_v2=sigma_v2, rate_cps=max(r1, r2), n_sig_unit=float(r.n_signal), b1b2=b1 * b2,
                tau_c=float(coherence_time_s(lam, w)), sigma_pair=float(pair_sigma_s(det, det, obs)),
                readout_scale=scale, r1=float(r.rate1_cps), r2=float(r.rate2_cps))


def run(campaign, cat, opts, out_dir: Path) -> dict:
    target = campaign.target
    if not hasattr(target, "theta_ld_mas"):
        raise SystemExit("runner nightmovie needs a single-star target")
    array = campaign.array
    if array is None or len(array.stations) != 2:
        raise SystemExit("runner nightmovie needs a two-station array (instrument.array)")
    if len(campaign.backends) != 1:
        raise SystemExit("runner nightmovie takes exactly one backend (one filter, one detector)")
    b = campaign.backends[0]
    spec, det, pol = b.spectrograph, b.detector, b.polarization_mode
    arr = Array(tuple(replace(s, detector=det) for s in array.stations), array.site)
    t1, t2 = arr.stations[0].telescope, arr.stations[1].telescope
    opt = campaign.option
    block_minutes = float(opt("night.block_minutes", 5.0))
    min_alt = float(opt("night.min_alt_deg", 30.0))
    lam, w = float(spec.channel_centers_nm[0]), float(spec.channel_widths_nm[0])
    rng = np.random.default_rng(int(opt("options.seed", 7)))
    alts = list(opt("options.theta_alternatives_pct", [-5.0, 5.0]))
    puls = opt("options.pulsation")
    tgt = prepare_single(target, spec)
    theta0 = float(tgt.theta_ld_mas)
    pupils = (t1.diameter_m, t2.diameter_m)

    mids, block_s, bvec, alt = _night(arr, target.dec_deg, block_minutes, min_alt)
    blen = np.hypot(bvec[:, 0], bvec[:, 1])
    t_h = (mids - mids[0]) * 1.0                     # hours since the first block (sidereal ~ solar here)
    if puls:
        P_h = float(puls["period_days"]) * 24.0
        theta_t = theta0 * (1.0 + float(puls["amplitude_frac"])
                            * np.sin(2 * np.pi * (t_h / P_h + float(puls.get("phase0", 0.0)))))
    else:
        theta_t = np.full(mids.size, theta0)
    bud = _budget(tgt, lam, w, block_s, t1, t2, det, pol, spec.throughput)
    v2_true = np.array([float(single_star_vis2(replace(tgt, theta_ld_mas=th), float(bl), lam, pupils)[0, 0])
                        for th, bl in zip(theta_t, blen)])
    v2_meas = v2_true + rng.normal(0.0, bud["sigma_v2"], size=v2_true.size)
    # the model curves
    B_grid = np.linspace(max(50.0, 0.5 * blen.min()), 1.15 * blen.max(), 400)
    curves = {0.0: np.array([float(single_star_vis2(tgt, float(bb), lam, pupils)[0, 0]) for bb in B_grid])}
    for p in alts:
        curves[p] = np.array([float(single_star_vis2(replace(tgt, theta_ld_mas=theta0 * (1 + p / 100.0)),
                                                     float(bb), lam, pupils)[0, 0]) for bb in B_grid])
    null_m = 1.22 * lam * 1e-9 / (tgt.drawn_diameter_mas * MAS)
    # the coincidence histogram per block
    hbin = float(opt("options.histogram.bin_ps", 3.125)) * 1e-12
    half = float(opt("options.histogram.lag_half_range_ps", 400.0)) * 1e-12
    lags = np.arange(-int(round(half / hbin)), int(round(half / hbin)) + 1) * hbin
    kern = np.exp(-0.5 * (lags / bud["sigma_pair"]) ** 2) / (np.sqrt(2 * np.pi) * bud["sigma_pair"]) * hbin
    acc = bud["b1b2"] * block_s * hbin                      # accidentals per bin per block
    hist = np.zeros_like(lags)
    hist_blocks = []
    for v2 in v2_true:
        expect = acc * (1.0 + bud["n_sig_unit"] * v2 / (bud["b1b2"] * block_s) * (kern / hbin))
        hist = hist + rng.poisson(expect)
        hist_blocks.append(hist.copy())
    hist_blocks = np.array(hist_blocks)
    # matched-filter view: excess over the accidentals, smoothed with the normalized kernel, in sigma units
    kern_n = kern / kern.sum()
    def mf(h, T):
        excess = h - acc * T
        sm = np.convolve(excess, kern_n[::-1], mode="same")
        noise = np.sqrt(acc * T * np.sum(kern_n ** 2))        # shot noise of the weighted sum
        return sm / noise
    mf_blocks = np.array([mf(hist_blocks[k], k + 1) for k in range(mids.size)])
    # the expected matched-filter bump after all blocks (in the same units)
    mf_expect = (bud["n_sig_unit"] * v2_true.sum() * np.convolve(kern, kern_n[::-1], mode="same")
                 / np.sqrt(acc * mids.size * np.sum(kern_n ** 2)))
    nbin_show = max(1, int(opt("options.display_bin_blocks", 6)))

    # Fisher precision of theta from the night (and per pulsation-phase bin for act two)
    dth = 0.01 * theta0
    dv = np.array([(float(single_star_vis2(replace(tgt, theta_ld_mas=theta0 + dth), float(bl), lam, pupils)[0, 0])
                    - float(single_star_vis2(replace(tgt, theta_ld_mas=theta0 - dth), float(bl), lam, pupils)[0, 0]))
                   / (2 * dth) for bl in blen])
    fisher_blocks = (dv / bud["sigma_v2"]) ** 2
    sig_theta_night = 1.0 / np.sqrt(fisher_blocks.sum())
    nights_5pct = (sig_theta_night / (0.05 * theta0)) ** 2

    print(f"=== {target.name}: theta_LD {theta0:.3f} mas, V {target.v_mag:.2f}, {b.name} on "
          f"{arr.stations[0].name}+{arr.stations[1].name} ===")
    print(f"  {mids.size} blocks of {block_minutes:g} min above {min_alt:g} deg; B {blen.min():.0f}-{blen.max():.0f} m, "
          f"first null at {null_m:.0f} m; |V|^2 {v2_true.min():.3g}-{v2_true.max():.3g}")
    print(f"  rates {bud['r1']:.2e} / {bud['r2']:.2e} cps, tau_c {bud['tau_c'] * 1e12:.2f} ps, pair sigma "
          f"{bud['sigma_pair'] * 1e12:.0f} ps, sigma(|V|^2) per block {bud['sigma_v2']:.3g}"
          f"{' READOUT-LIMITED x%.2f' % bud['readout_scale'] if bud['readout_scale'] < 1 else ''}")
    print(f"  one night: sigma(theta)/theta = {sig_theta_night / theta0:.3f} -> {nights_5pct:.2g} nights to 5 %")

    fold = None
    if puls:
        n_nights = int(puls.get("fold_nights", 40))
        nb = int(puls.get("phase_bins", 8))
        P_h = float(puls["period_days"]) * 24.0
        A = float(puls["amplitude_frac"])
        # each night starts at a random pulsation phase; blocks fall into phase bins
        phase_bin_fisher = np.zeros(nb)
        est_sum = np.zeros(nb)        # Fisher-weighted theta estimates
        per_night = []
        for n in range(n_nights):
            ph0 = rng.uniform()
            ph = np.mod(t_h / P_h + ph0, 1.0)
            th_n = theta0 * (1.0 + A * np.sin(2 * np.pi * ph))
            idx = np.minimum((ph * nb).astype(int), nb - 1)
            for k, bl in enumerate(blen):
                f = fisher_blocks[k]
                # a noisy theta estimate from this block: true theta + noise of 1/sqrt(f)
                est = th_n[k] + rng.normal(0.0, 1.0 / np.sqrt(f)) if f > 0 else th_n[k]
                phase_bin_fisher[idx[k]] += f
                est_sum[idx[k]] += f * est
            with np.errstate(invalid="ignore", divide="ignore"):
                theta_bins = np.where(phase_bin_fisher > 0, est_sum / phase_bin_fisher, np.nan)
                sig_bins = np.where(phase_bin_fisher > 0, 1.0 / np.sqrt(phase_bin_fisher), np.nan)
            per_night.append((theta_bins.copy(), sig_bins.copy()))
        fold = dict(n_nights=n_nights, phase_bins=nb, period_h=P_h, amplitude_frac=A, per_night=per_night)
        last_sig = np.nanmedian(per_night[-1][1]) / theta0
        print(f"  pulsation: P {P_h:.2f} h, amplitude {100 * A:.1f} %; after {n_nights} nights the "
              f"phase-binned diameter has sigma/theta {last_sig:.4f} per bin "
              f"({'resolves' if last_sig < A / 2 else 'does not resolve'} the {100 * A:.1f} % breathing)")

    multi = None
    n_more = int(opt("options.nights", 0) or 0)
    if n_more > 1 and not puls:
        v2_nights = [v2_meas]
        hist_tot = hist_blocks[-1].copy()
        mf_nights = [mf_blocks[-1]]
        th_est, th_sig = [], []
        f_night = float(fisher_blocks.sum())
        for j in range(1, n_more):
            v2_nights.append(v2_true + rng.normal(0.0, bud["sigma_v2"], size=v2_true.size))
            h = np.zeros_like(lags)
            for v2 in v2_true:
                expect = acc * (1.0 + bud["n_sig_unit"] * v2 / (bud["b1b2"] * block_s) * (kern / hbin))
                h = h + rng.poisson(expect)
            hist_tot = hist_tot + h
            mf_nights.append(mf(hist_tot, (j + 1) * mids.size))
        for j in range(n_more):
            sig = 1.0 / np.sqrt(f_night * (j + 1))
            th_sig.append(float(sig))
            # the Fisher-weighted estimate from the nights so far (statistically faithful draw)
            th_est.append(float(theta0 + rng.normal(0.0, sig)))
        multi = dict(n_nights=n_more, v2_nights=np.array(v2_nights), mf_nights=mf_nights,
                     theta_est=th_est, theta_sig=th_sig)
        print(f"  {n_more} nights: sigma(theta)/theta {th_sig[-1] / theta0:.3f} "
              f"({100 * th_sig[-1] / theta0:.1f} %); matched-filter bump {mf_nights[-1][int(np.argmin(np.abs(lags)))]:+.1f} sigma")

    out = {"target": target.name, "backend": b.name, "theta_ld_mas": theta0, "wavelength_nm": lam,
           "block_minutes": block_minutes, "n_blocks": int(mids.size), "baseline_m": blen.tolist(),
           "altitude_deg": alt.tolist(), "vis2_true": v2_true.tolist(), "vis2_measured": v2_meas.tolist(),
           "sigma_vis2": bud["sigma_v2"], "null_m": float(null_m), "rate_cps": bud["rate_cps"],
           "tau_c_ps": bud["tau_c"] * 1e12, "sigma_pair_ps": bud["sigma_pair"] * 1e12,
           "readout_scale": bud["readout_scale"], "sigma_theta_night_frac": float(sig_theta_night / theta0),
           "nights_to_5pct": float(nights_5pct)}
    if multi:
        out["nights"] = {"n_nights": multi["n_nights"], "sigma_theta_frac": [x / theta0 for x in multi["theta_sig"]],
                         "theta_est_mas": multi["theta_est"]}
    if fold:
        out["pulsation"] = {k: v for k, v in fold.items() if k != "per_night"}
        out["pulsation"]["final_sigma_theta_frac_per_bin"] = [float(x / theta0) for x in fold["per_night"][-1][1]]

    if opts.figures:
        path = out_dir / f"{campaign.name}.mp4"
        _render(path, target, b, arr, mids, alt, bvec, blen, v2_true, v2_meas, bud, B_grid, curves, alts,
                null_m, lags, mf_blocks, mf_expect, theta0, theta_t, lam, fold, nbin_show,
                fps=int(opt("options.fps", 12)), dpi=int(opt("options.dpi", 110)), multi=multi)
        out["movie"] = str(path)
    return out


def _render(path, target, backend, arr, mids, alt, bvec, blen, v2_true, v2_meas, bud, B_grid, curves, alts,
            null_m, lags, mf_blocks, mf_expect, theta0, theta_t, lam, fold, nbin_show, fps=12, dpi=110,
            multi=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    n = mids.size
    n_fold = fold["n_nights"] if fold else 0
    n_multi = multi["n_nights"] - 1 if multi else 0        # night 1 is act one
    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.5, 1.2], wspace=0.32, left=0.05, right=0.985, bottom=0.14, top=0.86)
    ax_uv, ax_v, ax_h = (fig.add_subplot(gs[0, k]) for k in range(3))
    fig.suptitle(f"{target.name}: one night on {arr.stations[0].name} + {arr.stations[1].name}, "
                 f"{backend.name}  (θ = {theta0:.3f} mas, V = {target.v_mag:.2f})", fontsize=11)

    # uv panel: the full track faint, the current point bold
    lim = 1.1 * np.abs(bvec).max()
    ax_uv.plot(bvec[:, 0], bvec[:, 1], color="0.8", lw=1)
    ax_uv.plot(-bvec[:, 0], -bvec[:, 1], color="0.8", lw=1)
    circ = plt.Circle((0, 0), null_m, fill=False, ls="--", color="tab:red", lw=1)
    ax_uv.add_patch(circ)
    uv_pt, = ax_uv.plot([], [], "o", color="tab:blue", ms=7)
    uv_pt2, = ax_uv.plot([], [], "o", color="tab:blue", ms=7, alpha=0.4)
    ax_uv.set_xlim(-lim, lim); ax_uv.set_ylim(-lim, lim); ax_uv.set_aspect("equal")
    ax_uv.set_xlabel("u [m] (east)"); ax_uv.set_ylabel("v [m] (north)")
    ax_uv.set_title("projected baseline; dashed: first null", fontsize=9)
    uv_txt = ax_uv.text(0.03, 0.97, "", transform=ax_uv.transAxes, va="top", fontsize=9)

    # visibility panel
    for p, c in curves.items():
        ax_v.plot(B_grid, c, color="k" if p == 0 else "0.6", lw=1.6 if p == 0 else 0.9,
                  ls="-" if p == 0 else "--", label="model θ" if p == 0 else f"θ {p:+g} %")
    ax_v.axvline(null_m, color="tab:red", ls="--", lw=0.8)
    pts = ax_v.errorbar([], [], yerr=[], fmt="o", color="tab:blue", ms=5, capsize=2, lw=1)
    cur, = ax_v.plot([], [], "o", color="tab:orange", ms=8, zorder=5)
    breathe, = ax_v.plot([], [], color="tab:orange", lw=1.2, alpha=0.8)
    sig_bin = bud["sigma_v2"] / np.sqrt(nbin_show)
    bin_min = int(round((mids[1] - mids[0]) * 60 * nbin_show)) if n > 1 else 0
    ax_v.set_xlabel("projected baseline [m]"); ax_v.set_ylabel(r"$|V|^2$")
    ymax = max(0.05, 1.25 * max(curves[0.0][B_grid >= 0.9 * blen.min()].max(), 2.5 * sig_bin))
    ax_v.set_ylim(-1.5 * sig_bin, ymax); ax_v.set_xlim(B_grid[0], B_grid[-1])
    ax_v.legend(fontsize=8, loc="upper right")
    ax_v.set_title(f"measurements in {bin_min} min bins, σ(|V|²) = {sig_bin:.3g} per bin", fontsize=9)
    v_txt = ax_v.text(0.03, 0.97, "", transform=ax_v.transAxes, va="top", fontsize=9)

    # histogram panel
    lags_ps = lags * 1e12
    mf_line, = ax_h.plot(lags_ps, mf_blocks[0], color="0.3", lw=1.0)
    exp_line, = ax_h.plot(lags_ps, mf_expect, color="tab:red", lw=1.2, ls="--", label="expected signal")
    ax_h.axhline(0, color="0.6", lw=0.8); ax_h.axhspan(-1, 1, color="0.9", zorder=0)
    ax_h.set_xlabel("lag τ [ps]"); ax_h.set_ylabel("excess coincidences / shot noise  [σ]")
    ax_h.set_ylim(min(-4, 1.1 * mf_blocks.min()), max(5, 1.25 * max(mf_expect.max(), mf_blocks.max())))
    ax_h.set_title(f"g² bump, matched-filtered (σ_t {bud['sigma_pair'] * 1e12:.0f} ps, τ_c {bud['tau_c'] * 1e12:.1f} ps)",
                   fontsize=9)
    ax_h.legend(fontsize=8, loc="lower right")
    h_txt = ax_h.text(0.03, 0.97, "", transform=ax_h.transAxes, va="top", fontsize=9)
    i0 = int(np.argmin(np.abs(lags)))

    def act1(k):
        uv_pt.set_data([bvec[k, 0]], [bvec[k, 1]]); uv_pt2.set_data([-bvec[k, 0]], [-bvec[k, 1]])
        uv_txt.set_text(f"H = {mids[k]:+.2f} h\nalt {alt[k]:.0f}°\nB = {blen[k]:.0f} m")
        # completed bins as filled points, the running bin hollow
        nb_done = (k + 1) // nbin_show
        xs = [blen[i * nbin_show:(i + 1) * nbin_show].mean() for i in range(nb_done)]
        ys = [v2_meas[i * nbin_show:(i + 1) * nbin_show].mean() for i in range(nb_done)]
        es = [sig_bin] * nb_done
        rest = (k + 1) % nbin_show
        if rest:
            xs.append(blen[nb_done * nbin_show:k + 1].mean()); ys.append(v2_meas[nb_done * nbin_show:k + 1].mean())
            es.append(bud["sigma_v2"] / np.sqrt(rest))
        pts.remove()
        new = ax_v.errorbar(xs, ys, yerr=es, fmt="o", color="tab:blue", ms=5, capsize=2, lw=1,
                            markerfacecolor=["tab:blue"] * nb_done + (["white"] if rest else []) if False else "tab:blue")
        cur.set_data([blen[k]], [v2_true[k]])
        if fold:
            # the disk breathing: |V|^2 depends on theta B / lambda, so the curve for theta(t) is the
            # model curve evaluated at B theta(t) / theta0
            breathe.set_data(B_grid, np.interp(B_grid * theta_t[k] / theta0, B_grid, curves[0.0]))
        side = "first lobe" if blen[k] < null_m else "beyond the first null"
        v_txt.set_text(f"block {k + 1}/{n}: |V|² = {v2_true[k]:.3g} ({side})"
                       + (f"\nθ(t) = {theta_t[k]:.4f} mas (pulsation)" if fold else ""))
        mf_line.set_ydata(mf_blocks[k])
        T = k + 1
        exp_now = mf_expect[i0] * (v2_true[:T].sum() / max(v2_true.sum(), 1e-30)) * np.sqrt(n / T)
        h_txt.set_text(f"{T} block(s), {T * (mids[1] - mids[0]) if n > 1 else 0:.1f} h\nbump at τ = 0: {mf_blocks[k][i0]:+.1f} σ "
                       f"(expected {exp_now:+.1f} σ)")
        return new

    # act two (pulsation fold): reuse the histogram axis
    if fold:
        nb, P_h, A = fold["phase_bins"], fold["period_h"], fold["amplitude_frac"]
        ph_c = (np.arange(nb) + 0.5) / nb
        ph_fine = np.linspace(0, 1, 200)

    def act2(j):
        ax_h.cla()
        th_b, sg_b = fold["per_night"][j]
        ax_h.plot(ph_fine, theta0 * (1 + A * np.sin(2 * np.pi * ph_fine)), color="k", lw=1.2, label="θ(φ) model")
        ax_h.errorbar(ph_c, th_b, yerr=sg_b, fmt="o", color="tab:blue", capsize=3, label=f"{j + 1} night(s), {nb} phase bins")
        ax_h.axhline(theta0, color="0.6", ls=":", lw=0.8)
        ax_h.set_ylim(theta0 * (1 - 4 * A), theta0 * (1 + 4 * A)); ax_h.set_xlim(0, 1)
        ax_h.set_xlabel("pulsation phase"); ax_h.set_ylabel("θ [mas]")
        ax_h.set_title(f"the diameter folded on the {P_h:.2f} h pulsation", fontsize=9)
        ax_h.legend(fontsize=8, loc="upper right")
        s_bin = float(np.nanmedian(sg_b)) / theta0
        need = (j + 1) * (s_bin / (A / 3.0)) ** 2 if s_bin > 0 else np.inf
        v_txt.set_text(f"act two: {j + 1} of {fold['n_nights']} nights folded\nσ(θ) per bin {100 * s_bin:.2f} % "
                       f"vs amplitude {100 * A:.1f} %\n(~{need:.0f} nights for σ = amplitude/3)")

    band_lo, = ax_v.plot([], [], color="tab:blue", lw=0.9, alpha=0.7)
    band_hi, = ax_v.plot([], [], color="tab:blue", lw=0.9, alpha=0.7)
    nb_full = n // nbin_show + (1 if n % nbin_show else 0)
    bin_slices = [slice(i * nbin_show, min((i + 1) * nbin_show, n)) for i in range(nb_full)]
    bin_B = np.array([blen[sl].mean() for sl in bin_slices])
    scale_v = 1.0  # placeholder to keep closures simple

    def act3(j):
        """Night j+1 (j >= 1) accumulated: averaged points, growing bump, narrowing theta band."""
        nonlocal pts
        nn = j + 1
        v2n = multi["v2_nights"][:nn]
        ys = np.array([v2n[:, sl].mean() for sl in bin_slices])
        es = np.array([bud["sigma_v2"] / np.sqrt(nn * (sl.stop - sl.start)) for sl in bin_slices])
        pts.remove()
        pts = ax_v.errorbar(bin_B, ys, yerr=es, fmt="o", color="tab:blue", ms=5, capsize=2, lw=1)
        th, sg = multi["theta_est"][j], multi["theta_sig"][j]
        band_lo.set_data(B_grid, np.interp(B_grid * (th - sg) / theta0, B_grid, curves[0.0]))
        band_hi.set_data(B_grid, np.interp(B_grid * (th + sg) / theta0, B_grid, curves[0.0]))
        cur.set_data([], [])
        v_txt.set_text(f"{nn} nights averaged ({nn * n} blocks)\nθ = {th:.4f} ± {sg:.4f} mas "
                       f"({100 * sg / theta0:.1f} %); blue band: θ ± σ")
        mf_line.set_ydata(multi["mf_nights"][j])
        exp_line.set_ydata(mf_expect * np.sqrt(nn))          # the bump grows as sqrt(nights), the noise stays at 1 sigma
        T = nn * n
        h_txt.set_text(f"{nn} nights, {T * (mids[1] - mids[0]) if n > 1 else 0:.0f} h on source\n"
                       f"bump at τ = 0: {multi['mf_nights'][j][i0]:+.1f} σ (expected {mf_expect[i0] * np.sqrt(nn):+.1f} σ)")
        ax_h.set_ylim(min(-4, 1.1 * multi["mf_nights"][j].min()), max(5, 1.3 * mf_expect.max() * np.sqrt(nn)))
        uv_txt.set_text(f"night {nn} of {multi['n_nights']}\nsame track each night")

    frames = n + n_fold + n_multi
    writer = FFMpegWriter(fps=fps, metadata={"title": f"hbtsim {target.name}"}) if shutil.which("ffmpeg") else None
    if writer is None:
        frames_dir = path.with_suffix("")
        frames_dir.mkdir(exist_ok=True)
        for f in range(frames):
            if f < n:
                pts = act1(f)
            elif multi:
                act3(f - n + 1)
            else:
                act2(f - n)
            fig.savefig(frames_dir / f"frame_{f:04d}.png", dpi=dpi)
        print(f"  ffmpeg not found: {frames} PNG frames in {frames_dir}")
        plt.close(fig)
        return
    with writer.saving(fig, str(path), dpi=dpi):
        for f in range(frames):
            if f < n:
                pts = act1(f)
                writer.grab_frame()
                if f == n - 1:
                    for _ in range(fps):          # hold the finished night
                        writer.grab_frame()
            elif multi:
                act3(f - n + 1)
                for _ in range(max(1, fps // 3)):          # a third of a second per night
                    writer.grab_frame()
                if f == frames - 1:
                    for _ in range(2 * fps):
                        writer.grab_frame()
            else:
                act2(f - n)
                writer.grab_frame()
                if f == frames - 1:
                    for _ in range(2 * fps):
                        writer.grab_frame()
    plt.close(fig)
    print(f"  wrote {path} ({frames} frames at {fps} fps)")
