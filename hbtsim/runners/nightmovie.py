"""Runner `nightmovie`: one night of a single star on a two-telescope array
as a movie -- the uv track, the measurements walking along the disk's
|V|^2(B) curve through its null, and the coincidence histogram building up.

Campaign fields: target (a single star), instrument.array (two stations),
one backend (filter + detector) -- or several, which are then rotated one
per night (run_rotation) -- night.block_minutes (the frame step),
options:
  fps, dpi                      movie encoding (default 12, 110)
  theta_alternatives_pct        extra |V|^2 curves at theta x (1 + p/100) (default [-5, 5])
  seed                          the noise realization
  display_bin_blocks            blocks averaged per displayed |V|^2 point (default 6: 30 min of 5-min blocks)
  histogram                     {bin_ps, lag_half_range_ps} of the coincidence panel, which shows the
                                excess over the accidentals smoothed with the matched filter, in units
                                of its shot noise, so the g2 bump rises out of the noise over the night
  schedule                      the nights of run_rotation: backend names, baselines in metres or dicts
                                {backend, baseline_m, pa_deg, nights} (a generator array rebuilt per night)
  yscale                        "linear" (default) or "log" for the visibility panel of run_rotation
  oblateness                    true: the target's `ellipse` against the equal-area disk (run_ellipse) on
                                any array, with `schedule` entries {backend, side_m, spacings_m, pa_deg,
                                nights} rebuilding the array per night; sigma(q) accumulates over the nights
  nights                        optional: a second act accumulates this many nights (each a new
                                noise realization of the same track): the binned points average
                                down, the g2 bump rises out of the matched-filter noise, and the
                                diameter's error band narrows
  pulsation                     optional {period_days, amplitude_frac, phase0, fold_nights,
                                phase_bins}: theta(t) = theta0 (1 + A sin(2 pi (t - t0)/P))
                                breathes along the night, and a second act folds
                                fold_nights nights on pulsation phase to show the
                                diameter curve emerging (beta Cep).
A target with a `chromatic_diameter` table is drawn with theta x factor(lambda)
per filter, and run_rotation fits the diameter per wavelength and reports
their ratio.  A binary target switches to hbtsim.runners.nightmovie_binary
(nights through the orbit: fringes, fitted separations, the angular semi-major
axis).
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
    array = campaign.array
    if array is None or len(array.stations) < 2:
        raise SystemExit("runner nightmovie needs an array (instrument.array)")
    single = hasattr(target, "theta_ld_mas")
    if single and campaign.option("options.oblateness", False):
        if getattr(target, "ellipse", None) is None:
            raise SystemExit("options.oblateness needs a target with an `ellipse`")
        return run_ellipse(campaign, cat, opts, out_dir)         # any array, any schedule
    if hasattr(target, "period_days"):                          # a binary: nights through the orbit
        from hbtsim.runners.nightmovie_binary import run_binary
        return run_binary(campaign, cat, opts, out_dir)
    if single and (len(campaign.backends) > 1 or campaign.option("options.schedule")):
        return run_rotation(campaign, cat, opts, out_dir)        # a different filter and/or array each night (any pairs)
    if not single:
        raise SystemExit("runner nightmovie needs a single-star or binary target")
    if len(array.stations) != 2:
        raise SystemExit("runner nightmovie needs a two-station array for the single-night mode")
    if len(campaign.backends) != 1:
        raise SystemExit("runner nightmovie takes one backend (one filter, one detector), or several to rotate")
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
    if tgt.theta_factor(lam) != 1.0:                            # a wavelength-dependent (TiO-band) diameter
        tgt = replace(tgt, theta_ld_mas=tgt.theta_ld_mas * tgt.theta_factor(lam))
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


# ---------------------------------------------------------------------------
# Filter rotation: a different filter each night
# ---------------------------------------------------------------------------
def _night_model(target, backend, arr, block_minutes, min_alt, du=0.05):
    """One night's configuration: every pair of the array along its uv
    track, the photon budget, the model |V|^2 per pair and block and its
    derivatives with respect to the diameter and the linear limb-darkening
    coefficient.  Per-pair arrays carry a leading pair axis."""
    from hbtsim.runners.schedule import with_ld_offset
    spec, det, pol = backend.spectrograph, backend.detector, backend.polarization_mode
    h0, h1 = hour_angle_window(target.dec_deg, arr.site.latitude_deg, min_alt)
    mids = hour_angle_blocks(h0, h1, block_minutes)
    if mids.size == 0:
        raise SystemExit("the target never rises above the altitude limit from this site")
    block_s = (h1 - h0) / mids.size * 3600.0
    pairs = arr.pairs()
    bvec = np.array([[pr[2] for pr in arr.projected(float(H), target.dec_deg).pairs()] for H in mids]).transpose(1, 0, 2)
    blen = np.hypot(bvec[..., 0], bvec[..., 1])                     # (n_pairs, n_blocks)
    lat, dec = np.radians(arr.site.latitude_deg), np.radians(target.dec_deg)
    alt = np.degrees(np.arcsin(np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(np.radians(mids * 15.0))))
    lam, w = float(spec.channel_centers_nm[0]), float(spec.channel_widths_nm[0])
    tgt = prepare_single(target, spec)
    fac = tgt.theta_factor(lam)
    if fac != 1.0:                                              # the drawn diameter at this wavelength
        tgt = replace(tgt, theta_ld_mas=tgt.theta_ld_mas * fac)
    theta0 = float(tgt.theta_ld_mas)
    dth = 0.01 * theta0
    t_up, t_dn = with_ld_offset(tgt, du), with_ld_offset(tgt, -du)
    v2, dv, dvu, sig, buds, pup = [], [], [], [], [], []
    for i, j, _ in pairs:
        t1, t2 = arr.stations[i].telescope, arr.stations[j].telescope
        pupils = (t1.diameter_m, t2.diameter_m)
        bud = _budget(tgt, lam, w, block_s, t1, t2, det, pol, spec.throughput)
        bl = blen[len(v2)]
        v2.append(np.array([float(single_star_vis2(tgt, float(b), lam, pupils)[0, 0]) for b in bl]))
        dv.append(np.array([(float(single_star_vis2(replace(tgt, theta_ld_mas=theta0 + dth), float(b), lam, pupils)[0, 0])
                             - float(single_star_vis2(replace(tgt, theta_ld_mas=theta0 - dth), float(b), lam, pupils)[0, 0]))
                            / (2 * dth) for b in bl]))
        if t_up is not None:
            dvu.append(np.array([(float(single_star_vis2(t_up, float(b), lam, pupils)[0, 0])
                                  - float(single_star_vis2(t_dn, float(b), lam, pupils)[0, 0])) / (2 * du) for b in bl]))
        else:
            dvu.append(np.zeros(bl.size))
        sig.append(bud["sigma_v2"]); buds.append(bud); pup.append(pupils)
    v2, dv, dvu, sig = np.array(v2), np.array(dv), np.array(dvu), np.array(sig)
    fisher = (dv / sig[:, None]) ** 2                                  # (n_pairs, n_blocks)
    F2 = np.array([[np.sum(dv * dv / sig[:, None] ** 2), np.sum(dv * dvu / sig[:, None] ** 2)],
                   [np.sum(dv * dvu / sig[:, None] ** 2), np.sum(dvu * dvu / sig[:, None] ** 2)]])
    ground = [float(np.hypot(pr[2][0], pr[2][1])) for pr in pairs]
    names = [f"{arr.stations[i].name}-{arr.stations[j].name}" for i, j, _ in pairs]
    return dict(label=backend.name, lam=lam, w=w, tgt=tgt, theta0=theta0, v2=v2, sigma=sig, bud=buds[0], buds=buds,
                fisher=fisher, F2=F2, snr2=float(np.sum((v2 / sig[:, None]) ** 2)),
                null_m=1.22 * lam * 1e-9 / (tgt.drawn_diameter_mas * MAS),
                mids=mids, block_s=block_s, bvec=bvec, blen=blen, alt=alt, pupils=pup[0], n_pairs=len(pairs),
                pair_names=names, ground_baselines_m=ground, ground_baseline_m=max(ground),
                ground_pa_deg=float(np.degrees(np.arctan2(pairs[0][2][0], pairs[0][2][1])) % 360.0),
                u0=float(tgt.star.ld_coeff(lam)) if tgt.star.ld_profile is None else None,
                theta_factor=fac, stations=tuple(st.name for st in arr.stations))


def run_rotation(campaign, cat, opts, out_dir: Path) -> dict:
    """A different configuration each night: `options.schedule` lists the
    nights (backend names, baselines in metres, or dicts {backend,
    baseline_m, pa_deg, nights}); without it the backends rotate over
    `options.nights`.  Per-night Fisher information on the diameter (and
    jointly on the linear limb-darkening coefficient) accumulates, and the
    movie shows the measurements of every night in equivalent baseline
    B lambda0 / lambda."""
    from hbtsim.runners.schedule import array_for, describe, normalize_schedule
    target = campaign.target
    backends = list(campaign.backends)
    opt = campaign.option
    block_minutes = float(opt("night.block_minutes", 5.0))
    min_alt = float(opt("night.min_alt_deg", 30.0))
    n_nights = int(opt("options.nights", 20))
    rng = np.random.default_rng(int(opt("options.seed", 7)))
    alts = list(opt("options.theta_alternatives_pct", [-5.0, 5.0]))
    nbin_show = max(1, int(opt("options.display_bin_blocks", 6)))
    yscale = str(opt("options.yscale", "linear"))

    sched_opt = opt("options.schedule")
    if sched_opt:
        entries = normalize_schedule(sched_opt, campaign, backends)
    else:
        entries = [{"backend_index": j % len(backends), "nights": 1} for j in range(n_nights)]
    nights = [e for e in entries for _ in range(int(e.get("nights", 1)))]
    n_nights = len(nights)
    keys, per, sched = [], [], []
    for e in nights:
        key = (e["backend_index"], e.get("baseline_m"), e.get("pa_deg"), e.get("side_m"),
               tuple(e["spacings_m"]) if "spacings_m" in e else None)
        if key not in keys:
            b = backends[e["backend_index"]]
            arr = array_for(cat, campaign, e, detector=b.detector)
            if len(arr.stations) < 2:
                raise SystemExit("runner nightmovie needs an array of at least two stations")
            keys.append(key)
            per.append(_night_model(target, b, arr, block_minutes, min_alt))
        sched.append(keys.index(key))
    theta0 = per[0]["theta0"]
    lam0 = per[0]["lam"]
    multi_baseline = len({tuple(q["ground_baselines_m"]) for q in per}) > 1
    st = per[0]["stations"]
    print(f"=== {target.name}: theta_LD {theta0:.3f} mas, V {target.v_mag:.2f}; {len(per)} configuration(s) over "
          f"{n_nights} nights on {'+'.join(st)} ({per[0]['n_pairs']} pair(s) per night) ===")
    for q in per:
        gb = ", ".join(f"{x:.0f}" for x in q["ground_baselines_m"])
        print(f"  {q['label'][:46]:46s} lambda {q['lam']:.0f} nm, ground B {gb} m PA {q['ground_pa_deg']:.0f}: "
              f"B {q['blen'].min():.0f}-{q['blen'].max():.0f} m, null at {q['null_m']:.1f} m, |V|^2 {q['v2'].min():.2g}-{q['v2'].max():.2g}, "
              f"sigma/block {q['sigma'][0]:.3g}, one night: sigma(theta)/theta {1 / np.sqrt(q['fisher'].sum()) / theta0:.3f}, "
              f"detection {np.sqrt(q['snr2']):.2f} sigma")
    meas = []                       # (configuration index, measured |V|^2 per block)
    fisher_tot, snr2_f = 0.0, np.zeros(len(per))
    F_tot = np.zeros((2, 2))
    sig_theta, sig_joint, sig_u, det_sig = [], [], [], []
    for j, ci in enumerate(sched):
        q = per[ci]
        meas.append((ci, q["v2"] + rng.normal(0.0, 1.0, size=q["v2"].shape) * q["sigma"][:, None]))
        fisher_tot += float(q["fisher"].sum())
        F_tot += q["F2"]
        snr2_f[ci] += q["snr2"]
        sig_theta.append(1.0 / np.sqrt(fisher_tot))
        if F_tot[1, 1] > 0 and np.linalg.det(F_tot) > 0:
            cov = np.linalg.inv(F_tot)
            sig_joint.append(float(np.sqrt(cov[0, 0]))); sig_u.append(float(np.sqrt(cov[1, 1])))
        else:
            sig_joint.append(float("inf")); sig_u.append(float("inf"))
        det_sig.append(np.sqrt(snr2_f.copy()))
    # the diameter per wavelength (a chromatic target draws a different disk per filter)
    by_lam = {}
    for ci in sched:
        q = per[ci]
        e = by_lam.setdefault(q["lam"], {"theta_mas": q["theta0"], "fisher": 0.0, "nights": 0, "theta_factor": q["theta_factor"]})
        e["fisher"] += float(q["fisher"].sum()); e["nights"] += 1
    theta_by_wavelength = {lam_: {"theta_mas": e["theta_mas"], "theta_factor": e["theta_factor"], "nights": e["nights"],
                                  "sigma_theta_frac": float(1.0 / np.sqrt(e["fisher"]) / e["theta_mas"]) if e["fisher"] > 0 else None}
                           for lam_, e in by_lam.items()}
    chromatic = None
    if len(by_lam) >= 2:
        l_lo, l_hi = min(by_lam), max(by_lam)
        a, b2 = theta_by_wavelength[l_lo], theta_by_wavelength[l_hi]
        if a["sigma_theta_frac"] is not None and b2["sigma_theta_frac"] is not None:
            ratio = a["theta_mas"] / b2["theta_mas"]
            s_r = ratio * np.sqrt(a["sigma_theta_frac"] ** 2 + b2["sigma_theta_frac"] ** 2)
            chromatic = {"wavelengths_nm": [l_lo, l_hi], "diameter_ratio": float(ratio), "sigma_ratio": float(s_r),
                         "ratio_significance": float((ratio - 1.0) / s_r) if s_r > 0 else None}
            print(f"  diameters per wavelength: " + ", ".join(f"{lam_:.0f} nm {e['theta_mas']:.3f} mas +/- {100 * e['sigma_theta_frac']:.2f} % ({e['nights']} nights)"
                                                              for lam_, e in sorted(theta_by_wavelength.items()))
                  + f"; ratio theta({l_lo:.0f})/theta({l_hi:.0f}) = {ratio:.4f} +/- {s_r:.4f} ({chromatic['ratio_significance']:.1f} sigma from 1)")
    best = int(np.argmax([q["fisher"].sum() for q in per]))
    sig_best = 1.0 / np.sqrt(n_nights * per[best]["fisher"].sum())
    counts = {}
    for ci in sched:
        counts[ci] = counts.get(ci, 0) + 1
    print("  schedule: " + ", ".join(f"{c} x [{per[ci]['lam']:.0f} nm, {'/'.join(f'{x:.0f}' for x in per[ci]['ground_baselines_m'])} m]"
                                     for ci, c in counts.items()))
    print(f"  after {n_nights} nights: sigma(theta)/theta = {sig_theta[-1] / theta0:.4f} (theta alone), "
          f"{sig_joint[-1] / theta0:.4f} jointly with u, sigma(u) = {sig_u[-1]:.3f} (u = {per[0]['u0']}); every night on "
          f"the best configuration instead: {sig_best / theta0:.4f}; detection per configuration "
          + ", ".join(f"[{q['lam']:.0f} nm, {'/'.join(f'{x:.0f}' for x in q['ground_baselines_m'])} m] {d:.1f} sigma" for q, d in zip(per, det_sig[-1])))
    theta_est = [float(theta0 + rng.normal(0.0, s)) for s in sig_theta]
    out = {"target": target.name, "filters": [q["label"] for q in per], "wavelengths_nm": [q["lam"] for q in per],
           "nulls_m": [q["null_m"] for q in per], "n_nights": n_nights, "schedule": sched,
           "configs": [{"label": q["label"], "wavelength_nm": q["lam"], "ground_baseline_m": q["ground_baseline_m"],
                        "ground_baselines_m": q["ground_baselines_m"], "pairs": q["pair_names"],
                        "ground_pa_deg": q["ground_pa_deg"], "null_m": q["null_m"], "n_blocks": int(q["mids"].size),
                        "vis2_min": float(q["v2"].min()), "vis2_max": float(q["v2"].max()), "sigma_vis2": float(q["sigma"][0]),
                        "readout_scale": q["bud"]["readout_scale"], "rate_cps": q["bud"]["rate_cps"]} for q in per],
           "baselines_m_per_night": [per[ci]["ground_baseline_m"] for ci in sched],
           "n_pairs": per[0]["n_pairs"],
           "sigma_theta_frac": [s / theta0 for s in sig_theta],
           "sigma_theta_frac_joint": [s / theta0 for s in sig_joint], "sigma_u": sig_u, "u_linear": per[0]["u0"],
           "sigma_theta_frac_best_single": sig_best / theta0,
           "best_single_filter": per[best]["label"], "detection_sigma_per_filter": det_sig[-1].tolist(),
           "one_night_sigma_theta_frac": [1 / np.sqrt(q["fisher"].sum()) / theta0 for q in per],
           "baseline_m": per[0]["blen"][0].tolist(), "theta_ld_mas": theta0,
           "theta_by_wavelength": {f"{k:.1f}": v for k, v in theta_by_wavelength.items()}, "chromatic": chromatic}
    if opts.figures:
        path = out_dir / f"{campaign.name}.mp4"
        ratio_txt = (f"θ({chromatic['wavelengths_nm'][0]:.0f})/θ({chromatic['wavelengths_nm'][1]:.0f}) = "
                     f"{chromatic['diameter_ratio']:.3f} ± {chromatic['sigma_ratio']:.3f}" if chromatic else "")
        _render_rotation(path, target, per, sched, meas, sig_theta, sig_joint, sig_u, theta_est, det_sig,
                         lam0, alts, nbin_show, fps=int(opt("options.fps", 12)), dpi=int(opt("options.dpi", 110)),
                         yscale=yscale, multi_baseline=multi_baseline, ratio_txt=ratio_txt)
        out["movie"] = str(path)
    return out


def _render_rotation(path, target, per, sched, meas, sig_theta, sig_joint, sig_u, theta_est, det_sig,
                     lam0, alts, nbin_show, fps=12, dpi=110, yscale="linear", multi_baseline=False, ratio_txt=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    theta0 = per[0]["theta0"]
    pupils = per[0]["pupils"]
    st = per[0]["stations"]
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(per)))
    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.6, 1.1], wspace=0.32, left=0.05, right=0.985, bottom=0.14, top=0.86)
    ax_uv, ax_v, ax_d = (fig.add_subplot(gs[0, k]) for k in range(3))
    what = "a different baseline each night" if multi_baseline else "a different filter each night"
    fig.suptitle(f"{target.name}: {what} on {' + '.join(st)} (θ = {theta0:.3f} mas, V = {target.v_mag:.2f})", fontsize=11)

    lim = 1.1 * max(np.abs(q["bvec"]).max() for q in per) * 1.3
    tracks = []
    for q, c in zip(per, colors):
        lines = []
        for p in range(q["n_pairs"]):
            lines.append(ax_uv.plot(q["bvec"][p, :, 0], q["bvec"][p, :, 1], color=c, lw=1, alpha=0.4)[0])
            lines.append(ax_uv.plot(-q["bvec"][p, :, 0], -q["bvec"][p, :, 1], color=c, lw=1, alpha=0.4)[0])
        tracks.append(lines)
    circles = {}
    for q, c in zip(per, colors):
        if q["lam"] not in circles:
            circ = plt.Circle((0, 0), q["null_m"], fill=False, ls="--", color=c, lw=1, alpha=0.9)
            ax_uv.add_patch(circ); circles[q["lam"]] = circ
    n_pairs_max = max(q["n_pairs"] for q in per)
    uv_pts = [ax_uv.plot([], [], "o", color="k", ms=7)[0] for _ in range(n_pairs_max)]
    ax_uv.set_xlim(-lim, lim); ax_uv.set_ylim(-lim, lim); ax_uv.set_aspect("equal")
    ax_uv.set_xlabel("u [m] (east)"); ax_uv.set_ylabel("v [m] (north)")
    ax_uv.set_title("projected baselines of the nights; dashed: first null", fontsize=9)
    uv_txt = ax_uv.text(0.03, 0.97, "", transform=ax_uv.transAxes, va="top", fontsize=9)

    # the visibility panel in equivalent baseline at lam0
    Beq_min = min(q["blen"].min() * lam0 / q["lam"] for q in per)
    Beq_max = max(q["blen"].max() * lam0 / q["lam"] for q in per)
    B_grid = np.linspace(0.5 * Beq_min if yscale != "log" else 0.3 * Beq_min, 1.08 * Beq_max, 500)
    tgt0 = per[0]["tgt"]
    curve0 = np.array([float(single_star_vis2(tgt0, float(bb), lam0, pupils)[0, 0]) for bb in B_grid])
    ax_v.plot(B_grid, curve0, color="k", lw=1.6, label=f"model θ at {lam0:.0f} nm")
    for q, c in zip(per, colors):                                # a chromatic disk: its own curve per wavelength
        if q["lam"] != lam0 and q["theta_factor"] != per[0]["theta_factor"]:
            ax_v.plot(B_grid, [float(single_star_vis2(q["tgt"], float(bb * q["lam"] / lam0), q["lam"], pupils)[0, 0]) for bb in B_grid],
                      color=c, lw=1.2, ls="-.", label=f"model θ at {q['lam']:.0f} nm")
    for p in alts:
        ax_v.plot(B_grid, np.interp(B_grid * (1 + p / 100.0), B_grid, curve0), color="0.6", lw=0.9, ls="--", label=f"θ {p:+g} %")
    band_lo, = ax_v.plot([], [], color="tab:blue", lw=0.9, alpha=0.7)
    band_hi, = ax_v.plot([], [], color="tab:blue", lw=0.9, alpha=0.7)
    for q, c in zip(per, colors):
        ax_v.plot([], [], "o", color=c, ms=5, label=("/".join(f"{x:.0f}" for x in q["ground_baselines_m"]) + " m" if multi_baseline else "") + f" {q['lam']:.0f} nm")
    ax_v.axvline(1.22 * lam0 * 1e-9 / (tgt0.drawn_diameter_mas * MAS), color="k", ls=":", lw=0.8)
    ax_v.set_xlabel(f"equivalent baseline B λ₀/λ at λ₀ = {lam0:.0f} nm  [m]"); ax_v.set_ylabel(r"$|V|^2$")
    sig_min = min(float(np.min(q["sigma"])) for q in per) / np.sqrt(nbin_show)
    if yscale == "log":
        ax_v.set_yscale("log"); ax_v.set_ylim(max(1e-4, 0.3 * sig_min), 1.6)
    else:
        ax_v.set_ylim(-1.5 * sig_min, max(0.05, 1.25 * curve0[B_grid >= 0.9 * Beq_min].max()))
    ax_v.set_xlim(B_grid[0], B_grid[-1])
    ax_v.legend(fontsize=7, loc="upper right", ncol=2)
    v_txt = ax_v.text(0.03, 0.97, "", transform=ax_v.transAxes, va="top", fontsize=9)
    err_containers = []

    # the diameter panel: sigma(theta)/theta vs nights, and the detection per configuration
    nights_ax = np.arange(1, len(sched) + 1)
    ax_d.plot(nights_ax, 100 * np.array(sig_theta) / theta0, color="tab:blue", lw=1.5, label="this schedule, θ alone")
    sj = 100 * np.array(sig_joint) / theta0
    ax_d.plot(nights_ax, np.where(np.isfinite(sj), sj, np.nan), color="tab:blue", lw=1.2, ls="--", label="θ jointly with limb darkening u")
    best = int(np.argmax([q["fisher"].sum() for q in per]))
    ax_d.plot(nights_ax, 100 / np.sqrt(nights_ax * per[best]["fisher"].sum()) / theta0, color="0.5", lw=1.2, ls=":",
              label=("/".join(f"{x:.0f}" for x in per[best]["ground_baselines_m"]) + " m, " if multi_baseline else "") + f"{per[best]['lam']:.0f} nm every night")
    d_pt, = ax_d.plot([], [], "o", color="tab:blue", ms=7)
    ax_d.set_xlabel("nights"); ax_d.set_ylabel("σ(θ)/θ  [%]"); ax_d.set_ylim(0, min(30, 110 * sig_theta[0] / theta0))
    ax_d.set_xlim(0, len(sched) + 1); ax_d.axhline(5, color="tab:red", ls=":", lw=0.8)
    ax_d.legend(fontsize=8, loc="upper right"); ax_d.set_title("diameter precision as the nights accumulate", fontsize=9)
    d_txt = ax_d.text(0.03, 0.03, "", transform=ax_d.transAxes, va="bottom", fontsize=8)

    def bin_slices(nq):
        nb = nq // nbin_show + (1 if nq % nbin_show else 0)
        return [slice(i * nbin_show, min((i + 1) * nbin_show, nq)) for i in range(nb)]

    def frame_night(j, k):
        """Night j (its configuration), block k of the night (k = n_j - 1: the whole night)."""
        ci, v2m = meas[j]
        q = per[ci]
        n = q["mids"].size
        for p, pt in enumerate(uv_pts):
            if p < q["n_pairs"]:
                pt.set_data([q["bvec"][p, k, 0]], [q["bvec"][p, k, 1]])
            else:
                pt.set_data([], [])
        for cj, lines in enumerate(tracks):
            for t in lines:
                t.set_linewidth(2.0 if cj == ci else 1.0); t.set_alpha(1.0 if cj == ci else 0.35)
        for lamc, circ in circles.items():
            circ.set_linewidth(2.2 if lamc == q["lam"] else 0.8); circ.set_alpha(1.0 if lamc == q["lam"] else 0.5)
        gb = "/".join(f"{x:.0f}" for x in q["ground_baselines_m"])
        uv_txt.set_text(f"night {j + 1}/{len(sched)}: {q['lam']:.0f} nm, ground B {gb} m\n"
                        f"H = {q['mids'][k]:+.2f} h, alt {q['alt'][k]:.0f}°\n"
                        f"B = {'/'.join(f'{x:.1f}' for x in q['blen'][:, k])} m")
        for ec in err_containers:
            ec.remove()
        err_containers.clear()
        for jj in range(j + 1):
            cj, vj = meas[jj]
            qj = per[cj]
            nj = qj["mids"].size
            kmax = nj if jj < j else k + 1
            xs, ys, es = [], [], []
            for p in range(qj["n_pairs"]):
                for sl in bin_slices(nj):
                    if sl.start >= kmax:
                        break
                    sl2 = slice(sl.start, min(sl.stop, kmax))
                    xs.append(qj["blen"][p, sl2].mean() * lam0 / qj["lam"]); ys.append(vj[p, sl2].mean())
                    es.append(qj["sigma"][p] / np.sqrt(sl2.stop - sl2.start))
            ys, es = np.array(ys), np.array(es)
            if yscale == "log":
                lo = np.minimum(es, ys - ax_v.get_ylim()[0] * 1.001)
                ys_plot = np.maximum(ys, ax_v.get_ylim()[0] * 1.001)
                err_containers.append(ax_v.errorbar(xs, ys_plot, yerr=[np.maximum(lo, 0), es], fmt="o", color=colors[cj], ms=4,
                                                    capsize=2, lw=0.9, alpha=1.0 if jj == j else 0.55))
            else:
                err_containers.append(ax_v.errorbar(xs, ys, yerr=es, fmt="o", color=colors[cj], ms=4, capsize=2, lw=0.9,
                                                    alpha=1.0 if jj == j else 0.55))
        th, sg = theta_est[j], sig_theta[j]
        if k == n - 1:
            band_lo.set_data(B_grid, np.interp(B_grid * (th - sg) / theta0, B_grid, curve0))
            band_hi.set_data(B_grid, np.interp(B_grid * (th + sg) / theta0, B_grid, curve0))
            joint = f", {100 * sig_joint[j] / theta0:.1f} % with u" if np.isfinite(sig_joint[j]) else ""
            v_txt.set_text(f"{j + 1} night(s)\nθ = {th:.4f} ± {sg:.4f} mas\n({100 * sg / theta0:.1f} %{joint})")
            d_pt.set_data([j + 1], [100 * sg / theta0])
            su = (f"; σ(u) = {sig_u[j]:.3f}" if np.isfinite(sig_u[j]) else "") + (f"\n{ratio_txt} (all nights)" if ratio_txt and j == len(sched) - 1 else "")
            d_txt.set_text("detection: " + ", ".join(f"{qq['ground_baseline_m']:.0f} m {d:.1f}σ" if multi_baseline else
                                                      f"{qq['lam']:.0f} nm {d:.1f}σ" for qq, d in zip(per, det_sig[j])) + su)
        else:
            v_txt.set_text(f"night {j + 1}, block {k + 1}/{n}\n|V|² = {'/'.join(f'{x:.3g}' for x in q['v2'][:, k])} at {q['lam']:.0f} nm\n(null {q['null_m']:.1f} m)")

    writer = FFMpegWriter(fps=fps, metadata={"title": f"hbtsim {target.name} schedule"}) if shutil.which("ffmpeg") else None
    n0 = per[sched[0]]["mids"].size
    step = max(1, n0 // 30)           # ~30 frames for the first night
    if writer is None:
        frames_dir = path.with_suffix(""); frames_dir.mkdir(exist_ok=True)
        f = 0
        for k in range(0, n0, step):
            frame_night(0, k); fig.savefig(frames_dir / f"frame_{f:04d}.png", dpi=dpi); f += 1
        for j in range(len(sched)):
            frame_night(j, per[sched[j]]["mids"].size - 1); fig.savefig(frames_dir / f"frame_{f:04d}.png", dpi=dpi); f += 1
        print(f"  ffmpeg not found: {f} PNG frames in {frames_dir}")
        plt.close(fig); return
    with writer.saving(fig, str(path), dpi=dpi):
        for k in list(range(0, n0, step)) + [n0 - 1]:
            frame_night(0, k); writer.grab_frame()
        for _ in range(fps):
            writer.grab_frame()
        for j in range(len(sched)):
            frame_night(j, per[sched[j]]["mids"].size - 1)
            for _ in range(max(1, fps // 2)):
                writer.grab_frame()
        for _ in range(2 * fps):
            writer.grab_frame()
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Oblateness: an elliptical disk along the track, nights accumulating
# ---------------------------------------------------------------------------
def _ellipse_model(target, backend, arr, block_minutes, min_alt):
    """One night's configuration of the oblateness mode: every pair of the
    array along its uv track, the oblate and the equal-area circular |V|^2
    per pair and block, the per-pair sigma and the Fisher matrix on
    (theta_major, q) with the orientation held fixed."""
    from hbtsim.iact import pair_track, single_vis2_fn
    from hbtsim.single import ellipse_vis
    spec, det, pol = backend.spectrograph, backend.detector, backend.polarization_mode
    lam, w = float(spec.channel_centers_nm[0]), float(spec.channel_widths_nm[0])
    tgt = prepare_single(target, spec)
    star = tgt.star
    ell = target.ellipse
    th_maj, q, pa = float(ell.theta_major_mas), float(ell.axis_ratio), float(ell.pa_deg)

    def trk(thm, qq):
        fn = single_vis2_fn(lambda bb, l: ellipse_vis(bb, l, star, thm, qq, pa)[0], True)
        return pair_track(arr, target.dec_deg, lam, fn, block_minutes=block_minutes, min_alt_deg=min_alt)
    t_ell = trk(th_maj, q)
    if t_ell.hour_angle_h.size == 0:
        raise SystemExit("the target never rises above the altitude limit from this site")
    v2_ell = t_ell.vis2                                         # (n_pairs, n_blocks)
    th_eq = th_maj / np.sqrt(q)                                 # equal-area circular disk
    v2_circ = trk(th_eq, 1.0).vis2
    d_th, d_q = 0.01 * th_maj, 0.01
    g_th = (trk(th_maj + d_th, q).vis2 - trk(th_maj - d_th, q).vis2) / (2 * d_th)
    g_q = (trk(th_maj, q + d_q).vis2 - trk(th_maj, q - d_q).vis2) / (2 * d_q)
    pairs = arr.pairs()
    sig, buds, pup = [], [], []
    for i, j, _ in pairs:
        t1, t2 = arr.stations[i].telescope, arr.stations[j].telescope
        buds.append(_budget(tgt, lam, w, t_ell.block_s, t1, t2, det, pol, spec.throughput))
        sig.append(buds[-1]["sigma_v2"]); pup.append((t1.diameter_m, t2.diameter_m))
    sig = np.array(sig)
    F = np.array([[np.sum(g_th * g_th / sig[:, None] ** 2), np.sum(g_th * g_q / sig[:, None] ** 2)],
                  [np.sum(g_q * g_th / sig[:, None] ** 2), np.sum(g_q * g_q / sig[:, None] ** 2)]])
    mids = t_ell.hour_angle_h
    lat, dec = np.radians(arr.site.latitude_deg), np.radians(target.dec_deg)
    alt = np.degrees(np.arcsin(np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(np.radians(mids * 15.0))))
    ground = [float(np.hypot(pr[2][0], pr[2][1])) for pr in pairs]
    return dict(label=backend.name, lam=lam, w=w, tgt=tgt, star=star, mids=mids, block_s=t_ell.block_s, alt=alt,
                bvec=t_ell.baseline_vec_m, blen=t_ell.baseline_len_m, pa_track=t_ell.position_angle_deg % 180.0,
                v2_ell=v2_ell, v2_circ=v2_circ, sigma=sig, buds=buds, bud=buds[0], F=F,
                snr2=float(np.sum((v2_ell / sig[:, None]) ** 2)), n_pairs=len(pairs), pupils=pup,
                pair_names=[f"{arr.stations[i].name}-{arr.stations[j].name}" for i, j, _ in pairs],
                ground_baselines_m=ground, ground_pa_deg=float(np.degrees(np.arctan2(pairs[0][2][0], pairs[0][2][1])) % 360.0),
                stations=tuple(st.name for st in arr.stations), th_maj=th_maj, q=q, pa=pa, th_eq=th_eq)


def run_ellipse(campaign, cat, opts, out_dir: Path) -> dict:
    """The target's `ellipse` (axis ratio q, major axis at pa_deg) against the
    equal-area circular disk.  Along the night the projected baseline rotates
    through ~90 deg of position angle, so the effective baseline
    sqrt(b_maj^2 + (b_min/q)^2) differs from |B| by up to (q - 1): the
    measurements deviate from the circular curve in a PA-dependent way.  The
    Fisher matrix on (theta_major, q) with the orientation held fixed gives
    sigma(q) per night and the nights to a 3 sigma detection of q - 1.
    Any array (every pair counts) and, with `options.schedule`, a different
    configuration each night (e.g. a triangle rotated by `pa_deg` on the
    second night to sample more position angles)."""
    from hbtsim.runners.schedule import array_for, normalize_schedule
    target = campaign.target
    backends = list(campaign.backends)
    opt = campaign.option
    block_minutes = float(opt("night.block_minutes", 5.0)); min_alt = float(opt("night.min_alt_deg", 30.0))
    n_nights = int(opt("options.nights", 40)); n_sigma = float(opt("options.detection_sigma", 3.0))
    rng = np.random.default_rng(int(opt("options.seed", 7)))
    nbin_show = max(1, int(opt("options.display_bin_blocks", 6)))
    sched_opt = opt("options.schedule")
    entries = (normalize_schedule(sched_opt, campaign, backends) if sched_opt
               else [{"backend_index": 0, "nights": n_nights}])
    nights = [e for e in entries for _ in range(int(e.get("nights", 1)))]
    n_nights = len(nights)
    keys, per, sched = [], [], []
    for e in nights:
        key = (e["backend_index"], e.get("baseline_m"), e.get("pa_deg"), e.get("side_m"),
               tuple(e["spacings_m"]) if "spacings_m" in e else None)
        if key not in keys:
            b = backends[e["backend_index"]]
            arr = array_for(cat, campaign, e, detector=b.detector) if sched_opt else \
                Array(tuple(replace(s, detector=b.detector) for s in campaign.array.stations), campaign.array.site)
            if len(arr.stations) < 2:
                raise SystemExit("runner nightmovie needs an array of at least two stations")
            keys.append(key)
            per.append(_ellipse_model(target, b, arr, block_minutes, min_alt))
        sched.append(keys.index(key))
    first = per[0]
    th_maj, q, pa, th_eq = first["th_maj"], first["q"], first["pa"], first["th_eq"]
    # the cumulative Fisher information over the nights
    F_tot = np.zeros((2, 2)); sig_q, sig_th = [], []
    for ci in sched:
        F_tot = F_tot + per[ci]["F"]
        cov = np.linalg.inv(F_tot)
        sig_q.append(float(np.sqrt(cov[1, 1]))); sig_th.append(float(np.sqrt(cov[0, 0])))
    sig_q, sig_th = np.array(sig_q), np.array(sig_th)
    cov1 = np.linalg.inv(first["F"])
    sig_q_night, sig_th_night = float(np.sqrt(cov1[1, 1])), float(np.sqrt(cov1[0, 0]))
    sig_q_known = 1.0 / np.sqrt(first["F"][1, 1])               # if theta were known
    nights_detect = (n_sigma * sig_q_night / (q - 1.0)) ** 2
    pa_all = np.concatenate([per[ci]["pa_track"].ravel() for ci in set(sched)])
    pa_cov = 5.0 * np.unique((pa_all // 5.0).astype(int)).size   # position angles sampled, in 5-degree bins
    print(f"=== {target.name}: ellipse theta_major {th_maj:.4f} mas, axis ratio {q:.3f} at PA {pa:.0f} deg "
          f"(equal-area theta {th_eq:.4f}); {len(per)} configuration(s) over {n_nights} nights on "
          f"{'+'.join(first['stations'])} ({first['n_pairs']} pair(s)) ===")
    for ci, m in enumerate(per):
        c1 = np.linalg.inv(m["F"])
        print(f"  {m['label'][:40]:40s} ground B {', '.join(f'{x:.0f}' for x in m['ground_baselines_m'])} m PA {m['ground_pa_deg']:.0f}: "
              f"{m['mids'].size} blocks of {block_minutes:g} min, B {m['blen'].min():.0f}-{m['blen'].max():.0f} m, baseline PA "
              f"{m['pa_track'].min():.0f}-{m['pa_track'].max():.0f} deg; |V|^2 {m['v2_ell'].min():.2g}-{m['v2_ell'].max():.2g}, "
              f"max |ellipse - circle| {np.abs(m['v2_ell'] - m['v2_circ']).max():.3g} (sigma per block {m['sigma'][0]:.3g}); "
              f"one night: sigma(q) {np.sqrt(c1[1, 1]):.4f}, sigma(theta_major)/theta {np.sqrt(c1[0, 0]) / th_maj:.3f}, "
              f"detection {np.sqrt(m['snr2']):.1f} sigma")
    print(f"  first configuration: sigma(q) = {sig_q_night:.4f} (theta free; {sig_q_known:.4f} with theta known); "
          f"q - 1 = {q - 1:.3f} at {n_sigma:g} sigma after {nights_detect:.0f} such nights")
    print(f"  the {n_nights} scheduled nights: sigma(q) = {sig_q[-1]:.4f}, sigma(theta_major)/theta = {sig_th[-1] / th_maj:.4f}, "
          f"q - 1 at {(q - 1) / sig_q[-1]:.1f} sigma; position angles covered {pa_cov:.0f} deg of 180")
    meas = [(ci, per[ci]["v2_ell"] + rng.normal(0.0, 1.0, size=per[ci]["v2_ell"].shape) * per[ci]["sigma"][:, None]) for ci in sched]
    q_est = [float(q + rng.normal(0.0, s)) for s in sig_q]
    out = {"target": target.name, "backend": backends[0].name, "theta_major_mas": th_maj, "axis_ratio": q, "pa_deg": pa,
           "theta_equal_area_mas": th_eq, "n_blocks": int(first["mids"].size), "baseline_m": first["blen"][0].tolist(),
           "baseline_pa_deg": first["pa_track"][0].tolist(), "vis2_ellipse": first["v2_ell"][0].tolist(),
           "vis2_circle": first["v2_circ"][0].tolist(), "sigma_vis2": float(first["sigma"][0]),
           "sigma_q_night": sig_q_night, "sigma_q_night_theta_known": float(sig_q_known),
           "sigma_theta_major_frac_night": sig_th_night / th_maj, "nights_to_detect": float(nights_detect),
           "n_nights": n_nights, "sigma_q": sig_q.tolist(), "sigma_theta_major_frac": (sig_th / th_maj).tolist(),
           "significance": float((q - 1) / sig_q[-1]), "n_pairs": first["n_pairs"], "pa_coverage_deg": float(pa_cov),
           "schedule": sched,
           "configs": [{"backend": m["label"], "wavelength_nm": m["lam"], "ground_baselines_m": m["ground_baselines_m"],
                        "ground_pa_deg": m["ground_pa_deg"], "pair_names": m["pair_names"], "n_blocks": int(m["mids"].size),
                        "sigma_vis2": m["sigma"].tolist(), "sigma_q_night": float(np.sqrt(np.linalg.inv(m["F"])[1, 1])),
                        "detection_sigma": float(np.sqrt(m["snr2"]))} for m in per]}
    if opts.figures:
        path = out_dir / f"{campaign.name}.mp4"
        _render_ellipse(path, target, per, sched, meas, sig_q, q_est, n_sigma, nbin_show,
                        fps=int(opt("options.fps", 12)), dpi=int(opt("options.dpi", 110)))
        out["movie"] = str(path)
    return out


def _render_ellipse(path, target, per, sched, meas, sig_q, q_est, n_sigma, nbin_show, fps=12, dpi=110):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Ellipse as MplEllipse
    from hbtsim.single import single_star_vis2

    first = per[0]
    th_maj, q, pa, th_eq, lam0 = first["th_maj"], first["q"], first["pa"], first["th_eq"], first["lam"]
    n_nights = len(meas)
    labels = sorted({m["label"] for m in per})
    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.6, 1.1], wspace=0.32, left=0.05, right=0.985, bottom=0.14, top=0.86)
    ax_uv, ax_v, ax_q = (fig.add_subplot(gs[0, k]) for k in range(3))
    fig.suptitle(f"{target.name}: is it oblate?  axis ratio {q:.3f}, major axis at PA {pa:.0f}°, "
                 f"{'+'.join(first['stations'])}, {' / '.join(labels)}", fontsize=11)

    pair_colors = ["tab:blue", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
    lim = 1.15 * max(np.abs(m["bvec"]).max() for m in per)
    for m in per:
        for pth in range(m["n_pairs"]):
            bv = m["bvec"][pth]
            ax_uv.plot(bv[:, 0], bv[:, 1], color="0.85", lw=1); ax_uv.plot(-bv[:, 0], -bv[:, 1], color="0.85", lw=1)
    q_show = 1.0 + 10.0 * (q - 1.0)                              # flattening exaggerated x10 for the eye
    ex = MplEllipse((0, 0), width=0.45 * lim / q_show, height=0.45 * lim, angle=-pa, fill=False, color="tab:orange", lw=1.5)
    ax_uv.add_patch(ex)
    ax_uv.text(0, -0.3 * lim, "star outline (flattening ×10)", ha="center", fontsize=7, color="tab:orange")
    uv_pts = [ax_uv.plot([], [], "o", color=pair_colors[pth % len(pair_colors)], ms=7)[0] for pth in range(max(m["n_pairs"] for m in per))]
    ax_uv.set_xlim(-lim, lim); ax_uv.set_ylim(-lim, lim); ax_uv.set_aspect("equal")
    ax_uv.set_xlabel("u [m] (east)"); ax_uv.set_ylabel("v [m] (north)")
    ax_uv.set_title("the projected baselines sweep the position angle", fontsize=9)
    uv_txt = ax_uv.text(0.03, 0.97, "", transform=ax_uv.transAxes, va="top", fontsize=9)

    b_eq = [m["blen"] * lam0 / m["lam"] for m in per]            # equivalent baseline at lambda0
    bmin, bmax = min(b.min() for b in b_eq), max(b.max() for b in b_eq)
    B_grid = np.linspace(0.6 * bmin, 1.1 * bmax, 300)
    tgt_c = replace(target, theta_ld_mas=th_eq / target.star.radius_scale)
    curve_c = np.array([float(single_star_vis2(tgt_c, float(bb), lam0, first["pupils"][0])[0, 0]) for bb in B_grid])
    ax_v.plot(B_grid, curve_c, color="k", lw=1.5, label=f"circular disk, θ = {th_eq:.3f} mas")
    # the oblate disk seen along its major and its minor axis: the two envelopes of the PA-dependent model
    from hbtsim.single import ellipse_vis
    for ang, ls, lbl in ((pa, "-", "oblate disk, baseline along the major axis (narrow)"),
                         (pa + 90.0, "--", "oblate disk, baseline along the minor axis (wide)")):
        bv = np.c_[B_grid * np.sin(np.radians(ang)), B_grid * np.cos(np.radians(ang))]
        v2 = np.abs(np.asarray(ellipse_vis(bv, lam0, first["star"], th_maj, q, pa)[0])) ** 2
        ax_v.plot(B_grid, v2.ravel(), color="tab:orange", lw=1.3, ls=ls, label=lbl)
    pts_sc = ax_v.scatter([], [], c=[], cmap="twilight", vmin=0, vmax=180, s=28, zorder=5)
    err_containers = []
    sig_min = min(m["sigma"].min() for m in per) / np.sqrt(nbin_show)
    ax_v.set_xlabel(f"projected baseline [m] (equivalent at {lam0:.0f} nm)"); ax_v.set_ylabel(r"$|V|^2$")
    ax_v.set_ylim(-1.5 * sig_min, max(0.05, 1.25 * curve_c[B_grid >= 0.9 * bmin].max())); ax_v.set_xlim(B_grid[0], B_grid[-1])
    ax_v.legend(fontsize=8, loc="upper right")
    cb = fig.colorbar(pts_sc, ax=ax_v, pad=0.01, fraction=0.04); cb.set_label("baseline PA [deg]", fontsize=8)
    v_txt = ax_v.text(0.03, 0.97, "", transform=ax_v.transAxes, va="top", fontsize=9)

    nights_ax = np.arange(1, n_nights + 1)
    ax_q.plot(nights_ax, (q - 1) / sig_q, color="tab:blue", lw=1.5)
    ax_q.axhline(n_sigma, color="tab:red", ls=":", lw=0.9, label=f"{n_sigma:g}σ")
    q_pt, = ax_q.plot([], [], "o", color="tab:blue", ms=7)
    ax_q.set_xlabel("nights"); ax_q.set_ylabel("(q − 1) / σ(q)"); ax_q.set_xlim(0, n_nights + 1)
    ax_q.set_ylim(0, max(n_sigma * 1.3, 1.1 * (q - 1) / sig_q[-1]))
    ax_q.set_title("significance of the oblateness (θ marginalized, PA known)", fontsize=9)
    ax_q.legend(fontsize=8, loc="upper left")
    q_txt = ax_q.text(0.97, 0.05, "", transform=ax_q.transAxes, ha="right", va="bottom", fontsize=9)

    def bins_of(n):
        nb = n // nbin_show + (1 if n % nbin_show else 0)
        return [slice(i * nbin_show, min((i + 1) * nbin_show, n)) for i in range(nb)]

    def draw(j, k):
        ci = sched[j]; m = per[ci]; n = m["mids"].size
        for pth, pt in enumerate(uv_pts):
            if pth < m["n_pairs"]:
                pt.set_data([m["bvec"][pth, k, 0]], [m["bvec"][pth, k, 1]])
            else:
                pt.set_data([], [])
        uv_txt.set_text(f"night {j + 1}/{n_nights}: {m['label'][:28]}\nH = {m['mids'][k]:+.2f} h, alt {m['alt'][k]:.0f}°\n"
                        + "\n".join(f"{nm}: B {m['blen'][pth, k]:.0f} m at PA {m['pa_track'][pth, k]:.0f}°"
                                    for pth, nm in enumerate(m["pair_names"])))
        for ec in err_containers:
            ec.remove()
        err_containers.clear()
        xs, ys, es, cs = [], [], [], []
        for cc in sorted(set(sched[:j + 1])):                   # the nights so far, averaged per configuration
            mm = per[cc]; nn = mm["mids"].size
            js = [jj for jj in range(j + 1) if sched[jj] == cc]
            for pth in range(mm["n_pairs"]):
                for sl in bins_of(nn):
                    vals = []
                    for jj in js:
                        stop = sl.stop if jj < j else min(sl.stop, k + 1)
                        if stop > sl.start:
                            vals.append(meas[jj][1][pth, sl.start:stop])
                    if not vals:
                        continue
                    v = np.concatenate(vals)
                    xs.append(b_eq[cc][pth][sl].mean()); ys.append(v.mean()); cs.append(mm["pa_track"][pth][sl].mean())
                    es.append(mm["sigma"][pth] / np.sqrt(v.size))
        pts_sc.set_offsets(np.c_[xs, ys] if xs else np.zeros((0, 2))); pts_sc.set_array(np.array(cs))
        if xs:
            err_containers.append(ax_v.errorbar(xs, ys, yerr=es, fmt="none", ecolor="0.4", capsize=2, lw=0.8, zorder=4))
        v_txt.set_text(f"{j + 1} night(s)" + (f", block {k + 1}/{n}" if k < n - 1 else "")
                       + f"\nq = {q_est[j]:.4f} ± {sig_q[j]:.4f}")
        if k == n - 1:
            q_pt.set_data([j + 1], [(q - 1) / sig_q[j]])
            q_txt.set_text(f"after {j + 1} nights: {(q - 1) / sig_q[j]:.1f}σ")

    writer = FFMpegWriter(fps=fps, metadata={"title": f"hbtsim {target.name} oblateness"}) if shutil.which("ffmpeg") else None
    # every night with a new configuration is swept block by block; a repeat of a configuration lands as one frame
    def sweep_nights():
        seen = set()
        for j in range(n_nights):
            n = per[sched[j]]["mids"].size
            if sched[j] not in seen:
                seen.add(sched[j])
                step = max(1, n // 40)
                yield j, list(range(0, n, step)) + [n - 1], True
            else:
                yield j, [n - 1], False
    if writer is None:
        frames_dir = path.with_suffix(""); frames_dir.mkdir(exist_ok=True); f = 0
        for j, ks, _ in sweep_nights():
            for k in ks:
                draw(j, k); fig.savefig(frames_dir / f"frame_{f:04d}.png", dpi=dpi); f += 1
        print(f"  ffmpeg not found: {f} PNG frames in {frames_dir}"); plt.close(fig); return
    with writer.saving(fig, str(path), dpi=dpi):
        for j, ks, swept in sweep_nights():
            for k in ks:
                draw(j, k); writer.grab_frame()
            for _ in range(fps if swept else max(1, fps // 3)):
                writer.grab_frame()
        for _ in range(2 * fps):
            writer.grab_frame()
    plt.close(fig)
    print(f"  wrote {path}")
