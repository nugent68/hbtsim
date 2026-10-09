"""Runner `specmovie`: a compact star (a uniform-disk target such as a white
dwarf) on a multi-telescope network with a spectrograph, one night as a
movie -- every pair's channels landing on the disk's |V|^2 against spatial
frequency B/lambda, and the diameter converging as the blocks accumulate.

Campaign fields: target (type uniform_disk), instrument.array (two or more
stations), one backend (spectrograph + detector), night.block_minutes,
options:
  theta_true_mas, truth_ld_u    the drawn disk (default the target's)
  glare_fraction                incoherent light from a bright neighbour (Sirius A for
                                Sirius B) as a fraction of the target's detected rate: it
                                adds to the accidentals and nothing to the signal (default 0)
  display_bins                  wavelength bins for the display and the fit (default 40)
  compare_pair                  the pair fitted alone for comparison (default the first)
  reference_diameters           [{label, mas, sigma_mas}] drawn on the diameter panel
  seed, fps, dpi
Outputs <name>.mp4 and results.json (per-pair and network precision for the
night, the realization's fit, rates and readout scales).
"""

from __future__ import annotations

import shutil
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from hbtsim.bispectrum import Array
from hbtsim.geometry import hour_angle_blocks, hour_angle_window
from hbtsim.montecarlo import fit_ud, ud_vis2
from hbtsim.snr import Observation, g2_snr, incident_rate, polarization_streams, readout_scale


def _fin(xs):
    return [float(x) if np.isfinite(x) else None for x in xs]


def run(campaign, cat, opts, out_dir: Path) -> dict:
    target = campaign.target
    if not hasattr(target, "theta_mas"):
        raise SystemExit("runner specmovie needs a uniform-disk target (type uniform_disk)")
    array = campaign.array
    if array is None or len(array.stations) < 2 or array.site is None:
        raise SystemExit("runner specmovie needs instrument.array with a site and at least two stations")
    if len(campaign.backends) != 1:
        raise SystemExit("runner specmovie takes one backend (spectrograph + detector)")
    b = campaign.backends[0]
    spec, det, pol = b.spectrograph, b.detector, b.polarization_mode
    arr = Array(tuple(replace(s, detector=det) for s in array.stations), array.site)
    opt = campaign.option
    block_minutes = float(opt("night.block_minutes", 5.0))
    min_alt = float(opt("night.min_alt_deg", 30.0))
    theta0 = float(opt("options.theta_true_mas", target.theta_mas))
    ld_u = float(opt("options.truth_ld_u", target.ld_u))
    glare = float(opt("options.glare_fraction", 0.0))
    nbins = int(opt("options.display_bins", 40))
    rng = np.random.default_rng(int(opt("options.seed", 11)))
    refs = opt("options.reference_diameters", []) or []

    h0, h1 = hour_angle_window(target.dec_deg, arr.site.latitude_deg, min_alt)
    mids = hour_angle_blocks(h0, h1, block_minutes)
    if mids.size == 0:
        raise SystemExit("the target never rises above the altitude limit from this site")
    block_s = (h1 - h0) / mids.size * 3600.0
    nm, w = spec.channel_centers_nm, spec.channel_widths_nm
    pairs = arr.pairs()
    names = [f"{arr.stations[i].name}-{arr.stations[j].name}" for i, j, _ in pairs]
    bvec = np.array([[pr[2] for pr in arr.projected(float(H), target.dec_deg).pairs()] for H in mids]).transpose(1, 0, 2)
    blen = np.hypot(bvec[..., 0], bvec[..., 1])                      # (n_pairs, n_blocks)
    n_streams, frac, p2, _ = polarization_streams(pol)
    obs = Observation(wavelength_nm=nm, filter_width_nm=w, t_int_s=block_s, polarization_mode=pol,
                      backend_throughput=spec.throughput)
    mag = np.full(nm.size, float(target.mag_ab))

    # sigma(|V|^2) per pair and channel for one block (no airmass term: constant along the night)
    sig = np.zeros((len(pairs), nm.size))
    rate_tel, scales = np.zeros((len(pairs), 2)), np.ones(len(pairs))
    for p, (i, j, _) in enumerate(pairs):
        t1, t2 = arr.stations[i].telescope, arr.stations[j].telescope
        r1 = np.asarray(incident_rate(mag, t1, det, obs)) * n_streams
        r2 = np.asarray(incident_rate(mag, t2, det, obs)) * n_streams
        scale = min(readout_scale(det, float(r1.sum()), float(r1.max()) / det.n_pixels),
                    readout_scale(det, float(r2.sum()), float(r2.max()) / det.n_pixels))
        mag_eff = mag - 2.5 * np.log10(scale) if scale < 1.0 else mag
        sky = glare * 0.5 * (r1 + r2) * scale / n_streams                  # the neighbour's light, per channel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = g2_snr(1.0, mag_eff, replace(obs, sky_cps=sky), telescope1=t1, telescope2=t2, detector1=det)
        sig[p] = 1.0 / np.asarray(r.snr, dtype=float)
        rate_tel[p] = (float(np.sum(r.rate1_cps)) * n_streams, float(np.sum(r.rate2_cps)) * n_streams)
        scales[p] = scale
    v2_true = ud_vis2(theta0, blen[..., None], nm[None, None, :], ld_u)      # (n_pairs, n_blocks, n_ch)
    v2_meas = v2_true + rng.normal(0.0, 1.0, v2_true.shape) * sig[:, None, :]

    # wavelength bins (display and the running fit)
    edges = np.linspace(float(nm[0]), float(nm[-1]) + 1e-9, nbins + 1)
    idx = np.clip(np.searchsorted(edges, nm, side="right") - 1, 0, nbins - 1)
    wgt = 1.0 / sig ** 2
    lamb = np.array([np.sum(nm[idx == q] * wgt[0, idx == q]) / np.sum(wgt[0, idx == q]) for q in range(nbins)])
    sigb = np.array([[1.0 / np.sqrt(np.sum(wgt[p, idx == q])) for q in range(nbins)] for p in range(len(pairs))])

    def binned(a):
        out = np.zeros(a.shape[:2] + (nbins,))
        for q in range(nbins):
            m = idx == q
            out[..., q] = np.sum(a[..., m] * wgt[:, None, m], axis=-1) / np.sum(wgt[:, m], axis=-1)[:, None]
        return out
    v2b_true, v2b_meas = binned(v2_true), binned(v2_meas)

    cmp_name = str(opt("options.compare_pair", names[0]))
    pc = names.index(cmp_name) if cmp_name in names else 0
    th_net, sg_net, th_pair, sg_pair = [], [], [], []
    for k in range(mids.size):
        V = v2b_meas[:, :k + 1, :].reshape(-1, nbins).T
        S = np.repeat(sigb[:, None, :], k + 1, axis=1).reshape(-1, nbins).T
        th, s = fit_ud(V, S, blen[:, :k + 1].reshape(-1), lamb, theta0, ld_u)
        th_net.append(th); sg_net.append(s)
        Vp = v2b_meas[pc, :k + 1, :].T
        Sp = np.repeat(sigb[pc][None, :], k + 1, axis=0).T
        th, s = fit_ud(Vp, Sp, blen[pc, :k + 1], lamb, theta0, ld_u)
        th_pair.append(th); sg_pair.append(s)
    # the exact (all-channel) analytic precision of the night, per pair and for the network
    per_pair = []
    for p in range(len(pairs)):
        _, s = fit_ud(v2_true[p].T, np.repeat(sig[p][None, :], mids.size, axis=0).T, blen[p], nm, theta0, ld_u)
        per_pair.append(s / theta0)
    _, s_all = fit_ud(v2_true.reshape(-1, nm.size).T, np.repeat(sig[:, None, :], mids.size, axis=1).reshape(-1, nm.size).T,
                      blen.reshape(-1), nm, theta0, ld_u)
    net_prec = s_all / theta0
    hours = mids.size * block_s / 3600.0

    print(f"{target.name}: theta {theta0:.4f} mas (u = {ld_u:g}), AB {target.mag_ab:.2f}, dec {target.dec_deg:+.1f}; "
          f"{'+'.join(s.name for s in arr.stations)} at {arr.site.name}, {b.name}")
    print(f"  {mids.size} blocks of {block_minutes:g} min ({hours:.1f} h above {min_alt:g} deg), {nm.size} channels, "
          f"glare fraction {glare:g}" + ("" if glare == 0 else " (incoherent neighbour light added to the accidentals)"))
    for p in range(len(pairs)):
        print(f"  {names[p]:8s} B {blen[p].min():5.0f}-{blen[p].max():5.0f} m, |V|^2 {v2_true[p].min():.3f}-{v2_true[p].max():.3f}, "
              f"rate {rate_tel[p, 0]:.2e}/{rate_tel[p, 1]:.2e} cps/tel{' x%.2f readout' % scales[p] if scales[p] < 1 else ''}: "
              f"sigma(theta)/theta {100 * per_pair[p]:6.2f} % in the night")
    print(f"  NETWORK: sigma(theta)/theta {100 * net_prec:.2f} % in {hours:.1f} h (all channels); the realization's fit "
          f"{th_net[-1]:.5f} +/- {sg_net[-1]:.5f} mas ({100 * sg_net[-1] / theta0:.2f} %), {cmp_name} alone "
          f"{th_pair[-1]:.5f} +/- {sg_pair[-1]:.5f} mas ({100 * sg_pair[-1] / theta0:.1f} %)")

    out = {"target": target.name, "backend": b.name, "n_channels": int(nm.size), "block_minutes": block_minutes,
           "n_blocks": int(mids.size), "hours": hours, "theta_true_mas": theta0, "ld_u": ld_u, "glare_fraction": glare,
           "pairs": [{"pair": names[p], "baseline_min_m": float(blen[p].min()), "baseline_max_m": float(blen[p].max()),
                      "vis2_min": float(v2_true[p].min()), "vis2_max": float(v2_true[p].max()),
                      "rate_cps_per_telescope": [float(x) for x in rate_tel[p]], "readout_scale": float(scales[p]),
                      "sigma_theta_frac_night": float(per_pair[p])} for p in range(len(pairs))],
           "network_sigma_theta_frac_night": float(net_prec),
           "fit_theta_mas": _fin(th_net), "fit_sigma_mas": _fin(sg_net),
           "compare_pair": cmp_name, "pair_fit_theta_mas": _fin(th_pair), "pair_fit_sigma_mas": _fin(sg_pair)}
    if opts.figures:
        path = out_dir / f"{campaign.name}.mp4"
        _render(path, target, b, arr, names, mids, blen, lamb, v2b_true, v2b_meas, sigb, theta0, ld_u, th_net, sg_net,
                th_pair, sg_pair, pc, net_prec, refs, glare, fps=int(opt("options.fps", 8)), dpi=int(opt("options.dpi", 110)))
        out["movie"] = str(path)
    return out


def _render(path, target, backend, arr, names, mids, blen, lamb, v2b_true, v2b_meas, sigb, theta0, ld_u, th_net, sg_net,
            th_pair, sg_pair, pc, net_prec, refs, glare, fps=8, dpi=110):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    n_pairs, n = blen.shape
    t_h = (np.arange(n) + 1) * ((mids[-1] - mids[0]) / max(n - 1, 1)) if n > 1 else np.ones(1)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_pairs] if n_pairs <= 10 else plt.cm.hsv(np.linspace(0, 1, n_pairs))
    fig = plt.figure(figsize=(14.5, 5.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.6, 1.2], wspace=0.3, left=0.045, right=0.985, bottom=0.14, top=0.85)
    ax_uv, ax_v, ax_t = (fig.add_subplot(gs[0, k]) for k in range(3))
    fig.suptitle(f"{target.name}: one night on {'+'.join(s.name for s in arr.stations)}, {backend.name}  "
                 f"(θ = {1000 * theta0:.1f} µas, AB {target.mag_ab:.2f}"
                 + (f", neighbour glare {100 * glare:.0f} % of the star" if glare > 0 else "") + ")", fontsize=11)

    # uv coverage: tracks of every pair
    uv = np.array([[pr[2] for pr in arr.projected(float(H), target.dec_deg).pairs()] for H in mids]).transpose(1, 0, 2)
    lim = 1.1 * np.abs(uv).max()
    for p in range(n_pairs):
        ax_uv.plot(uv[p, :, 0], uv[p, :, 1], color=colors[p], lw=1, alpha=0.5)
        ax_uv.plot(-uv[p, :, 0], -uv[p, :, 1], color=colors[p], lw=1, alpha=0.5)
    uv_now = [ax_uv.plot([], [], "o", color=colors[p], ms=5)[0] for p in range(n_pairs)]
    uv_now2 = [ax_uv.plot([], [], "o", color=colors[p], ms=5, alpha=0.4)[0] for p in range(n_pairs)]
    ax_uv.set_xlim(-lim, lim); ax_uv.set_ylim(-lim, lim); ax_uv.set_aspect("equal")
    ax_uv.set_xlabel("u [m] (east)"); ax_uv.set_ylabel("v [m] (north)")
    ax_uv.set_title(f"uv coverage: {n_pairs} pairs", fontsize=9)
    uv_txt = ax_uv.text(0.03, 0.97, "", transform=ax_uv.transAxes, va="top", fontsize=9)

    # |V|^2 against spatial frequency
    x_all = (blen[:, :, None] / (lamb[None, None, :] * 1e-9)) * 1e-9        # (n_pairs, n_blocks, nbins) in Glambda
    xg = np.linspace(0.0, 1.08 * x_all.max(), 300)
    ax_v.plot(xg, ud_vis2(theta0, xg * 1e9, 1e9, ld_u), color="k", lw=1.5, label=f"disk θ = {1000 * theta0:.1f} µas")
    fit_line, = ax_v.plot([], [], color="tab:red", lw=1.2, ls="--", label="network fit")
    nowpts = [ax_v.plot([], [], ".", color=colors[p], ms=4, alpha=0.6)[0] for p in range(n_pairs)]
    acc = ax_v.errorbar([], [], yerr=[], fmt="o", color="0.25", ms=4, capsize=2, lw=1, label="all blocks so far, binned in B/λ")
    nxb = 60
    xedges = np.linspace(0.0, 1.0001 * x_all.max(), nxb + 1)
    ax_v.set_xlim(0, xg[-1]); ax_v.set_ylim(min(0.3, 0.9 * ud_vis2(theta0, x_all.max() * 1e9, 1e9, ld_u)), 1.08)
    ax_v.set_xlabel("spatial frequency B/λ [Gλ]"); ax_v.set_ylabel(r"$|V|^2$")
    ax_v.set_title(f"every pair, every channel ({len(lamb)} wavelength bins shown): the disk's visibility", fontsize=9)
    ax_v.legend(fontsize=8, loc="lower left")
    v_txt = ax_v.text(0.97, 0.97, "", transform=ax_v.transAxes, va="top", ha="right", fontsize=9)

    # diameter against time
    for i_ref, ref in enumerate(refs):
        col = ["tab:orange", "tab:green"][i_ref % 2]
        m, sm = 1000 * float(ref["mas"]), 1000 * float(ref.get("sigma_mas", 0.0))
        if sm > 0:
            ax_t.axhspan(m - sm, m + sm, color=col, alpha=0.2, lw=0)
        ax_t.axhline(m, color=col, lw=1, label=f"{ref['label']}: {m:.1f} µas")
    ax_t.axhline(1000 * theta0, color="0.4", ls="--", lw=1, label=f"truth {1000 * theta0:.1f} µas")
    net_pts = ax_t.errorbar([], [], yerr=[], fmt="o", color="tab:red", ms=4, capsize=2, lw=1, label="network")
    pair_pts = ax_t.errorbar([], [], yerr=[], fmt="o", mfc="none", color="tab:blue", ms=4, capsize=2, lw=0.8, alpha=0.7,
                             label=f"{names[pc]} alone")
    ax_t.set_xlim(0, 1.05 * (t_h[-1] if n > 1 else 1)); ax_t.set_ylim(1000 * theta0 * 0.6, 1000 * theta0 * 1.4)
    ax_t.set_xlabel("hours into the night"); ax_t.set_ylabel("diameter [µas]")
    ax_t.set_title("the diameter from the blocks so far", fontsize=9)
    ax_t.legend(fontsize=8, loc="upper right")
    t_txt = ax_t.text(0.03, 0.03, "", transform=ax_t.transAxes, va="bottom", fontsize=9)

    def frame(k):
        nonlocal acc, net_pts, pair_pts
        for p in range(n_pairs):
            uv_now[p].set_data([uv[p, k, 0]], [uv[p, k, 1]]); uv_now2[p].set_data([-uv[p, k, 0]], [-uv[p, k, 1]])
            nowpts[p].set_data(x_all[p, k], v2b_meas[p, k])
        uv_txt.set_text(f"H = {mids[k]:+.2f} h\nblock {k + 1} of {n}")
        # accumulate all blocks so far into spatial-frequency bins
        X = x_all[:, :k + 1, :].ravel(); Y = v2b_meas[:, :k + 1, :].ravel()
        W = np.repeat(1.0 / sigb[:, None, :] ** 2, k + 1, axis=1).ravel()
        ib = np.clip(np.searchsorted(xedges, X, side="right") - 1, 0, nxb - 1)
        xs, ys, es = [], [], []
        for q in range(nxb):
            m = ib == q
            if m.any():
                ws = W[m]; xs.append(np.sum(X[m] * ws) / ws.sum()); ys.append(np.sum(Y[m] * ws) / ws.sum()); es.append(1 / np.sqrt(ws.sum()))
        acc.remove()
        acc = ax_v.errorbar(xs, ys, yerr=es, fmt="o", color="0.25", ms=4, capsize=2, lw=1)
        fit_line.set_data(xg, ud_vis2(th_net[k], xg * 1e9, 1e9, ld_u))
        v_txt.set_text(f"network fit so far: θ = {1000 * th_net[k]:.2f} ± {1000 * sg_net[k]:.2f} µas")
        net_pts.remove(); pair_pts.remove()
        net_pts = ax_t.errorbar(t_h[:k + 1], 1000 * np.array(th_net[:k + 1]), yerr=1000 * np.array(sg_net[:k + 1]),
                                fmt="o", color="tab:red", ms=4, capsize=2, lw=1)
        okp = np.isfinite(sg_pair[:k + 1]) & (np.array(sg_pair[:k + 1]) < 0.3 * theta0)   # shown once it says something
        pair_pts = ax_t.errorbar(t_h[:k + 1][okp], 1000 * np.array(th_pair[:k + 1])[okp], yerr=1000 * np.array(sg_pair[:k + 1])[okp],
                                 fmt="o", mfc="none", color="tab:blue", ms=4, capsize=2, lw=0.8, alpha=0.7)
        t_txt.set_text(f"after {t_h[k]:.1f} h: network {100 * sg_net[k] / theta0:.1f} %, {names[pc]} alone "
                       + (f"{100 * sg_pair[k] / theta0:.0f} %" if np.isfinite(sg_pair[k]) else "unconstrained")
                       + (f"\nwhole night, all channels: {100 * net_prec:.2f} %" if k == n - 1 else ""))

    writer = FFMpegWriter(fps=fps, bitrate=2400) if shutil.which("ffmpeg") else None
    if writer is None:
        d = path.with_suffix(""); d.mkdir(exist_ok=True)
        for k in range(n):
            frame(k); fig.savefig(d / f"frame_{k:04d}.png", dpi=dpi)
        plt.close(fig); return
    with writer.saving(fig, str(path), dpi):
        for k in range(n):
            frame(k); writer.grab_frame()
        for _ in range(3 * fps):
            writer.grab_frame()
    plt.close(fig)
