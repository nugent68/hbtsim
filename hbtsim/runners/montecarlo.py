"""Runner `montecarlo`: the EON-SII design-study Monte Carlo rerun with
hbtsim's photon budget, and a ladder that closes the gap to the study's
quoted precision one assumption at a time (formerly
scripts/eonsii_montecarlo.py).

The design study (Schweizer et al. 2026, arXiv:2608.17444, Sec. 6.2)
simulates Sirius B (V = 8.44, 30 uas) for 10 h at three zenith angles with
1000 channels and recovers 0.02955 +/- 0.0014 mas (4.7 %) with the
MCP-PMT; its Table 2 quotes 1.5 / 6 / 37.5 h (MCP-PMT) and 0.33 / 1.3 /
8.2 h (SPAD) to 10 / 5 / 2 % diameter precision at 1.5-2 km.  Its
estimator constants (b_el, F, eta_vis, dt_res) are not published.

The campaign's target is a uniform-disk target, its instrument array a
two-station pair -- or a multi-station network, in which case every pair
is run analytically and combined (run_network); every backend is run in
turn.  Knobs under `options`:
theta_true_mas, truth_ld_u, zenith_angles_deg, t_total_h,
blocks_per_zenith, n_realizations, estimators, seed, bin_ps,
lag_half_range_ps, sideband_sigma, background (a list; the first is the
accidental model of the main Monte Carlo, every further one is rerun with
the matched filter only), paper.{theta_mas, sigma_mas,
hours_to_10pct.{mcp, spad}} (the comparison numbers; a backend maps to
"mcp" when its detector name contains "MCP", else "spad").
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..estimators import implied_dt_res, optimal_box_half_width
from ..montecarlo import MCConfig, expected_counts, fit_ud, run_mc
from . import write_tables

ESTIMATORS = ("matched", "box_opt", "box_sigma_raw", "box_tdc_raw")


def analytic_theta_precision(cfg: MCConfig) -> float:
    """sigma(theta)/theta in cfg.t_total_h from the analytic matched-filter
    sigma of every channel and block."""
    e = expected_counts(cfg)
    _, s = fit_ud(e.vis2_true, e.analytic_sigma_vis2(), e.b_proj_m, e.nm, cfg.theta_true_mas)
    return s / cfg.theta_true_mas


def hours_to(prec_frac_10h: float, target: float, t_total_h: float = 10.0) -> float:
    return t_total_h * (prec_frac_10h / target) ** 2


def best_baseline(cfg: MCConfig, grid=np.arange(1000.0, 3601.0, 100.0)):
    precs = [analytic_theta_precision(replace(cfg, ground_baseline_m=b)) for b in grid]
    k = int(np.argmin(precs))
    return float(grid[k]), float(precs[k])


def paper_key(detector) -> str:
    """'mcp' for an MCP-PMT detector, 'spad' otherwise."""
    return "mcp" if "MCP" in str(getattr(detector, "name", "")).upper() else "spad"


def _finite(x):
    x = float(x)
    return x if math.isfinite(x) else None


def _stats(st) -> dict:
    return {k: _finite(getattr(st, k)) for k in ("theta_mean", "theta_std", "bias_frac",
                                                   "precision_frac", "sigma_pred", "pull_std")}


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def run(campaign, cat, opts, out_dir: Path) -> dict:
    target = campaign.target
    array = campaign.array
    opt = campaign.option
    theta = float(opt("options.theta_true_mas", target.theta_mas))
    n_real = int(opt("options.n_realizations", 100))
    estimators = tuple(opt("options.estimators", ESTIMATORS))
    seed = int(opt("options.seed", 11))
    backgrounds = list(opt("options.background", ["singles", "sideband"])) or ["singles"]
    paper = opt("options.paper") or {}
    paper_mc = (dict(theta=paper["theta_mas"], sigma=paper["sigma_mas"])
                if "theta_mas" in paper and "sigma_mas" in paper else None)
    paper_h10 = dict(paper.get("hours_to_10pct") or {})
    cfg_kw = dict(theta_true_mas=theta, ld_u=float(opt("options.truth_ld_u", 0.0)),
                  zenith_angles_deg=tuple(opt("options.zenith_angles_deg", (45.0, 52.5, 60.0))),
                  t_total_h=float(opt("options.t_total_h", 10.0)),
                  blocks_per_zenith=int(opt("options.blocks_per_zenith", 1)),
                  bin_ps=float(opt("options.bin_ps", 3.125)),
                  lag_half_range_ps=float(opt("options.lag_half_range_ps", 400.0)),
                  sideband_sigma=float(opt("options.sideband_sigma", 6.0)),
                  background=backgrounds[0])
    t_tot = cfg_kw["t_total_h"]
    if len(array.stations) > 2:
        return run_network(campaign, cat, opts, out_dir, target, array, cfg_kw, theta, n_real, seed)
    results = {}
    for b in campaign.backends:
        det = b.detector if b.detector is not None else array.stations[0].detector
        key = paper_key(det)
        h10_paper = paper_h10.get(key)
        cfg = MCConfig.from_target(target, array, b.spectrograph, detector=det,
                                   polarization_mode=b.polarization_mode, **cfg_kw)
        e = expected_counts(cfg)
        sig = float(np.median(e.sigma_pair))
        print(f"\n=== {b.name} [{det.name}]: {target.name}, theta {theta} mas, {t_tot:g} h at zenith "
              f"{cfg.zenith_angles_deg} deg from {cfg.site.name}, PA {cfg.ground_pa_deg:.0f} "
              f"{cfg.ground_baseline_m:.0f} m (projected {', '.join(f'{x:.0f}' for x in e.b_proj_m)} m), "
              f"{cfg.spectrograph.n_channels} ch, {cfg.polarization_mode} ===")
        print(f"  per-channel rate {np.median(e.rates):.0f} cps, per telescope "
              f"{e.rates[:, 0].sum():.2e} cps; tau_c {np.median(e.tau_c) * 1e12:.2f} ps; "
              f"pair sigma {sig * 1e12:.1f} ps; |V|^2 {e.vis2_true.min():.2f}-{e.vis2_true.max():.2f}")

        # ---- Monte Carlo (Poisson histograms) ----
        mc = run_mc(cfg, n_real=n_real, estimators=estimators, seed=seed)
        extra = [(f"matched, {bg} accidentals", f"matched_{bg}",
                  run_mc(replace(cfg, background=bg), n_real=n_real, estimators=("matched",),
                         seed=seed).stats["matched"]) for bg in backgrounds[1:]]
        print(f"  Monte Carlo ({n_real} realizations, Poisson coincidence histograms, "
              f"accidentals from the {backgrounds[0]}):")
        mc_out = {}
        for label, name, st in [(n, n, s) for n, s in mc.stats.items()] + extra:
            print(f"    {label:30s} theta = {st.theta_mean:.5f} mas (bias {100 * st.bias_frac:+.1f} %), "
                  f"scatter {100 * st.precision_frac:.2f} %, fitted sigma {100 * st.sigma_pred / theta:.2f} %, "
                  f"pull std {st.pull_std:.2f}")
            mc_out[name] = _stats(st)
        base = analytic_theta_precision(cfg)
        h = {p: hours_to(base, p, t_tot) for p in (0.10, 0.05, 0.02)}
        print(f"    analytic matched-filter precision: {100 * base:.2f} % in {t_tot:g} h "
              f"(hours to 10/5/2 %: {h[0.10]:.1f} / {h[0.05]:.1f} / {h[0.02]:.0f})")
        paper_ratio = None
        if key == "mcp" and paper_mc is not None:
            paper_ratio = mc.stats["matched"].precision_frac / (paper_mc["sigma"] / paper_mc["theta"])
            print(f"    design study MC: {paper_mc['theta']} +/- {paper_mc['sigma']} mas "
                  f"({100 * paper_mc['sigma'] / paper_mc['theta']:.1f} %) -> our scatter is "
                  f"{paper_ratio:.1f}x theirs")

        # ---- the ladder (analytic, one assumption at a time) ----
        print(f"  Gap ladder (hours to 10 %"
              + (f"; paper {h10_paper} h" if h10_paper is not None else "") + "):")
        rows = []
        prec = base
        rows.append((f"our budget, matched filter, {cfg.polarization_mode}", prec))
        prec_pbs = analytic_theta_precision(replace(cfg, polarization_mode="pbs"))
        rows.append(("+ polarizing beamsplitter (p2 -> 1 per stream)", prec_pbs))
        # a literal Eq. 5 with eta = 1 on the unpolarized photons: S/N x 2 relative to p2 = 1/2
        prec_eta = base / 2.0
        rows.append(("  or: Eq. 5 with eta = 1 on unpolarized light (x2 S/N)", prec_eta))
        for label, dt in (("sigma_pair", sig), (f"{cfg.bin_ps} ps TDC bin", cfg.bin_ps * 1e-12)):
            f = np.sqrt(np.sqrt(np.pi) * sig / dt)          # full-capture box of half-width... as dt_res
            rows.append((f"  + Eq. 5 with dt_res = {label}, no capture correction (x{f:.2f} S/N)", prec_eta / f))
        b_opt, prec_b = best_baseline(replace(cfg, polarization_mode="pbs"))
        rows.append((f"pbs + diameter-optimal baseline ({b_opt:.0f} m)", prec_b))
        prec_qe = analytic_theta_precision(replace(cfg, polarization_mode="pbs", ground_baseline_m=b_opt,
                                                   detector=replace(det, pde_table_nm=tuple(
                                                       (l, min(1.0, 1.3 * p)) for l, p in det.pde_table_nm))))
        rows.append(("  + QE x 1.3", prec_qe))
        prec_ext = analytic_theta_precision(replace(cfg, polarization_mode="pbs", ground_baseline_m=b_opt,
                                                    extinction="izana"))
        rows.append(("  + Izana extinction (X - 1) beyond the zenith budget", prec_ext))
        g = run_mc(replace(cfg, polarization_mode="pbs", ground_baseline_m=b_opt, channel_corr=0.06),
                   n_real=max(50, n_real // 2), mode="gaussian", seed=5).stats["matched"]
        rows.append(("  + 6 % adjacent-channel correlation (Gaussian MC)", g.precision_frac))
        ladder = []
        for label, p in rows:
            hh = hours_to(p, 0.10, t_tot)
            tail = f"  (x{hh / h10_paper:.1f} paper)" if h10_paper else ""
            print(f"    {label:62s} {100 * p:6.2f} % in {t_tot:g} h -> {hh:8.2f} h{tail}")
            ladder.append([label.strip(), _finite(p), _finite(hh)])
        t_ours = hours_to(prec_b, 0.10, t_tot)
        dt_res_ps = None
        if h10_paper:
            dt_res_ps = implied_dt_res(h10_paper, t_ours, sig) * 1e12
            print(f"  dt_res that Eq. 5 needs to match the paper at the pbs+optimal-baseline budget: "
                  f"eta = 1: {dt_res_ps:.2f} ps "
                  f"(sqrt(pi) sigma = {np.sqrt(np.pi) * sig * 1e12:.1f} ps; optimal honest box half-width "
                  f"{optimal_box_half_width(sig) * 1e12:.1f} ps)")
        write_tables(out_dir, [[lab, "nan" if p is None else f"{100 * p:.2f}",
                                "nan" if hh is None else f"{hh:.2f}"] for lab, p, hh in ladder],
                     ["assumption", f"precision in {t_tot:g} h [%]", "hours to 10 %"],
                     name=f"ladder_{_slug(b.name)}", latex=opts.latex)
        results[b.name] = {
            "detector": det.name, "paper_key": key, "n_channels": cfg.spectrograph.n_channels,
            "baseline_m": cfg.ground_baseline_m, "projected_baselines_m": [float(x) for x in e.b_proj_m],
            "pair_sigma_ps": sig * 1e12,
            "analytic_precision_10h": _finite(base), "t_total_h": t_tot,
            "hours_to_10pct": _finite(h[0.10]), "hours_to_5pct": _finite(h[0.05]),
            "hours_to_2pct": _finite(h[0.02]), "paper_hours_to_10pct": h10_paper,
            "scatter_over_paper": None if paper_ratio is None else _finite(paper_ratio),
            "mc": mc_out, "ladder": ladder,
            "optimal_baseline_m": b_opt, "implied_dt_res_ps": None if dt_res_ps is None else _finite(dt_res_ps),
        }
    return results


def run_network(campaign, cat, opts, out_dir, target, array, cfg_kw, theta, n_real, seed) -> dict:
    """A multi-station array: every pair is its own two-station Monte Carlo
    configuration (the analytic matched-filter precision per pair, the
    Poisson Monte Carlo on the best pair), and the network's precision is
    the inverse-variance sum over the pairs, which all measure the same
    diameter in the same hours."""
    from ..bispectrum import Array
    t_tot = cfg_kw["t_total_h"]
    results = {}
    for b in campaign.backends:
        det = b.detector if b.detector is not None else array.stations[0].detector
        print(f"\n=== {b.name} [{det.name}]: {target.name}, theta {theta} mas, {t_tot:g} h at zenith "
              f"{cfg_kw['zenith_angles_deg']} deg, every pair of {'+'.join(st.name for st in array.stations)} ===")
        pairs, inv_var = [], 0.0
        best = None
        for i, j, _ in array.pairs():
            sub = Array((array.stations[i], array.stations[j]), array.site)
            cfg = MCConfig.from_target(target, sub, b.spectrograph, detector=det,
                                       polarization_mode=b.polarization_mode, **cfg_kw)
            e = expected_counts(cfg)
            prec = analytic_theta_precision(cfg)
            inv_var += 1.0 / prec ** 2 if np.isfinite(prec) and prec > 0 else 0.0
            row = dict(pair=f"{array.stations[i].name}-{array.stations[j].name}", baseline_m=cfg.ground_baseline_m,
                       projected_baselines_m=[float(x) for x in e.b_proj_m], vis2_min=float(e.vis2_true.min()),
                       vis2_max=float(e.vis2_true.max()), rate_per_telescope_cps=float(e.rates[:, 0].sum()),
                       analytic_precision=_finite(prec), hours_to_10pct=_finite(hours_to(prec, 0.10, t_tot)))
            pairs.append(row)
            print(f"  {row['pair']:8s} B {cfg.ground_baseline_m:5.0f} m (projected {', '.join(f'{x:.0f}' for x in e.b_proj_m)}), "
                  f"|V|^2 {e.vis2_true.min():.3f}-{e.vis2_true.max():.3f}, rate {e.rates[:, 0].sum():.2e} cps/tel: "
                  f"sigma(theta)/theta {100 * prec:7.2f} % in {t_tot:g} h -> {hours_to(prec, 0.10, t_tot):9.1f} h to 10 %")
            if best is None or prec < best[0]:
                best = (prec, cfg, row["pair"])
        net = 1.0 / np.sqrt(inv_var) if inv_var > 0 else float("inf")
        h = {p: hours_to(net, p, t_tot) for p in (0.10, 0.05, 0.02)}
        print(f"  NETWORK ({len(pairs)} pairs, inverse-variance sum): sigma(theta)/theta {100 * net:.2f} % in {t_tot:g} h; "
              f"hours to 10/5/2 %: {h[0.10]:.1f} / {h[0.05]:.1f} / {h[0.02]:.1f}")
        mc_out = {}
        if n_real > 0 and best is not None:
            mc = run_mc(best[1], n_real=n_real, estimators=("matched",), seed=seed)
            st = mc.stats["matched"]
            print(f"  Poisson Monte Carlo on the best pair {best[2]} ({n_real} realizations): theta = {st.theta_mean:.5f} mas "
                  f"(bias {100 * st.bias_frac:+.1f} %), scatter {100 * st.precision_frac:.2f} %, fitted sigma "
                  f"{100 * st.sigma_pred / theta:.2f} % (analytic {100 * best[0]:.2f} %)")
            mc_out = {"pair": best[2], **_stats(st)}
        write_tables(out_dir, [[r["pair"], f"{r['baseline_m']:.0f}", f"{r['vis2_min']:.3f}-{r['vis2_max']:.3f}",
                                "nan" if r["analytic_precision"] is None else f"{100 * r['analytic_precision']:.2f}",
                                "nan" if r["hours_to_10pct"] is None else f"{r['hours_to_10pct']:.1f}"] for r in pairs]
                     + [["network", "", "", f"{100 * net:.2f}", f"{h[0.10]:.1f}"]],
                     ["pair", "B [m]", "|V|^2", f"precision in {t_tot:g} h [%]", "hours to 10 %"],
                     name=f"pairs_{_slug(b.name)}", latex=opts.latex)
        results[b.name] = {"detector": det.name, "n_channels": b.spectrograph.n_channels, "t_total_h": t_tot,
                           "pairs": pairs, "network_precision": _finite(net),
                           "hours_to_10pct": _finite(h[0.10]), "hours_to_5pct": _finite(h[0.05]), "hours_to_2pct": _finite(h[0.02]),
                           "mc_best_pair": mc_out}
    return results
