"""The EON-SII design-study Monte Carlo, rerun with hbtsim's photon budget,
and a ladder that closes the gap to the study's quoted precision one
assumption at a time.

The design study (Schweizer et al. 2026, arXiv:2608.17444, Sec. 6.2)
simulates Sirius B (V = 8.44, 30 uas) for 10 h at three zenith angles with
1000 channels and recovers 0.02955 +/- 0.0014 mas (4.7 %) with the
MCP-PMT; its Table 2 quotes 1.5 / 6 / 37.5 h (MCP-PMT) and 0.33 / 1.3 /
8.2 h (SPAD) to 10 / 5 / 2 % diameter precision at 1.5-2 km.  Its
estimator constants (b_el, F, eta_vis, dt_res) are not published.

    .venv/bin/python scripts/eonsii_montecarlo.py [--n-real 100] [--detector mcp|spad]
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from hbtsim.catalog import Catalog
from hbtsim.estimators import implied_dt_res, optimal_box_half_width
from hbtsim.montecarlo import MCConfig, expected_counts, fit_ud, run_mc

CAT = Catalog(env=False)

PAPER_MC = dict(theta=0.02955, sigma=0.0014)                  # MCP-PMT, 10 h
PAPER_H10 = {"mcp": 1.5, "spad": 0.33}                        # hours to 10 %


def analytic_theta_precision(cfg: MCConfig) -> float:
    """sigma(theta)/theta in cfg.t_total_h from the analytic matched-filter
    sigma of every channel and block."""
    e = expected_counts(cfg)
    _, s = fit_ud(e.vis2_true, e.analytic_sigma_vis2(), e.b_proj_m, e.nm, cfg.theta_true_mas)
    return s / cfg.theta_true_mas


def hours_to(prec_frac_10h: float, target: float) -> float:
    return 10.0 * (prec_frac_10h / target) ** 2


def best_baseline(cfg: MCConfig, grid=np.arange(1000.0, 3601.0, 100.0)):
    precs = [analytic_theta_precision(replace(cfg, ground_baseline_m=b)) for b in grid]
    k = int(np.argmin(precs))
    return float(grid[k]), float(precs[k])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-real", type=int, default=100)
    ap.add_argument("--detector", choices=("mcp", "spad"), default=None, help="default: both")
    ap.add_argument("--theta", type=float, default=0.0295, help="true diameter [mas] (paper MC: 0.030)")
    args = ap.parse_args()

    for key in ((args.detector,) if args.detector else ("mcp", "spad")):
        det = CAT.load_detector("eonsii_mcp_pmt" if key == "mcp" else "eonsii_spad")
        cfg = MCConfig.from_target(CAT.load_target("sirius_b"),
                                   CAT.load_array("eonsii_pair_teide", baseline_m=1750.0, pa_deg=90.0),
                                   CAT.load_spectrograph("eonsii_1000ch"),
                                   detector=det, theta_true_mas=args.theta)
        e = expected_counts(cfg)
        sig = float(np.median(e.sigma_pair))
        print(f"\n=== {det.name}: Sirius B, theta {args.theta} mas, 10 h at zenith "
              f"{cfg.zenith_angles_deg} deg from Teide, E-W {cfg.ground_baseline_m:.0f} m "
              f"(projected {', '.join(f'{b:.0f}' for b in e.b_proj_m)} m), 1000 ch ===")
        print(f"  per-channel rate {np.median(e.rates):.0f} cps, per telescope "
              f"{e.rates[:, 0].sum():.2e} cps; tau_c {np.median(e.tau_c) * 1e12:.2f} ps; "
              f"pair sigma {sig * 1e12:.1f} ps; |V|^2 {e.vis2_true.min():.2f}-{e.vis2_true.max():.2f}")

        # ---- Monte Carlo (Poisson histograms) ----
        mc = run_mc(cfg, n_real=args.n_real,
                    estimators=("matched", "box_opt", "box_sigma_raw", "box_tdc_raw"), seed=11)
        mc_sb = run_mc(replace(cfg, background="sideband"), n_real=args.n_real,
                       estimators=("matched",), seed=11).stats["matched"]
        print(f"  Monte Carlo ({args.n_real} realizations, Poisson coincidence histograms, "
              f"accidentals from the singles rates):")
        for name, st in list(mc.stats.items()) + [("matched, sideband accidentals", mc_sb)]:
            print(f"    {name:30s} theta = {st.theta_mean:.5f} mas (bias {100 * st.bias_frac:+.1f} %), "
                  f"scatter {100 * st.precision_frac:.2f} %, fitted sigma {100 * st.sigma_pred / args.theta:.2f} %, "
                  f"pull std {st.pull_std:.2f}")
        base = analytic_theta_precision(cfg)
        print(f"    analytic matched-filter precision: {100 * base:.2f} % in 10 h "
              f"(hours to 10/5/2 %: {hours_to(base, .10):.1f} / {hours_to(base, .05):.1f} / {hours_to(base, .02):.0f})")
        if key == "mcp":
            print(f"    design study MC: {PAPER_MC['theta']} +/- {PAPER_MC['sigma']} mas "
                  f"({100 * PAPER_MC['sigma'] / PAPER_MC['theta']:.1f} %) -> our scatter is "
                  f"{mc.stats['matched'].precision_frac / (PAPER_MC['sigma'] / PAPER_MC['theta']):.1f}x theirs")

        # ---- the ladder (analytic, one assumption at a time) ----
        print(f"  Gap ladder (hours to 10 %; paper {PAPER_H10[key]} h):")
        rows = []
        prec = base
        rows.append(("our budget, matched filter, unpolarized", prec))
        prec_pbs = analytic_theta_precision(replace(cfg, polarization_mode="pbs"))
        rows.append(("+ polarizing beamsplitter (p2 -> 1 per stream)", prec_pbs))
        # a literal Eq. 5 with eta = 1 on the unpolarized photons: S/N x 2 relative to p2 = 1/2
        prec_eta = base / 2.0
        rows.append(("  or: Eq. 5 with eta = 1 on unpolarized light (x2 S/N)", prec_eta))
        for label, dt in (("sigma_pair", sig), ("3.125 ps TDC bin", 3.125e-12)):
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
                   n_real=max(50, args.n_real // 2), mode="gaussian", seed=5).stats["matched"]
        rows.append(("  + 6 % adjacent-channel correlation (Gaussian MC)", g.precision_frac))
        for label, p in rows:
            h = hours_to(p, 0.10)
            print(f"    {label:62s} {100 * p:6.2f} % in 10 h -> {h:8.2f} h  (x{h / PAPER_H10[key]:.1f} paper)")
        t_ours = hours_to(prec_b, 0.10)
        print(f"  dt_res that Eq. 5 needs to match the paper at the pbs+optimal-baseline budget: "
              f"eta = 1: {implied_dt_res(PAPER_H10[key], t_ours, sig) * 1e12:.2f} ps "
              f"(sqrt(pi) sigma = {np.sqrt(np.pi) * sig * 1e12:.1f} ps; optimal honest box half-width "
              f"{optimal_box_half_width(sig) * 1e12:.1f} ps)")


if __name__ == "__main__":
    main()
