"""Red-clump angular sizes with intensity interferometry (Kim & Kaiser 2026,
PASP 138, 044202) on hbtsim's photon budget with NewEra profiles.

Three questions, for HD 17652 (beta For, V = 4.46, 1.835 mas) and HD 360
(V = 5.99, 0.906 mas), two 4 m telescopes:

  (1) Radius convention.  How far apart are the tau = 1 radius, the
      apparent limb and the outer boundary of the spherical models, and
      how does the uniform-disk diameter relate to them per band?
  (2) Multiplexing.  Their single H-band filter (sigma_s < 0.007 in 2 h
      at ~100 m) against many optical channels with real detectors.
  (3) Chromaticity.  theta_UD(lambda) across 400-950 nm at R = 5000.

No 4800 K / log g 2.5 NewEra model exists yet (the cool-star box holds
log g >= 4.0 at these temperatures, and log g 0-2 only below 4000 K or
at 5000 K / log g 0).  Stand-ins, clearly labelled: the 4800 K log g 4.5
dwarf (same T_eff, wrong gravity) and the 5000 K log g 0 supergiant
(bracketing the gravity from the other side).  The script ends with the
model request that would replace them.

    .venv/bin/python scripts/redclump_ii.py [--stand-in dwarf|supergiant|linear] [--hours 2]
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import replace

import numpy as np

from hbtsim.diameter import BANDS, KK_DETECTOR, KK_TELESCOPE, scale_precision_scan
from hbtsim.sed import load_star_tables, with_tables
from hbtsim.single import (HD_17652, HD_360, prepare_single, single_star_vis2,
                           ud_diameter_per_channel)
from hbtsim.snr import (EON_SII_TELESCOPE, EONSII_SPAD, EONSII_SPECTROGRAPH, SPAD_LAMBDA,
                        SPAD_LAMBDA_NG, Spectrograph)

DATA = "data/newera_redclump"
STAND_INS = {"dwarf": "lte04800-4.50-0.0", "supergiant": "lte05000-0.00-0.0"}
TABLES = {"dwarf": ("newera_lte04700-4.50-0.0", "newera_lte04800-4.50-0.0"),
          "supergiant": ("newera_lte05000-0.00-0.0",)}
SPEC_OPT_KK = Spectrograph(lambda_min_nm=400.0, lambda_max_nm=950.0, n_channels=1000, throughput=1.0,
                           name="1000 ch 400-950 nm (KK detector)")
SPEC_320 = Spectrograph(n_channels=320)
SPEC_R5000 = Spectrograph.from_resolving_power(5000.0)
KK_TARGET_SIGMA_S = 0.007          # their HD 17652 H-band result in 2 h


def table_path(model: str, ir: bool) -> str:
    return os.path.join(DATA, f"newera_{model}_{'380-2500nm_0.1nm' if ir else '380-1000nm_0.02nm'}.npz")


def attach(target, model: str | None, ir: bool):
    """The target with a stand-in NewEra table (or the linear law when None)."""
    if model is None:
        return target
    ft, ld = load_star_tables(table_path(model, ir))
    return replace(target, star=with_tables(target.star, ft, ld))


def radius_conventions():
    print("(1) Radius conventions of the spherical models (from the binned tables)")
    print(f"  {'model':26s} {'R_out/R_tau1':>13s} {'R_out/R_edge':>13s} {'R_edge/R_tau1':>14s}")
    for name in ("newera_lte04700-4.50-0.0", "newera_lte04800-4.50-0.0", "newera_lte05000-0.00-0.0"):
        d = np.load(os.path.join(DATA, f"{name}_380-1000nm_0.02nm.npz"))
        r_t = float(d["r_outer_over_tau1"]) if "r_outer_over_tau1" in d else np.nan
        mu_e = float(d["mu_edge"]) if "mu_edge" in d else np.nan
        r_e = 1.0 / np.sqrt(1.0 - mu_e**2)
        print(f"  {name[7:]:26s} {r_t:13.4f} {r_e:13.4f} {r_e / r_t:14.4f}")
    print("  (R_tau1: Rosseland tau = 1; R_edge: where the continuum intensity drops to half;"
          " R_out: the model's outer boundary.  The 4800 K / log g 2.5 giant lies between the rows.)")


def ud_over_ld(target, model, x=1.5):
    """theta_UD / theta_LD per band at pi theta B / lambda = x, 4 m pupils."""
    tgt = attach(target, model, ir=True)
    out = {}
    for band, spec in BANDS.items():
        p = prepare_single(tgt, spec)
        nm = spec.channel_centers_nm
        b = x * nm[0] * 1e-9 / (np.pi * tgt.drawn_diameter_mas * 4.8481368e-9)
        v2 = single_star_vis2(p, b, nm, (4.0, 4.0))[:, 0]
        out[band] = float(ud_diameter_per_channel(v2, b, nm, (4.0, 4.0))[0]) / tgt.theta_ld_mas
    return out


def scan(target, spec, hours, **kw):
    b = np.arange(20.0, 301.0, 5.0)
    bb, sig, best = scale_precision_scan(target, b, spec, t_int_s=hours * 3600.0, **kw)
    return best, sig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stand-in", choices=("dwarf", "supergiant", "linear"), default="dwarf")
    ap.add_argument("--star", choices=("hd17652", "hd360", "both"), default="both")
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    model = None if args.stand_in == "linear" else STAND_INS[args.stand_in]
    label = {"dwarf": "4800 K / log g 4.5 dwarf STAND-IN", "supergiant": "5000 K / log g 0 supergiant STAND-IN",
             "linear": "blackbody + linear law"}[args.stand_in]

    radius_conventions()
    stars = {"hd17652": HD_17652, "hd360": HD_360}
    keys = ("hd17652", "hd360") if args.star == "both" else (args.star,)
    for key in keys:
        t0 = stars[key]
        t_ir = attach(t0, model, ir=True)
        print(f"\n=== {t0.name}: theta_LD {t0.theta_ld_mas} mas, V {t0.v_mag}; profile: {label}; "
              f"drawn diameter {t_ir.drawn_diameter_mas:.4f} mas (r_outer {t_ir.star.radius_scale:.4f}); "
              f"model AB(551) - V = {t_ir.ab_mag(551.0) - t0.v_mag:+.3f}")
        if model is not None:
            r = ud_over_ld(t0, model)
            print("  theta_UD / theta_LD at x = 1.5 per band: "
                  + ", ".join(f"{b} {v:.4f}" for b, v in r.items()))

        print(f"\n(2) Scale precision sigma_s in {args.hours:g} h, two 4 m telescopes, baseline scan 20-300 m")
        print(f"  Kim & Kaiser instrument (throughput 0.3, 42.4 ps FWHM, one filter):")
        for band in ("V", "R", "I", "H", "K"):
            best, sig = scan(t_ir, BANDS[band], args.hours)
            print(f"    {band}: sigma_s = {best.sigma_s:.4f} at B = {best.baseline_m:.0f} m "
                  f"(|V|^2 {best.vis2[0]:.2f}; rate {best.snr.total_rate_cps[0]:.2e} cps/tel)"
                  + (f"  <- paper: < {KK_TARGET_SIGMA_S} at ~100 m" if band == "H" and key == "hd17652" else ""))
        t_opt = attach(t0, model, ir=False)
        print(f"  Multiplexed optical backends (hours to sigma_s = {KK_TARGET_SIGMA_S}):")
        cases = [
            ("KK detector, 1000 ch 400-950 nm", SPEC_OPT_KK, KK_TELESCOPE, KK_DETECTOR, "unpolarized"),
            ("SPAD Lambda 320 ch, time-tag (1e8 cps)", SPEC_320, KK_TELESCOPE, SPAD_LAMBDA, "unpolarized"),
            ("SPAD Lambda 320 ch, correlator", SPEC_320, KK_TELESCOPE, SPAD_LAMBDA_NG, "unpolarized"),
            ("R = 5000 (4325 ch), correlator", SPEC_R5000, KK_TELESCOPE, SPAD_LAMBDA_NG, "unpolarized"),
            ("R = 5000, correlator + PBS", SPEC_R5000, KK_TELESCOPE, SPAD_LAMBDA_NG, "pbs"),
            ("EON-SII 1000 ch 400-550 nm, QUASAR SPAD (1 GHz link)", EONSII_SPECTROGRAPH,
             EON_SII_TELESCOPE, EONSII_SPAD, "unpolarized"),
        ]
        for name, spec, tel, det, pol in cases:
            best, sig = scan(t_opt, spec, args.hours, telescope=tel, detector=det, polarization_mode=pol)
            h_to = args.hours * (best.sigma_s / KK_TARGET_SIGMA_S) ** 2
            lim = f" READOUT-LIMITED x{best.snr.readout_scale:.2g}" if best.snr.readout_limited else ""
            print(f"    {name:52s} sigma_s = {best.sigma_s:.4f} at B = {best.baseline_m:.0f} m -> "
                  f"{h_to:7.2f} h; rate {best.snr.total_rate_cps[0]:.2e} cps/tel{lim}")

        if model is not None:
            print("\n(3) Chromaticity: theta_UD(lambda) at R = 5000 over 400-950 nm at the optical-best baseline")
            best, _ = scan(t_opt, SPEC_R5000, args.hours, telescope=KK_TELESCOPE, detector=SPAD_LAMBDA_NG)
            p = prepare_single(t_opt, SPEC_R5000)
            nm = SPEC_R5000.channel_centers_nm
            v2 = single_star_vis2(p, best.baseline_m, nm, (4.0, 4.0))[:, 0]
            th = ud_diameter_per_channel(v2, best.baseline_m, nm, (4.0, 4.0))
            for lo, hi in ((400, 450), (450, 500), (500, 550), (550, 650), (650, 800), (800, 950)):
                m = (nm >= lo) & (nm < hi)
                print(f"    {lo}-{hi} nm: theta_UD / theta_LD median {np.median(th[m]) / t0.theta_ld_mas:.4f}, "
                      f"range {th[m].min() / t0.theta_ld_mas:.4f}-{th[m].max() / t0.theta_ld_mas:.4f}")
            print(f"    per-channel sigma(theta_UD)/theta in {args.hours:g} h: median "
                  f"{np.median(best.sigma_vis2 / np.abs(best.dvis2_ds)):.3g}; B = {best.baseline_m:.0f} m")
            if not args.no_figures:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
                ax[0].plot(nm, v2, lw=0.6)
                ax[0].set_ylabel(r"$|V|^2$ (4 m pupils)")
                ax[1].plot(nm, th / t0.theta_ld_mas, lw=0.6)
                ax[1].set_ylabel(r"$\theta_{UD}/\theta_{LD}$")
                ax[1].set_xlabel("Vacuum wavelength [nm]")
                ax[0].set_title(f"{t0.name}: {label}, B = {best.baseline_m:.0f} m, R = 5000")
                fig.tight_layout()
                out = f"output/redclump_{key}_{args.stand_in}_chromatic.png"
                fig.savefig(out, dpi=140)
                plt.close(fig)
                print(f"    wrote {out}")

    print("\nModel request (Hamburg): NewEra HSR-RF at [M/H] = 0 for T_eff 4700, 4800, 4900 K x "
          "log g 2.0, 2.5, 3.0 (the red clump; 9 models), optionally 4800 K / 2.5 at [M/H] -0.5 and +0.5.")


if __name__ == "__main__":
    main()
