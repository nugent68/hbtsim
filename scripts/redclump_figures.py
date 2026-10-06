"""Figures for docs/redclump_note (red-clump angular sizes with intensity
interferometry, Kim & Kaiser 2026, on hbtsim's budget):

  fig 1  limb geometry of the stand-in NewEra models (I(mu) in V and H,
         the tau = 1 radius, the apparent limb and the outer boundary)
  fig 2  sigma_s vs baseline, 2 h, two 4 m telescopes: their single
         filters (V R I H K) and the multiplexed optical backends
  fig 3  theta_UD / theta_LD across 400-950 nm at R = 5000 with 2 h
         error bars in 50 nm bins, dwarf and supergiant stand-ins

    .venv/bin/python scripts/redclump_figures.py
"""

from __future__ import annotations

import os
import warnings
from dataclasses import replace

import numpy as np

from hbtsim.catalog import Catalog
from hbtsim.diameter import scale_precision, scale_precision_scan
from hbtsim.sed import load_star_tables, with_tables
from hbtsim.single import prepare_single, single_star_vis2, ud_diameter_per_channel

CAT = Catalog(env=False)
HD_17652 = CAT.load_target("hd17652")
HD_360 = CAT.load_target("hd360")
KK_TELESCOPE = CAT.load_telescope("kk_4m")
KK_DETECTOR = CAT.load_detector("kk_ideal")
BANDS = {b: CAT.load_spectrograph(f"filter_{s}_{b.lower()}")
         for b, s in (("B", "johnson"), ("V", "johnson"), ("R", "cousins"), ("I", "cousins"),
                      ("H", "2mass"), ("K", "2mass"))}
EON_SII_TELESCOPE = CAT.load_telescope("eonsii_4m")
EONSII_SPAD = CAT.load_detector("eonsii_spad")
EONSII_SPECTROGRAPH = CAT.load_spectrograph("eonsii_1000ch")
SPAD_LAMBDA_NG = CAT.load_detector("spad_lambda_ng")

DATA = "data/newera_redclump"
OUT = "docs/redclump_note"
MODELS = {"dwarf (4800 K, log g 4.5)": "newera_lte04800-4.50-0.0",
          "supergiant (5000 K, log g 0)": "newera_lte05000-0.00-0.0"}
SPEC_R5000 = CAT.load_spectrograph("r5000_400_950")
SPEC_OPT_KK = CAT.load_spectrograph("kk_1000ch_400_950")


def attach(target, model, ir):
    ft, ld = load_star_tables(os.path.join(DATA, f"{model}_{'380-2500nm_0.1nm' if ir else '380-1000nm_0.02nm'}.npz"))
    return replace(target, star=with_tables(target.star, ft, ld))


def fig_limb(plt):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (label, model) in zip(axes, MODELS.items()):
        d = np.load(os.path.join(DATA, f"{model}_380-2500nm_0.1nm.npz"))
        mu = np.asarray(d["mu"], float)
        wl = np.asarray(d["wavelength_nm"], float)
        r = np.sqrt(np.clip(1.0 - mu**2, 0.0, 1.0))               # in units of R_out
        for lam, c in ((551.0, "C0"), (1630.0, "C3")):
            k = int(np.argmin(np.abs(wl - lam)))
            row = np.asarray(d["intensity"][k], float)
            ax.plot(r, row / row.max(), color=c, lw=1.2, label=f"{lam:.0f} nm")
        r_t = 1.0 / float(d["r_outer_over_tau1"])
        mu_e = float(d["mu_edge"])
        r_e = np.sqrt(1.0 - mu_e**2)
        ax.axvline(r_t, color="k", ls="--", lw=0.8, label=r"$R_{\tau=1}$")
        ax.axvline(r_e, color="0.5", ls=":", lw=0.8, label=r"$R_{\rm limb}$ (half intensity)")
        ax.axvline(1.0, color="k", lw=0.8, label=r"$R_{\rm out}$")
        ax.set_xlim(0.6 if "super" in label else 0.9, 1.005)
        ax.set_xlabel(r"$r / R_{\rm out}$")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$I(r)/I(0)$")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_limb.png"), dpi=160)
    plt.close(fig)


def fig_sigma_s(plt, hours=2.0):
    b = np.arange(20.0, 301.0, 5.0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    model = MODELS["dwarf (4800 K, log g 4.5)"]
    for ax, target in zip(axes, (HD_17652, HD_360)):
        t_ir = attach(target, model, ir=True)
        t_opt = attach(target, model, ir=False)
        for band, c in zip("VRIHK", ("C0", "C1", "C2", "C3", "C4")):
            _, sig, _ = scale_precision_scan(t_ir, b, BANDS[band], t_int_s=hours * 3600.0,
                                             telescope=KK_TELESCOPE, detector=KK_DETECTOR)
            ax.plot(b, sig, color=c, lw=1.2, label=f"{band} filter (KK instrument)")
        cases = [("R = 5000 SPAD Lambda, correlator", SPEC_R5000, KK_TELESCOPE, SPAD_LAMBDA_NG, "k", "-"),
                 ("EON-SII 1000 ch, QUASAR SPAD, 1 GHz link", EONSII_SPECTROGRAPH, EON_SII_TELESCOPE,
                  EONSII_SPAD, "k", "--"),
                 ("1000 ch 400-950 nm, ideal detector", SPEC_OPT_KK, KK_TELESCOPE, KK_DETECTOR, "0.5", ":")]
        for name, spec, tel, det, c, ls in cases:
            _, sig, _ = scale_precision_scan(t_opt, b, spec, t_int_s=hours * 3600.0, telescope=tel,
                                             detector=det)
            ax.plot(b, sig, color=c, ls=ls, lw=1.4, label=name)
        ax.axhline(0.007, color="C3", lw=0.6, alpha=0.5)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.0)
        ax.set_xlabel("Baseline [m]")
        ax.set_title(target.name, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel(r"$\sigma_s$ (fractional scale precision, 2 h)")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_sigma_s.png"), dpi=160)
    plt.close(fig)


def fig_chromatic(plt, hours=2.0):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    nm = SPEC_R5000.channel_centers_nm
    for (label, model), c in zip(MODELS.items(), ("C0", "C3")):
        t_opt = attach(HD_17652, model, ir=False)
        p = prepare_single(t_opt, SPEC_R5000)
        best = scale_precision(t_opt, 50.0, SPEC_R5000, telescope=KK_TELESCOPE, detector=SPAD_LAMBDA_NG,
                               t_int_s=hours * 3600.0)
        v2 = single_star_vis2(p, 50.0, nm, (4.0, 4.0))[:, 0]
        th = ud_diameter_per_channel(v2, 50.0, nm, (4.0, 4.0)) / HD_17652.theta_ld_mas
        ax.plot(nm, th, color=c, lw=0.5, alpha=0.8, label=f"{label} stand-in")
        # 50 nm bins: inverse-variance mean with the 2 h per-channel errors
        sig_ch = best.sigma_vis2 / np.abs(best.dvis2_ds)      # fractional, per channel
        edges = np.arange(400.0, 951.0, 50.0)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (nm >= lo) & (nm < hi)
            w = 1.0 / sig_ch[m] ** 2
            ax.errorbar(0.5 * (lo + hi), np.sum(w * th[m]) / np.sum(w), yerr=np.sqrt(1.0 / np.sum(w)) * np.mean(th[m]),
                        fmt="o", ms=3, color=c, capsize=2)
    ax.set_xlabel("Vacuum wavelength [nm]")
    ax.set_ylabel(r"$\theta_{UD}/\theta_{LD}$")
    ax.set_title("HD 17652: R = 5000, B = 50 m, 4 m pupils; points: 50 nm bins with 2 h errors", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_chromatic.png"), dpi=160)
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_limb(plt)
    print("wrote fig_limb.png")
    fig_chromatic(plt)
    print("wrote fig_chromatic.png")
    fig_sigma_s(plt)
    print("wrote fig_sigma_s.png")


if __name__ == "__main__":
    main()
