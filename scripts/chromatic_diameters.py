"""Chromatic (Balmer-core vs continuum) diameters of Sirius A and Vega with
EON-SII (hbtsim.chromatic, hbtsim.single, NewEra tables).

EON-SII's 400-550 nm spectrograph holds H-delta, H-gamma and H-beta at
once.  For each star, backend and readout strategy the script finds the
first-lobe baseline that maximizes the Asimov significance of "theta_UD
(lambda) is smooth" and prints the per-line signal (core theta_UD over
the continuum fit), per-line and total significance per night, the
continuum chromaticity, the tagged channels and nights to 5 sigma.

    .venv/bin/python scripts/chromatic_diameters.py [--star sirius|vega|both]
        [--backend spad|spad-pbs|mcp|r7500|all] [--readout link|subset|correlator|all]
        [--nights 1] [--plot]
    # regression against the 2026-09-25 1 m study (Sirius H-beta +3.5 %, ~5 sigma/night):
    .venv/bin/python scripts/chromatic_diameters.py --instrument c2pu --hbeta-window
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import replace

import numpy as np

from hbtsim.catalog import Catalog
from hbtsim.chromatic import (BALMER_VAC_NM, chromatic_signature, line_masks,
                              night_seconds, optimal_baseline)
from hbtsim.single import attach_newera_single

CAT = Catalog(env=False)
STARS = {"sirius": "sirius_a", "vega": "vega"}           # CLI key -> catalog target
# (label, spectrograph, detector, polarization) read off the catalog's counting backends
BACKENDS = {key: (b.name, b.spectrograph, b.detector, b.polarization_mode)
            for key, b in ((k, CAT.load_backend(n)) for k, n in
                           (("mcp", "eonsii_mcp"), ("spad", "eonsii_spad"),
                            ("spad-pbs", "eonsii_spad_pbs"), ("r7500", "eonsii_r7500_spad")))}
EON_SII_TELESCOPE = CAT.load_telescope("eonsii_4m")
TEIDE = CAT.load_site("teide")
EONSII_MIN_SPACING_M = CAT.raw("array", "eonsii_triangle_paranal")["generator"]["min_spacing_m"]


def readout_config(readout: str, detector):
    """The CLI readout strategy as (channel_selection, detector): "link"
    reads every channel through the detector's time-tag link, "subset"
    tags only the line and reference channels, "correlator" reads every
    channel with an on-board correlator (no link-rate cap)."""
    if readout == "link":
        return "all", detector
    if readout == "subset":
        return "subset", detector
    if readout == "correlator":
        return "all", replace(detector, readout="correlator", max_total_cps=None)
    raise ValueError(f"unknown readout {readout!r}")


def describe(r, nights):
    s = r.significance * np.sqrt(nights)
    lines = ", ".join(f"{k} {r.line_signal_pct.get(k, np.nan):+.2f} % / {v * np.sqrt(nights):.1f} sigma"
                      for k, v in r.line_significance.items())
    tag = f"{int(r.tagged.sum())} of {r.nm.size} ch tagged" if r.channel_selection == "subset" else f"{r.nm.size} ch"
    where = ""
    if r.channel_selection == "subset":
        m = line_masks(r.nm)
        on = [k for k in BALMER_VAC_NM if k in m and (r.tagged & (m[k]["core"] | m[k]["wing"])).any()]
        where = f" (lines covered: {', '.join(on) or 'none'}; continuum refs {int((r.tagged & m['continuum']).sum())})"
    return (f"B {r.baseline_m:5.1f} m, |V|^2 {np.median(r.vis2):.2f}; {tag}{where}; readout scale "
            f"{r.readout_scale:.3g}; total {s:.1f} sigma in {nights:g} night(s) "
            f"-> {r.nights_to(5.0) :.2g} nights to 5 sigma | {lines} | continuum "
            f"{r.continuum_chromaticity_pct:+.2f} % ({r.nm[0]:.0f}->{r.nm[-1]:.0f} nm)")


def plot(r, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(r.nm, r.vis2, lw=0.8)
    ax[0].set_ylabel(r"$|V|^2$ (pupil-averaged)")
    ok = r.tagged & np.isfinite(r.sigma_theta)
    ax[1].plot(r.nm, r.theta_ud, lw=0.8, color="C0", label=r"model $\theta_{UD}$")
    ax[1].errorbar(r.nm[ok], r.theta_ud[ok], yerr=r.sigma_theta[ok], fmt=".", ms=2, lw=0.5,
                   color="C1", label="tagged, 1-night error")
    ax[1].plot(r.nm, r.continuum, "k--", lw=0.8, label="continuum fit")
    for k, l in BALMER_VAC_NM.items():
        for a in ax:
            a.axvline(l, color="0.7", lw=0.5)
    ax[1].set_ylabel(r"$\theta_{UD}$ [mas]")
    ax[1].set_xlabel("Vacuum wavelength [nm]")
    ax[1].legend(fontsize=8)
    ax[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"    wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--star", choices=("sirius", "vega", "both"), default="both")
    ap.add_argument("--backend", choices=tuple(BACKENDS) + ("all",), default="all")
    ap.add_argument("--readout", choices=("link", "subset", "correlator", "all"), default="all")
    ap.add_argument("--nights", type=float, default=1.0)
    ap.add_argument("--deg", type=int, default=1, help="local continuum polynomial degree")
    ap.add_argument("--newera-dir", default="data/newera")
    ap.add_argument("--instrument", choices=("eonsii", "c2pu"), default="eonsii")
    ap.add_argument("--hbeta-window", action="store_true",
                    help="c2pu regression: 320 ch at R = 5000 around H-beta, correlator, 6 h, flat continuum")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message=".*dead-time.*")

    stars = ("sirius", "vega") if args.star == "both" else (args.star,)
    for key in stars:
        tgt, rep = attach_newera_single(CAT.load_target(STARS[key]), args.newera_dir)
        print(f"\n=== {tgt.name}: theta_LD {tgt.theta_ld_mas} mas, drawn {tgt.drawn_diameter_mas:.4f} mas "
              f"(r_outer {tgt.star.radius_scale:.5f}); V {tgt.v_mag:+.2f}, model AB(550) - V = "
              f"{tgt.v_check():+.3f}\n    {rep}")
        if args.instrument == "c2pu":
            spec = CAT.load_spectrograph("hbeta_window_r5000")
            sel, det = readout_config("correlator", CAT.load_detector("spad_lambda_ng"))
            # t_int_s is given, so the site only labels the night; C2PU is at Calern
            r = chromatic_signature(tgt, 1.7 * 486.27e-9 / (np.pi * tgt.drawn_diameter_mas * 4.8481368e-9),
                                    spec, telescope=CAT.load_telescope("c2pu_1m"), detector=det,
                                    site=CAT.load_site("calern"), t_int_s=6 * 3600.0,
                                    channel_selection=sel, deg=0 if args.hbeta_window else args.deg)
            print(f"  2 x 1 m (C2PU), {spec.name}, SPAD Lambda correlator, 6 h: {describe(r, args.nights)}")
            continue
        print(f"  EON-SII from Teide: {night_seconds(tgt, TEIDE) / 3600:.1f} h above 30 deg per night; "
              f"baselines >= {EONSII_MIN_SPACING_M:g} m (4 m units)")
        keys = tuple(BACKENDS) if args.backend == "all" else (args.backend,)
        readouts = ("link", "subset", "correlator") if args.readout == "all" else (args.readout,)
        for bk in keys:
            label, spec, det, pol = BACKENDS[bk]
            for ro in readouts:
                sel, det_ro = readout_config(ro, det)
                b, r = optimal_baseline(tgt, spec, telescope=EON_SII_TELESCOPE, detector=det_ro,
                                        site=TEIDE, channel_selection=sel, polarization_mode=pol,
                                        deg=args.deg, min_baseline_m=EONSII_MIN_SPACING_M)
                print(f"  {label:32s} {ro:10s}: {describe(r, args.nights)}")
                if args.plot and bk == "spad" and ro == "subset":
                    os.makedirs("output", exist_ok=True)
                    plot(r, f"output/chromatic_{key}_eonsii.png",
                         f"{tgt.name}: EON-SII {label}, {ro} readout, B = {b:.1f} m")


if __name__ == "__main__":
    main()
