"""Like-for-like comparison with Kim & Kaiser (2026) Table 1 and their
H-band results: the zero-baseline inverse noise 1/sigma(|V|^2) per band
in 2 h on their instrument (two 4 m, throughput 0.3, sigma_t 16.6 ps),
from (a) their photometry through their formula, (b) hbtsim's matched
filter with the model spectrum anchored to that photometry, and (c) the
unanchored stand-in model; then the H-band sigma_s against their quoted
values.

    .venv/bin/python scripts/redclump_compare.py
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np

from hbtsim.diameter import BANDS, KK_DETECTOR, KK_TELESCOPE, kim_kaiser_sigma_vis2, scale_precision_scan
from hbtsim.sed import load_star_tables, with_tables
from hbtsim.single import HD_17652, HD_360
from hbtsim.snr import AB_ZERO_FNU, C_LIGHT, H_PLANCK, Observation, g2_snr

# their Table 1: 1/sigma(|V|^2) in 2 h (V, R, I, H, K)
KK_TABLE1 = {"HD 17652": dict(V=28.93, R=40.37, I=69.70, H=188.53, K=168.63),
             "HD 360": dict(V=7.07, R=9.81, I=17.02, H=47.31, K=41.82)}
KK_SIGMA_S = {"HD 17652": ("H", "< 0.007 at ~100 m", "H, K: <~ 0.01 at ~150 m; optical ~0.1 at 50-70 m"),
              "HD 360": ("H", "< 0.03 at ~100 m", "")}
MODEL = "data/newera_redclump/newera_lte04800-4.50-0.0_380-2500nm_0.1nm.npz"


def snr0_formula(ab_mag, nm):
    """Their Eq. 7 at |V|^2 = 1 from an AB magnitude."""
    f_nu = AB_ZERO_FNU * 10 ** (-0.4 * ab_mag)
    dgamma = 0.3 * KK_TELESCOPE.area_m2 * f_nu / (H_PLANCK * C_LIGHT / (nm * 1e-9))
    return 1.0 / kim_kaiser_sigma_vis2(dgamma, 7200.0, KK_DETECTOR.jitter_sigma_s)


def snr0_hbtsim(target, band):
    spec = BANDS[band]
    nm, w = spec.channel_centers_nm, spec.channel_widths_nm
    obs = Observation(wavelength_nm=nm, filter_width_nm=w, t_int_s=7200.0, backend_throughput=1.0)
    r = g2_snr(np.ones(1), target.ab_mag(nm), obs, telescope1=KK_TELESCOPE, detector1=KK_DETECTOR)
    return float(np.asarray(r.snr)[0])


def main():
    warnings.filterwarnings("ignore")
    ft, ld = load_star_tables(MODEL)
    for tgt in (HD_17652, HD_360):
        key = tgt.name.split(" (")[0]
        anchored = replace(tgt, star=with_tables(tgt.star, ft, ld))
        bare = replace(anchored, mag_anchors=())
        print(f"\n=== {key}: 1/sigma(|V|^2) at zero baseline, 2 h, their instrument "
              f"(sigma_t {KK_DETECTOR.jitter_sigma_s * 1e12:.1f} ps)")
        print(f"  {'band':4s} {'their Table 1':>13s} {'their formula, their phot.':>27s} {'hbtsim, anchored model':>23s} "
              f"{'ratio':>6s} {'unanchored stand-in':>20s}")
        for band in "VRIHK":
            nm = BANDS[band].channel_centers_nm[0]
            phot = [a for a in tgt.mag_anchors if abs(a[0] - nm) < 1.0]
            s_formula = snr0_formula(phot[0][1], nm) if phot else np.nan
            s_anch = snr0_hbtsim(anchored, band)
            s_bare = snr0_hbtsim(bare, band)
            theirs = KK_TABLE1[key][band]
            print(f"  {band:4s} {theirs:13.2f} {s_formula:27.2f} {s_anch:23.2f} {s_anch / theirs:6.3f} {s_bare:20.2f}")
        print("  (R and I: they interpolate fluxes between V, H, K and Gaia; hbtsim interpolates the "
              "anchor offset in log lambda over the model spectrum)")
        b = np.arange(20.0, 301.0, 5.0)
        print(f"  sigma_s in 2 h, anchored model, baseline scanned:")
        for band in "VRIHK":
            _, sig, best = scale_precision_scan(anchored, b, BANDS[band], t_int_s=7200.0)
            note = ""
            if band == "H":
                note = f"   <- paper: {KK_SIGMA_S[key][1]}"
            print(f"    {band}: {best.sigma_s:.4f} at {best.baseline_m:.0f} m (|V|^2 {best.vis2[0]:.2f}){note}")
        if KK_SIGMA_S[key][2]:
            print(f"    paper, other bands: {KK_SIGMA_S[key][2]}")


if __name__ == "__main__":
    main()
