"""Cross-check of hbtsim's photon-counting g2 SNR against the EON-SII
design paper (arXiv:2608.17444) on its own targets.

The paper's Table gives hours to 10 / 5 / 2 % diameter precision for
single white dwarfs with two 4 m telescopes at ~1.5 km:

    Sirius B    V = 8.44, 28.5 uas   MCP-PMT 1.5 / 6 / 37.5 h   SPAD 0.33 / 1.3 / 8.2 h
    40 Eri B    V = 9.52, 24.3 uas   MCP-PMT 10 / 40 / --  h    SPAD 2 / 8 / 50 h
    Procyon B   V = 10.70, 32.6 uas  --                         SPAD 25-30 / 100 / -- h

Here each star is a uniform disk with a mild linear limb darkening
(u = 0.3), observed at the baseline where |V|^2 ~ 0.6 (the paper's
1.5 km for Sirius B), with hbtsim.snr.EON_SII_TELESCOPE and the two
detector presets over the 1000-channel spectrograph.  Diameter precision
follows from sigma(|V|^2) through d ln|V|^2 / d ln theta.  Each row also
evaluates the design study's photon-level Eq. 5 (hbtsim.estimators) with
eta = p2 and dt_res = sqrt(pi) sigma_pair, which equals the matched filter,
and the dt_res Eq. 5 would need to reproduce the paper's hours.  The full
Monte Carlo and gap ladder are in scripts/eonsii_montecarlo.py.

    .venv/bin/python scripts/eonsii_crosscheck.py
"""

from __future__ import annotations

import numpy as np

from hbtsim.estimators import implied_dt_res, matched_filter_equivalents, photon_level_snr
from hbtsim.limbdark import visibility_ld_disk
from hbtsim.params import MAS
from hbtsim.snr import (EON_SII_TELESCOPE, EONSII_MCP_PMT, EONSII_SPAD, EONSII_SPECTROGRAPH,
                        Observation, g2_snr, incident_rate, readout_scale)

TARGETS = [  # name, V (taken as AB at 475 nm; white dwarfs are hot: B-V ~ 0), theta [uas],
    ("Sirius B", 8.44, 28.5, {"mcp": (1.5, 6.0, 37.5), "spad": (0.33, 1.3, 8.2)}),
    ("40 Eri B", 9.52, 24.3, {"mcp": (10.0, 40.0, np.nan), "spad": (2.0, 8.0, 50.0)}),
    ("Procyon B", 10.70, 32.6, {"spad": (27.5, 100.0, np.nan)}),
]
U_LD = 0.3
PRECISIONS = (0.10, 0.05, 0.02)


def main():
    spec = EONSII_SPECTROGRAPH
    nm = spec.channel_centers_nm
    widths = spec.channel_widths_nm
    for name, vmag, theta_uas, paper in TARGETS:
        theta = theta_uas * 1e-3 * MAS
        # baseline: |V|^2 ~ 0.6 at the band centre (x ~ 1.5 for u = 0.3)
        x_target = 1.5
        b = x_target * 475e-9 / (np.pi * theta)
        x = np.pi * theta * b / (nm * 1e-9)
        v2 = visibility_ld_disk(x, U_LD) ** 2
        # d ln|V|^2 / d ln theta from the analytic law (finite difference)
        dv = (visibility_ld_disk(x * 1.001, U_LD) ** 2 - visibility_ld_disk(x * 0.999, U_LD) ** 2) / 0.002
        dln = dv / v2                           # d ln V2 / d ln theta
        print(f"\n=== {name}: V = {vmag}, theta = {theta_uas} uas, baseline {b:.0f} m, "
              f"|V|^2 = {v2.min():.2f}-{v2.max():.2f} across 400-550 nm ===")
        for key, det in (("mcp", EONSII_MCP_PMT), ("spad", EONSII_SPAD)):
            if key not in paper:
                continue
            for pol in ("unpolarized", "pbs"):
                obs = Observation(wavelength_nm=nm, filter_width_nm=widths, t_int_s=3600.0,
                                  polarization_mode=pol, backend_throughput=spec.throughput)
                r = g2_snr(v2, vmag, obs, telescope1=EON_SII_TELESCOPE, detector1=det)
                tot = float(np.sum(incident_rate(vmag, EON_SII_TELESCOPE, det, obs)))
                sc = readout_scale(det, tot)
                snr_h = float(np.sqrt(np.sum(r.snr**2))) * sc      # per hour, all channels
                # fractional precision on |V|^2 (channels combined, at the
                # median |V|^2) and on theta after 1 h; time scales as 1/t
                sig_v2_rel = 1.0 / snr_h
                sig_th_rel = sig_v2_rel / float(np.median(np.abs(dln)))
                hours = [(p / sig_th_rel) ** -2 for p in PRECISIONS]
                hours_v2 = [(p / sig_v2_rel) ** -2 for p in PRECISIONS]
                ref = paper[key]
                print(f"  {det.name[:34]:34s} {pol:11s}: rate {np.median(r.rate1_cps):7.0f} cps/ch, "
                      f"{tot:.2e} cps/tel; SNR/h (all ch) {snr_h:6.2f}; hours to 10/5/2 % on theta: "
                      f"{hours[0]:7.2f} {hours[1]:7.2f} {hours[2]:7.1f}  (on |V|^2: "
                      f"{hours_v2[0]:.1f} {hours_v2[1]:.1f} {hours_v2[2]:.0f});  paper: "
                      f"{ref[0]} / {ref[1]} / {ref[2]} h  -> ratio ours/paper {hours[0] / ref[0]:.2f}")
                sig = float(np.median(r.sigma_pair_s))
                p2 = 0.5 if pol == "unpolarized" else 1.0
                eq = matched_filter_equivalents(sig, p2)
                n_str = 1 if pol == "unpolarized" else 2
                snr5 = photon_level_snr(v2, r.rate1_cps, r.rate2_cps, r.tau_c_s, 3600.0,
                                        eq["dt_res_s"], eta=eq["eta"])
                snr5_h = float(np.sqrt(n_str * np.sum(snr5**2))) * sc
                print(f"  {'':34s} {'':11s}  Eq. 5 (eta = p2, dt_res = sqrt(pi) sigma = "
                      f"{eq['dt_res_s'] * 1e12:.1f} ps): SNR/h {snr5_h:6.2f} (matched {snr_h:.2f}); "
                      f"dt_res that reproduces the paper: {implied_dt_res(ref[0], hours[0], sig) * 1e12:.2f} ps")
        print(f"  (tau_c per channel {float(np.median(r.tau_c_s)) * 1e12:.2f} ps, pair kernel sigma "
              f"{float(np.median(r.sigma_pair_s)) * 1e12:.1f} ps, p2 = {'1/2' if pol == 'unpolarized' else '1'})")


if __name__ == "__main__":
    main()
