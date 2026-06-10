"""SNR of g2(B) for the binary with a given telescope pair.

    python -m hbtsim.snr_cli                      # C2PU defaults: 15 m, 1 m
    python -m hbtsim.snr_cli --baseline 15 50 100 --diameter 1.0
    python -m hbtsim.snr_cli --time 7200 --filter-width 1 --phase 0.1

|V|^2(B) is taken from the FFT pipeline at the requested orbital phase
(baseline along the projected separation axis), and the source brightness
from the anchored blackbody model of Beta Aurigae.
"""

from __future__ import annotations

import argparse

import numpy as np

from . import hbt
from .orbit import SkyPositions, sky_positions
from .params import BETA_AUR, GridConfig
from .render import render_image
from .snr import (C2PU, SPAD_LAMBDA, Detector, Observation, Telescope,
                  g2_snr, system_ab_mag)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="g2(B) SNR for a telescope pair")
    p.add_argument("--baseline", type=float, nargs="+", default=[15.0],
                   help="baseline(s) in m (C2PU: 15)")
    p.add_argument("--diameter", type=float, default=C2PU.diameter_m,
                   help="telescope diameter in m (C2PU: 1.0)")
    p.add_argument("--throughput", type=float, default=C2PU.throughput,
                   help="optics+atmosphere throughput, excl. detector PDE")
    p.add_argument("--wavelengths", type=float, nargs="+", default=[400.0, 800.0],
                   help="observing wavelengths in nm")
    p.add_argument("--filter-width", type=float, default=10.0,
                   help="filter FWHM in nm")
    p.add_argument("--time", type=float, default=3600.0,
                   help="integration time in s")
    p.add_argument("--phase", type=float, default=0.0,
                   help="orbital phase in [0,1); 0 = greatest separation")
    p.add_argument("--n-pixels", type=int, default=SPAD_LAMBDA.n_pixels,
                   help="detector pixels the light is spread over")
    p.add_argument("--mag", type=float, default=None,
                   help="override source AB magnitude (default: Beta Aur model)")
    args = p.parse_args(argv)

    system, grid = BETA_AUR, GridConfig()
    tel = Telescope(diameter_m=args.diameter, throughput=args.throughput)
    det = Detector(name=SPAD_LAMBDA.name, pde_table_nm=SPAD_LAMBDA.pde_table_nm,
                   jitter_fwhm_ps=SPAD_LAMBDA.jitter_fwhm_ps,
                   dead_time_ns=SPAD_LAMBDA.dead_time_ns,
                   dark_cps_per_pixel=SPAD_LAMBDA.dark_cps_per_pixel,
                   n_pixels=args.n_pixels)

    psi = 2.0 * np.pi * args.phase
    pos = SkyPositions(*(np.asarray(v) for v in sky_positions(psi, system)))

    print(f"Source    : {system.name}, orbital phase {args.phase:.3f} "
          f"(projected separation {float(pos.rho):.3f} mas)")
    print(f"Telescopes: 2 x {tel.diameter_m:.2f} m, throughput {tel.throughput:.2f}")
    print(f"Detectors : {det.name} (jitter {det.jitter_fwhm_ps:.0f} ps FWHM, "
          f"dead time {det.dead_time_ns:.0f} ns, {det.dark_cps_per_pixel:.0f} cps "
          f"dark/pixel, {det.n_pixels} pixel(s))")
    print(f"Filter    : {args.filter_width:.1f} nm FWHM; "
          f"integration {args.time:.0f} s\n")

    hdr = (f"{'lam[nm]':>8} {'B[m]':>7} {'mag(AB)':>8} {'rate[Mcps]':>11} "
           f"{'tau_c[fs]':>10} {'|V|^2':>7} {'N_sig':>10} {'N_bkg':>12} {'SNR':>8}")
    print(hdr)
    print("-" * len(hdr))
    for lam_nm in args.wavelengths:
        mag = system_ab_mag(system, lam_nm) if args.mag is None else args.mag
        img = render_image(pos, system, lam_nm, grid)
        v2map = hbt.vis2_map(img, grid.pad)
        v2 = np.asarray(hbt.vis2_of_baseline(
            v2map, np.asarray(args.baseline, dtype=float), lam_nm * 1e-9,
            float(pos.pa), grid))
        for b_m, vis2 in zip(args.baseline, v2):
            obs = Observation(wavelength_nm=lam_nm,
                              filter_width_nm=args.filter_width,
                              t_int_s=args.time)
            r = g2_snr(float(vis2), mag, obs, telescope1=tel, detector1=det)
            print(f"{lam_nm:8.0f} {b_m:7.1f} {mag:8.2f} "
                  f"{r.rate1_cps / 1e6:11.3f} {r.tau_c_s * 1e15:10.1f} "
                  f"{r.vis2:7.4f} {r.n_signal:10.1f} {r.n_background:12.0f} "
                  f"{r.snr:8.2f}")


if __name__ == "__main__":
    main()
