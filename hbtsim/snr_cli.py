"""SNR of g2(B) for the binary with a given telescope pair.

Default: spectral mode -- the light is dispersed over the SPAD Lambda's
320-pixel linear array (400-950 nm, ~1.7 nm channels), each pixel pair
measuring g2 independently; channel SNRs add in quadrature.

    python -m hbtsim.snr_cli                          # C2PU: B = 50 m, 2 x 1 m
    python -m hbtsim.snr_cli --baseline 15 50 100     # several baselines
    python -m hbtsim.snr_cli --mode narrowband --baseline 15 \
        --wavelengths 400 800 --filter-width 10       # the old filter setup

In spectral mode each channel's |V|^2 comes from the batched render +
exact-DFT pipeline (hbtsim.spectral, GPU-accelerated under JAX; ~10 ms
per channel on CPU) -- pass --vis2-method analytic for the instant
out-of-eclipse approximation.  Narrowband mode renders once per filter.
Source brightness is the anchored blackbody model of the chosen --system
in both modes.
"""

from __future__ import annotations

import argparse

import numpy as np

from . import hbt
from .orbit import SkyPositions, sky_positions
from .params import SYSTEMS, GridConfig
from .render import render_image
from .snr import (C2PU, SPAD_LAMBDA, Detector, Observation, Spectrograph,
                  Telescope, g2_snr, spectral_g2_snr, system_ab_mag)


def narrowband(args, system, grid, tel, det) -> None:
    psi = 2.0 * np.pi * args.phase
    pos = SkyPositions(*(np.asarray(v) for v in sky_positions(psi, system)))
    print(f"Filter    : {args.filter_width:.1f} nm FWHM at "
          f"{', '.join(f'{w:.0f}' for w in args.wavelengths)} nm\n")

    hdr = (f"{'lam[nm]':>8} {'B[m]':>7} {'mag(AB)':>8} {'rate[Mcps]':>11} "
           f"{'tau_c[fs]':>10} {'|V|^2':>7} {'N_sig':>10} {'N_bkg':>12} {'SNR':>8}")
    print(hdr)
    print("-" * len(hdr))
    for lam_nm in args.wavelengths:
        mag = system_ab_mag(system, lam_nm) if args.mag is None else args.mag
        img = render_image(pos, system, lam_nm, grid)
        v2 = np.asarray(hbt.vis2_along_pa(
            img, np.asarray(args.baseline, dtype=float), lam_nm * 1e-9,
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


def spectral(args, system, tel, det) -> None:
    spec = Spectrograph(lambda_min_nm=args.lambda_min,
                        lambda_max_nm=args.lambda_max,
                        n_channels=args.channels)
    print(f"Spectro   : {spec.n_channels} channels x "
          f"{spec.channel_width_nm:.2f} nm over "
          f"{spec.lambda_min_nm:.0f}-{spec.lambda_max_nm:.0f} nm "
          f"(1 pixel/channel)\n")

    for b_m in args.baseline:
        res = spectral_g2_snr(system, b_m, spectrograph=spec,
                              t_int_s=args.time, telescope1=tel,
                              detector1=det, orbital_phase=args.phase,
                              vis2_method=args.vis2_method,
                              chunk_size=args.chunk)
        print(f"Baseline {b_m:.1f} m ({res.vis2_method} |V|^2): "
              f"total SNR = {res.snr_total:.2f} "
              f"(best channel {res.snr.max():.2f} at "
              f"{res.channel_nm[res.snr.argmax()]:.0f} nm; "
              f"total rate {res.rate_cps.sum() / 1e6:.1f} Mcps/telescope)")
        hdr = (f"  {'lam[nm]':>8} {'mag(AB)':>8} {'rate[Mcps/ch]':>14} "
               f"{'|V|^2':>7} {'SNR/ch':>8}")
        print(hdr)
        step = max(1, spec.n_channels // 8)
        for k in range(0, spec.n_channels, step):
            print(f"  {res.channel_nm[k]:8.1f} {res.mag_ab[k]:8.2f} "
                  f"{res.rate_cps[k] / 1e6:14.3f} {res.vis2[k]:7.4f} "
                  f"{res.snr[k]:8.3f}")
        print()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="g2(B) SNR for a telescope pair")
    p.add_argument("--system", choices=sorted(SYSTEMS), default="betaaur")
    p.add_argument("--mode", choices=("spectral", "narrowband"),
                   default="spectral")
    p.add_argument("--baseline", type=float, nargs="+", default=[50.0],
                   help="baseline(s) in m")
    p.add_argument("--diameter", type=float, default=C2PU.diameter_m,
                   help="telescope diameter in m (C2PU: 1.0)")
    p.add_argument("--throughput", type=float, default=C2PU.throughput,
                   help="optics+atmosphere throughput, excl. detector PDE")
    p.add_argument("--time", type=float, default=3600.0,
                   help="integration time in s")
    p.add_argument("--phase", type=float, default=0.0,
                   help="orbital phase in [0,1); 0 = greatest separation")
    # spectral mode
    p.add_argument("--channels", type=int, default=320,
                   help="spectral channels (SPAD Lambda: 320)")
    p.add_argument("--lambda-min", type=float, default=400.0)
    p.add_argument("--lambda-max", type=float, default=950.0)
    p.add_argument("--vis2-method", choices=("render", "analytic"),
                   default="render",
                   help="per-channel |V|^2: rendered image + exact DFT "
                        "(valid in eclipses) or analytic binary (instant, "
                        "out of eclipse only)")
    p.add_argument("--chunk", type=int, default=None,
                   help="channels per GPU/CPU batch (default: auto)")
    # narrowband mode
    p.add_argument("--wavelengths", type=float, nargs="+", default=[400.0, 800.0])
    p.add_argument("--filter-width", type=float, default=10.0,
                   help="filter FWHM in nm")
    p.add_argument("--n-pixels", type=int, default=1,
                   help="pixels the light is spread over (narrowband mode)")
    p.add_argument("--mag", type=float, default=None,
                   help="override source AB magnitude (narrowband mode)")
    args = p.parse_args(argv)

    system, grid = SYSTEMS[args.system], GridConfig()
    tel = Telescope(diameter_m=args.diameter, throughput=args.throughput)
    det = Detector(name=SPAD_LAMBDA.name, pde_table_nm=SPAD_LAMBDA.pde_table_nm,
                   jitter_fwhm_ps=SPAD_LAMBDA.jitter_fwhm_ps,
                   dead_time_ns=SPAD_LAMBDA.dead_time_ns,
                   dark_cps_per_pixel=SPAD_LAMBDA.dark_cps_per_pixel,
                   n_pixels=args.n_pixels)

    pos = sky_positions(2.0 * np.pi * args.phase, system)
    print(f"Source    : {system.name}, orbital phase {args.phase:.3f} "
          f"(projected separation {float(pos.rho):.3f} mas)")
    print(f"Telescopes: 2 x {tel.diameter_m:.2f} m, throughput {tel.throughput:.2f}")
    print(f"Detectors : {det.name} (jitter {det.jitter_fwhm_ps:.0f} ps FWHM, "
          f"dead time {det.dead_time_ns:.0f} ns, {det.dark_cps_per_pixel:.0f} cps "
          f"dark/pixel)")
    print(f"Integration: {args.time:.0f} s")

    if args.mode == "spectral":
        spectral(args, system, tel, det)
    else:
        narrowband(args, system, grid, tel, det)


if __name__ == "__main__":
    main()
