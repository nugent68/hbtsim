"""Benchmark the batched spectral |V|^2(B, lambda) pipeline.

    python scripts/bench_spectral.py                    # default device
    JAX_PLATFORMS=cpu python scripts/bench_spectral.py  # force CPU

Reports compile time, steady-state wall time and ms/channel for
spectral_vis2 (render + exact DFT), then the end-to-end spectral SNR
with a cross-check against the analytic visibility (out of eclipse only).
"""

from __future__ import annotations

import argparse
import time

import jax
import numpy as np

from hbtsim import hbt
from hbtsim.orbit import positions_at
from hbtsim.params import BETA_AUR, GridConfig
from hbtsim.snr import Spectrograph, spectral_g2_snr
from hbtsim.spectral import _auto_chunk, spectral_vis2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--channels", type=int, default=320)
    p.add_argument("--chunk", type=int, default=None)
    p.add_argument("--baseline", type=float, default=50.0)
    p.add_argument("--phase", type=float, default=0.0)
    p.add_argument("--repeat", type=int, default=3)
    args = p.parse_args()

    chunk = _auto_chunk() if args.chunk is None else args.chunk
    print(f"jax {jax.__version__}, backend {jax.default_backend()}, "
          f"devices {jax.devices()}")
    print(f"{args.channels} channels (400-950 nm), chunk {chunk}, "
          f"B = {args.baseline} m, phase {args.phase}")
    print(f"estimated peak memory ~ {chunk * 0.03:.1f} GiB "
          f"({chunk} x ~30 MB/channel of render temporaries)\n")

    system, grid = BETA_AUR, GridConfig()
    pos = positions_at(system, args.phase)
    nm = np.linspace(400.0, 950.0, args.channels)
    baselines = np.arange(10.0, 151.0, 10.0)

    # --- spectral_vis2: compile vs steady state ---
    t0 = time.perf_counter()
    v2 = np.asarray(spectral_vis2(pos, baselines, nm, system, grid, chunk_size=chunk))
    t_first = time.perf_counter() - t0
    print(f"spectral_vis2 first call (compile + run): {t_first:8.2f} s")

    times = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        v2 = np.asarray(spectral_vis2(pos, baselines, nm, system, grid, chunk_size=chunk))
        times.append(time.perf_counter() - t0)
    t_best = min(times)
    print(f"spectral_vis2 steady state (best of {args.repeat}): "
          f"{t_best:8.2f} s  ({1e3 * t_best / args.channels:.1f} ms/channel)")

    # --- end-to-end SNR + cross-check ---
    spec = Spectrograph(n_channels=args.channels)
    t0 = time.perf_counter()
    res = spectral_g2_snr(system, args.baseline, spectrograph=spec,
                          orbital_phase=args.phase, vis2_method="render",
                          chunk_size=chunk)
    t_snr = time.perf_counter() - t0
    print(f"\nspectral_g2_snr (rnd):  total SNR = {res.snr_total:.2f}  "
          f"[{t_snr:.2f} s]")
    try:
        ana = spectral_g2_snr(system, args.baseline, spectrograph=spec,
                              orbital_phase=args.phase,
                              vis2_method="analytic")
        rel = abs(res.snr_total - ana.snr_total) / ana.snr_total
        print(f"spectral_g2_snr (ana):  total SNR = {ana.snr_total:.2f}  "
              f"(render vs analytic: {100 * rel:.2f}%)")
    except ValueError as exc:
        print(f"analytic cross-check skipped: {exc}")


if __name__ == "__main__":
    main()
