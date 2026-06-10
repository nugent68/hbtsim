"""Command-line entry point: python -m hbtsim [--frames N] [--out PATH]"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

from .movie import make_movie, precompute_frames
from .params import BETA_AUR, GridConfig, MovieConfig


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="HBT simulation of Beta Aurigae")
    p.add_argument("--frames", type=int, default=240, help="frames per orbit")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--out", default="output/betaaur_hbt.mp4")
    p.add_argument("--baseline-pa", default="follow",
                   help='"follow" (projected separation axis) or fixed PA in degrees')
    args = p.parse_args(argv)

    pa = args.baseline_pa if args.baseline_pa == "follow" else float(args.baseline_pa)
    cfg = replace(MovieConfig(), n_frames=args.frames, fps=args.fps, baseline_pa=pa)
    grid = GridConfig()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"Precomputing {cfg.n_frames} frames for {BETA_AUR.name} ...")
    fd = precompute_frames(BETA_AUR, grid, cfg)
    print(f"Writing movie to {args.out} ...")
    make_movie(fd, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
