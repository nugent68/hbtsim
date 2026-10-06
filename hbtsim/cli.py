"""`hbtsim movie`: the three-panel orbit / lightcurve / g2(B) movie.

    hbtsim movie --target betaaur --frames 240
    hbtsim movie --target algol --newera-dir data/newera --allow-extrapolation
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from .cli_common import add_catalog_options, catalog_from, ensure_parent, resolve_target
from .movie import make_movie, precompute_frames
from .params import GridConfig, MovieConfig


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="HBT simulation movie of a binary star")
    add_catalog_options(p)
    p.add_argument("--frames", type=int, default=240, help="frames per orbit")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--out", default=None,
                   help="output path (default output/<target>_hbt.mp4)")
    p.add_argument("--baseline-pa", default="follow",
                   help='"follow" (projected separation axis) or fixed PA in degrees')
    args = p.parse_args(argv)

    cat = catalog_from(args)
    system = resolve_target(cat, args)
    out = args.out or f"output/{args.target}_hbt.mp4"
    pa = args.baseline_pa if args.baseline_pa == "follow" else float(args.baseline_pa)
    cfg = replace(MovieConfig(), n_frames=args.frames, fps=args.fps, baseline_pa=pa)
    grid = GridConfig().fit_orbit(system)

    ensure_parent(out)
    print(f"Precomputing {cfg.n_frames} frames for {system.name} ...")
    fd = precompute_frames(system, grid, cfg)
    print(f"Writing movie to {out} ...")
    make_movie(fd, out)
    print("Done.")


if __name__ == "__main__":
    main()
