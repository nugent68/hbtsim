"""Bin a NewEra PHOENIX HSR-RF file (angle-resolved intensities) to a
small .npz table for hbtsim.sed.load_star_tables.

    python scripts/prepare_newera.py \\
        ../data/lte06000-5.00-0.0.PHOENIX-NewEra-ACES-COND-2023.HSR-RFs.h5 \\
        --out data/newera_06000_5.00.npz --step 0.02 --range 380 1000

Reads /PHOENIX_RF/{wl, flux, mu, Intensities} in chunks (the full file
is ~9 GB), averages flux and I(mu) into bins of --step nm, normalizes
I(mu)/I(1) and writes wavelength_nm, flux [W m^-2 m^-1], mu, intensity.
Requires h5py (pip install 'hbtsim[sed]').
"""

from __future__ import annotations

import argparse
import time

from hbtsim.sed import read_newera_hsr_rf, save_star_tables


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5")
    p.add_argument("--out", required=True)
    p.add_argument("--step", type=float, default=0.02, help="bin width [nm]")
    p.add_argument("--range", type=float, nargs=2, default=(380.0, 1000.0),
                   metavar=("MIN", "MAX"), help="wavelength range [nm]")
    p.add_argument("--chunk", type=int, default=200_000)
    args = p.parse_args()

    t0 = time.perf_counter()
    tab = read_newera_hsr_rf(args.h5, args.range[0], args.range[1], args.step,
                             chunk=args.chunk)
    save_star_tables(args.out, tab)
    print(f"wrote {args.out}: {tab['wavelength_nm'].size} bins of {args.step} nm, "
          f"{tab['mu'].size} mu values, in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
