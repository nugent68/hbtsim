"""Chromatic (Balmer-core vs continuum) diameters of Sirius A and Vega with
EON-SII (hbtsim.chromatic, hbtsim.single, NewEra tables).  This script moved
into the catalog campaign runner hbtsim/runners/chromatic.py; the campaigns
chromatic_sirius_vega_eonsii (EON-SII pair from Teide: every backend and
readout strategy at its optimal first-lobe baseline) and
chromatic_c2pu_regression (2 x 1 m, 320 ch at R = 5000 around H-beta, 6 h)
replace its command-line options.
"""

import sys


def main():
    print("This script moved into the catalog campaign runner; run:")
    print("  .venv/bin/python -m hbtsim run chromatic_sirius_vega_eonsii [--no-figures] "
          "[--set options.nights=1] [--set options.channel_selection='[\"subset\"]']")
    print("  .venv/bin/python -m hbtsim run chromatic_c2pu_regression --no-figures")
    return 2


if __name__ == "__main__":
    sys.exit(main())
