"""delta Vel closure phases with three EON-SII 4 m units at Paranal / CTAO-South:
diagnostics, a layout scan of the triangle side, four backends on the same
geometry, a multi-night campaign (uniform phases plus the eclipse windows),
nights to precision, figures and a blackbody-vs-NewEra snapshot.  This now
lives in the catalog campaign runner hbtsim/runners/g3_campaign.py, driven
by hbtsim/configs/campaigns/g3_deltavel_eonsii_paranal.json.
"""

import sys


def main():
    print("This script moved into the catalog campaign runner; run:")
    print("  .venv/bin/python -m hbtsim run g3_deltavel_eonsii_paranal [--no-figures] [--no-latex]")
    print("  .venv/bin/python -m hbtsim run g3_deltavel_eonsii_paranal --set options.side_m=20"
          "  # fixed side, no scan")
    print("  .venv/bin/python -m hbtsim run g3_deltavel_eonsii_paranal"
          " --set options.layout_scan.sides_m=[8,12,20] --set options.n_uniform=8")
    return 2


if __name__ == "__main__":
    sys.exit(main())
