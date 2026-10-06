"""Feasibility of three-telescope closure-phase intensity interferometry:
the figures and every number quoted in docs/three_telescope_feasibility.md
and the paper's Table 3 (per-pair |gamma|, the cos(phi_c) map, snapshot
sensitivities and one night along the uv track), plus the two-telescope
g2 numbers of Section 3.  This script moved into the catalog campaign
runners hbtsim/runners/g3.py and hbtsim/runners/g2.py; run the campaigns
below instead.
"""

import sys


def main():
    print("This script moved into the catalog campaign runner; run:")
    print("  hbtsim run g3_spica_vlt [--no-figures] [--no-track] [--phase PH]")
    print("  hbtsim run g3_deltavel_vlt [--set phases.extra=[0.25,0.5,0.9]]")
    print("  hbtsim run g3_betaaur_maunakea")
    print("  hbtsim run g3_algol_maunakea")
    print("  hbtsim run g2_c2pu | g2_keck | g2_eonsii   (the former --g2 --instrument ...)")
    sys.exit(2)


if __name__ == "__main__":
    main()
