"""Red-clump angular sizes with intensity interferometry (Kim & Kaiser 2026,
PASP 138, 044202) on hbtsim's photon budget with NewEra stand-in profiles:
radius conventions of the spherical models, one broad-band filter against
multiplexed optical backends, and theta_UD(lambda) at R = 5000.  This script
moved into the catalog campaign runner hbtsim/runners/scale.py; the campaigns
redclump_ii_dwarf (4800 K / log g 4.5 stand-in) and redclump_ii_supergiant
(5000 K / log g 0 stand-in) replace its --stand-in option, and --no-newera
gives the blackbody + linear-law case.
"""

import sys


def main():
    print("This script moved into the catalog campaign runner; run:")
    print("  .venv/bin/python -m hbtsim run redclump_ii_dwarf [--no-figures] [--set options.hours=2]")
    print("  .venv/bin/python -m hbtsim run redclump_ii_supergiant [--no-figures]")
    print("  .venv/bin/python -m hbtsim run redclump_ii_dwarf --no-newera   # blackbody + linear law")
    return 2


if __name__ == "__main__":
    sys.exit(main())
