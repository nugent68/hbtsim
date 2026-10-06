"""The EON-SII design-study Monte Carlo (Sirius B, 10 h, three zenith
angles, 1000 channels, MCP-PMT and SPAD) rerun with hbtsim's photon
budget, and the ladder that closes the gap to the study's quoted precision
one assumption at a time.  This now lives in the catalog campaign runner
hbtsim/runners/montecarlo.py, driven by
hbtsim/configs/campaigns/mc_sirius_b_eonsii.json.
"""

import sys


def main():
    print("This script moved into the catalog campaign runner; run:")
    print("  .venv/bin/python -m hbtsim run mc_sirius_b_eonsii")
    print("  .venv/bin/python -m hbtsim run mc_sirius_b_eonsii --set options.n_realizations=100"
          " --set options.theta_true_mas=0.0295")
    print("  .venv/bin/python -m hbtsim run mc_sirius_b_eonsii --set backends=[\"eonsii_mcp\"]"
          "  # one detector")
    return 2


if __name__ == "__main__":
    sys.exit(main())
