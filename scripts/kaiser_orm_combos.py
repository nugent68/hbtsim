"""Rank the 2- and 3-telescope combinations of the five ORM telescopes
(NOT, TNG, GTC, WHT, INT) for R. Kaiser's 2027 targets from the per-pair
rows of the g2_kaiser_orm_2027 campaign (results.json): for every target
and filter, the SNR per night of a combination is the quadrature sum of
its pairs' SNR, and the nights to the campaign's diameter precision come
from the summed Fisher information on theta.

    .venv/bin/python -m hbtsim.run_campaign g2_kaiser_orm_2027 --quiet --out OUT
    .venv/bin/python scripts/kaiser_orm_combos.py OUT/g2_kaiser_orm_2027/results.json [--filter 431] [--top 6]
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations

import numpy as np

TELS = ("NOT", "TNG", "GTC", "WHT", "INT")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results")
    ap.add_argument("--filter", default="431", help="backend label substring (default 431)")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=3.0)
    args = ap.parse_args()
    d = json.load(open(args.results))
    rows = [r for r in d["rows"] if args.filter in r["backend"]]
    targets = []
    for r in rows:
        if r["target"] not in targets:
            targets.append(r["target"])
    by = {(r["target"], r["pair"]): r for r in rows}
    prec = rows[0]["diameter_precision"]

    def pair_key(a, b):
        for k in (f"{a}-{b}", f"{b}-{a}"):
            if (targets[0], k) in by:
                return k
        raise KeyError((a, b))

    combos = [c for n in (2, 3) for c in combinations(TELS, n)]
    print(f"filter {args.filter}; nights to {args.sigma:g} sigma on |V|^2 and to {100 * prec:g} % on theta, "
          "per combination (one night = the target's track above 30 deg)\n")
    score = {c: [] for c in combos}
    for t in targets:
        out = []
        for c in combos:
            pairs = [pair_key(a, b) for a, b in combinations(c, 2)]
            snr2 = sum(by[(t, p)]["snr2_night"] for p in pairs)
            fish = sum(by[(t, p)]["fisher_theta"] for p in pairs)
            rho = by[(t, pairs[0])]["rho_mas"]
            n_det = (args.sigma ** 2 / snr2) if snr2 > 0 else np.inf
            n_th = (1.0 / np.sqrt(fish) / (prec * rho)) ** 2 if fish > 0 else np.inf
            out.append((c, np.sqrt(snr2), n_det, n_th, pairs))
            score[c].append(n_th)
        out.sort(key=lambda x: x[3])
        r0 = by[(t, out[0][4][0])]
        print(f"{t}: theta {r0['rho_mas']:.3f} mas, V {r0['v_mag']:.2f}, {r0['hours']:.1f} h above 30 deg")
        print(f"  {'telescopes':16s} {'SNR/night':>9s} {'nights 3sig':>11s} {'nights 5%':>9s}   pairs (baseline range m)")
        for c, s, nd, nt, pairs in out[:args.top]:
            bl = ", ".join(f"{p} {by[(t, p)]['baseline_min_m']:.0f}-{by[(t, p)]['baseline_max_m']:.0f}" for p in pairs)
            print(f"  {'+'.join(c):16s} {s:9.2f} {nd:11.2f} {nt:9.2f}   {bl}")
        print()
    print("over the four targets (sum of the nights to the diameter precision):")
    for n in (2, 3):
        ranked = sorted((c for c in combos if len(c) == n), key=lambda c: sum(score[c]))
        for c in ranked[:args.top]:
            print(f"  {'+'.join(c):16s} {sum(score[c]):7.2f} nights   " + "  ".join(f"{t.split()[0]}{t.split()[1][:3] if len(t.split()) > 1 else ''} {x:.2f}" for t, x in zip(targets, score[c])))
        print()


if __name__ == "__main__":
    main()
