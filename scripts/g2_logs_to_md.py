"""Turn output/logs/g2_<instrument>_<bb|newera>.txt (feasibility_g3.py --g2
--table output) into the markdown table of docs §5f.

    .venv/bin/python scripts/g2_logs_to_md.py > /tmp/g2_tables.md
"""
import glob
import os
import re

ROW = re.compile(r"^\s+(.+?)\s+SNR2/h =\s+([0-9.]+); rate ([0-9.e+]+) cps/tel"
                 r"( READOUT-LIMITED x([0-9.]+))?, dead-time load ([0-9.]+), \|V\|\^2 median ([0-9.]+)")
HEAD = re.compile(r"^\s+(.+?) \[(\w+(?:\(A\))?)\]: rho = ([0-9.]+) mas at phase ([0-9.]+); best baseline (\d+) m")


def parse(path):
    rows, cur = [], None
    for line in open(path):
        m = HEAD.match(line)
        if m:
            cur = dict(system=m.group(1).split(" (")[0].split(" A-B")[0], sed=m.group(2),
                       rho=float(m.group(3)), b=int(m.group(5)))
            continue
        m = ROW.match(line)
        if m and cur:
            rows.append(dict(cur, backend=m.group(1).strip(), snr=float(m.group(2)),
                             rate=float(m.group(3)), scale=m.group(5), load=float(m.group(6)),
                             v2=float(m.group(7))))
    return rows


def main():
    print("| Instrument | System | SED | B [m] | Backend | SNR₂/√h | rate/tel | readout |")
    print("|---|---|---|---|---|---|---|---|")
    for inst in ("c2pu", "keck", "eonsii"):
        for sed in ("bb", "newera"):
            path = f"output/logs/g2_{inst}_{sed}.txt"
            if not os.path.exists(path):
                continue
            for r in parse(path):
                lim = f"limited ×{r['scale']}" if r["scale"] else "—"
                print(f"| {inst} | {r['system']} | {r['sed']} | {r['b']} | {r['backend']} | "
                      f"{r['snr']:.1f} | {r['rate']:.1e} | {lim} |")


if __name__ == "__main__":
    main()
