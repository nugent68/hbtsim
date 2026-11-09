"""Cross-check of hbtsim's IACT path against Fig. 13 of the MAGIC-SII
performance paper (Abe et al. 2024, MNRAS 529, 4387; the sensitivity
plot T. Hassan circulated with the cycle-21 proposal templates): the
relative uncertainty on a uniform-disk diameter after 2.5 h as a function
of B magnitude, for MAGIC-I + II, MAGIC + LST-1 and MAGIC + four LSTs, in
dark time (their dashed lines) and under a full Moon (their points).  Their
simulated star has the position and diameter of gamma CrV (theta_UD
0.72 mas from the NSII, their Table 2; dec -17.5 deg).

hbtsim supplies the uv tracks of the catalog arrays (magic_lst1,
magic_lst4; MAGIC alone is the first two stations), the pupil-averaged
uniform-disk |V|^2 and a Fisher matrix on (theta, N0); the per-block
sigma(|V|^2) is the group's own analog formula (hbtsim.iact, their Eq. 6)
with the catalog's per-station mirror area, QE and optical efficiency
(magic_sii backend).  Two bounds are drawn: N0 free (the group fits
|V|^2 = N0 V^2(theta); a single baseline then needs the baseline to
change through the slot) and N0 fixed (the zero-baseline correlation
taken from calibrators).

Assumptions of ours, not theirs:
  - the 2.5 h slot is centred on the meridian (a single-baseline result
    depends on where the slot falls; MAGIC alone spans 0.012-0.040 at
    B = 0.75 over the night);
  - m_AB(425 nm) = B - 0.12 for an early-type star;
  - the full-Moon background: beta = 10^(-0.4 (B_sky - B)) with B_sky the
    integrated sky in the 0.1 deg pixel, default 6.4 (fitted to where their
    Moon points leave their dark lines, B > 4.5; it is a sky of ~18.5
    mag/arcsec^2).

    .venv/bin/python scripts/magic_fig13_crosscheck.py [--b-sky 6.4] [--out docs/figures/magic_fig13_crosscheck.png]
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from hbtsim.bispectrum import Array
from hbtsim.catalog import load_array, load_backend
from hbtsim.iact import pair_track, track_sigma
from hbtsim.single import ud_vis2

THETA_UD = 0.72          # mas, gamma CrV (NSII; Abe et al. 2024 Table 2)
DEC = -17.54             # deg
HOURS = 2.5
BLOCK_MIN = 15.0
B_TO_AB = -0.12          # m_AB(425) - B_Vega, early-type star (ASSUMED)
MAGS = np.arange(0.75, 6.01, 0.75)
LINE_MAGS = np.linspace(0.5, 6.5, 25)

# Abe et al. 2024 Fig. 13, digitized (full-Moon points; dark lines read at one magnitude,
# their slope is 0.4 dex per magnitude by construction)
FIG13_MOON = {
    "MAGIC": [(0.75, 0.0092), (1.5, 0.0165), (2.25, 0.034), (3.0, 0.075), (3.75, 0.20)],
    "MAGIC-LST1": [(0.75, 0.0027), (1.5, 0.0048), (2.25, 0.0088), (3.0, 0.019), (3.75, 0.037), (4.5, 0.085), (5.25, 0.21)],
    "MAGIC-4LST": [(2.25, 0.0033), (3.0, 0.0058), (3.75, 0.0118), (4.5, 0.027), (5.25, 0.062), (6.0, 0.165)],
}
FIG13_DARK = {"MAGIC": (0.75, 0.0085), "MAGIC-LST1": (0.75, 0.0025), "MAGIC-4LST": (2.25, 0.0031)}
COLOURS = {"MAGIC": "#1f77b4", "MAGIC-LST1": "#ff7f0e", "MAGIC-4LST": "#2ca02c"}   # as in their figure


def arrays():
    a3 = load_array("magic_lst1")
    a6 = load_array("magic_lst4")
    return {"MAGIC": Array(a3.stations[:2], a3.site), "MAGIC-LST1": a3, "MAGIC-4LST": a6}


def ud_fn(theta_mas):
    def fn(bvecs, lam, phase, diam):
        b = np.asarray(bvecs, float)
        return ud_vis2(theta_mas, np.hypot(b[:, 0], b[:, 1]), float(lam), diam)[0]
    return fn


def fisher(arr, backend, mag_ab, window, theta=THETA_UD):
    """(sigma(theta)/theta with N0 free, with N0 fixed, the track)."""
    kw = dict(block_minutes=BLOCK_MIN, hour_angle_window_h=window, min_alt_deg=0.0)
    lam = backend.lambda_nm
    tr = pair_track(arr, DEC, lam, ud_fn(theta), **kw)
    e = 0.01 * theta
    up = pair_track(arr, DEC, lam, ud_fn(theta + e), **kw).vis2
    dn = pair_track(arr, DEC, lam, ud_fn(theta - e), **kw).vis2
    g = np.stack([(up - dn) / (2 * e), tr.vis2])
    sig = track_sigma(tr, mag_ab, backend)
    w = np.broadcast_to(1.0 / sig[:, None] ** 2, g.shape[1:])
    F = np.einsum("ipk,jpk,pk->ij", g, g, w)
    return float(np.sqrt(np.linalg.inv(F)[0, 0]) / theta), float(1.0 / np.sqrt(F[0, 0]) / theta), tr


def beta_for(b_mag, b_sky):
    return 10.0 ** (-0.4 * (b_sky - b_mag))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--b-sky", type=float, default=6.4, help="integrated full-Moon sky per pixel, B mag (default 6.4)")
    ap.add_argument("--out", default=None, help="figure path (default: no figure)")
    args = ap.parse_args()
    window = (-HOURS / 2, HOURS / 2)
    bk0 = load_backend("magic_sii")
    dark = replace(bk0, beta=0.0)
    res = {}
    print(f"gamma CrV geometry, theta_UD {THETA_UD} mas, {HOURS} h on the meridian, Moon sky B = {args.b_sky} per pixel")
    for name, arr in arrays().items():
        rows, line = [], []
        for m in MAGS:
            fd, xd, tr = fisher(arr, dark, m + B_TO_AB, window)
            fm, xm, _ = fisher(arr, replace(bk0, beta=beta_for(m, args.b_sky)), m + B_TO_AB, window)
            rows.append((float(m), fd, xd, fm, xm))
        for m in LINE_MAGS:
            fd, xd, _ = fisher(arr, dark, m + B_TO_AB, window)
            line.append((float(m), fd, xd))
        res[name] = dict(rows=rows, line=np.array(line))
        print(f"\n{name}: {len(tr.pair_names)} pair(s); projected baselines first/last block "
              + ", ".join(f"{b[0]:.0f}-{b[-1]:.0f} m" for b in tr.baseline_len_m))
        print("    B   dark N0 free  dark N0 fixed  Moon N0 free  Moon N0 fixed   Fig.13 Moon   ours/theirs (free, fixed)")
        theirs = dict(FIG13_MOON[name])
        for m, fd, xd, fm, xm in rows:
            t = theirs.get(m)
            tail = f"{t:12.4f}   {fm / t:6.2f} {xm / t:6.2f}" if t else "          --"
            print(f"  {m:4.2f}  {fd:12.4f}  {xd:13.4f}  {fm:12.4f}  {xm:13.4f} {tail}")
        m0, y0 = FIG13_DARK[name]
        fd, xd, _ = fisher(arr, dark, m0 + B_TO_AB, window)
        print(f"  dark line at B = {m0}: Fig. 13 {y0:.4f}; ours N0 free {fd:.4f} ({fd / y0:.2f}x), N0 fixed {xd:.4f} ({xd / y0:.2f}x)")
    if args.out:
        plot(res, args.b_sky, args.out)
        print("wrote", args.out)


def plot(res, b_sky, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=150)
    first = True
    for name, d in res.items():
        c = COLOURS[name]
        lm, free, fixed = d["line"].T
        lab = (lambda s: s) if first else (lambda s: None)
        ax.fill_between(lm, fixed, free, color=c, alpha=0.13, lw=0)
        ax.plot(lm, free, color=c, lw=2.0, label=lab("hbtsim dark, N$_0$ free"))
        ax.plot(lm, fixed, color=c, lw=1.2, ls=":", label=lab("hbtsim dark, N$_0$ fixed"))
        m0, y0 = FIG13_DARK[name]
        ax.plot(lm, y0 * 10 ** (0.4 * (lm - m0)), color=c, lw=1.4, ls="--", label=lab("Abe et al. 2024 Fig. 13, dark (2.5 h)"))
        rows = np.array(d["rows"])
        ax.plot(rows[:, 0], rows[:, 3], "o", mfc="white", mec=c, mew=1.6, ms=7, label=lab("hbtsim full Moon, N$_0$ free"))
        tk = np.array(FIG13_MOON[name])
        ax.plot(tk[:, 0], tk[:, 1], "o", color=c, ms=6, label=lab("Abe et al. 2024 Fig. 13, full Moon"))
        k = {"MAGIC": 2.0, "MAGIC-LST1": 3.4, "MAGIC-4LST": 4.8}[name]
        ax.annotate(name, (k, np.interp(k, lm, free)), xytext=(8, -14), textcoords="offset points",
                    color=c, fontsize=12, rotation=33)
        first = False
    ax.set_yscale("log")
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(2e-3, 0.5)
    ax.set_xlabel("B mag", fontsize=13)
    ax.set_ylabel(r"$\Delta\theta_{UD}/\theta_{UD}$", fontsize=13)
    ax.set_title(r"$\gamma$ CrV geometry ($\theta_{UD}$ 0.72 mas, $\delta = -17.5°$), 425/26 nm, 2.5 h on the meridian; "
                 f"Moon sky B = {b_sky:g} per pixel", fontsize=9.5, loc="left")
    ax.grid(True, which="both", alpha=0.18, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    main()
