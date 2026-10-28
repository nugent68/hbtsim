"""Runner `iact`: nights of one target on a Cherenkov-telescope array
(MAGIC + LST-1, the CTAO-North layouts, VERITAS) with the groups' analog
photomultiplier sensitivity (hbtsim.iact): every pair's |V|^2 along the
night's uv track, sigma(|V|^2) per block from the backend's constants and
each station's mirror area, QE and optical efficiency, and the Fisher
precision accumulated night by night on

  - a single star: the diameter theta with the zero-baseline normalization
    N0 free (the groups fit |V|^2 = N0 V^2(theta)), plus the linear
    limb-darkening coefficient u with options.fit_u, or the ellipse
    (theta_major, axis ratio, PA) with options.ellipse (the target's
    `ellipse` entry drawn, the equal-area circle the alternative);
  - a binary: (ln a, Omega[, omega]) through the orbit, the phase advancing
    night to night (hbtsim.iact.binary_vis2_fn, pupil-averaged).

Campaign fields: target, instrument.array (an array whose stations carry
analog detectors), backends (analog, model "analog"; each is run on every
night and reported on its own), night.block_minutes / min_alt_deg, options:
  nights               number of nights (default 1)
  night_step_days      days between nights (default 1.0)
  phase0               orbital phase of the first night's start (binaries; default
                       maximum separation)
  pupils               average |V|^2 over the dish apertures (default true)
  fit_u                single stars: the linear limb-darkening coefficient free (default false)
  ellipse              single stars with an `ellipse`: fit the ellipse (default false)
  diameter_precision   the relative precision on theta (or a) to quote nights for (default 0.01)
  seed, display_bin_blocks   the simulated points of the figure (default 1, 3)
  figure               write <name>.png (default true; needs matplotlib)
Outputs: table.md (per backend and pair), table_nights.md (precision after
each night), <name>.png, results.json.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from hbtsim.iact import binary_vis2_fn, pair_track, single_vis2_fn, track_sigma
from hbtsim.orbit import max_separation_phase, positions_at
from hbtsim.single import ellipse_vis, single_star_vis2
from hbtsim.snr import system_ab_mag

from . import write_tables


def _is_single(target) -> bool:
    return hasattr(target, "theta_ld_mas")


# ---------------------------------------------------------------------------
# parameterized visibility models: params -> vis2_fn for pair_track
# ---------------------------------------------------------------------------
def _single_model(target, pupils: bool, fit_u: bool):
    """Circular limb-darkened disk: params (theta_mas[, u])."""
    from hbtsim.runners.schedule import with_ld_offset
    u0 = float(target.star.ld_coeff(425.0)) if target.star.ld_profile is None else None
    names = ["theta"] + (["u"] if fit_u else [])
    if fit_u and u0 is None:
        raise SystemExit("fit_u needs a star with a linear limb-darkening table, not a model profile")

    def model(params):
        t = replace(target, theta_ld_mas=float(params[0]))
        if fit_u:
            t = with_ld_offset(t, float(params[1]) - u0)

        def fn(bvecs, lambda_nm, phase, diameters):
            b = np.asarray(bvecs, dtype=float)
            return single_star_vis2(t, np.hypot(b[:, 0], b[:, 1]), float(lambda_nm), diameters if pupils else None)[0]
        return fn
    p0 = [float(target.theta_ld_mas)] + ([u0] if fit_u else [])
    eps = [0.01 * target.theta_ld_mas] + ([0.05] if fit_u else [])
    return names, p0, eps, model


def _ellipse_model(target, pupils: bool):
    """Elliptical outline: params (theta_major_mas, axis_ratio, pa_deg)."""
    e = target.ellipse
    if e is None:
        raise SystemExit(f"{target.name} has no ellipse entry for options.ellipse")

    def model(params):
        th, r, pa = (float(x) for x in params)
        return single_vis2_fn(lambda bv, nm: ellipse_vis(bv, nm, target.star, th, r, pa), pupils)
    return (["theta_major", "axis_ratio", "pa"], [e.theta_major_mas, e.axis_ratio, e.pa_deg],
            [0.01 * e.theta_major_mas, 0.01, 0.5], model)


def _binary_model(system, pupils: bool):
    """Two limb-darkened disks: params (ln a, Omega[, omega]); ln a through the
    distance (all angular sizes scale together, as in nightmovie_binary)."""
    names = ["lna", "Omega"] + (["omega"] if system.eccentricity > 0 else [])

    def model(params):
        s = replace(system, distance_pc=system.distance_pc * float(np.exp(-params[0])),
                    node_pa_deg=float(params[1]))
        if len(params) > 2:
            s = replace(s, arg_periastron_deg=float(params[2]))
        return binary_vis2_fn(s, pupils)
    p0 = [0.0, float(system.node_pa_deg or 0.0)] + ([float(system.arg_periastron_deg)] if len(names) > 2 else [])
    eps = [0.01, 0.5] + ([0.5] if len(names) > 2 else [])
    return names, p0, eps, model


# ---------------------------------------------------------------------------
# one night
# ---------------------------------------------------------------------------
def _track(array, target, lam, fn, block_minutes, min_alt, phase0, period):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pair_track(array, target.dec_deg, lam, fn, block_minutes=block_minutes, min_alt_deg=min_alt,
                          phase0=phase0, period_days=period)


def _night(array, target, backend, model, p0, eps, block_minutes, min_alt, phase0, period, pupils):
    """|V|^2 of every pair and block, sigma per block, and the Fisher matrix
    on (params..., N0) from the numerical derivatives of N0 |V|^2."""
    lam = backend.lambda_nm
    tr = _track(array, target, lam, model(p0), block_minutes, min_alt, phase0, period)
    if tr.hour_angle_h.size == 0:
        return None
    single = _is_single(target)
    mag = float(target.ab_mag(lam)) if single else float(system_ab_mag(target, lam))
    sig = track_sigma(tr, mag, backend)                         # (n_pair,) per block
    grads = []
    for i, e in enumerate(eps):
        up, dn = list(p0), list(p0)
        up[i] += e
        dn[i] -= e
        try:
            tu = _track(array, target, lam, model(up), block_minutes, min_alt, phase0, period)
            td = _track(array, target, lam, model(dn), block_minutes, min_alt, phase0, period)
            grads.append((tu.vis2 - td.vis2) / (2 * e))
        except ValueError:                                     # a perturbed orbit grazes an eclipse
            grads.append(np.zeros_like(tr.vis2))
    grads.append(tr.vis2)                                      # d(N0 V^2)/dN0 at N0 = 1
    g = np.stack(grads)                                        # (n_par + 1, n_pair, n_blk)
    w = 1.0 / sig[None, :, None] ** 2
    F = np.einsum("ipk,jpk,pk->ij", g, g, np.broadcast_to(w[0], g.shape[1:]))
    return dict(track=tr, sigma=sig, F=F, mag=mag, grads=g)


def _sigmas(F, names):
    """Marginal sigmas of the parameters (N0 profiled out) from a Fisher
    matrix, plus "<first>_alone": the first parameter with only N0 free
    (the groups' circular-disk fit with the shape fixed)."""
    try:
        cov = np.linalg.inv(F)
        s = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        s = np.full(F.shape[0], np.inf)
    out = dict(zip(list(names) + ["N0"], (float(x) for x in s)))
    sub = F[np.ix_([0, F.shape[0] - 1], [0, F.shape[0] - 1])]
    try:
        out[f"{names[0]}_alone"] = float(np.sqrt(max(np.linalg.inv(sub)[0, 0], 0.0)))
    except np.linalg.LinAlgError:
        out[f"{names[0]}_alone"] = float("inf")
    return out


def _label(name: str) -> str:
    """Display name of a parameter: ln a is reported as sigma(a)/a."""
    return "a" if name == "lna" else name


def _pct(x) -> str:
    return "inf" if x is None or not np.isfinite(x) else f"{100 * x:.3g} %"


def _fit_theta(track, sig, target, pupils, lam):
    """Least-squares circular diameter with N0 profiled analytically (the
    groups' uniform/limb-darkened-disk fit) to one night's simulated |V|^2."""
    v = track.vis2.ravel()
    w = np.repeat(1.0 / sig ** 2, track.vis2.shape[1])
    b = track.baseline_len_m.ravel()
    th0 = target.theta_ld_mas
    best = None
    for th in np.linspace(0.5 * th0, 1.5 * th0, 401):
        m = single_star_vis2(replace(target, theta_ld_mas=float(th)), b, lam, None)[0]
        n0 = np.sum(w * m * v) / max(np.sum(w * m * m), 1e-300)
        chi2 = float(np.sum(w * (v - n0 * m) ** 2))
        if best is None or chi2 < best[1]:
            best = (float(th), chi2, float(n0))
    return best


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def run(campaign, cat, opts, out_dir: Path) -> dict:
    warnings.filterwarnings("ignore", message=".*extrapolated.*")
    target, array = campaign.target, campaign.array
    if target is None or array is None:
        raise SystemExit(f"campaign {campaign.name}: runner iact needs a target and instrument.array")
    if array.site is None:
        raise SystemExit(f"campaign {campaign.name}: the array needs a site")
    if target.dec_deg is None:
        raise SystemExit(f"{target.name} has no declination")
    backends = [b for b in campaign.backends if hasattr(b, "b_el_hz")]
    if not backends:
        raise SystemExit(f"campaign {campaign.name}: runner iact needs analog backends (model 'analog')")
    opt = campaign.option
    n_nights = int(opt("options.nights", 1))
    step = float(opt("options.night_step_days", 1.0))
    block_minutes = float(opt("night.block_minutes", 20.0))
    min_alt = float(opt("night.min_alt_deg", 30.0))
    pupils = bool(opt("options.pupils", True))
    prec = float(opt("options.diameter_precision", 0.01))
    single = _is_single(target)
    if single:
        if opt("options.ellipse", False):
            names, p0, eps, model = _ellipse_model(target, pupils)
            scale_name, scale0 = "theta_major", p0[0]
        else:
            names, p0, eps, model = _single_model(target, pupils, bool(opt("options.fit_u", False)))
            scale_name, scale0 = "theta", p0[0]
        period, phase0 = None, 0.0
    else:
        names, p0, eps, model = _binary_model(target, pupils)
        scale_name, scale0 = "lna", 1.0
        period = target.period_days
        phase0 = opt("options.phase0")
        phase0 = max_separation_phase(target) if phase0 in (None, "max_separation") else float(phase0)
    stations = "+".join(s.name for s in array.stations)
    print(f"{target.name}: {'single star' if single else 'binary'} on {stations} at {array.site.name}, "
          f"{n_nights} night(s) of {block_minutes:g}-min blocks above {min_alt:g} deg; parameters {names} + N0"
          + (f"; phase0 {phase0:.3f}, P = {period:.4f} d" if period else ""))
    for s in array.stations:
        print(f"  {s.name}: {s.telescope.diameter_m:g} m, {s.telescope.area_m2:.0f} m^2, q {s.telescope.throughput:.2f}, "
              f"QE {s.detector.pde(backends[0].lambda_nm):.2f} at {backends[0].lambda_nm:.0f} nm")
    if single:
        print(f"  theta_LD = {target.theta_ld_mas:.3f} mas, V = {target.v_mag:.2f}, dec {target.dec_deg:+.1f}"
              + (f"; ellipse {target.ellipse.theta_major_mas:.3f} mas, ratio {target.ellipse.axis_ratio}, "
                 f"PA {target.ellipse.pa_deg}" if target.ellipse else ""))
    else:
        pos = positions_at(target, phase0)
        print(f"  a = {float(positions_at(target, 0.0).rho):.3f} mas (phase 0), rho = {float(pos.rho):.3f} mas at phase0, "
              f"e = {target.eccentricity}, dec {target.dec_deg:+.1f}")

    rng = np.random.default_rng(int(opt("options.seed", 1)))
    results, rows, night_rows, fig_data = {}, [], [], {}
    for bk in backends:
        lam = bk.lambda_nm
        print(f"\n=== {bk.name}: {lam:g} nm / {bk.dlambda_nm:g} nm, b_el {bk.b_el_hz / 1e6:g} MHz, F {bk.noise_factor:g}, "
              f"sigma_spec {bk.sigma_spec:g}, {bk.n_channels} channel(s) ===")
        F = np.zeros((len(names) + 1, len(names) + 1))
        nights, sig_after, first = [], [], None
        for n in range(n_nights):
            ph = (phase0 + n * step / period) % 1.0 if period else 0.0
            q = _night(array, target, bk, model, p0, eps, block_minutes, min_alt, ph, period, pupils)
            if q is None:
                print(f"  night {n + 1}: {target.name} never above {min_alt:g} deg from {array.site.name}")
                continue
            tr, sig = q["track"], q["sigma"]
            F += q["F"]
            s = _sigmas(F, names)
            rel = s[scale_name] / scale0
            rel_alone = s[f"{scale_name}_alone"] / scale0
            sig_after.append(rel)
            snr_night = float(np.sqrt(np.sum((tr.vis2 / sig[:, None]) ** 2)))
            hours = tr.hour_angle_h.size * tr.block_s / 3600.0
            extra = ", ".join(f"sigma({k}) {s[k]:.3g}" for k in names if k != scale_name)
            lab = _label(scale_name)
            print(f"  night {n + 1}: {tr.hour_angle_h.size} blocks ({hours:.1f} h)"
                  + (f", phase {ph:.3f}-{float(tr.phase[-1]) % 1:.3f}" if period else "")
                  + f"; |V|^2 {tr.vis2.min():.3f}-{tr.vis2.max():.3f}; S/N {snr_night:.0f}; after {n + 1} night(s): "
                  f"sigma({lab})/{lab} {_pct(rel)}" + (f" ({_pct(rel_alone)} with the shape fixed), {extra}" if extra else "")
                  + f", sigma(N0) {s['N0']:.3g}")
            nights.append(dict(night=n + 1, phase0=float(ph), n_blocks=int(tr.hour_angle_h.size), hours=float(hours),
                               snr=snr_night, sigma_rel_cumulative=float(rel),
                               sigma_rel_alone_cumulative=float(rel_alone) if np.isfinite(rel_alone) else None,
                               sigmas_cumulative={k: (None if not np.isfinite(v) else v) for k, v in s.items()}))
            if first is None:
                first = q
                for p, name in enumerate(tr.pair_names):
                    print(f"    {name:22s} B {tr.baseline_len_m[p].min():.0f}-{tr.baseline_len_m[p].max():.0f} m, "
                          f"PA {tr.position_angle_deg[p].min():.0f}..{tr.position_angle_deg[p].max():.0f} deg, "
                          f"|V|^2 {tr.vis2[p].min():.3f}-{tr.vis2[p].max():.3f}, sigma(|V|^2) {sig[p]:.4f} per block, "
                          f"{sig[p] * np.sqrt(tr.block_s / 3600.0):.4f} per hour")
                    rows.append([bk.name, name, f"{tr.baseline_len_m[p].min():.0f}-{tr.baseline_len_m[p].max():.0f}",
                                 f"{tr.vis2[p].min():.3f}-{tr.vis2[p].max():.3f}", f"{sig[p]:.4f}",
                                 f"{sig[p] * np.sqrt(tr.block_s / 3600.0):.4f}",
                                 f"{float(np.sqrt(np.sum((tr.vis2[p] / sig[p]) ** 2))):.0f}"])
        if not nights:
            continue
        s = _sigmas(F, names)
        rel = s[scale_name] / scale0
        n_done = len(nights)
        nights_to = n_done * (rel / prec) ** 2 if np.isfinite(rel) and rel > 0 else None
        lab = _label(scale_name)
        fit = None
        if single and not opt("options.ellipse", False) and first is not None:
            tr, sig = first["track"], first["sigma"]
            meas = tr.vis2 + rng.normal(size=tr.vis2.shape) * sig[:, None]
            fit = _fit_theta(replace(tr, vis2=meas), sig, target, pupils, lam)
            print(f"  simulated first night (seed {int(opt('options.seed', 1))}): circular fit theta = {fit[0]:.4f} mas "
                  f"(truth {target.theta_ld_mas:.4f}, Fisher sigma {s['theta']:.4f}), N0 = {fit[2]:.3f}, chi2/N = {fit[1] / tr.vis2.size:.2f}")
            fig_data[bk.name] = dict(track=tr, meas=meas, sigma=sig, fit=fit)
        elif first is not None:
            tr, sig = first["track"], first["sigma"]
            fig_data[bk.name] = dict(track=tr, meas=tr.vis2 + rng.normal(size=tr.vis2.shape) * sig[:, None], sigma=sig, fit=None)
        print(f"  after {n_done} night(s): sigma({lab})/{lab} = {_pct(rel)}"
              + (f" -> {nights_to:.3g} nights to {prec:.1%}" if nights_to else "")
              + (f"; {_pct(s[f'{scale_name}_alone'] / scale0)} with the shape fixed" if len(names) > 1 else "")
              + "".join(f"; sigma({k}) = {s[k]:.3g}" for k in names if k != scale_name))
        night_rows += [[bk.name, str(r["night"]), f"{r['hours']:.1f}", f"{r['snr']:.0f}", _pct(r["sigma_rel_cumulative"])]
                       + [f"{r['sigmas_cumulative'][k]:.3g}" if r["sigmas_cumulative"][k] is not None else "inf"
                          for k in names if k != scale_name] for r in nights]
        results[bk.name] = dict(lambda_nm=lam, nights=nights, sigma_rel=float(rel) if np.isfinite(rel) else None,
                                sigma_rel_alone=(float(s[f"{scale_name}_alone"] / scale0)
                                                 if np.isfinite(s[f"{scale_name}_alone"]) else None),
                                sigmas={k: (None if not np.isfinite(v) else v) for k, v in s.items()},
                                nights_to_precision=nights_to, precision=prec,
                                fisher=F.tolist(), parameters=list(names) + ["N0"],
                                fit_first_night=None if fit is None else dict(theta_mas=fit[0], chi2=fit[1], N0=fit[2]),
                                pairs=[dict(pair=nm, baseline_min_m=float(first["track"].baseline_len_m[p].min()),
                                            baseline_max_m=float(first["track"].baseline_len_m[p].max()),
                                            vis2_min=float(first["track"].vis2[p].min()), vis2_max=float(first["track"].vis2[p].max()),
                                            sigma_block=float(first["sigma"][p]),
                                            sigma_hour=float(first["sigma"][p] * np.sqrt(first["track"].block_s / 3600.0)))
                                       for p, nm in enumerate(first["track"].pair_names)])
    write_tables(out_dir, rows, ["backend", "pair", "B [m]", "|V|^2", "sigma per block", "sigma per hour", "S/N night"],
                 latex=bool(opts.latex))
    write_tables(out_dir, night_rows, ["backend", "night", "hours", "S/N", f"sigma({_label(scale_name)}) rel."]
                 + [f"sigma({k})" for k in names if k != scale_name], name="table_nights", latex=bool(opts.latex))
    out = dict(target=target.name, array=stations, site=array.site.name, single=single, parameters=list(names) + ["N0"],
               scale_parameter=scale_name, truth={k: v for k, v in zip(names, p0)}, n_nights=n_nights,
               block_minutes=block_minutes, pupils=pupils, backends=results)
    if single:
        out["theta_ld_mas"] = float(target.theta_ld_mas)
    else:
        out["a_mas"] = float(positions_at(target, 0.0).rho)
    if opts.figures and bool(opt("options.figure", True)) and fig_data:
        try:
            path = _figure(out_dir / f"{campaign.name}.png", campaign, target, array, backends, fig_data, results, names,
                           scale_name, int(opt("options.display_bin_blocks", 3)), single, pupils)
            out["figure"] = str(path)
            print(f"  wrote {path}")
        except ImportError:
            print("  matplotlib not installed: no figure")
    return out


# ---------------------------------------------------------------------------
# figure: uv coverage, |V|^2 against the projected baseline, precision by night
# ---------------------------------------------------------------------------
def _figure(path, campaign, target, array, backends, fig_data, results, names, scale_name, nbin, single, pupils):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a_uv, a_v, a_n) = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1, 1.5, 1]))
    bk0 = backends[0]
    d0 = fig_data[bk0.name]
    tr = d0["track"]
    kinds = sorted({tuple(sorted((si.telescope.diameter_m, sj.telescope.diameter_m))) for si, sj in tr.stations})
    colors = {k: f"C{i}" for i, k in enumerate(kinds)}
    for p, (si, sj) in enumerate(tr.stations):
        k = tuple(sorted((si.telescope.diameter_m, sj.telescope.diameter_m)))
        u, v = tr.baseline_vec_m[p, :, 0], tr.baseline_vec_m[p, :, 1]
        a_uv.plot(np.r_[u, -u], np.r_[v, -v], ".", ms=2, color=colors[k], alpha=0.6)
    for k, c in colors.items():
        a_uv.plot([], [], ".", color=c, label=f"{k[0]:g} m – {k[1]:g} m")
    a_uv.set_aspect("equal")
    a_uv.set_xlabel("u [m] (east)")
    a_uv.set_ylabel("v [m] (north)")
    a_uv.set_title(f"{tr.pair_names.__len__()} pairs, one night", fontsize=10)
    a_uv.legend(fontsize=7)
    a_uv.grid(alpha=0.3)
    # |V|^2 against baseline: the model curve (singles) and the first night's simulated points, binned
    for bi, bk in enumerate(backends):
        d = fig_data.get(bk.name)
        if d is None:
            continue
        t, meas, sig = d["track"], d["meas"], d["sigma"]
        nb = max(1, nbin)
        for p in range(meas.shape[0]):
            k = tuple(sorted((t.stations[p][0].telescope.diameter_m, t.stations[p][1].telescope.diameter_m)))
            m = meas.shape[1] // nb * nb
            if m == 0:
                continue
            bb = t.baseline_len_m[p, :m].reshape(-1, nb).mean(axis=1)
            vv = meas[p, :m].reshape(-1, nb).mean(axis=1)
            a_v.errorbar(bb, vv, yerr=sig[p] / np.sqrt(nb), fmt="o" if bi == 0 else "s", ms=2.5, lw=0.6, capsize=0,
                         color=colors[k], alpha=0.8 if bi == 0 else 0.5)
    if single:
        B = np.linspace(1.0, 1.05 * tr.baseline_len_m.max(), 400)
        ell = campaign.option("options.ellipse", False) and target.ellipse is not None
        for bi, bk in enumerate(backends):
            a_v.plot(B, single_star_vis2(target, B, bk.lambda_nm, None)[0], "-" if bi == 0 else "--", color="k", lw=1.0,
                     label=f"circular theta_LD = {target.theta_ld_mas:.3f} mas at {bk.lambda_nm:g} nm"
                           + (" (reference; points drawn from the ellipse)" if ell else ""))
    a_v.set_xlabel("projected baseline [m]")
    a_v.set_ylabel(r"$|V|^2$")
    a_v.set_ylim(-0.05, 1.05)
    a_v.set_title(f"{target.name.split(' (')[0]}: first night, {nbin} blocks per point"
                  + ("" if len(backends) == 1 else "; circles / squares = backends"), fontsize=10)
    if single:
        a_v.legend(fontsize=7)
    a_v.grid(alpha=0.3)
    # precision after each night
    for bi, bk in enumerate(backends):
        r = results.get(bk.name)
        if not r:
            continue
        x = [q["night"] for q in r["nights"]]
        y = [100 * q["sigma_rel_cumulative"] for q in r["nights"]]
        a_n.plot(x, y, "o-" if bi == 0 else "s--", color=f"C{bi}", label=bk.name[:40])
    a_n.set_xlabel("nights")
    a_n.set_ylabel(f"sigma({_label(scale_name)}) / {_label(scale_name)} [%]")
    a_n.set_yscale("log")
    a_n.set_title("precision, N0 free" + (f" ({', '.join(k for k in names if k != scale_name)} free)" if len(names) > 1 else ""),
                  fontsize=10)
    a_n.legend(fontsize=7)
    a_n.grid(alpha=0.3, which="both")
    fig.suptitle(campaign.label[:120], fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
