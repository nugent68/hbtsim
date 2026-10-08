"""Binary mode of the `nightmovie` runner: an eclipsing/spectroscopic pair
followed through its orbit on a two-telescope array, one night per frame
set -- the fringe pattern of each night, the fitted separation landing on
the apparent orbit night after night, and the angular semi-major axis
(hence the geometric distance) sharpening as the nights accumulate.

Campaign fields as the single-star mode (target now a binary, one
backend, instrument.array with two stations, night.block_minutes);
options:
  nights                 consecutive nights (default 8)
  phase0                 orbital phase at the middle of the first night (default 0.0)
  night_step_days        days between the nights' midpoints (default 1.0)
  gap                    optional {after_night, days}: a break in the run (a second orbit a season later)
  third_light_fraction   fraction of the collected light from unresolved companions
                         (adds photons, dilutes the fringe by (1 - f)^2; default 0)
  display_bin_blocks     blocks per displayed |V|^2 point (default 3)
  fit_grid_mas, fit_step_mas   half-width and step of the per-night (dx, dy) chi^2 grid
  global_fit, fit_scale_range, fit_scale_step, fit_node_step_deg
                         chi^2 of the nights so far over the orbit's angular scale (default
                         0.5-1.5 x the assumed a, step 0.01) and node-angle offset (0-180 deg,
                         step 1.5): the right panel shows the distance after each night with
                         the surviving fringe-alias solutions; the closing act shows the map
                         and the family of orbits allowed at 68 %
  reference_distances    [{label, pc, lo_pc, hi_pc}] drawn on the distance panel (e.g. Hipparcos)
  seed, fps, dpi

The model visibility is the rendered two-disk image (hbtsim.spectral.spectral_vis,
valid through the eclipses); the per-night fit and the Fisher precision use the
analytic two-disk visibility, so eclipse nights (disks overlapping) carry no
separation point -- they are the nights the fringe disappears.
"""

from __future__ import annotations

import shutil
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from hbtsim.bispectrum import Array, binary_vis_complex_analytic
from hbtsim.orbit import positions_at, sky_positions
from hbtsim.params import MAS, GridConfig, in_eclipse
from hbtsim.snr import (Observation, g2_snr, incident_rate, polarization_streams, readout_scale,
                        system_ab_mag)
from hbtsim.spectral import spectral_vis


def _budget_binary(system, lam, w, block_s, t1, t2, det, pol, throughput, f3):
    """sigma(|V|^2 of the pair) per block, with the companions' light f3
    adding to the rates and diluting the fringe by (1 - f3)^2."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mag_pair = float(system_ab_mag(system, lam))
    mag = mag_pair + 2.5 * np.log10(1.0 - f3)                 # brighter by the companions
    obs = Observation(wavelength_nm=lam, filter_width_nm=w, t_int_s=block_s, polarization_mode=pol,
                      backend_throughput=throughput)
    n_streams = polarization_streams(pol)[0]
    r1 = float(incident_rate(mag, t1, det, obs)) * n_streams
    r2 = float(incident_rate(mag, t2, det, obs)) * n_streams
    scale = min(readout_scale(det, r1), readout_scale(det, r2))
    mag_eff = mag - 2.5 * np.log10(scale) if scale < 1.0 else mag
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = g2_snr(1.0, mag_eff, obs, telescope1=t1, telescope2=t2, detector1=det)
    sigma_eff = 1.0 / float(r.snr)                             # on (1 - f3)^2 |V|^2
    return dict(sigma_v2=sigma_eff / (1.0 - f3) ** 2, sigma_eff=sigma_eff, mag_pair=mag_pair, mag_total=mag,
                rate_cps=max(r1, r2), readout_scale=scale, r1=float(r.rate1_cps), r2=float(r.rate2_cps),
                dilution=(1.0 - f3) ** 2)


def _pair_terms(system, pos, bvecs, lam):
    """A(u), C(u) of |V|^2 = A + C cos(2 pi u . dtheta) for the two disks
    (fluxes and disk visibilities at this epoch), from the analytic V at
    the true separation and at its mirror image: with V = p + q e^{-i phi},
    A = |p|^2 + |q|^2 and C = 2|p||q|."""
    from hbtsim.limbdark import star_disk_visibility
    b = np.asarray(bvecs, dtype=float)
    lam_m = lam * 1e-9
    b_len = np.hypot(b[:, 0], b[:, 1])
    f, v = [], []
    for s in (system.primary, system.secondary):
        theta_d = 2.0 * system.drawn_radius_mas(s) * MAS
        f.append(float(np.atleast_1d(s.surface_flux(lam))[0]) * theta_d ** 2)
        v.append(np.asarray(star_disk_visibility(s, np.pi * theta_d * b_len / lam_m, lam), dtype=float).ravel())
    norm = (f[0] + f[1]) ** 2
    a = (f[0] ** 2 * v[0] ** 2 + f[1] ** 2 * v[1] ** 2) / norm
    c = 2.0 * f[0] * f[1] * v[0] * v[1] / norm
    return a, c


def _fit_night(system, pos_track, bvecs, lam, v2_meas, sig, half, step):
    """chi^2 grid over the mid-night separation (dx, dy) [mas] of the
    secondary from the primary, with the ephemeris' differential motion
    within the night; returns the best point, its 1-sigma covariance from
    the chi^2 curvature, and the chi^2 map (for the 180-degree twin)."""
    u = bvecs / (lam * 1e-9)                                   # (K, 2) cycles/rad
    k_mid = len(pos_track) // 2
    dx_t = np.array([float(p.x2 - p.x1) for p in pos_track]) - float(pos_track[k_mid].x2 - pos_track[k_mid].x1)
    dy_t = np.array([float(p.y2 - p.y1) for p in pos_track]) - float(pos_track[k_mid].y2 - pos_track[k_mid].y1)
    a, c = _pair_terms(system, pos_track[k_mid], bvecs, lam)
    g = np.arange(-half, half + step / 2, step)
    gx, gy = np.meshgrid(g, g, indexing="ij")                  # (G, G)
    ph_t = 2 * np.pi * (u[:, 0] * dx_t + u[:, 1] * dy_t) * MAS   # (K,)
    chi2 = np.zeros_like(gx)
    for k in range(u.shape[0]):
        model = a[k] + c[k] * np.cos(2 * np.pi * (u[k, 0] * gx + u[k, 1] * gy) * MAS + ph_t[k])
        chi2 += ((v2_meas[k] - model) / sig) ** 2
    i, j = np.unravel_index(np.argmin(chi2), chi2.shape)
    best = np.array([gx[i, j], gy[i, j]])
    # curvature at the minimum (central differences on the grid)
    def c2(ii, jj):
        return chi2[np.clip(ii, 0, g.size - 1), np.clip(jj, 0, g.size - 1)]
    hxx = (c2(i + 1, j) - 2 * c2(i, j) + c2(i - 1, j)) / step ** 2
    hyy = (c2(i, j + 1) - 2 * c2(i, j) + c2(i, j - 1)) / step ** 2
    hxy = (c2(i + 1, j + 1) - c2(i + 1, j - 1) - c2(i - 1, j + 1) + c2(i - 1, j - 1)) / (4 * step ** 2)
    hess = 0.5 * np.array([[hxx, hxy], [hxy, hyy]])
    try:
        cov = np.linalg.inv(hess)
        if not np.all(np.isfinite(cov)) or cov[0, 0] <= 0 or cov[1, 1] <= 0:
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        cov = np.diag([step ** 2, step ** 2])
    return best, cov, chi2, g


def _fisher_scale_node(system, pos_phases, bvecs_nights, lam, sig):
    """Fisher information on (ln a_mas, Omega) from the out-of-eclipse
    blocks of every night: numerical derivatives of the analytic |V|^2
    with the distance (all angular sizes scale) and the node angle."""
    eps_d, eps_o = 0.01, 0.5                                   # 1 % in distance, 0.5 deg in Omega
    info = np.zeros((2, 2))
    sys_dp = replace(system, distance_pc=system.distance_pc * (1 + eps_d))
    sys_dm = replace(system, distance_pc=system.distance_pc * (1 - eps_d))
    sys_op = replace(system, node_pa_deg=(system.node_pa_deg or 0.0) + eps_o)
    sys_om = replace(system, node_pa_deg=(system.node_pa_deg or 0.0) - eps_o)
    for phases, bvecs in zip(pos_phases, bvecs_nights):
        for ph, b in zip(phases, bvecs):
            pos = positions_at(system, ph)
            if in_eclipse(system, pos):
                continue
            bb = b[None, :]
            dv_dlnd = (abs(binary_vis_complex_analytic(bb, lam, sys_dp, positions_at(sys_dp, ph)))[0] ** 2
                       - abs(binary_vis_complex_analytic(bb, lam, sys_dm, positions_at(sys_dm, ph)))[0] ** 2) / (2 * eps_d)
            dv_do = (abs(binary_vis_complex_analytic(bb, lam, sys_op, positions_at(sys_op, ph)))[0] ** 2
                     - abs(binary_vis_complex_analytic(bb, lam, sys_om, positions_at(sys_om, ph)))[0] ** 2) / (2 * eps_o)
            grad = np.array([-dv_dlnd, dv_do])                 # ln a_mas = -ln d
            info += np.outer(grad, grad) / sig ** 2
    return info


def _islands(dchi2, s_grid, om_grid):
    """The separate 95 % islands (Delta chi^2 < 6.17) of a (scale, node) map,
    with the node angle wrapping at 180 deg; each with its deepest point and
    its 68 % and 95 % extents in scale, deepest first."""
    from scipy import ndimage
    n = om_grid.size
    lab, _ = ndimage.label(np.concatenate([dchi2, dchi2, dchi2], axis=1) < 6.17)
    mid = lab[:, n:2 * n]
    out = []
    for L in set(np.unique(mid)) - {0}:
        m = mid == L
        i, j = np.unravel_index(int(np.argmin(np.where(m, dchi2, np.inf))), dchi2.shape)
        in68 = m & (dchi2 < 2.30)
        rng = lambda mm: (float(s_grid[mm.any(axis=1)].min()), float(s_grid[mm.any(axis=1)].max()))
        out.append(dict(s_best=float(s_grid[i]), om_best=float(om_grid[j]), dchi2_min=float(dchi2[i, j]),
                        s68=rng(in68) if in68.any() else None, s95=rng(m)))
    return sorted(out, key=lambda d: d["dchi2_min"])


def _global_fit(system, nights, bvec, lam, sig, s_grid, om_grid):
    """chi^2 of every out-of-eclipse block over the orbit's angular scale s
    (a_mas = s x the assumed value, all angular sizes scaling with it) and
    the node angle offset Omega: the two unknowns of a spectroscopic pair
    seen with one baseline.  The model cube is built once; the chi^2 is
    then accumulated night by night, so the result carries the fit after
    1, 2, ... N nights (the distance converging) as well as the final map
    and its islands (the fringe aliases)."""
    om = np.radians(om_grid)
    blocks = [(j, k) for j, nn in enumerate(nights) for k in range(len(nn["pos"])) if not nn["eclipse"][k]]
    M = np.zeros((len(blocks), s_grid.size, om_grid.size))
    for i, sc in enumerate(s_grid):
        sys_s = replace(system, distance_pc=system.distance_pc / sc)
        for n, (j, k) in enumerate(blocks):
            pos = positions_at(sys_s, float(nights[j]["phases"][k]))
            a, c = _pair_terms(sys_s, pos, bvec[k:k + 1], lam)
            dx, dy = float(pos.x2 - pos.x1), float(pos.y2 - pos.y1)
            u = bvec[k] / (lam * 1e-9)
            M[n, i] = a[0] + c[0] * np.cos(2 * np.pi * (u[0] * (dx * np.cos(om) + dy * np.sin(om))
                                                       + u[1] * (-dx * np.sin(om) + dy * np.cos(om))) * MAS)
    data = np.array([nights[j]["v2_meas"][k] for j, k in blocks])
    per_block = ((data[:, None, None] - M) / sig) ** 2
    night_of = np.array([j for j, k in blocks])
    per_night, chi2 = [], np.zeros((s_grid.size, om_grid.size))
    for j in range(len(nights)):
        sel = night_of == j
        if sel.any():
            chi2 = chi2 + per_block[sel].sum(axis=0)
        if not np.any(night_of <= j):
            per_night.append(None)
            continue
        d = chi2 - chi2.min()
        isl = _islands(d, s_grid, om_grid)
        per_night.append(dict(new_info=bool(sel.any()), s_best=isl[0]["s_best"], om_best=isl[0]["om_best"],
                              s68_best_island=isl[0]["s68"] or isl[0]["s95"], islands=isl, n_islands95=len(isl)))
    d = chi2 - chi2.min()
    i, j = np.unravel_index(int(np.argmin(chi2)), chi2.shape)
    in68 = d < 2.30
    s_rng = lambda m: (float(s_grid[m.any(axis=1)].min()), float(s_grid[m.any(axis=1)].max()))
    fin = per_night[-1]
    return dict(chi2=chi2, dchi2=d, s_grid=s_grid, om_grid=om_grid, s_best=float(s_grid[i]), om_best=float(om_grid[j]),
                s68=s_rng(in68), s95=s_rng(d < 6.17), s68_best_island=fin["s68_best_island"], n_islands95=fin["n_islands95"],
                om68_best_island=(float(om_grid[(in68 & (d < 6.17)).any(axis=0)].min()), float(om_grid[in68.any(axis=0)].max())),
                n_blocks=len(blocks), per_night=per_night)


def run_binary(campaign, cat, opts, out_dir: Path) -> dict:
    system = campaign.target
    array = campaign.array
    if len(campaign.backends) != 1:
        raise SystemExit("the binary night movie takes one backend")
    b = campaign.backends[0]
    spec, det, pol = b.spectrograph, b.detector, b.polarization_mode
    arr = Array(tuple(replace(s, detector=det) for s in array.stations), array.site)
    t1, t2 = arr.stations[0].telescope, arr.stations[1].telescope
    opt = campaign.option
    block_minutes = float(opt("night.block_minutes", 10.0))
    min_alt = float(opt("night.min_alt_deg", 30.0))
    lam, w = float(spec.channel_centers_nm[0]), float(spec.channel_widths_nm[0])
    rng = np.random.default_rng(int(opt("options.seed", 7)))
    n_nights = int(opt("options.nights", 8))
    phase0 = float(opt("options.phase0", 0.0))
    step_days = float(opt("options.night_step_days", 1.0))
    f3 = float(opt("options.third_light_fraction", 0.0))
    nbin_show = int(opt("options.display_bin_blocks", 3))
    fit_half = float(opt("options.fit_grid_mas", 1.3 * system.angular_semimajor_mas))
    fit_step = float(opt("options.fit_step_mas", 0.005))
    if system.dec_deg is None:
        raise SystemExit(f"{system.name} has no declination")

    from hbtsim.runners.nightmovie import _night
    mids, block_s, bvec, alt = _night(arr, system.dec_deg, block_minutes, min_alt)
    blen = np.hypot(bvec[:, 0], bvec[:, 1])
    bud = _budget_binary(system, lam, w, block_s, t1, t2, det, pol, spec.throughput, f3)
    grid = GridConfig().for_system(system)
    P = system.period_days
    print(f"{system.name}: a = {system.angular_semimajor_mas:.3f} mas, P = {P:.4f} d, "
          f"diameters {2 * system.angular_radius_mas(system.primary):.3f} / "
          f"{2 * system.angular_radius_mas(system.secondary):.3f} mas, d = {system.distance_pc:.0f} pc")
    print(f"  {arr.stations[0].name} + {arr.stations[1].name}, {b.name}: {mids.size} blocks of {block_minutes:g} min, "
          f"B {blen.min():.0f}-{blen.max():.0f} m; AB({lam:.0f}) pair {bud['mag_pair']:.2f}, "
          f"collected {bud['mag_total']:.2f} (third light {f3:.2f}, fringe dilution {bud['dilution']:.2f}); "
          f"rate {bud['rate_cps']:.3g} cps, readout scale {bud['readout_scale']:.2f}; "
          f"sigma(|V|^2 pair) {bud['sigma_v2']:.3g} per block")
    print(f"  fringe period lambda/a = {lam * 1e-9 / (system.angular_semimajor_mas * MAS):.0f} m at quadrature")

    nights = []
    phases_all, bvecs_all = [], []
    gap = opt("options.gap")                                   # {"after_night": 8, "days": 323}: a break in the run
    day_of = [j * step_days + (float(gap["days"]) if gap and j >= int(gap["after_night"]) else 0.0) for j in range(n_nights)]
    for j in range(n_nights):
        ph_mid = (phase0 + day_of[j] / P) % 1.0
        phases = (ph_mid + (mids - mids.mean()) / 24.0 / P) % 1.0
        pos_track = [positions_at(system, float(ph)) for ph in phases]
        ecl = np.array([bool(in_eclipse(system, p)) for p in pos_track])
        v2_true = np.array([float(np.abs(np.asarray(spectral_vis(p, bvec[k:k + 1], [lam], system, grid))[0, 0]) ** 2)
                            for k, p in enumerate(pos_track)])
        v2_meas = v2_true + rng.normal(0.0, bud["sigma_v2"], size=v2_true.size)
        night = dict(phase_mid=ph_mid, day=day_of[j], phases=phases, pos=pos_track, eclipse=ecl, v2_true=v2_true, v2_meas=v2_meas,
                     rho_mid=float(pos_track[len(pos_track) // 2].rho),
                     pa_mid=float(pos_track[len(pos_track) // 2].position_angle_deg))
        if ecl.mean() < 0.5:
            best, cov, chi2, g = _fit_night(system, pos_track, bvec, lam, v2_meas, bud["sigma_v2"], fit_half, fit_step)
            pm = pos_track[len(pos_track) // 2]
            truth = np.array([float(pm.x2 - pm.x1), float(pm.y2 - pm.y1)])
            if np.hypot(*(best - truth)) > np.hypot(*(best + truth)):
                best = -best                                   # the 180-degree twin: pick the branch on the orbit
            dchi2 = chi2 - chi2.min()
            area = float(np.sum(dchi2 <= 2.30) * fit_step ** 2)          # the 68 % region of two parameters
            in_region = bool(dchi2[int(np.argmin(np.abs(g - truth[0]))), int(np.argmin(np.abs(g - truth[1])))] <= 2.30)
            night.update(fit=best, cov=cov, truth=truth, dchi2=dchi2, grid=g, region_area=area, truth_in_region=in_region)
            print(f"  night {j + 1}: phase {ph_mid:.3f}, rho {night['rho_mid']:.3f} mas, |V|^2 {v2_true.min():.3f}-{v2_true.max():.3f}, "
                  f"best ({best[0]:+.3f}, {best[1]:+.3f}) vs true ({truth[0]:+.3f}, {truth[1]:+.3f}) mas; "
                  f"68 % region {area:.3f} mas^2 (truth {'inside' if in_region else 'outside'})")
        else:
            print(f"  night {j + 1}: phase {ph_mid:.3f}, rho {night['rho_mid']:.3f} mas -- eclipse "
                  f"({100 * ecl.mean():.0f} % of the blocks), |V|^2 {v2_true.min():.3f}-{v2_true.max():.3f}, no fringe")
        nights.append(night)
        phases_all.append(phases); bvecs_all.append(bvec)
        # the cumulative (a, Omega) precision after this night
        info = _fisher_scale_node(system, phases_all, bvecs_all, lam, bud["sigma_v2"])
        try:
            cov_so = np.linalg.inv(info)
            sig_lna, sig_om = float(np.sqrt(cov_so[0, 0])), float(np.sqrt(cov_so[1, 1]))
        except np.linalg.LinAlgError:
            sig_lna, sig_om = float("inf"), float("inf")
        night.update(sigma_lna=sig_lna, sigma_omega_deg=sig_om)
        print(f"    after {j + 1} night(s): sigma(a)/a = sigma(d)/d = {100 * sig_lna:.1f} %, sigma(Omega) = {sig_om:.1f} deg")

    gfit = None
    if any("fit" in nn for nn in nights) and opt("options.global_fit", True):
        s_lo, s_hi = (float(x) for x in opt("options.fit_scale_range", [0.5, 1.5]))
        s_grid = np.arange(s_lo, s_hi + 1e-9, float(opt("options.fit_scale_step", 0.01)))
        om_grid = np.arange(0.0, 180.0, float(opt("options.fit_node_step_deg", 1.5)))
        gfit = _global_fit(system, nights, bvec, lam, bud["sigma_v2"], s_grid, om_grid)
        a0 = system.angular_semimajor_mas
        d0 = system.distance_pc
        for j, pn in enumerate(gfit["per_night"]):
            if pn is None or not pn["new_info"]:
                continue
            lo, hi = pn["s68_best_island"]
            print(f"    global fit after night {j + 1}: d = {d0 / pn['s_best']:.0f} pc (68 % {d0 / hi:.0f}-{d0 / lo:.0f}), "
                  f"{pn['n_islands95']} island(s)" + ("" if pn["n_islands95"] == 1 else
                  "; others at d = " + ", ".join(f"{d0 / x['s_best']:.0f}" for x in pn["islands"][1:4]) + " pc"))
        print(f"  global fit of all {n_nights} nights ({gfit['n_blocks']} blocks out of eclipse) over scale x node angle: "
              f"best a = {gfit['s_best'] * a0:.3f} mas (d = {system.distance_pc / gfit['s_best']:.0f} pc), "
              f"node offset {gfit['om_best']:+.1f} deg; 68 % in the best island a = {gfit['s68_best_island'][0] * a0:.3f}-"
              f"{gfit['s68_best_island'][1] * a0:.3f} mas, 68 % overall {gfit['s68'][0] * a0:.3f}-{gfit['s68'][1] * a0:.3f} mas; "
              f"{gfit['n_islands95']} separate 95 % island(s)")

    out = {"target": system.name, "backend": b.name, "wavelength_nm": lam, "block_minutes": block_minutes,
           "n_blocks": int(mids.size), "n_nights": n_nights, "baseline_m": blen.tolist(),
           "semimajor_mas": float(system.angular_semimajor_mas), "distance_pc": float(system.distance_pc),
           "third_light_fraction": f3, "sigma_vis2_pair": bud["sigma_v2"], "rate_cps": bud["rate_cps"],
           "ab_mag_pair": bud["mag_pair"], "ab_mag_collected": bud["mag_total"],
           "nights": [{"phase_mid": n["phase_mid"], "day": n["day"], "rho_mid_mas": n["rho_mid"], "pa_mid_deg": n["pa_mid"],
                       "eclipse_fraction": float(n["eclipse"].mean()),
                       "vis2_true": n["v2_true"].tolist(),
                       "best_fit_mas": (n["fit"].tolist() if "fit" in n else None),
                       "truth_mas": (n["truth"].tolist() if "fit" in n else None),
                       "region68_area_mas2": n.get("region_area"),
                       "truth_in_region68": n.get("truth_in_region"),
                       "sigma_distance_frac_cumulative": n["sigma_lna"],
                       "sigma_omega_deg_cumulative": n["sigma_omega_deg"]} for n in nights]}
    if gfit is not None:
        a0 = system.angular_semimajor_mas
        out["global_fit"] = {"scale_best": gfit["s_best"], "a_best_mas": gfit["s_best"] * a0,
                             "distance_best_pc": system.distance_pc / gfit["s_best"], "node_offset_best_deg": gfit["om_best"],
                             "a68_mas": [x * a0 for x in gfit["s68"]], "a95_mas": [x * a0 for x in gfit["s95"]],
                             "a68_best_island_mas": [x * a0 for x in gfit["s68_best_island"]],
                             "node68_best_island_deg": list(gfit["om68_best_island"]), "n_islands95": gfit["n_islands95"],
                             "n_blocks": gfit["n_blocks"]}
        out["cumulative_fit"] = [None if pn is None else
                                 {"nights": j + 1, "new_info": pn["new_info"], "distance_best_pc": system.distance_pc / pn["s_best"],
                                  "distance68_pc": [system.distance_pc / pn["s68_best_island"][1], system.distance_pc / pn["s68_best_island"][0]],
                                  "a_best_mas": pn["s_best"] * a0, "n_islands95": pn["n_islands95"],
                                  "other_islands_pc": [system.distance_pc / x["s_best"] for x in pn["islands"][1:]]}
                                 for j, pn in enumerate(gfit["per_night"])]
    if opts.figures:
        path = out_dir / f"{campaign.name}.mp4"
        _render_binary(path, system, b, arr, mids, alt, bvec, blen, nights, bud, lam, nbin_show, f3,
                       fps=int(opt("options.fps", 15)), dpi=int(opt("options.dpi", 110)), gfit=gfit,
                       refs=opt("options.reference_distances", []) or [])
        out["movie"] = str(path)
    return out


def _render_binary(path, system, backend, arr, mids, alt, bvec, blen, nights, bud, lam, nbin_show, f3, fps=15, dpi=110,
                   gfit=None, refs=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Circle

    n = mids.size
    N = len(nights)
    r1 = system.angular_radius_mas(system.primary)
    r2 = system.angular_radius_mas(system.secondary)
    a_mas = system.angular_semimajor_mas
    d0 = system.distance_pc
    node0 = system.node_pa_deg or 0.0
    fig = plt.figure(figsize=(14.5, 5.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.6, 1.25], wspace=0.3, left=0.045, right=0.955, bottom=0.14, top=0.85)
    ax_sky, ax_v, ax_d = (fig.add_subplot(gs[0, k]) for k in range(3))
    fig.suptitle(f"{system.name.split(' (')[0]}: {N} nights on {arr.stations[0].name} + {arr.stations[1].name}, "
                 f"{backend.name}  (a = {a_mas:.3f} mas, P = {system.period_days:.3f} d, d = {d0:.0f} pc assumed)", fontsize=11)

    # sky panel: relative orbit (secondary about the primary), the disks, the baseline direction
    psi = np.linspace(0, 2 * np.pi, 400)
    porb = sky_positions(psi, system)
    ox, oy = np.asarray(porb.x2) - np.asarray(porb.x1), np.asarray(porb.y2) - np.asarray(porb.y1)
    ax_sky.plot(ox, oy, color="0.75", lw=1)
    lim = 1.25 * (a_mas * (1 + system.eccentricity) + r1)
    ax_sky.set_xlim(lim, -lim); ax_sky.set_ylim(-lim, lim); ax_sky.set_aspect("equal")   # East left
    ax_sky.set_xlabel("ΔRA [mas]  (east left)"); ax_sky.set_ylabel("ΔDec [mas]")
    ax_sky.set_title("the pair on the sky; line: baseline direction", fontsize=9)
    d1 = Circle((0, 0), r1, color="tab:blue", alpha=0.85, zorder=3)
    d2 = Circle((0, 0), r2, color="tab:orange", alpha=0.85, zorder=4)
    ax_sky.add_patch(d1); ax_sky.add_patch(d2)
    bline, = ax_sky.plot([], [], color="tab:red", lw=1.2, alpha=0.8, zorder=2)
    sky_txt = ax_sky.text(0.03, 0.97, "", transform=ax_sky.transAxes, va="top", fontsize=9)

    # fringe panel: |V|^2 of the pair along the night vs hour angle
    sig_bin = bud["sigma_v2"] / np.sqrt(nbin_show)
    bin_min = int(round((mids[1] - mids[0]) * 60 * nbin_show)) if n > 1 else 0
    model_line, = ax_v.plot([], [], color="k", lw=1.5, label="model (pair |V|²)")
    ecl_line, = ax_v.plot([], [], color="tab:purple", lw=2.5, alpha=0.5, label="disks overlapping (rendered)")
    pts = ax_v.errorbar([], [], yerr=[], fmt="o", color="tab:blue", ms=5, capsize=2, lw=1)
    cur, = ax_v.plot([], [], "o", color="tab:orange", ms=8, zorder=5)
    vmax = max(1.1 * max(nn["v2_true"].max() for nn in nights), 3 * sig_bin)
    ax_v.set_xlim(mids[0] - 0.1, mids[-1] + 0.1); ax_v.set_ylim(-1.5 * sig_bin, vmax)
    ax_v.set_xlabel("hour angle [h]"); ax_v.set_ylabel(r"$|V|^2$ of the pair")
    ax_v.set_title(f"the night's fringes: B {blen.min():.0f}–{blen.max():.0f} m, {bin_min} min bins, σ = {sig_bin:.3g} per bin",
                   fontsize=9)
    ax_v.legend(fontsize=8, loc="upper right")
    v_txt = ax_v.text(0.03, 0.97, "", transform=ax_v.transAxes, va="top", fontsize=9)

    # distance panel: the global fit after each night
    if gfit is not None:
        s_lo, s_hi = float(gfit["s_grid"][0]), float(gfit["s_grid"][-1])
    else:
        s_lo, s_hi = 0.5, 1.5
    ax_d.set_xlim(0.4, N + 0.6); ax_d.set_ylim(d0 / s_hi, d0 / s_lo)
    ax_d.set_xticks(range(1, N + 1))
    ax_d.set_xlabel("nights observed"); ax_d.set_ylabel("distance [pc] = a (AU, spectroscopic) / a (mas, fringes)")
    ax_d.axhline(d0, color="0.4", ls="--", lw=1, label=f"orbit model: {d0:.0f} pc")
    for i_ref, ref in enumerate(refs):
        col = ["tab:orange", "tab:green", "tab:purple", "tab:brown"][i_ref % 4]
        if "lo_pc" in ref and "hi_pc" in ref:                  # a band with its line; otherwise a thin line only
            ax_d.axhspan(float(ref["lo_pc"]), float(ref["hi_pc"]), color=col, alpha=0.18, lw=0)
            ax_d.axhline(float(ref["pc"]), color=col, lw=1, label=f"{ref['label']}: {float(ref['pc']):.0f} pc")
        else:
            ax_d.axhline(float(ref["pc"]), color=col, lw=0.8, ls=":", label=f"{ref['label']}: {float(ref['pc']):.0f} pc")
    ax_d.plot([], [], "o", color="tab:blue", label="deepest solution, 68 %")
    ax_d.plot([], [], "o", mfc="none", color="tab:blue", alpha=0.5, label="other fringe-alias solutions")
    ax_d.legend(fontsize=8, loc="upper right")
    ax_d.set_title("the distance after each night (fit of all nights so far)", fontsize=9)
    d_txt = ax_d.text(0.97, 0.03, "", transform=ax_d.transAxes, va="bottom", ha="right", fontsize=9)
    pending = ax_d.axvline(1, color="tab:blue", ls=":", lw=1)
    ax2 = ax_d.secondary_yaxis("right", functions=(lambda d: a_mas * d0 / np.maximum(d, 1e-9), lambda a: a_mas * d0 / np.maximum(a, 1e-9)))
    ax2.set_ylabel("a [mas]")

    def place_disks(pos):
        x1, y1, x2, y2 = (float(v) for v in (pos.x1, pos.y1, pos.x2, pos.y2))
        d1.center = (0.0, 0.0); d2.center = (x2 - x1, y2 - y1)
        d2.set_zorder(4 if bool(pos.front2) else 2)

    def night_result(j):
        """Draw the cumulative fit after night j (nothing if that night added no information)."""
        pn = gfit["per_night"][j] if gfit is not None else None
        if pn is None:
            return
        for k, isl in enumerate(pn["islands"]):
            db = d0 / isl["s_best"]
            lo, hi = isl["s68"] or isl["s95"]
            err = [[db - d0 / hi], [d0 / lo - db]]
            if k == 0:
                ax_d.errorbar([j + 1], [db], yerr=err, fmt="o", color="tab:blue", ms=6, capsize=3, lw=1.3, zorder=5)
            else:
                ax_d.errorbar([j + 1], [db], yerr=err, fmt="o", mfc="none", color="tab:blue", ms=5, capsize=2, lw=0.8, alpha=0.5, zorder=4)
        lo, hi = pn["s68_best_island"]
        d_txt.set_text(f"after {j + 1} night(s): d = {d0 / pn['s_best']:.0f} pc (68 %: {d0 / hi:.0f}–{d0 / lo:.0f})\n"
                       f"{pn['n_islands95']} solution(s) survive at 95 %"
                       + ("" if pn["new_info"] else "  — eclipse night, nothing added"))

    def frame(j, k):
        nn = nights[j]
        pos = nn["pos"][k]
        place_disks(pos)
        ang = np.arctan2(bvec[k, 1], bvec[k, 0])
        bline.set_data([-lim * np.cos(ang), lim * np.cos(ang)], [-lim * np.sin(ang), lim * np.sin(ang)])
        fr = lam * 1e-9 / max(float(pos.rho), 1e-3) / MAS
        day_lbl = f" (day {nn['day']:.0f})" if nn["day"] != j else ""
        sky_txt.set_text(f"night {j + 1}{day_lbl}, phase {nn['phases'][k]:.3f}\nH = {mids[k]:+.2f} h, alt {alt[k]:.0f}°\n"
                         f"ρ = {float(pos.rho):.3f} mas, B = {blen[k]:.0f} m\nfringe λ/ρ = {fr:.0f} m"
                         + ("\nECLIPSE" if nn["eclipse"][k] else ""))
        model_line.set_data(mids[:k + 1], nn["v2_true"][:k + 1])
        em = np.where(nn["eclipse"][:k + 1], nn["v2_true"][:k + 1], np.nan)
        ecl_line.set_data(mids[:k + 1], em)
        nb_done = (k + 1) // nbin_show
        xs = [mids[i * nbin_show:(i + 1) * nbin_show].mean() for i in range(nb_done)]
        ys = [nn["v2_meas"][i * nbin_show:(i + 1) * nbin_show].mean() for i in range(nb_done)]
        es = [sig_bin] * nb_done
        rest = (k + 1) % nbin_show
        if rest:
            xs.append(mids[nb_done * nbin_show:k + 1].mean()); ys.append(nn["v2_meas"][nb_done * nbin_show:k + 1].mean())
            es.append(bud["sigma_v2"] / np.sqrt(rest))
        nonlocal pts
        pts.remove()
        pts = ax_v.errorbar(xs, ys, yerr=es, fmt="o", color="tab:blue", ms=5, capsize=2, lw=1)
        cur.set_data([mids[k]], [nn["v2_true"][k]])
        v_txt.set_text(f"night {j + 1} of {N}" + ("  — eclipse: the fringe is gone" if nn["eclipse"].mean() >= 0.5 else ""))
        pending.set_xdata([j + 1, j + 1])
        if k == n - 1:
            night_result(j)

    # the final act: the chi^2 map in the sky panel's place, the fitted orbit in the fringe panel's place
    n_final = 4 * fps if gfit is not None else 0
    if gfit is not None:
        d, S, OMg = gfit["dchi2"], gfit["s_grid"], gfit["om_grid"]
        cand = np.argwhere(d < 2.30)
        rng_draw = np.random.default_rng(1)
        if cand.shape[0] > 50:
            cand = cand[rng_draw.choice(cand.shape[0], 50, replace=False)]
        cand = cand[np.argsort(d[cand[:, 0], cand[:, 1]])[::-1]]       # worst first, the best orbit last

        def orbit_xy(sc, omg):
            pp = sky_positions(psi, replace(system, distance_pc=d0 / sc, node_pa_deg=node0 + omg))
            return np.asarray(pp.x2) - np.asarray(pp.x1), np.asarray(pp.y2) - np.asarray(pp.y1)

        ax_chi = fig.add_axes(ax_sky.get_position()); ax_chi.set_visible(False)
        ax_chi.contourf(OMg, S, d, levels=[0, 2.30, 6.17], colors=["tab:blue", "lightsteelblue"], alpha=0.8)
        ax_chi.plot([0.0], [1.0], "+", color="k", ms=11, mew=1.5, label="truth")
        ax_chi.plot([gfit["om_best"]], [gfit["s_best"]], "x", color="tab:red", ms=8, mew=1.5, label="best fit")
        ax_chi.set_xlabel("node angle offset [°]"); ax_chi.set_ylabel("orbit scale a / a₀ = d₀ / d")
        ax_chi.set_title("χ² of all nights: Δχ² < 2.3 (68 %), < 6.2 (95 %)", fontsize=9)
        ax_chi.legend(fontsize=8, loc="upper right")
        ax_orb = fig.add_axes(ax_v.get_position()); ax_orb.set_visible(False)
        ax_orb.plot(ox, oy, color="0.75", lw=1.5)
        ax_orb.plot([0], [0], "+", color="k", ms=8)
        ax_orb.set_xlim(lim, -lim); ax_orb.set_ylim(-lim, lim); ax_orb.set_aspect("equal")
        ax_orb.set_xlabel("ΔRA [mas]"); ax_orb.set_ylabel("ΔDec [mas]")
        ax_orb.set_title("the orbit fitted to all nights: thin, orbits allowed at 68 %; red, best fit; grey, truth", fontsize=9)
        best_line, = ax_orb.plot([], [], color="tab:red", lw=1.6, zorder=6)
        fam = []

        def final(k):
            if k == 0:
                ax_sky.set_visible(False); ax_v.set_visible(False)
                ax_chi.set_visible(True); ax_orb.set_visible(True)
                pending.set_visible(False)
            m = min(cand.shape[0], int(np.ceil((k + 1) / max(1, (n_final // 2)) * cand.shape[0])))
            while len(fam) < m:
                i, j = cand[len(fam)]
                x, y = orbit_xy(S[i], OMg[j])
                fam.append(ax_orb.plot(x, y, color="tab:blue", lw=0.7, alpha=0.3, zorder=4)[0])
            if k >= n_final // 2:
                x, y = orbit_xy(gfit["s_best"], gfit["om_best"])
                best_line.set_data(x, y)
                lo, hi = gfit["s68_best_island"]
                d_txt.set_text(f"all {N} nights: d = {d0 / gfit['s_best']:.0f} pc ({d0 / hi:.0f}–{d0 / lo:.0f}), truth {d0:.0f}\n"
                               f"a = {gfit['s_best'] * a_mas:.3f} mas ({lo * a_mas:.3f}–{hi * a_mas:.3f}), node {node0 + gfit['om_best']:.0f}°\n"
                               f"{gfit['n_islands95']} solution(s) at 95 %: the fringe aliases")

    writer = FFMpegWriter(fps=fps, bitrate=2400) if shutil.which("ffmpeg") else None
    if writer is None:
        frames_dir = path.with_suffix("")
        frames_dir.mkdir(exist_ok=True)
        f = 0
        for j in range(N):
            for k in range(n):
                frame(j, k); fig.savefig(frames_dir / f"frame_{f:04d}.png", dpi=dpi); f += 1
        for k in range(n_final):
            final(k); fig.savefig(frames_dir / f"frame_{f:04d}.png", dpi=dpi); f += 1
        plt.close(fig)
        return
    with writer.saving(fig, str(path), dpi):
        for j in range(N):
            for k in range(n):
                frame(j, k)
                writer.grab_frame()
            for _ in range(fps // 2):                           # a pause at each night's end
                writer.grab_frame()
        for k in range(n_final):
            final(k)
            writer.grab_frame()
        for _ in range(2 * fps):                                # hold the result
            writer.grab_frame()
    plt.close(fig)
