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
  third_light_fraction   fraction of the collected light from unresolved companions
                         (adds photons, dilutes the fringe by (1 - f)^2; default 0)
  display_bin_blocks     blocks per displayed |V|^2 point (default 3)
  fit_grid_mas, fit_step_mas   half-width and step of the per-night (dx, dy) chi^2 grid
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
    for j in range(n_nights):
        ph_mid = (phase0 + j * step_days / P) % 1.0
        phases = (ph_mid + (mids - mids.mean()) / 24.0 / P) % 1.0
        pos_track = [positions_at(system, float(ph)) for ph in phases]
        ecl = np.array([bool(in_eclipse(system, p)) for p in pos_track])
        v2_true = np.array([float(np.abs(np.asarray(spectral_vis(p, bvec[k:k + 1], [lam], system, grid))[0, 0]) ** 2)
                            for k, p in enumerate(pos_track)])
        v2_meas = v2_true + rng.normal(0.0, bud["sigma_v2"], size=v2_true.size)
        night = dict(phase_mid=ph_mid, phases=phases, pos=pos_track, eclipse=ecl, v2_true=v2_true, v2_meas=v2_meas,
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

    out = {"target": system.name, "backend": b.name, "wavelength_nm": lam, "block_minutes": block_minutes,
           "n_blocks": int(mids.size), "n_nights": n_nights, "baseline_m": blen.tolist(),
           "semimajor_mas": float(system.angular_semimajor_mas), "distance_pc": float(system.distance_pc),
           "third_light_fraction": f3, "sigma_vis2_pair": bud["sigma_v2"], "rate_cps": bud["rate_cps"],
           "ab_mag_pair": bud["mag_pair"], "ab_mag_collected": bud["mag_total"],
           "nights": [{"phase_mid": n["phase_mid"], "rho_mid_mas": n["rho_mid"], "pa_mid_deg": n["pa_mid"],
                       "eclipse_fraction": float(n["eclipse"].mean()),
                       "vis2_true": n["v2_true"].tolist(),
                       "best_fit_mas": (n["fit"].tolist() if "fit" in n else None),
                       "truth_mas": (n["truth"].tolist() if "fit" in n else None),
                       "region68_area_mas2": n.get("region_area"),
                       "truth_in_region68": n.get("truth_in_region"),
                       "sigma_distance_frac_cumulative": n["sigma_lna"],
                       "sigma_omega_deg_cumulative": n["sigma_omega_deg"]} for n in nights]}
    if opts.figures:
        path = out_dir / f"{campaign.name}.mp4"
        _render_binary(path, system, b, arr, mids, alt, bvec, blen, nights, bud, lam, nbin_show, f3,
                       fps=int(opt("options.fps", 15)), dpi=int(opt("options.dpi", 110)))
        out["movie"] = str(path)
    return out


def _render_binary(path, system, backend, arr, mids, alt, bvec, blen, nights, bud, lam, nbin_show, f3, fps=15, dpi=110):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Circle

    n = mids.size
    r1 = system.angular_radius_mas(system.primary)
    r2 = system.angular_radius_mas(system.secondary)
    a_mas = system.angular_semimajor_mas
    fig = plt.figure(figsize=(14.5, 5.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.6, 1.1], wspace=0.3, left=0.045, right=0.985, bottom=0.14, top=0.85)
    ax_sky, ax_v, ax_orb = (fig.add_subplot(gs[0, k]) for k in range(3))
    fig.suptitle(f"{system.name.split(' (')[0]}: {len(nights)} nights on {arr.stations[0].name} + {arr.stations[1].name}, "
                 f"{backend.name}  (a = {a_mas:.3f} mas, P = {system.period_days:.3f} d, d = {system.distance_pc:.0f} pc assumed)",
                 fontsize=11)

    # sky panel: relative orbit (secondary about the primary), the disks, the baseline direction
    psi = np.linspace(0, 2 * np.pi, 400)
    porb = sky_positions(psi, system)
    ax_sky.plot(np.asarray(porb.x2) - np.asarray(porb.x1), np.asarray(porb.y2) - np.asarray(porb.y1), color="0.75", lw=1)
    lim = 1.25 * (a_mas + r1)
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

    # orbit panel: fitted separations on the apparent orbit
    ax_orb.plot(np.asarray(porb.x2) - np.asarray(porb.x1), np.asarray(porb.y2) - np.asarray(porb.y1), color="0.75", lw=1)
    ax_orb.plot([0], [0], "+", color="k", ms=8)
    ax_orb.set_xlim(lim, -lim); ax_orb.set_ylim(-lim, lim); ax_orb.set_aspect("equal")
    ax_orb.set_xlabel("ΔRA [mas]"); ax_orb.set_ylabel("ΔDec [mas]")
    ax_orb.set_title("this night's 68 % region of the separation (stripes: one baseline\n"
                     "direction fixes it only along itself); dashed: orbit at a ± 1σ", fontsize=9)
    orb_txt = ax_orb.text(0.03, 0.03, "", transform=ax_orb.transAxes, va="bottom", fontsize=9)
    orb_true, = ax_orb.plot([], [], "o", color="tab:orange", ms=6, zorder=5)
    ox, oy = np.asarray(porb.x2) - np.asarray(porb.x1), np.asarray(porb.y2) - np.asarray(porb.y1)
    band = [ax_orb.plot([], [], color="tab:blue", lw=1, ls="--")[0] for _ in range(2)]
    regions = []

    def place_disks(pos):
        x1, y1, x2, y2 = (float(v) for v in (pos.x1, pos.y1, pos.x2, pos.y2))
        d1.center = (0.0, 0.0); d2.center = (x2 - x1, y2 - y1)
        d2.set_zorder(4 if bool(pos.front2) else 2)

    def frame(j, k):
        nn = nights[j]
        pos = nn["pos"][k]
        place_disks(pos)
        orb_true.set_data([float(pos.x2 - pos.x1)], [float(pos.y2 - pos.y1)])
        ang = np.arctan2(bvec[k, 1], bvec[k, 0])
        bline.set_data([-lim * np.cos(ang), lim * np.cos(ang)], [-lim * np.sin(ang), lim * np.sin(ang)])
        fr = lam * 1e-9 / max(float(pos.rho), 1e-3) / MAS
        sky_txt.set_text(f"night {j + 1}, phase {nn['phases'][k]:.3f}\nH = {mids[k]:+.2f} h, alt {alt[k]:.0f}°\n"
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
        v_txt.set_text(f"night {j + 1} of {len(nights)}" + ("  — eclipse: the fringe is gone" if nn["eclipse"].mean() >= 0.5 else ""))
        if k == n - 1:
            for r in regions:                                  # only the latest night's region stays
                r.remove()
            regions.clear()
            if "fit" in nn:
                g = nn["grid"]
                regions.append(ax_orb.contourf(g, g, nn["dchi2"].T, levels=[0.0, 2.30], colors=["tab:blue"], alpha=0.4))
            else:
                regions.append(ax_orb.plot([0], [0], "x", color="tab:purple", ms=12, mew=2)[0])
            sd = nn["sigma_lna"]
            for sgn, ln in zip((1 - sd, 1 + sd), band):
                ln.set_data(sgn * ox, sgn * oy)
            orb_txt.set_text(f"after {j + 1} night(s):\nσ(a)/a = σ(d)/d = {100 * sd:.1f} %  "
                             f"({system.distance_pc:.0f} ± {system.distance_pc * sd:.0f} pc)\n"
                             f"σ(Ω) = {nn['sigma_omega_deg']:.1f}°"
                             + (f"\nthird light {100 * f3:.0f} %: fringe × {bud['dilution']:.2f}" if f3 > 0 else ""))

    writer = FFMpegWriter(fps=fps, bitrate=2400) if shutil.which("ffmpeg") else None
    if writer is None:
        frames_dir = path.with_suffix("")
        frames_dir.mkdir(exist_ok=True)
        f = 0
        for j in range(len(nights)):
            for k in range(n):
                frame(j, k); fig.savefig(frames_dir / f"frame_{f:04d}.png", dpi=dpi); f += 1
        plt.close(fig)
        return
    with writer.saving(fig, str(path), dpi):
        for j in range(len(nights)):
            for k in range(n):
                frame(j, k)
                writer.grab_frame()
            for _ in range(fps // 2):                           # a pause at each night's end
                writer.grab_frame()
    plt.close(fig)
