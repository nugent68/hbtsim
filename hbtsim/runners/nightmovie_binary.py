"""Binary mode of the `nightmovie` runner: an eclipsing/spectroscopic pair
followed through its orbit, one night per frame set -- the fringe pattern of
each night on every pair of the array, the fitted separation landing on the
apparent orbit night after night, and the angular semi-major axis (hence the
geometric distance), the node angle and, for an eccentric orbit, the argument
of periastron sharpening as the nights accumulate.

Campaign fields as the single-star mode (target now a binary, one backend --
or several named in the schedule --, instrument.array with two or more
stations, night.block_minutes); options:
  nights                 consecutive nights (default 8; ignored with a schedule)
  schedule               the nights' configurations as in run_rotation: backend names, baselines
                         in metres or dicts {backend, baseline_m, pa_deg, side_m, spacings_m, nights}
                         (a generator array rebuilt per night; every pair of the array is used)
  pupils                 true: the model |V|^2 is averaged over the two pupils (hbtsim.aperture
                         quadrature) -- needed when the fringe period lambda/rho approaches the
                         telescope diameter (Capella on 1.65 m trucks; default false)
  phase0                 orbital phase at the middle of the first night (default 0.0)
  night_step_days        days between the nights' midpoints (default 1.0)
  gap                    optional {after_night, days}: a break in the run (a second orbit a season later)
  third_light_fraction   fraction of the collected light from unresolved companions
                         (adds photons, dilutes the fringe by (1 - f)^2; default 0)
  display_bin_blocks     blocks per displayed |V|^2 point (default 3)
  fit_grid_mas, fit_step_mas   half-width and step of the per-night (dx, dy) chi^2 grid
  global_fit, fit_scale_range, fit_scale_step, fit_node_step_deg, fit_periastron_step_deg
                         chi^2 of the nights so far over the orbit's angular scale (default
                         0.5-1.5 x the assumed a, step 0.01), the node-angle offset (0-180 deg,
                         step 1.5) and, when fit_periastron_step_deg is given and e > 0, the
                         argument of periastron (0-360 deg): the right panel shows the distance
                         after each night with the surviving fringe-alias solutions; the closing
                         act shows the map and the family of orbits allowed at 68 %
  fit_theta_range, fit_theta_step
                         when given, the stars' angular diameters are a free parameter of the
                         global fit and the Fisher matrix: a common factor k on both radii
                         (their ratio and the SED fixed) over [lo, hi] in steps -- the fringe
                         phase pins the orbit's scale, the disk envelope k, so a and the
                         diameters separate (two sides of a triangle give the lever); the fit
                         then reports theta_1, theta_2 beside a and d
  reference_distances    [{label, pc, lo_pc, hi_pc}] drawn on the distance panel (e.g. Hipparcos)
  reference_periastron   {label, omega_deg, sigma_deg[, expected_advance_deg]}: an earlier orbit's
                         periastron; the result carries the apsidal advance omega_fit - omega_ref
                         (|V|^2 is even in the separation, so omega is measured modulo 180 and the
                         advance on the branch nearest expected_advance_deg) and its uncertainty
  render_n_max           largest image grid for the rendered eclipse blocks (default 4096)
  seed, fps, dpi

The model visibility is the analytic two-disk visibility out of eclipse
(pupil-averaged with `pupils`) and the rendered two-disk image
(hbtsim.spectral.spectral_vis) for the blocks within the near-eclipse margin,
whose grid is built only when such a block occurs (Capella's wide orbit never
fits one).  The per-night fit and the Fisher precision use the analytic
visibility, so eclipse nights (disks overlapping) carry no separation point --
they are the nights the fringe disappears.
"""

from __future__ import annotations

import shutil
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from hbtsim.bispectrum import Array, binary_vis_complex_analytic
from hbtsim.geometry import hour_angle_blocks, hour_angle_window
from hbtsim.orbit import positions_at, sky_positions
from hbtsim.params import MAS, GridConfig, in_eclipse
from hbtsim.snr import (Observation, g2_snr, incident_rate, polarization_streams, readout_scale,
                        system_ab_mag)
from hbtsim.spectral import spectral_vis

# Delta chi^2 of the 68 % and 95 % regions for 2 and 3 fitted parameters
_DCHI2 = {2: (2.30, 6.17), 3: (3.53, 7.81), 4: (4.72, 9.49)}


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


def _quad_for(system, d1, d2, lam):
    """The pupil-pair quadrature for the shortest fringe period of the orbit."""
    from hbtsim.aperture import fringe_period_m, pupil_pair_quadrature_for
    rho_max = system.angular_semimajor_mas * (1.0 + system.eccentricity)
    return pupil_pair_quadrature_for(d1, d2, fringe_period_m(rho_max, lam))


def _v2_analytic(system, pos, bvecs, lam, quad=None):
    """|V|^2 of the pair at one epoch for baseline vectors (K, 2), averaged
    over the two pupils when a quadrature is given.  Raises ValueError in
    (or near) eclipse."""
    b = np.atleast_2d(np.asarray(bvecs, dtype=float))
    if quad is None:
        return np.abs(binary_vis_complex_analytic(b, lam, system, pos)) ** 2
    pts = quad.points(b).reshape(-1, 2)
    v2 = np.abs(binary_vis_complex_analytic(pts, lam, system, pos)) ** 2
    return quad.reduce(v2.reshape(b.shape[0], -1))


def _binary_night(system, backend, arr, block_minutes, min_alt, f3, pupils):
    """One night's configuration: every pair of the array along its uv
    track (bvec (n_pairs, n_blocks, 2)), the per-pair photon budget and
    sigma(|V|^2) per block, and the pupil quadratures when requested."""
    spec, det, pol = backend.spectrograph, backend.detector, backend.polarization_mode
    h0, h1 = hour_angle_window(system.dec_deg, arr.site.latitude_deg, min_alt)
    mids = hour_angle_blocks(h0, h1, block_minutes)
    if mids.size == 0:
        raise SystemExit("the target never rises above the altitude limit from this site")
    block_s = (h1 - h0) / mids.size * 3600.0
    pairs = arr.pairs()
    bvec = np.array([[pr[2] for pr in arr.projected(float(H), system.dec_deg).pairs()] for H in mids]).transpose(1, 0, 2)
    blen = np.hypot(bvec[..., 0], bvec[..., 1])
    lat, dec = np.radians(arr.site.latitude_deg), np.radians(system.dec_deg)
    alt = np.degrees(np.arcsin(np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(np.radians(mids * 15.0))))
    lam, w = float(spec.channel_centers_nm[0]), float(spec.channel_widths_nm[0])
    sig, buds, quads, diam = [], [], [], []
    for i, j, _ in pairs:
        t1, t2 = arr.stations[i].telescope, arr.stations[j].telescope
        buds.append(_budget_binary(system, lam, w, block_s, t1, t2, det, pol, spec.throughput, f3))
        sig.append(buds[-1]["sigma_v2"]); diam.append((t1.diameter_m, t2.diameter_m))
        quads.append(_quad_for(system, t1.diameter_m, t2.diameter_m, lam) if pupils else None)
    ground = [float(np.hypot(pr[2][0], pr[2][1])) for pr in pairs]
    return dict(label=backend.name, backend=backend, lam=lam, w=w, mids=mids, block_s=block_s, alt=alt, bvec=bvec, blen=blen,
                sigma=np.array(sig), buds=buds, bud=buds[0], quads=quads, n_pairs=len(pairs), pupils=diam,
                pair_names=[f"{arr.stations[i].name}-{arr.stations[j].name}" for i, j, _ in pairs],
                ground_baselines_m=ground, stations=tuple(st.name for st in arr.stations))


def _fit_night(system, pos_track, ecl, bvecs, lam, v2_meas, sig, half, step, quads):
    """chi^2 grid over the mid-night separation (dx, dy) [mas] of the
    secondary from the primary, with the ephemeris' differential motion
    within the night, over every pair (bvecs (P, K, 2), v2_meas (P, K),
    sig (P,)) and the out-of-eclipse blocks; returns the best point, its
    1-sigma covariance from the chi^2 curvature, and the chi^2 map (for
    the 180-degree twin).  With pupil quadratures the fringe term carries
    the pupils' envelope W(dtheta) (the cross-correlation's transform,
    common to every block) and A, C are their pupil averages."""
    lam_m = lam * 1e-9
    k_mid = len(pos_track) // 2
    dx_t = np.array([float(p.x2 - p.x1) for p in pos_track]) - float(pos_track[k_mid].x2 - pos_track[k_mid].x1)
    dy_t = np.array([float(p.y2 - p.y1) for p in pos_track]) - float(pos_track[k_mid].y2 - pos_track[k_mid].y1)
    g = np.arange(-half, half + step / 2, step)
    gx, gy = np.meshgrid(g, g, indexing="ij")                  # (G, G)
    chi2 = np.zeros_like(gx)
    for p in range(bvecs.shape[0]):
        u = bvecs[p] / lam_m                                   # (K, 2) cycles/rad
        quad = quads[p]
        if quad is None:
            a, c = _pair_terms(system, pos_track[k_mid], bvecs[p], lam)
            env_c, env_s = 1.0, 0.0
        else:
            pts = quad.points(bvecs[p]).reshape(-1, 2)         # (K M, 2)
            am, cm = _pair_terms(system, pos_track[k_mid], pts, lam)
            am, cm = am.reshape(bvecs.shape[1], -1), cm.reshape(bvecs.shape[1], -1)
            a, c = quad.reduce(am), quad.reduce(cm)
            o = quad.offsets_m / lam_m                         # (M, 2)
            ph_o = 2 * np.pi * (o[:, 0, None, None] * gx + o[:, 1, None, None] * gy) * MAS
            # the envelope weighted by the fringe amplitude across the pupil at the mid-night baseline
            wc = quad.weights * cm[k_mid] / c[k_mid]
            env_c = np.tensordot(wc, np.cos(ph_o), axes=([0], [0]))
            env_s = np.tensordot(wc, np.sin(ph_o), axes=([0], [0]))
        ph_t = 2 * np.pi * (u[:, 0] * dx_t + u[:, 1] * dy_t) * MAS   # (K,)
        for k in range(u.shape[0]):
            if ecl[k]:
                continue
            arg = 2 * np.pi * (u[k, 0] * gx + u[k, 1] * gy) * MAS + ph_t[k]
            model = a[k] + c[k] * (np.cos(arg) * env_c - np.sin(arg) * env_s)
            chi2 += ((v2_meas[p, k] - model) / sig[p]) ** 2
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


def _theta_scaled(system, k):
    """The system with both stars' radii scaled by k -- the angular
    diameters free of the orbit's scale -- with log g pinned so the SED
    and limb darkening stay those of the catalog star."""
    if k == 1.0:
        return system
    def sc(star):
        return replace(star, radius_rsun=star.radius_rsun * k, logg=star.log_g)
    return replace(system, primary=sc(system.primary), secondary=sc(system.secondary))


def _fisher_night(system, night, with_omega, with_theta=False):
    """Fisher information of one night on (ln a_mas, Omega[, omega][, ln k]):
    numerical derivatives of the analytic (pupil-averaged) |V|^2 of every
    pair and out-of-eclipse block with the distance (all angular sizes
    scale), the node angle, the argument of periastron and, with
    with_theta, the diameter factor k.  Blocks where a perturbed orbit
    grazes the eclipse margin are skipped."""
    names = ["lna", "Omega"] + (["omega"] if with_omega else []) + (["lnk"] if with_theta else [])
    eps = {"lna": 0.01, "Omega": 0.5, "omega": 0.5, "lnk": 0.01}
    def pert(name, sgn):
        if name == "lna":                                      # ln a_mas = -ln d
            return replace(system, distance_pc=system.distance_pc * (1 - sgn * eps[name]))
        if name == "Omega":
            return replace(system, node_pa_deg=(system.node_pa_deg or 0.0) + sgn * eps[name])
        if name == "lnk":
            return _theta_scaled(system, 1 + sgn * eps[name])
        return replace(system, arg_periastron_deg=system.arg_periastron_deg + sgn * eps[name])
    systems = {nm: (pert(nm, +1), pert(nm, -1)) for nm in names}
    lam, P = night["lam"], night["bvec"].shape[0]
    info = np.zeros((len(names), len(names)))
    for k, ph in enumerate(night["phases"]):
        if night["eclipse"][k]:
            continue
        try:
            grad = np.zeros((len(names), P))
            for n_i, nm in enumerate(names):
                sp, sm = systems[nm]
                pp, pm = positions_at(sp, float(ph)), positions_at(sm, float(ph))
                for p in range(P):
                    b = night["bvec"][p, k:k + 1]
                    grad[n_i, p] = (float(_v2_analytic(sp, pp, b, lam, night["quads"][p])[0])
                                    - float(_v2_analytic(sm, pm, b, lam, night["quads"][p])[0])) / (2 * eps[nm])
        except ValueError:                                     # a perturbed orbit grazes the eclipse margin
            continue
        gw = grad / night["sigma"][None, :]
        info += gw @ gw.T
    return names, info


def _islands(dchi2, axes):
    """The separate 95 % islands of a Delta chi^2 map; axes = [(name, grid,
    wrap period or None)] in the map's axis order (scale, node[, periastron]
    [, diameter factor]), the angle axes wrapping with their periods.  Each
    island carries its deepest point and its 68 % and 95 % extents per axis
    as <name>_best, <name>68, <name>95 (the extents of a wrapped axis follow
    the island across the wrap; absent axes w, k give None), deepest first."""
    from scipy import ndimage
    nd = dchi2.ndim
    t68, t95 = _DCHI2[nd]
    names = [a[0] for a in axes]
    grids = [np.asarray(a[1], dtype=float) for a in axes]
    wraps = [a[2] for a in axes]
    tiled = np.tile(dchi2, tuple(3 if w else 1 for w in wraps))
    ext = [np.concatenate([g - w, g, g + w]) if w else g for g, w in zip(grids, wraps)]
    lab, _ = ndimage.label(tiled < t95)
    centre = tuple(slice(g.size, 2 * g.size) if w else slice(None) for g, w in zip(grids, wraps))
    mid = lab[centre]
    out = []
    for L in set(np.unique(mid)) - {0}:
        m_c = mid == L
        idx = np.unravel_index(int(np.argmin(np.where(m_c, dchi2, np.inf))), dchi2.shape)
        comp = lab == L                                        # the island in the tiled map (crosses the wrap)
        def extent(mask):
            if not mask.any():
                return None
            return [(float(ext[ax][np.any(mask, axis=tuple(o for o in range(nd) if o != ax))].min()),
                     float(ext[ax][np.any(mask, axis=tuple(o for o in range(nd) if o != ax))].max())) for ax in range(nd)]
        e95 = extent(comp)
        e68 = extent(comp & (tiled < t68))
        isl = dict(dchi2_min=float(dchi2[idx]))
        for ax, nm in enumerate(names):
            isl[f"{nm}_best"] = float(grids[ax][idx[ax]])
            isl[f"{nm}95"] = tuple(e95[ax])
            isl[f"{nm}68"] = tuple(e68[ax]) if e68 else None
        for nm in ("w", "k"):
            if nm not in names:
                isl[f"{nm}_best"] = isl[f"{nm}68"] = isl[f"{nm}95"] = None
        out.append(isl)
    return sorted(out, key=lambda d: d["dchi2_min"])


def _global_fit(system, nights, s_grid, om_grid, w_grid=None, k_grid=None):
    """chi^2 of every out-of-eclipse block of every pair over the orbit's
    angular scale s (a_mas = s x the assumed value, all angular sizes
    scaling with it), the node angle offset Omega, when w_grid is given
    the argument of periastron omega and, when k_grid is given, a common
    factor k on the two diameters (so the disks measure s x k and the
    fringe s: the diameters come free of the distance).  Positions are
    computed once per (omega, block) at s = 1 and scaled; the node
    rotation is analytic; A, C once per (s, k, pair).  The chi^2 is
    accumulated night by night, so the result carries the fit after 1,
    2, ... N nights as well as the final map and its islands (the fringe
    aliases)."""
    fit_w = w_grid is not None
    fit_k = k_grid is not None
    W = np.asarray(w_grid if fit_w else [system.arg_periastron_deg], dtype=float)
    Kd = np.asarray(k_grid if fit_k else [1.0], dtype=float)
    om = np.radians(om_grid)
    cos_o, sin_o = np.cos(om)[None, :, None], np.sin(om)[None, :, None]
    shape = (s_grid.size, om_grid.size) + ((W.size,) if fit_w else ()) + ((Kd.size,) if fit_k else ())
    chi2 = np.zeros(shape)
    sys_sk = [[_theta_scaled(replace(system, distance_pc=system.distance_pc / sc), float(kk)) for kk in Kd] for sc in s_grid]
    axes = [("s", s_grid, None), ("om", om_grid, 180.0)] + ([("w", W, 180.0)] if fit_w else []) + ([("k", Kd, None)] if fit_k else [])
    per_night, n_blocks = [], 0
    for j, nn in enumerate(nights):
        ks = [k for k in range(len(nn["phases"])) if not nn["eclipse"][k]]
        new = bool(ks)
        if ks:
            lam_m = nn["lam"] * 1e-9
            phases = np.asarray(nn["phases"])[ks]
            dx0 = np.zeros((W.size, len(ks))); dy0 = np.zeros_like(dx0)
            for iw, wv in enumerate(W):
                sw = replace(system, arg_periastron_deg=float(wv)) if fit_w else system
                pos = sky_positions(2 * np.pi * phases, sw)
                dx0[iw] = np.asarray(pos.x2) - np.asarray(pos.x1); dy0[iw] = np.asarray(pos.y2) - np.asarray(pos.y1)
            # the separation rotated by the node offset: (K, OM, W)
            dxr = dx0.T[:, None, :] * cos_o + dy0.T[:, None, :] * sin_o
            dyr = -dx0.T[:, None, :] * sin_o + dy0.T[:, None, :] * cos_o
            for p in range(nn["bvec"].shape[0]):
                bv = nn["bvec"][p, ks]                         # (K, 2)
                u = bv / lam_m
                ph0 = (u[:, 0, None, None] * dxr + u[:, 1, None, None] * dyr) * MAS    # cycles at s = 1
                quad = nn["quads"][p]
                if quad is not None:
                    o = quad.offsets_m / lam_m                 # (M, 2)
                    ph_o = (o[:, 0, None, None, None] * dxr[None] + o[:, 1, None, None, None] * dyr[None]) * MAS
                    pts = quad.points(bv).reshape(-1, 2)
                data = np.asarray(nn["v2_meas"])[p, ks]
                sig = nn["sigma"][p]
                for i, sc in enumerate(s_grid):
                    if quad is None:
                        fr = np.cos(2 * np.pi * sc * ph0)
                    else:
                        arg = 2 * np.pi * sc * ph0
                        cos_po, sin_po = np.cos(2 * np.pi * sc * ph_o), np.sin(2 * np.pi * sc * ph_o)
                    for ik in range(Kd.size):
                        if quad is None:
                            a, c = _pair_terms(sys_sk[i][ik], None, bv, nn["lam"])
                            model = a[:, None, None] + c[:, None, None] * fr          # (K, OM, W)
                        else:
                            am, cm = _pair_terms(sys_sk[i][ik], None, pts, nn["lam"])
                            am, cm = am.reshape(len(ks), -1), cm.reshape(len(ks), -1)
                            a = quad.reduce(am)
                            wc = quad.weights[None, :] * cm                       # (K, M): amplitude-weighted envelope
                            env_c = np.einsum("km,mkow->kow", wc, cos_po)
                            env_s = np.einsum("km,mkow->kow", wc, sin_po)
                            model = a[:, None, None] + np.cos(arg) * env_c - np.sin(arg) * env_s
                        r = (((data[:, None, None] - model) / sig) ** 2).sum(axis=0)   # (OM, W)
                        if not fit_w:
                            r = r[:, 0]
                        if fit_k:
                            chi2[i, ..., ik] += r
                        else:
                            chi2[i] += r
                n_blocks += len(ks)
        d = chi2 - chi2.min()
        isl = _islands(d, axes)
        per_night.append(dict(new_info=new, s_best=isl[0]["s_best"], om_best=isl[0]["om_best"], w_best=isl[0]["w_best"],
                              k_best=isl[0]["k_best"], k68_best_island=(isl[0]["k68"] or isl[0]["k95"]) if fit_k else None,
                              s68_best_island=isl[0]["s68"] or isl[0]["s95"], islands=isl, n_islands95=len(isl)))
    d = chi2 - chi2.min()
    idx = np.unravel_index(int(np.argmin(chi2)), chi2.shape)
    t68, t95 = _DCHI2[chi2.ndim]
    s_rng = lambda m: (float(s_grid[m.any(axis=tuple(range(1, chi2.ndim)))].min()),
                       float(s_grid[m.any(axis=tuple(range(1, chi2.ndim)))].max()))
    fin = per_night[-1]
    best = fin["islands"][0]
    return dict(chi2=chi2, dchi2=d, s_grid=s_grid, om_grid=om_grid, w_grid=W if fit_w else None, fit_w=fit_w,
                k_grid=Kd if fit_k else None, fit_k=fit_k,
                s_best=float(s_grid[idx[0]]), om_best=float(om_grid[idx[1]]), w_best=float(W[idx[2]]) if fit_w else None,
                k_best=float(Kd[idx[-1]]) if fit_k else None, k68_best_island=fin["k68_best_island"],
                s68=s_rng(d < t68), s95=s_rng(d < t95), s68_best_island=fin["s68_best_island"], n_islands95=fin["n_islands95"],
                om68_best_island=best["om68"] or best["om95"], w68_best_island=(best["w68"] or best["w95"]) if fit_w else None,
                n_blocks=n_blocks, per_night=per_night)


def _advance(w_best, w68, ref):
    """The apsidal advance omega_fit - omega_ref modulo 180 (the branch
    nearest ref["expected_advance_deg"] when given, else in (-90, 90])
    and its uncertainty: the fit's 68 % half-width and the reference's
    sigma in quadrature."""
    adv = (w_best - float(ref["omega_deg"])) % 180.0
    if "expected_advance_deg" in ref:
        exp = float(ref["expected_advance_deg"])
        adv = exp + ((adv - exp + 90.0) % 180.0 - 90.0)
    elif adv > 90.0:
        adv -= 180.0
    s_adv = float(np.hypot(0.5 * (w68[1] - w68[0]), float(ref.get("sigma_deg", 0.0))))
    return float(adv), s_adv


def run_binary(campaign, cat, opts, out_dir: Path) -> dict:
    from hbtsim.runners.schedule import array_for, normalize_schedule
    system = campaign.target
    backends = list(campaign.backends)
    opt = campaign.option
    block_minutes = float(opt("night.block_minutes", 10.0))
    min_alt = float(opt("night.min_alt_deg", 30.0))
    rng = np.random.default_rng(int(opt("options.seed", 7)))
    phase0 = float(opt("options.phase0", 0.0))
    step_days = float(opt("options.night_step_days", 1.0))
    f3 = float(opt("options.third_light_fraction", 0.0))
    pupils = bool(opt("options.pupils", False))
    nbin_show = int(opt("options.display_bin_blocks", 3))
    fit_half = float(opt("options.fit_grid_mas", 1.3 * system.angular_semimajor_mas))
    fit_step = float(opt("options.fit_step_mas", 0.005))
    n_max = int(opt("options.render_n_max", 4096))
    if system.dec_deg is None:
        raise SystemExit(f"{system.name} has no declination")
    sched_opt = opt("options.schedule")
    if sched_opt:
        entries = normalize_schedule(sched_opt, campaign, backends)
    else:
        if len(backends) != 1:
            raise SystemExit("the binary night movie takes one backend (or a schedule naming them)")
        if len(campaign.array.stations) != 2:
            raise SystemExit("the binary night movie needs a two-station array (or a schedule of generator arrays)")
        entries = [{"backend_index": 0, "nights": int(opt("options.nights", 8))}]
    plan = [e for e in entries for _ in range(int(e.get("nights", 1)))]
    n_nights = len(plan)
    keys, cfgs, sched = [], [], []
    for e in plan:
        key = (e["backend_index"], e.get("baseline_m"), e.get("pa_deg"), e.get("side_m"),
               tuple(e["spacings_m"]) if "spacings_m" in e else None)
        if key not in keys:
            b = backends[e["backend_index"]]
            arr = array_for(cat, campaign, e, detector=b.detector) if sched_opt else \
                Array(tuple(replace(s, detector=b.detector) for s in campaign.array.stations), campaign.array.site)
            if len(arr.stations) < 2:
                raise SystemExit("the binary night movie needs an array of at least two stations")
            keys.append(key)
            cfgs.append(_binary_night(system, b, arr, block_minutes, min_alt, f3, pupils))
        sched.append(keys.index(key))
    c0 = cfgs[0]
    lam = c0["lam"]
    P_orb = system.period_days
    with_omega = system.eccentricity > 0.0
    k_rng = opt("options.fit_theta_range")
    with_theta = bool(k_rng)
    k_grid = np.arange(float(k_rng[0]), float(k_rng[1]) + 1e-9, float(opt("options.fit_theta_step", 0.01))) if with_theta else None
    print(f"{system.name}: a = {system.angular_semimajor_mas:.3f} mas, P = {P_orb:.4f} d, e = {system.eccentricity:.3f}"
          + (f" (omega {system.arg_periastron_deg:.1f} deg)" if with_omega else "") + ", "
          f"diameters {2 * system.angular_radius_mas(system.primary):.3f} / "
          f"{2 * system.angular_radius_mas(system.secondary):.3f} mas, d = {system.distance_pc:.0f} pc; "
          f"{len(cfgs)} configuration(s) over {n_nights} nights" + (", pupil-averaged model" if pupils else "")
          + (", diameters free" if with_theta else ""))
    for c in cfgs:
        bud = c["bud"]
        print(f"  {'+'.join(c['stations'])} ({', '.join(c['pair_names'])}), {c['label']}: {c['mids'].size} blocks of {block_minutes:g} min, "
              f"ground B {', '.join(f'{x:.0f}' for x in c['ground_baselines_m'])} m, projected {c['blen'].min():.0f}-{c['blen'].max():.0f} m; "
              f"AB({c['lam']:.0f}) pair {bud['mag_pair']:.2f}, collected {bud['mag_total']:.2f} (third light {f3:.2f}, fringe dilution "
              f"{bud['dilution']:.2f}); rate {bud['rate_cps']:.3g} cps, readout scale {bud['readout_scale']:.2f}; "
              f"sigma(|V|^2 pair) {', '.join(f'{s:.3g}' for s in c['sigma'])} per block")
        print(f"    fringe period lambda/a = {c['lam'] * 1e-9 / (system.angular_semimajor_mas * MAS):.0f} m at quadrature"
              + (f"; pupils {c['pupils'][0][0]:.2f} m" if pupils else ""))

    grid = None
    nights = []
    gap = opt("options.gap")                                   # {"after_night": 8, "days": 323}: a break in the run
    day_of = [j * step_days + (float(gap["days"]) if gap and j >= int(gap["after_night"]) else 0.0) for j in range(n_nights)]
    F_tot = None
    for j in range(n_nights):
        c = cfgs[sched[j]]
        mids, bvec = c["mids"], c["bvec"]
        ph_mid = (phase0 + day_of[j] / P_orb) % 1.0
        phases = (ph_mid + (mids - mids.mean()) / 24.0 / P_orb) % 1.0
        pos_track = [positions_at(system, float(ph)) for ph in phases]
        ecl = np.array([bool(in_eclipse(system, p)) for p in pos_track])
        v2_true = np.zeros(bvec.shape[:2])
        for k, p in enumerate(pos_track):
            if ecl[k]:
                if grid is None:
                    try:
                        grid = GridConfig().for_system(system, n_max=n_max)
                    except ValueError as err:
                        raise SystemExit(f"an eclipse block needs the rendered two-disk image: {err}") from err
                for ip in range(bvec.shape[0]):
                    v2_true[ip, k] = float(np.abs(np.asarray(spectral_vis(p, bvec[ip, k:k + 1], [c["lam"]], system, grid))[0, 0]) ** 2)
            else:
                for ip in range(bvec.shape[0]):
                    v2_true[ip, k] = float(_v2_analytic(system, p, bvec[ip, k:k + 1], c["lam"], c["quads"][ip])[0])
        v2_meas = v2_true + rng.normal(0.0, 1.0, size=v2_true.shape) * c["sigma"][:, None]
        night = dict(ci=sched[j], lam=c["lam"], bvec=bvec, sigma=c["sigma"], quads=c["quads"], phase_mid=ph_mid, day=day_of[j],
                     phases=phases, pos=pos_track, eclipse=ecl, v2_true=v2_true, v2_meas=v2_meas,
                     rho_mid=float(pos_track[len(pos_track) // 2].rho),
                     pa_mid=float(pos_track[len(pos_track) // 2].position_angle_deg))
        if ecl.mean() < 0.5:
            best, cov, chi2, g = _fit_night(system, pos_track, ecl, bvec, c["lam"], v2_meas, c["sigma"], fit_half, fit_step, c["quads"])
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
                  f"68 % region {area:.3f} mas^2 (truth {'inside' if in_region else 'outside'})"
                  + (f"; {100 * ecl.mean():.0f} % of the blocks near eclipse" if ecl.any() else ""))
        else:
            print(f"  night {j + 1}: phase {ph_mid:.3f}, rho {night['rho_mid']:.3f} mas -- eclipse "
                  f"({100 * ecl.mean():.0f} % of the blocks), |V|^2 {v2_true.min():.3f}-{v2_true.max():.3f}, no fringe")
        nights.append(night)
        # the cumulative (a, Omega[, omega]) precision after this night
        names, info = _fisher_night(system, night, with_omega, with_theta)
        F_tot = info if F_tot is None else F_tot + info
        try:
            cov_so = np.linalg.inv(F_tot)
            sigs = [float(np.sqrt(cov_so[i, i])) for i in range(len(names))]
        except np.linalg.LinAlgError:
            sigs = [float("inf")] * len(names)
        night.update(sigma_lna=sigs[0], sigma_omega_deg=sigs[1], sigma_periastron_deg=sigs[2] if with_omega else None,
                     sigma_lnk=sigs[-1] if with_theta else None)
        print(f"    after {j + 1} night(s): sigma(a)/a = sigma(d)/d = {100 * sigs[0]:.2f} %, sigma(Omega) = {sigs[1]:.2f} deg"
              + (f", sigma(omega) = {sigs[2]:.2f} deg" if with_omega else "")
              + (f", sigma(theta)/theta = {100 * sigs[-1]:.2f} %" if with_theta else ""))

    gfit = None
    w_step = opt("options.fit_periastron_step_deg")
    ref_w = opt("options.reference_periastron")
    if any("fit" in nn for nn in nights) and opt("options.global_fit", True):
        s_lo, s_hi = (float(x) for x in opt("options.fit_scale_range", [0.5, 1.5]))
        s_grid = np.arange(s_lo, s_hi + 1e-9, float(opt("options.fit_scale_step", 0.01)))
        om_grid = np.arange(0.0, 180.0, float(opt("options.fit_node_step_deg", 1.5)))
        # |V|^2 is even in the separation, so omega and omega + 180 are indistinguishable: fit omega modulo 180
        w_grid = np.arange(0.0, 180.0, float(w_step)) if (w_step and with_omega) else None
        gfit = _global_fit(system, nights, s_grid, om_grid, w_grid, k_grid)
        a0 = system.angular_semimajor_mas
        d0 = system.distance_pc
        for j, pn in enumerate(gfit["per_night"]):
            if pn is None or not pn["new_info"]:
                continue
            lo, hi = pn["s68_best_island"]
            print(f"    global fit after night {j + 1}: d = {d0 / pn['s_best']:.1f} pc (68 % {d0 / hi:.1f}-{d0 / lo:.1f}), "
                  f"{pn['n_islands95']} island(s)" + ("" if pn["n_islands95"] == 1 else
                  "; others at d = " + ", ".join(f"{d0 / x['s_best']:.0f}" for x in pn["islands"][1:4]) + " pc")
                  + (f"; omega {pn['w_best']:.0f} deg" if gfit["fit_w"] else "")
                  + (f"; theta x {pn['k_best']:.3f}" if gfit["fit_k"] else ""))
        print(f"  global fit of all {n_nights} nights ({gfit['n_blocks']} pair-blocks out of eclipse) over scale x node"
              + (" x periastron" if gfit["fit_w"] else "") + (" x diameters" if gfit["fit_k"] else "") + f": best a = {gfit['s_best'] * a0:.4f} mas (d = {d0 / gfit['s_best']:.1f} pc), "
              f"node offset {gfit['om_best']:+.2f} deg; 68 % in the best island a = {gfit['s68_best_island'][0] * a0:.4f}-"
              f"{gfit['s68_best_island'][1] * a0:.4f} mas, 68 % overall {gfit['s68'][0] * a0:.4f}-{gfit['s68'][1] * a0:.4f} mas; "
              f"{gfit['n_islands95']} separate 95 % island(s)")
        if gfit["fit_k"]:
            k68 = gfit["k68_best_island"]
            th1, th2 = 2 * system.angular_radius_mas(system.primary), 2 * system.angular_radius_mas(system.secondary)
            kk = gfit["k_best"] * gfit["s_best"]                       # the diameters measure s x k
            print(f"  diameters (factor k on the catalog radii, free of the distance): k = {gfit['k_best']:.3f} (68 % {k68[0]:.3f}-{k68[1]:.3f}); "
                  f"theta = {kk * th1:.3f} / {kk * th2:.3f} mas (truth {th1:.3f} / {th2:.3f}), "
                  f"sigma(theta)/theta {100 * 0.5 * (k68[1] - k68[0]) / gfit['k_best']:.1f} %")
        if gfit["fit_w"]:
            w68 = gfit["w68_best_island"]
            print(f"  argument of periastron (modulo 180; the spectroscopic orbit picks the half): {gfit['w_best']:.1f} deg "
                  f"(68 % {w68[0]:.1f}-{w68[1]:.1f}; truth {system.arg_periastron_deg % 180.0:.1f})")
            if ref_w:
                adv, s_adv = _advance(gfit["w_best"], w68, ref_w)
                print(f"  apsidal advance since {ref_w['label']} (omega {float(ref_w['omega_deg']):.0f} deg): {adv:+.1f} +/- {s_adv:.1f} deg"
                      + (f" (branch nearest the expected {float(ref_w['expected_advance_deg']):.0f} deg)" if "expected_advance_deg" in ref_w else " (modulo 180)"))

    out = {"target": system.name, "backend": c0["label"], "wavelength_nm": lam, "block_minutes": block_minutes,
           "n_blocks": int(c0["mids"].size), "n_nights": n_nights, "baseline_m": c0["blen"][0].tolist(),
           "n_pairs": c0["n_pairs"], "pupils": pupils, "schedule": sched,
           "configs": [{"backend": c["label"], "wavelength_nm": c["lam"], "stations": list(c["stations"]), "pair_names": c["pair_names"],
                        "ground_baselines_m": c["ground_baselines_m"], "n_blocks": int(c["mids"].size),
                        "sigma_vis2_pair": c["sigma"].tolist()} for c in cfgs],
           "semimajor_mas": float(system.angular_semimajor_mas), "distance_pc": float(system.distance_pc),
           "eccentricity": float(system.eccentricity), "arg_periastron_deg": float(system.arg_periastron_deg),
           "third_light_fraction": f3, "sigma_vis2_pair": float(c0["sigma"][0]), "rate_cps": c0["bud"]["rate_cps"],
           "ab_mag_pair": c0["bud"]["mag_pair"], "ab_mag_collected": c0["bud"]["mag_total"],
           "nights": [{"config": n["ci"], "phase_mid": n["phase_mid"], "day": n["day"], "rho_mid_mas": n["rho_mid"], "pa_mid_deg": n["pa_mid"],
                       "eclipse_fraction": float(n["eclipse"].mean()),
                       "vis2_true": n["v2_true"][0].tolist(), "vis2_true_pairs": n["v2_true"].tolist(),
                       "best_fit_mas": (n["fit"].tolist() if "fit" in n else None),
                       "truth_mas": (n["truth"].tolist() if "fit" in n else None),
                       "region68_area_mas2": n.get("region_area"),
                       "truth_in_region68": n.get("truth_in_region"),
                       "sigma_distance_frac_cumulative": n["sigma_lna"],
                       "sigma_omega_deg_cumulative": n["sigma_omega_deg"],
                       "sigma_periastron_deg_cumulative": n["sigma_periastron_deg"],
                       "sigma_theta_frac_cumulative": n["sigma_lnk"]} for n in nights]}
    if gfit is not None:
        a0 = system.angular_semimajor_mas
        out["global_fit"] = {"scale_best": gfit["s_best"], "a_best_mas": gfit["s_best"] * a0,
                             "distance_best_pc": system.distance_pc / gfit["s_best"], "node_offset_best_deg": gfit["om_best"],
                             "a68_mas": [x * a0 for x in gfit["s68"]], "a95_mas": [x * a0 for x in gfit["s95"]],
                             "a68_best_island_mas": [x * a0 for x in gfit["s68_best_island"]],
                             "node68_best_island_deg": list(gfit["om68_best_island"]), "n_islands95": gfit["n_islands95"],
                             "n_blocks": gfit["n_blocks"], "fit_periastron": gfit["fit_w"], "fit_diameters": gfit["fit_k"]}
        if gfit["fit_k"]:
            k68 = gfit["k68_best_island"]
            th1, th2 = 2 * system.angular_radius_mas(system.primary), 2 * system.angular_radius_mas(system.secondary)
            kk = gfit["k_best"] * gfit["s_best"]
            out["global_fit"].update(theta_scale_best=gfit["k_best"], theta_scale68=list(k68),
                                     theta_primary_best_mas=kk * th1, theta_secondary_best_mas=kk * th2,
                                     theta_primary68_mas=[k68[0] * gfit["s_best"] * th1, k68[1] * gfit["s_best"] * th1],
                                     theta_primary_truth_mas=th1, theta_secondary_truth_mas=th2,
                                     sigma_theta_frac=0.5 * (k68[1] - k68[0]) / gfit["k_best"])
        if gfit["fit_w"]:
            w68 = gfit["w68_best_island"]
            out["global_fit"].update(omega_best_deg=gfit["w_best"], omega68_deg=list(w68),
                                     omega_truth_deg=float(system.arg_periastron_deg), omega_truth_mod180_deg=float(system.arg_periastron_deg % 180.0))
            if ref_w:
                out["global_fit"].update(omega_advance_deg=adv, sigma_omega_advance_deg=s_adv, reference_periastron=ref_w)
        out["cumulative_fit"] = [None if pn is None else
                                 {"nights": j + 1, "new_info": pn["new_info"], "distance_best_pc": system.distance_pc / pn["s_best"],
                                  "distance68_pc": [system.distance_pc / pn["s68_best_island"][1], system.distance_pc / pn["s68_best_island"][0]],
                                  "a_best_mas": pn["s_best"] * a0, "n_islands95": pn["n_islands95"], "omega_best_deg": pn["w_best"],
                                  "theta_scale_best": pn["k_best"],
                                  "other_islands_pc": [system.distance_pc / x["s_best"] for x in pn["islands"][1:]]}
                                 for j, pn in enumerate(gfit["per_night"])]
    if opts.figures:
        path = out_dir / f"{campaign.name}.mp4"
        _render_binary(path, system, cfgs, nights, nbin_show, f3, fps=int(opt("options.fps", 15)), dpi=int(opt("options.dpi", 110)),
                       gfit=gfit, refs=opt("options.reference_distances", []) or [], ref_w=ref_w)
        out["movie"] = str(path)
    return out


def _render_binary(path, system, cfgs, nights, nbin_show, f3, fps=15, dpi=110, gfit=None, refs=(), ref_w=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Circle

    N = len(nights)
    c0 = cfgs[0]
    lam = c0["lam"]
    r1 = system.angular_radius_mas(system.primary)
    r2 = system.angular_radius_mas(system.secondary)
    a_mas = system.angular_semimajor_mas
    d0 = system.distance_pc
    node0 = system.node_pa_deg or 0.0
    w0 = system.arg_periastron_deg
    pair_colors = ["tab:blue", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
    P_max = max(c["n_pairs"] for c in cfgs)
    labels = sorted({c["label"] for c in cfgs})
    fig = plt.figure(figsize=(14.5, 5.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.6, 1.25], wspace=0.3, left=0.045, right=0.955, bottom=0.14, top=0.85)
    ax_sky, ax_v, ax_d = (fig.add_subplot(gs[0, k]) for k in range(3))
    fig.suptitle(f"{system.name.split(' (')[0]}: {N} nights on {'+'.join(c0['stations'])}, {' / '.join(labels)}  "
                 f"(a = {a_mas:.3f} mas, P = {system.period_days:.3f} d, d = {d0:.0f} pc assumed)", fontsize=11)

    # sky panel: relative orbit (secondary about the primary), the disks, the baseline directions
    psi = np.linspace(0, 2 * np.pi, 400)
    porb = sky_positions(psi, system)
    ox, oy = np.asarray(porb.x2) - np.asarray(porb.x1), np.asarray(porb.y2) - np.asarray(porb.y1)
    ax_sky.plot(ox, oy, color="0.75", lw=1)
    lim = 1.25 * (a_mas * (1 + system.eccentricity) + r1)
    ax_sky.set_xlim(lim, -lim); ax_sky.set_ylim(-lim, lim); ax_sky.set_aspect("equal")   # East left
    ax_sky.set_xlabel("ΔRA [mas]  (east left)"); ax_sky.set_ylabel("ΔDec [mas]")
    ax_sky.set_title("the pair on the sky; lines: baseline directions", fontsize=9)
    d1 = Circle((0, 0), r1, color="tab:blue", alpha=0.85, zorder=3)
    d2 = Circle((0, 0), r2, color="tab:orange", alpha=0.85, zorder=4)
    ax_sky.add_patch(d1); ax_sky.add_patch(d2)
    blines = [ax_sky.plot([], [], color=pair_colors[p % len(pair_colors)], lw=1.2, alpha=0.7, zorder=2)[0] for p in range(P_max)]
    sky_txt = ax_sky.text(0.03, 0.97, "", transform=ax_sky.transAxes, va="top", fontsize=9)

    # fringe panel: |V|^2 of the pair along the night vs hour angle, one colour per telescope pair
    sig_min = min(c["sigma"].min() for c in cfgs) / np.sqrt(nbin_show)
    model_lines = [ax_v.plot([], [], color=pair_colors[p % len(pair_colors)], lw=1.5)[0] for p in range(P_max)]
    ecl_lines = [ax_v.plot([], [], color="tab:purple", lw=2.5, alpha=0.5)[0] for p in range(P_max)]
    ax_v.plot([], [], color="k", lw=1.5, label="model (pair |V|²), one colour per telescope pair")
    ax_v.plot([], [], color="tab:purple", lw=2.5, alpha=0.5, label="disks overlapping (rendered)")
    pts = [ax_v.errorbar([], [], yerr=[], fmt="o", color=pair_colors[p % len(pair_colors)], ms=5, capsize=2, lw=1) for p in range(P_max)]
    curs = [ax_v.plot([], [], "o", color="tab:orange", ms=8, zorder=5)[0] for p in range(P_max)]
    h_lo = min(c["mids"][0] for c in cfgs); h_hi = max(c["mids"][-1] for c in cfgs)
    vmax = max(1.1 * max(nn["v2_true"].max() for nn in nights), 3 * sig_min)
    ax_v.set_xlim(h_lo - 0.1, h_hi + 0.1); ax_v.set_ylim(-1.5 * sig_min, vmax)
    ax_v.set_xlabel("hour angle [h]"); ax_v.set_ylabel(r"$|V|^2$ of the pair")
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
    ax_d.axhline(d0, color="0.4", ls="--", lw=1, label=f"orbit model: {d0:.1f} pc")
    for i_ref, ref in enumerate(refs):
        col = ["tab:orange", "tab:green", "tab:purple", "tab:brown"][i_ref % 4]
        if "lo_pc" in ref and "hi_pc" in ref:                  # a band with its line; otherwise a thin line only
            ax_d.axhspan(float(ref["lo_pc"]), float(ref["hi_pc"]), color=col, alpha=0.18, lw=0)
            ax_d.axhline(float(ref["pc"]), color=col, lw=1, label=f"{ref['label']}: {float(ref['pc']):.1f} pc")
        else:
            ax_d.axhline(float(ref["pc"]), color=col, lw=0.8, ls=":", label=f"{ref['label']}: {float(ref['pc']):.2f} pc")
    ax_d.plot([], [], "o", color="tab:blue", label="deepest solution, 68 %")
    ax_d.plot([], [], "o", mfc="none", color="tab:blue", alpha=0.5, label="other fringe-alias solutions")
    ax_d.legend(fontsize=8, loc="upper right")
    ax_d.set_title("the distance after each night (fit of all nights so far)", fontsize=9)
    d_txt = ax_d.text(0.97, 0.03, "", transform=ax_d.transAxes, va="bottom", ha="right", fontsize=8.5)
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
            err = [[max(db - d0 / hi, 0.0)], [max(d0 / lo - db, 0.0)]]
            if k == 0:
                ax_d.errorbar([j + 1], [db], yerr=err, fmt="o", color="tab:blue", ms=6, capsize=3, lw=1.3, zorder=5)
            else:
                ax_d.errorbar([j + 1], [db], yerr=err, fmt="o", mfc="none", color="tab:blue", ms=5, capsize=2, lw=0.8, alpha=0.5, zorder=4)
        lo, hi = pn["s68_best_island"]
        d_txt.set_text(f"after {j + 1} night(s): d = {d0 / pn['s_best']:.1f} pc (68 %: {d0 / hi:.1f}–{d0 / lo:.1f})\n"
                       f"{pn['n_islands95']} solution(s) survive at 95 %"
                       + (f"\nω = {pn['w_best']:.0f}°" if gfit["fit_w"] else "")
                       + (f"\nθ = {pn['k_best'] * pn['s_best'] * 2 * r1:.3f} / {pn['k_best'] * pn['s_best'] * 2 * r2:.3f} mas" if gfit["fit_k"] else "")
                       + ("" if pn["new_info"] else "  — eclipse night, nothing added"))

    def frame(j, k):
        nn = nights[j]
        c = cfgs[nn["ci"]]
        mids, bvec, blen, n = c["mids"], c["bvec"], c["blen"], c["mids"].size
        pos = nn["pos"][k]
        place_disks(pos)
        for p, bl in enumerate(blines):
            if p < c["n_pairs"]:
                ang = np.arctan2(bvec[p, k, 1], bvec[p, k, 0])
                bl.set_data([-lim * np.cos(ang), lim * np.cos(ang)], [-lim * np.sin(ang), lim * np.sin(ang)])
            else:
                bl.set_data([], [])
        fr = c["lam"] * 1e-9 / max(float(pos.rho), 1e-3) / MAS
        day_lbl = f" (day {nn['day']:.0f})" if nn["day"] != j else ""
        sky_txt.set_text(f"night {j + 1}{day_lbl}, phase {nn['phases'][k]:.3f}\nH = {mids[k]:+.2f} h, alt {c['alt'][k]:.0f}°\n"
                         f"ρ = {float(pos.rho):.3f} mas, B = {', '.join(f'{x:.0f}' for x in blen[:, k])} m\nfringe λ/ρ = {fr:.1f} m"
                         + ("\nECLIPSE" if nn["eclipse"][k] else ""))
        ax_v.set_title(f"the night's fringes: {', '.join(c['pair_names'])}, B {blen.min():.0f}–{blen.max():.0f} m, "
                       f"{int(round((mids[1] - mids[0]) * 60 * nbin_show)) if n > 1 else 0} min bins", fontsize=9)
        for p in range(P_max):
            if p >= c["n_pairs"]:
                model_lines[p].set_data([], []); ecl_lines[p].set_data([], []); curs[p].set_data([], [])
                pts[p].remove(); pts[p] = ax_v.errorbar([], [], yerr=[], fmt="o", color=pair_colors[p % len(pair_colors)], ms=5)
                continue
            model_lines[p].set_data(mids[:k + 1], nn["v2_true"][p, :k + 1])
            em = np.where(nn["eclipse"][:k + 1], nn["v2_true"][p, :k + 1], np.nan)
            ecl_lines[p].set_data(mids[:k + 1], em)
            nb_done = (k + 1) // nbin_show
            xs = [mids[i * nbin_show:(i + 1) * nbin_show].mean() for i in range(nb_done)]
            ys = [nn["v2_meas"][p, i * nbin_show:(i + 1) * nbin_show].mean() for i in range(nb_done)]
            es = [c["sigma"][p] / np.sqrt(nbin_show)] * nb_done
            rest = (k + 1) % nbin_show
            if rest:
                xs.append(mids[nb_done * nbin_show:k + 1].mean()); ys.append(nn["v2_meas"][p, nb_done * nbin_show:k + 1].mean())
                es.append(c["sigma"][p] / np.sqrt(rest))
            pts[p].remove()
            pts[p] = ax_v.errorbar(xs, ys, yerr=es, fmt="o", color=pair_colors[p % len(pair_colors)], ms=5, capsize=2, lw=1)
            curs[p].set_data([mids[k]], [nn["v2_true"][p, k]])
        gb = sorted(set(round(float(x)) for x in c["ground_baselines_m"]))
        cfg_lbl = c["label"][:24] if len(labels) > 1 else (f"B {gb[0]} m" if len(gb) == 1 else f"B {gb[0]}–{gb[-1]} m")
        v_txt.set_text(f"night {j + 1} of {N}" + (f": {cfg_lbl}" if len(cfgs) > 1 else "")
                       + ("  — eclipse: the fringe is gone" if nn["eclipse"].mean() >= 0.5 else ""))
        pending.set_xdata([j + 1, j + 1])
        if k == n - 1:
            night_result(j)

    # the final act: the chi^2 map in the sky panel's place, the fitted orbit in the fringe panel's place
    n_final = 4 * fps if gfit is not None else 0
    if gfit is not None:
        d, S, OMg = gfit["dchi2"], gfit["s_grid"], gfit["om_grid"]
        t68, t95 = _DCHI2[d.ndim]
        d_map = d.reshape(d.shape[0], d.shape[1], -1).min(axis=2)   # the profile over the periastron and the diameters
        cand = np.argwhere(d < t68)
        rng_draw = np.random.default_rng(1)
        if cand.shape[0] > 50:
            cand = cand[rng_draw.choice(cand.shape[0], 50, replace=False)]
        cand = cand[np.argsort(d[tuple(cand.T)])[::-1]]        # worst first, the best orbit last

        def orbit_xy(sc, omg, wv):
            sysf = replace(system, distance_pc=d0 / sc, node_pa_deg=node0 + omg)
            if wv is not None:
                sysf = replace(sysf, arg_periastron_deg=float(wv))
            pp = sky_positions(psi, sysf)
            return np.asarray(pp.x2) - np.asarray(pp.x1), np.asarray(pp.y2) - np.asarray(pp.y1)

        ax_chi = fig.add_axes(ax_sky.get_position()); ax_chi.set_visible(False)
        # the node axis centred on the best fit (it wraps at 180), the view zoomed to the best island
        om_c = ((OMg - gfit["om_best"] + 90.0) % 180.0) - 90.0 + gfit["om_best"]
        o = np.argsort(om_c)
        ax_chi.contourf(om_c[o], S, d_map[:, o], levels=[0, t68, t95], colors=["tab:blue", "lightsteelblue"], alpha=0.8)
        truth_om = ((0.0 - gfit["om_best"] + 90.0) % 180.0) - 90.0 + gfit["om_best"]
        ax_chi.plot([truth_om], [1.0], "+", color="k", ms=11, mew=1.5, label="truth")
        ax_chi.plot([gfit["om_best"]], [gfit["s_best"]], "x", color="tab:red", ms=8, mew=1.5, label="best fit")
        isl0 = gfit["per_night"][-1]["islands"][0]
        h_om = max(3.0 * 0.5 * (isl0["om95"][1] - isl0["om95"][0]), 3.0 * abs(truth_om - gfit["om_best"]), 2.0)
        h_s = max(3.0 * 0.5 * (isl0["s95"][1] - isl0["s95"][0]), 3.0 * abs(1.0 - gfit["s_best"]), 0.01)
        ax_chi.set_xlim(gfit["om_best"] - min(h_om, 90.0), gfit["om_best"] + min(h_om, 90.0))
        ax_chi.set_ylim(max(S[0], gfit["s_best"] - h_s), min(S[-1], gfit["s_best"] + h_s))
        ax_chi.set_xlabel("node angle offset [°]"); ax_chi.set_ylabel("orbit scale a / a₀ = d₀ / d")
        prof = [nm for nm, f in (("ω", gfit["fit_w"]), ("θ", gfit["fit_k"])) if f]
        ax_chi.set_title(f"χ² of all nights: Δχ² < {t68:.1f} (68 %), < {t95:.1f} (95 %)" + (f", profiled over {', '.join(prof)}" if prof else ""), fontsize=9)
        ax_chi.legend(fontsize=8, loc="upper right")
        ax_orb = fig.add_axes(ax_v.get_position()); ax_orb.set_visible(False)
        ax_orb.plot(ox, oy, color="0.75", lw=1.5)
        ax_orb.plot([0], [0], "+", color="k", ms=8)
        ax_orb.set_xlim(lim, -lim); ax_orb.set_ylim(-lim, lim); ax_orb.set_aspect("equal")
        ax_orb.set_xlabel("ΔRA [mas]"); ax_orb.set_ylabel("ΔDec [mas]")
        ax_orb.set_title("the orbit fitted to all nights: thin, orbits allowed at 68 %; red, best fit; grey, truth", fontsize=9)
        best_line, = ax_orb.plot([], [], color="tab:red", lw=1.6, zorder=6)
        if gfit["fit_w"]:
            w68 = gfit["w68_best_island"]
            w_txt = (f"ω = {gfit['w_best']:.0f}° mod 180 (68 %: {w68[0]:.0f}–{w68[1]:.0f}), truth {w0 % 180.0:.0f}°")
            if ref_w:
                adv, s_adv = _advance(gfit["w_best"], w68, ref_w)
                w_txt += f"\nadvance since {ref_w['label'].split(' (')[0].split(':')[0]} (ω {float(ref_w['omega_deg']):.0f}°): {adv:+.0f} ± {s_adv:.0f}°"
        else:
            w_txt = ""
        if gfit["fit_k"]:
            k68 = gfit["k68_best_island"]
            kk = gfit["k_best"] * gfit["s_best"]
            w_txt += ("\n" if w_txt else "") + (f"θ = {kk * 2 * r1:.3f} / {kk * 2 * r2:.3f} mas (68 % ±{100 * 0.5 * (k68[1] - k68[0]) / gfit['k_best']:.1f} %), "
                                                 f"truth {2 * r1:.3f} / {2 * r2:.3f}")
        fam = []

        def final(k):
            if k == 0:
                ax_sky.set_visible(False); ax_v.set_visible(False)
                ax_chi.set_visible(True); ax_orb.set_visible(True)
                pending.set_visible(False)
            m = min(cand.shape[0], int(np.ceil((k + 1) / max(1, (n_final // 2)) * cand.shape[0])))
            while len(fam) < m:
                idx = cand[len(fam)]
                x, y = orbit_xy(S[idx[0]], OMg[idx[1]], gfit["w_grid"][idx[2]] if gfit["fit_w"] else None)
                fam.append(ax_orb.plot(x, y, color="tab:blue", lw=0.7, alpha=0.3, zorder=4)[0])
            if k >= n_final // 2:
                x, y = orbit_xy(gfit["s_best"], gfit["om_best"], gfit["w_best"])
                best_line.set_data(x, y)
                lo, hi = gfit["s68_best_island"]
                d_txt.set_text(f"all {N} nights: d = {d0 / gfit['s_best']:.1f} pc ({d0 / hi:.1f}–{d0 / lo:.1f}), truth {d0:.1f}\n"
                               f"a = {gfit['s_best'] * a_mas:.3f} mas ({lo * a_mas:.3f}–{hi * a_mas:.3f}), node {node0 + gfit['om_best']:.1f}°\n"
                               f"{gfit['n_islands95']} solution(s) at 95 %: the fringe aliases" + (f"\n{w_txt}" if w_txt else ""))

    writer = FFMpegWriter(fps=fps, bitrate=2400) if shutil.which("ffmpeg") else None
    if writer is None:
        frames_dir = path.with_suffix("")
        frames_dir.mkdir(exist_ok=True)
        f = 0
        for j in range(N):
            for k in range(cfgs[nights[j]["ci"]]["mids"].size):
                frame(j, k); fig.savefig(frames_dir / f"frame_{f:04d}.png", dpi=dpi); f += 1
        for k in range(n_final):
            final(k); fig.savefig(frames_dir / f"frame_{f:04d}.png", dpi=dpi); f += 1
        plt.close(fig)
        return
    with writer.saving(fig, str(path), dpi):
        for j in range(N):
            for k in range(cfgs[nights[j]["ci"]]["mids"].size):
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
