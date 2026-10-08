"""Runner `g2`: two-telescope g2 sensitivity of the campaign's binaries on
one instrument (the former scripts/feasibility_g3.py --g2 and the markdown
table scripts/g2_logs_to_md.py built from its logs).

For every target: the baseline along the separation axis (scanned over
options.baseline_scan_m {start, stop, step}) that maximizes the total SNR
of the first backend at the snapshot phase (options.phase:
"max_separation" or a number), then SNR per sqrt(hour) of every backend
at that baseline, the total rate per telescope, the readout state and
the pupil smearing D/P.  options.section3_numbers (default false) also
prints the two-telescope numbers of the paper's Section 3 (beta Aur and
Algol on C2PU and Keck with the 320-channel SPAD Lambda).  table.md has
one row per (target, backend).

Array mode (instrument.array instead of instrument.telescope, e.g. the
LPQI-Pathfinder NOT + TNG pair): the baselines are the array's station
pairs projected along one night's uv track (night.block_minutes blocks
above night.min_alt_deg, iact.pair_track); per block and channel the
pupil-averaged |V|^2 and the photon-budget sigma(|V|^2) of the unequal
pair (snr.g2_snr with telescope1/telescope2) add in quadrature to one
night's SNR per pair, then over pairs.  options.one_backend_per_night
(true: a filter wheel, one filter per night) makes the "filter set" total
the SUM of the nights of the backends sharing a detector, never a
quadrature sum.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from dataclasses import replace

from hbtsim.aperture import fringe_smearing_factor
from hbtsim.bispectrum import Array
from hbtsim.orbit import max_separation_phase, positions_at
from hbtsim.params import MAS
from hbtsim.snr import (Observation, g2_snr, incident_rate, polarization_streams, readout_scale,
                        spectral_g2_snr, system_ab_mag)

from . import write_tables


def _tabled(system) -> str:
    return {(True, True): "NewEra", (True, False): "NewEra(A)",
            (False, True): "NewEra(B)", (False, False): "blackbody"}[
        (system.primary.flux_table is not None, system.secondary.flux_table is not None)]


def g2_numbers(beta_aur, algol, c2pu, keck, spad_lambda, spad_lambda_ng, spec_320):
    """The two-telescope numbers of the paper's Section 3."""
    print("\n=== Two-telescope g2 (Section 3) ===")
    pos = positions_at(beta_aur, 0.0)
    for lam in (400.0, 800.0):
        obs = Observation(wavelength_nm=lam, filter_width_nm=10.0, t_int_s=3600.0)
        from hbtsim.hbt import binary_vis2_analytic
        for b in (15.0, 50.0):
            v2 = float(binary_vis2_analytic(b, lam, beta_aur, float(pos.rho))[0])
            r = g2_snr(v2, system_ab_mag(beta_aur, lam), obs, telescope1=c2pu, detector1=spad_lambda)
            print(f"  Beta Aur, C2PU, 10 nm filter at {lam:.0f} nm, B = {b:.0f} m: "
                  f"|V|^2 = {v2:.3f}, SNR2/h = {r.snr:.2f}")
    for det, lab in ((spad_lambda, "time-tag"), (spad_lambda_ng, "correlator")):
        r = spectral_g2_snr(beta_aur, 50.0, spectrograph=spec_320, telescope1=c2pu,
                            detector1=det, vis2_method="analytic")
        print(f"  Beta Aur, C2PU, 320 ch, B = 50 m, {lab}: SNR2/h = {r.snr_total:.2f} "
              f"(rate {r.total_rate_cps[0]:.2e} cps/tel, limited={r.readout_limited}, "
              f"scale {r.readout_scale:.2f})")
    for sysm in (algol, beta_aur):
        for det, lab in ((spad_lambda, "time-tag"), (spad_lambda_ng, "correlator")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = spectral_g2_snr(sysm, 85.0, spectrograph=spec_320, telescope1=keck,
                                    detector1=det, vis2_method="analytic")
            sig = 1.0 / r.snr * r.vis2
            sig_unit = 1.0 / (r.snr / np.maximum(r.vis2, 1e-12))
            print(f"  {sysm.name.split()[0]}, Keck 85 m, 320 ch, {lab}: SNR2/h = "
                  f"{r.snr_total:.1f}; sigma(|V|^2) per channel-hour "
                  f"{np.median(sig_unit):.3g} (median), {sig_unit.min():.3g}-{sig_unit.max():.3g}; "
                  f"rate {r.total_rate_cps[0]:.2e} cps/tel, limited={r.readout_limited}, "
                  f"dead-time load {r.dead_time_load_max:.2f}")


def g2_table(instrument: str, tel, backends, systems, b_scan, phase_spec="max_separation",
             vis2_method="analytic", latex=False):
    """Two-telescope g2 sensitivity of every system on one instrument: the
    baseline (along the separation, scanned) that maximizes the total SNR
    of the instrument's first backend at quadrature, then SNR/h for each
    backend at that baseline, the total rate per telescope and the pupil
    smearing D/P.  Returns (rows, best baselines by target)."""
    b_scan = np.asarray(b_scan, float)
    print(f"\n=== Two-telescope g2: {instrument} (2 x {tel.diameter_m:g} m, "
          f"{tel.area_m2:.1f} m^2 each, throughput {tel.throughput:.2f}) ===")
    rows, best = [], {}
    for system in systems:
        phase = (max_separation_phase(system) if phase_spec in ("max_separation", None)
                 else float(phase_spec))
        pos = positions_at(system, phase)
        label0, spec0, det0, pol0 = backends[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tot = [spectral_g2_snr(system, b, spectrograph=spec0, telescope1=tel, detector1=det0,
                                   polarization_mode=pol0, orbital_phase=phase,
                                   vis2_method=vis2_method).snr_total for b in b_scan]
        b_best = float(b_scan[int(np.argmax(tot))])
        d_over_p = tel.diameter_m * float(pos.rho) * MAS / (spec0.lambda_min_nm * 1e-9)
        tabled = _tabled(system)
        short = system.name.split()[0]
        best[short] = b_best
        print(f"\n  {system.name} [{tabled}]: rho = {float(pos.rho):.2f} mas at phase {phase:.3f}; "
              f"best baseline {b_best:.0f} m (first backend), D/P = {d_over_p:.2f} at "
              f"{spec0.lambda_min_nm:.0f} nm, fringe contrast retained "
              f"{fringe_smearing_factor(tel.diameter_m, tel.diameter_m, float(pos.rho) * MAS, spec0.lambda_min_nm * 1e-9):.2f}")
        for label, spec, det, pol in backends:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = spectral_g2_snr(system, b_best, spectrograph=spec, telescope1=tel,
                                    detector1=det, polarization_mode=pol, orbital_phase=phase,
                                    vis2_method=vis2_method)
            print(f"    {label:34s} SNR2/h = {r.snr_total:8.1f}; rate {r.total_rate_cps[0]:.2e} cps/tel"
                  f"{' READOUT-LIMITED x%.2f' % r.readout_scale if r.readout_limited else ''}, "
                  f"dead-time load {r.dead_time_load_max:.2f}, |V|^2 median {np.median(r.vis2):.3f}")
            rows.append({"target": short, "instrument": instrument, "backend": label,
                         "baseline_m": b_best, "snr_total": float(r.snr_total),
                         "rate_cps": float(r.total_rate_cps[0]),
                         "readout_limited": bool(r.readout_limited),
                         "readout_scale": float(r.readout_scale),
                         "dead_time_load": float(r.dead_time_load_max),
                         "vis2_median": float(np.median(r.vis2)), "tabled": tabled,
                         "phase": float(phase), "rho_mas": float(pos.rho)})
    if latex:
        print("\n  LaTeX rows (target & instrument & backend & B & SNR2/sqrt(h)):")
        for r in rows:
            print(f"  {r['target']} & {r['instrument']} & {r['backend']} & {r['baseline_m']:.0f} m & "
                  f"{r['snr_total']:.1f} \\\\")
    return rows, best


def _is_single(target) -> bool:
    return hasattr(target, "theta_ld_mas")


def _tabled_any(target) -> str:
    if _is_single(target):
        return "NewEra" if target.star.flux_table is not None else "blackbody"
    return _tabled(target)


def _single_vis2_fn(target, pupils):
    """pair_track vis2_fn for a SingleStar: the pupil-averaged limb-darkened
    disk (hbtsim.single.single_star_vis2) at the projected baseline length."""
    from hbtsim.single import single_star_vis2

    def fn(bvecs, lambda_nm, phase, diameters):
        b = np.asarray(bvecs, dtype=float)
        lengths = np.hypot(b[:, 0], b[:, 1])
        return single_star_vis2(target, lengths, float(lambda_nm), diameters if pupils else None)[0]
    return fn


def g2_array_track(array, backends, systems, *, block_minutes=30.0, min_alt_deg=30.0,
                   phase_spec="max_separation", one_backend_per_night=True, pupils=True,
                   n_sigma=3.0, diameter_precision=0.05):
    """One night of every (target, backend) on the array's station pairs.
    Targets may be binaries (BinarySystem: fringes of the two disks along
    the track) or single stars (SingleStar: the limb-darkened disk; the row
    then also carries nights_diameter, the nights to a relative diameter
    precision `diameter_precision` from the Fisher information of |V|^2 on
    theta along the track).  Returns (rows, filter_set_nights)."""
    from hbtsim.iact import binary_vis2_fn, pair_track
    from hbtsim.single import prepare_single
    if array.site is None:
        raise SystemExit("runner g2 array mode needs an array with a site")
    names = "+".join(s.name for s in array.stations)
    print(f"\n=== Two-telescope g2 along the night's uv track: {names} at {array.site.name} ===")
    for s in array.stations:
        print(f"  {s.name}: {s.telescope.diameter_m:g} m, {s.telescope.area_m2:.1f} m^2, "
              f"throughput {s.telescope.throughput:.2f}")
    rows, set_nights = [], {}
    for system in systems:
        if system.dec_deg is None:
            raise SystemExit(f"{system.name} has no declination")
        single = _is_single(system)
        # binaries keep the first word (the docs' "Beta", "Algol", "Spica"); single stars
        # need the constellation too ("gamma Peg" vs "gamma Cas")
        tabled = _tabled_any(system)
        short = system.name.split(" (")[0].strip() if single else system.name.split()[0]
        if single:
            phase, period, rho = 0.0, None, float(system.theta_ld_mas)
            print(f"\n  {system.name} [{tabled}]: theta_LD = {rho:.3f} mas (drawn {system.drawn_diameter_mas:.3f}), "
                  f"V = {system.v_mag:.2f}, dec {system.dec_deg:+.1f}")
        else:
            phase = (max_separation_phase(system) if phase_spec in ("max_separation", None)
                     else float(phase_spec))
            pos = positions_at(system, phase)
            period, rho = system.period_days, float(pos.rho)
            print(f"\n  {system.name} [{tabled}]: rho = {rho:.2f} mas at phase {phase:.3f}, "
                  f"dec {system.dec_deg:+.1f}")
        per_det, per_det_fisher = {}, {}
        for label, spec, det, pol in backends:
            arr = Array(tuple(replace(s, detector=det) for s in array.stations), array.site)
            n_streams = polarization_streams(pol)[0]
            nm, widths = spec.channel_centers_nm, spec.channel_widths_nm
            pairs = arr.pairs()
            snr2 = np.zeros(len(pairs))
            fisher = np.zeros(len(pairs))          # sum (d|V|^2/dtheta / sigma)^2 over blocks, singles only
            v2min, v2max = np.full(len(pairs), np.inf), np.full(len(pairs), -np.inf)
            bmin, bmax = np.full(len(pairs), np.inf), np.full(len(pairs), -np.inf)
            rate = np.zeros(len(pairs))
            scale_min, load_max, n_blocks, hours = 1.0, 0.0, 0, 0.0
            if single:
                tgt = prepare_single(system, spec)
                vis_fn = _single_vis2_fn(tgt, pupils)
                dth = 0.01 * tgt.theta_ld_mas
                vis_fn_up = _single_vis2_fn(replace(tgt, theta_ld_mas=tgt.theta_ld_mas + dth), pupils)
                vis_fn_dn = _single_vis2_fn(replace(tgt, theta_ld_mas=tgt.theta_ld_mas - dth), pupils)
            else:
                tgt, vis_fn = system, binary_vis2_fn(system, pupils)
            for lam, w in zip(nm, widths):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tr = pair_track(arr, system.dec_deg, float(lam), vis_fn, block_minutes=block_minutes,
                                    min_alt_deg=min_alt_deg, phase0=phase, period_days=period)
                    if single:
                        up = pair_track(arr, system.dec_deg, float(lam), vis_fn_up, block_minutes=block_minutes,
                                        min_alt_deg=min_alt_deg)
                        dn = pair_track(arr, system.dec_deg, float(lam), vis_fn_dn, block_minutes=block_minutes,
                                        min_alt_deg=min_alt_deg)
                if tr.hour_angle_h.size == 0:
                    continue
                n_blocks, hours = tr.hour_angle_h.size, tr.hour_angle_h.size * tr.block_s / 3600.0
                mag = float(tgt.ab_mag(float(lam))) if single else float(system_ab_mag(system, float(lam)))
                obs = Observation(wavelength_nm=float(lam), filter_width_nm=float(w), t_int_s=tr.block_s,
                                  polarization_mode=pol, backend_throughput=spec.throughput)
                for p, (i, j, _) in enumerate(pairs):
                    t1, t2 = arr.stations[i].telescope, arr.stations[j].telescope
                    r1 = float(incident_rate(mag, t1, det, obs)) * n_streams
                    r2 = float(incident_rate(mag, t2, det, obs)) * n_streams
                    scale = min(readout_scale(det, r1), readout_scale(det, r2))
                    mag_eff = mag - 2.5 * np.log10(scale) if scale < 1.0 else mag
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r = g2_snr(tr.vis2[p], mag_eff, obs, telescope1=t1, telescope2=t2,
                                   detector1=det)
                        snr_b = np.asarray(r.snr, float)
                        if single:
                            sig = np.asarray(g2_snr(1.0, mag_eff, obs, telescope1=t1, telescope2=t2,
                                                    detector1=det).snr, float)   # sigma(|V|^2) = 1 / SNR(|V|^2 = 1)
                            dv = (up.vis2[p] - dn.vis2[p]) / (2.0 * dth)
                            fisher[p] += float(np.sum((dv * sig) ** 2))
                    snr2[p] += float(np.sum(snr_b ** 2))
                    v2min[p], v2max[p] = min(v2min[p], tr.vis2[p].min()), max(v2max[p], tr.vis2[p].max())
                    bmin[p], bmax[p] = min(bmin[p], tr.baseline_len_m[p].min()), max(bmax[p], tr.baseline_len_m[p].max())
                    rate[p] += max(r1, r2)
                    scale_min = min(scale_min, scale)
                    load_max = max(load_max, float(np.max(r.dead_time_load)))
            if n_blocks == 0:
                print(f"    {label:40s} never above {min_alt_deg:g} deg from {array.site.name}")
                continue
            snr_night = float(np.sqrt(snr2.sum()))
            nights = (n_sigma / snr_night) ** 2 if snr_night > 0 else np.inf
            per_det.setdefault(det.name, []).append(nights)
            nights_theta = None
            if single:
                f_tot = float(fisher.sum())
                sig_theta = 1.0 / np.sqrt(f_tot) if f_tot > 0 else np.inf        # one night
                nights_theta = (sig_theta / (diameter_precision * tgt.theta_ld_mas)) ** 2
                per_det_fisher.setdefault(det.name, []).append(f_tot)
            print(f"    {label:40s} {n_blocks} x {block_minutes:g} min ({hours:.1f} h): "
                  f"SNR2/night = {snr_night:.3g}; nights to {n_sigma:g} sigma on |V|^2: "
                  f"{nights:.3g}; |V|^2 {v2min.min():.3g}-{v2max.max():.3g}; "
                  + (f"nights to {100 * diameter_precision:g} % on theta: {nights_theta:.3g}; " if single else "")
                  + f"rate {rate.max():.2e} cps/tel{' READOUT-LIMITED x%.1e' % scale_min if scale_min < 1 else ''}, "
                  f"dead-time load {load_max:.2f}")
            for p, (i, j, _) in enumerate(pairs):
                s_p = float(np.sqrt(snr2[p]))
                print(f"      {arr.stations[i].name}-{arr.stations[j].name}: B {bmin[p]:.0f}-{bmax[p]:.0f} m, "
                      f"|V|^2 {v2min[p]:.3g}-{v2max[p]:.3g}, SNR2/night {s_p:.3g}")
                rows.append({"target": short, "tabled": tabled, "backend": label, "detector": det.name,
                             "pair": f"{arr.stations[i].name}-{arr.stations[j].name}",
                             "baseline_min_m": float(bmin[p]), "baseline_max_m": float(bmax[p]),
                             "vis2_min": float(v2min[p]), "vis2_max": float(v2max[p]),
                             "snr_night": s_p, "snr_night_all_pairs": snr_night,
                             "nights_detection": None if not np.isfinite(nights) else float(nights),
                             "nights_diameter": (None if nights_theta is None or not np.isfinite(nights_theta)
                                                 else float(nights_theta)),
                             "diameter_precision": diameter_precision if single else None,
                             "n_blocks": int(n_blocks), "hours": float(hours),
                             "rate_cps": float(rate[p]), "readout_limited": scale_min < 1.0,
                             "readout_scale": float(scale_min), "dead_time_load": float(load_max),
                             "phase": float(phase), "rho_mas": rho, "single": single,
                             "v_mag": float(system.v_mag) if single else None})
        if per_det:
            set_nights[short] = {}
            for dname, lst in per_det.items():
                tot = float(sum(lst)) if one_backend_per_night else float(1.0 / np.sqrt(sum(1.0 / x**2 for x in lst)))
                entry = {"detection": None if not np.isfinite(tot) else tot}
                if dname in per_det_fisher:
                    # one filter per night: the Fisher information of the set is the sum of
                    # the per-filter informations, each bought with its own night
                    f_set = sum(per_det_fisher[dname])
                    k = len(per_det_fisher[dname])
                    sig_set = 1.0 / np.sqrt(f_set / k) if f_set > 0 else np.inf   # per night of the rotation
                    entry["diameter"] = None if not np.isfinite(sig_set) else float((sig_set / (diameter_precision * system.theta_ld_mas)) ** 2)
                set_nights[short][dname] = entry
                rule = "one filter per night: nights ADD" if one_backend_per_night else "simultaneous: quadrature"
                print(f"    filter set on {dname}: {tot:.3g} nights to {n_sigma:g} sigma ({rule})"
                      + (f"; rotating the filters, {entry['diameter']:.3g} nights to {100 * diameter_precision:g} % on theta"
                         if entry.get("diameter") else ""))
    return rows, set_nights


def _scan(campaign) -> np.ndarray:
    s = campaign.option("options.baseline_scan_m") or {}
    start = float(s.get("start", 10.0))
    stop = float(s.get("stop", 300.0))
    step = float(s.get("step", 5.0))
    return np.arange(start, stop + step / 2.0, step)      # stop inclusive


def run(campaign, cat, opts, out_dir: Path) -> dict:
    warnings.filterwarnings("ignore", message=".*extrapolated.*")
    warnings.filterwarnings("ignore", message=".*dead-time.*")

    tel = campaign.telescope
    backends = [(b.name, b.spectrograph, b.detector, b.polarization_mode) for b in campaign.backends]
    if not backends:
        raise SystemExit(f"campaign {campaign.name}: runner g2 needs at least one backend")
    if tel is None and campaign.array is not None:
        inst = campaign.option("instrument.array")
        rows, set_nights = g2_array_track(
            campaign.array, backends, campaign.targets,
            block_minutes=float(campaign.option("night.block_minutes", 30.0)),
            min_alt_deg=float(campaign.option("night.min_alt_deg", 30.0)),
            phase_spec=campaign.option("options.phase", "max_separation"),
            one_backend_per_night=bool(campaign.option("options.one_backend_per_night", True)),
            n_sigma=float(campaign.option("options.detection_sigma", 3.0)),
            diameter_precision=float(campaign.option("options.diameter_precision", 0.05)))
        any_single = any(r["single"] for r in rows)
        def _f(x, fmt=".3g"):
            return "inf" if x is None else format(x, fmt)
        table = [[r["target"], r["tabled"], r["backend"], r["pair"],
                  f"{r['baseline_min_m']:.0f}-{r['baseline_max_m']:.0f}",
                  f"{r['vis2_min']:.3g}-{r['vis2_max']:.3g}", f"{r['snr_night']:.3g}",
                  _f(r["nights_detection"])] + ([_f(r["nights_diameter"]) if r["single"] else "-"] if any_single else [])
                 for r in rows]
        header = ["target", "SED", "backend", "pair", "B [m]", "|V|^2", "SNR2/night",
                  f"nights ({campaign.option('options.detection_sigma', 3.0):g} sigma)"]
        if any_single:
            header.append(f"nights ({100 * float(campaign.option('options.diameter_precision', 0.05)):g} % theta)")
        write_tables(out_dir, table, header, latex=bool(opts.latex))
        return {"mode": "array_track", "instrument": inst, "rows": rows, "filter_set_nights": set_nights,
                "one_backend_per_night": bool(campaign.option("options.one_backend_per_night", True))}
    if tel is None:
        raise SystemExit(f"campaign {campaign.name}: runner g2 needs instrument.telescope or instrument.array")
    inst = campaign.option("instrument.telescope")
    instrument = inst if isinstance(inst, str) else (tel.name or "telescope")
    names = ([campaign.spec["target"]] if "target" in campaign.spec else []) \
        + list(campaign.spec.get("targets", ()))
    by_name = dict(zip(names, campaign.targets))
    latex = bool(opts.latex)

    if campaign.option("options.section3_numbers", False):
        missing = [k for k in ("betaaur", "algol") if k not in by_name]
        if missing:
            print(f"  section3_numbers: campaign lacks target(s) {missing}; skipped")
        else:
            g2_numbers(by_name["betaaur"], by_name["algol"],
                       cat.load_telescope("c2pu_1m"), cat.load_telescope("keck_10m"),
                       cat.load_detector("spad_lambda"), cat.load_detector("spad_lambda_ng"),
                       cat.load_spectrograph("spad_lambda_320"))

    rows, best = g2_table(instrument, tel, backends, campaign.targets, _scan(campaign),
                          phase_spec=campaign.option("options.phase", "max_separation"),
                          vis2_method=campaign.option("options.vis2_method", "analytic"),
                          latex=latex)

    table = [[r["target"], r["tabled"], r["backend"], f"{r['baseline_m']:.0f}",
              f"{r['snr_total']:.1f}", f"{r['rate_cps']:.1e}",
              f"limited x{r['readout_scale']:.2f}" if r["readout_limited"] else "-"] for r in rows]
    write_tables(out_dir, table, ["target", "SED", "backend", "B [m]", "SNR2/sqrt(h)",
                                  "rate cps/tel", "limited"], latex=latex)
    return {"instrument": instrument, "telescope_m": float(tel.diameter_m),
            "rows": rows, "best_baselines": best}
