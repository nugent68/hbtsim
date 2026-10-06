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
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from hbtsim.aperture import fringe_smearing_factor
from hbtsim.orbit import max_separation_phase, positions_at
from hbtsim.params import MAS
from hbtsim.snr import Observation, g2_snr, spectral_g2_snr, system_ab_mag

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
    if tel is None:
        raise SystemExit(f"campaign {campaign.name}: runner g2 needs instrument.telescope")
    inst = campaign.option("instrument.telescope")
    instrument = inst if isinstance(inst, str) else (tel.name or "telescope")
    backends = [(b.name, b.spectrograph, b.detector, b.polarization_mode) for b in campaign.backends]
    if not backends:
        raise SystemExit(f"campaign {campaign.name}: runner g2 needs at least one backend")
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
