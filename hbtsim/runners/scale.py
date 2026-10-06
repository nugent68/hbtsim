"""Runner `scale`: angular-scale precision of single stars (the red-clump
comparison with Kim & Kaiser 2026, PASP 138, 044202) on hbtsim's photon
budget with model-atmosphere stand-ins (hbtsim.diameter, hbtsim.single).

Three questions per target:

  (1) Radius convention.  How far apart are the tau = 1 radius, the
      apparent limb and the outer boundary of the spherical models
      (options.radius_convention_models, read from the atmosphere
      resource), and how does the uniform-disk diameter relate to the
      limb-darkened one per band (theta_UD / theta_LD at x = 1.5)?
  (2) Multiplexing.  One broad-band filter per band (options.filters on
      the campaign telescope and options.filter_detector) against the
      campaign's multiplexed backends and the EON-SII case
      (options.eonsii_case), all scanned over options.baseline_scan_m in
      options.hours; hours to options.target_sigma_s.
  (3) Chromaticity.  theta_UD(lambda) per window at the best baseline of
      options.chromaticity_backend (default: the campaign backend with the
      most channels).

The stand-in named by the campaign's `atmosphere.stand_in` is attached by
run_campaign from the 380-1000 nm tables; the broad-band part re-attaches
the 380-2500 nm table of the same model (falling back to the target as
loaded when it is missing).
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from hbtsim.diameter import scale_precision_scan
from hbtsim.runners import write_tables
from hbtsim.sed import load_star_tables, with_tables
from hbtsim.single import prepare_single, single_star_vis2, ud_diameter_per_channel

RADIUS_CONVENTION_MODELS = ("lte04700-4.50-0.0", "lte04800-4.50-0.0", "lte05000-0.00-0.0")
CHROMATICITY_WINDOWS_NM = ((400, 450), (450, 500), (500, 550), (550, 650), (650, 800), (800, 950))


def table_path(data_dir: str, model: str, ir: bool) -> str:
    return os.path.join(data_dir, f"newera_{model}_{'380-2500nm_0.1nm' if ir else '380-1000nm_0.02nm'}.npz")


def attach(target, model: str | None, ir: bool, data_dir: str | None):
    """The target with a stand-in NewEra table (the target as given when
    model is None or the table is missing)."""
    if model is None or data_dir is None:
        return target
    path = table_path(data_dir, model, ir)
    if not os.path.exists(path):
        print(f"  stand-in table {path} not found: keeping the target's tables as loaded")
        return target
    ft, ld = load_star_tables(path)
    return replace(target, star=with_tables(target.star, ft, ld))


def radius_conventions(data_dir: str | None, models=RADIUS_CONVENTION_MODELS) -> dict:
    print("(1) Radius conventions of the spherical models (from the binned tables)")
    out = {}
    if data_dir is None or not os.path.isdir(data_dir):
        print(f"  (skipped: NewEra directory {data_dir} not found)")
        return out
    print(f"  {'model':26s} {'R_out/R_tau1':>13s} {'R_out/R_edge':>13s} {'R_edge/R_tau1':>14s}")
    for model in models:
        path = table_path(data_dir, model, ir=False)
        if not os.path.exists(path):
            print(f"  {model:26s} (table {path} missing; skipped)")
            continue
        d = np.load(path)
        r_t = float(d["r_outer_over_tau1"]) if "r_outer_over_tau1" in d else np.nan
        mu_e = float(d["mu_edge"]) if "mu_edge" in d else np.nan
        r_e = 1.0 / np.sqrt(1.0 - mu_e**2)
        print(f"  {model:26s} {r_t:13.4f} {r_e:13.4f} {r_e / r_t:14.4f}")
        out[model] = {"r_outer_over_tau1": _finite(r_t), "r_outer_over_edge": _finite(r_e),
                      "r_edge_over_tau1": _finite(r_e / r_t)}
    print("  (R_tau1: Rosseland tau = 1; R_edge: where the continuum intensity drops to half;"
          " R_out: the model's outer boundary.  The 4800 K / log g 2.5 giant lies between the rows.)")
    return out


def ud_over_ld(tgt, bands: dict, pupil_m: float, x=1.5) -> dict:
    """theta_UD / theta_LD per band at pi theta B / lambda = x (tgt with its
    IR tables attached)."""
    out = {}
    pup = (pupil_m, pupil_m)
    for band, spec in bands.items():
        p = prepare_single(tgt, spec)
        nm = spec.channel_centers_nm
        b = x * nm[0] * 1e-9 / (np.pi * tgt.drawn_diameter_mas * 4.8481368e-9)
        v2 = single_star_vis2(p, b, nm, pup)[:, 0]
        out[band] = float(ud_diameter_per_channel(v2, b, nm, pup)[0]) / tgt.theta_ld_mas
    return out


def scan(target, spec, hours, baselines, **kw):
    """Baseline scan: (best ScaleResult, sigma_s per baseline)."""
    bb, sig, best = scale_precision_scan(target, baselines, spec, t_int_s=hours * 3600.0, **kw)
    return best, sig


def _finite(x):
    x = float(x)
    return x if math.isfinite(x) else None


def _band_label(name: str) -> str:
    """"filter_johnson_v" -> "V"."""
    return name.rsplit("_", 1)[-1].upper()


def _baselines(spec: dict | None) -> np.ndarray:
    spec = spec or {}
    return np.arange(float(spec.get("start", 20.0)), float(spec.get("stop", 300.0)) + 1e-9,
                     float(spec.get("step", 5.0)))


def _best_record(best, hours, target_sigma):
    return {"sigma_s": _finite(best.sigma_s), "baseline_m": float(best.baseline_m),
            "vis2": _finite(best.vis2[0]), "rate_cps": _finite(best.snr.total_rate_cps[0]),
            "hours_to_target": _finite(hours * (best.sigma_s / target_sigma) ** 2),
            "readout_limited": bool(best.snr.readout_limited),
            "readout_scale": _finite(best.snr.readout_scale), "n_channels": int(best.n_channels)}


def run(campaign, cat, opts, out_dir: Path) -> dict:
    warnings.filterwarnings("ignore")
    atm = campaign.spec.get("atmosphere") or {}
    use = atm.get("use", True) and not opts.no_newera
    model = atm.get("stand_in") if use else None
    data_dir = opts.newera_dir
    if data_dir is None and atm.get("resource"):
        data_dir = cat.load_resource(atm["resource"]).resolved_path
    label = f"NewEra STAND-IN {model}" if model else "blackbody + linear law"

    telescope = campaign.telescope
    if telescope is None:
        raise ValueError(f"campaign {campaign.name}: needs instrument.telescope")
    pupil_m = telescope.diameter_m
    hours = float(campaign.option("options.hours", 2.0))
    target_sigma = float(campaign.option("options.target_sigma_s", 0.007))
    baselines = _baselines(campaign.option("options.baseline_scan_m"))
    bands = {_band_label(n): cat.load_spectrograph(n) for n in campaign.option("options.filters", ())}
    det_name = campaign.option("options.filter_detector")
    filter_detector = cat.load_detector(det_name) if det_name else (campaign.backends[0].detector
                                                                    if campaign.backends else None)
    cases = [(b.name, b.spectrograph, telescope, b.detector, b.polarization_mode) for b in campaign.backends]
    eon = campaign.option("options.eonsii_case")
    if eon:
        be = cat.load_backend(eon["backend"])
        cases.append((f"EON-SII {be.name}", be.spectrograph, cat.load_telescope(eon["telescope"]),
                      be.detector, be.polarization_mode))
    chrom_name = campaign.option("options.chromaticity_backend")
    if chrom_name:
        chrom = cat.load_backend(chrom_name)
    else:
        chrom = max(campaign.backends, key=lambda b: b.spectrograph.n_channels) if campaign.backends else None

    results = {"stand_in": model or "linear", "hours": hours, "target_sigma_s": target_sigma}
    results["radius_conventions"] = radius_conventions(
        data_dir, campaign.option("options.radius_convention_models", RADIUS_CONVENTION_MODELS))
    rows = []
    spec_names = ([campaign.spec["target"]] if "target" in campaign.spec else []) + list(
        campaign.spec.get("targets", ()))
    for key, t_opt in zip(spec_names, campaign.targets):
        # t_opt carries the 380-1000 nm stand-in (run_campaign); the IR table serves the broad bands
        t_ir = attach(t_opt, model, True, data_dir)
        print(f"\n=== {t_opt.name}: theta_LD {t_opt.theta_ld_mas} mas, V {t_opt.v_mag}; profile: {label}; "
              f"drawn diameter {t_ir.drawn_diameter_mas:.4f} mas (r_outer {t_ir.star.radius_scale:.4f}); "
              f"model AB(551) - V = {t_ir.ab_mag(551.0) - t_opt.v_mag:+.3f}")
        rec = results[t_opt.name] = {"theta_ud_over_ld": {}, "filters": {}, "multiplexed": {}}
        if model is not None and bands:
            rec["theta_ud_over_ld"] = ud_over_ld(t_ir, bands, pupil_m)
            print("  theta_UD / theta_LD at x = 1.5 per band: "
                  + ", ".join(f"{b} {v:.4f}" for b, v in rec["theta_ud_over_ld"].items()))

        print(f"\n(2) Scale precision sigma_s in {hours:g} h, two {pupil_m:g} m telescopes, baseline scan "
              f"{baselines[0]:g}-{baselines[-1]:g} m")
        if bands and filter_detector is not None:
            print(f"  {telescope.name or 'campaign telescope'} with {filter_detector.name}, one filter per band:")
        for band, spec in bands.items():
            if filter_detector is None:
                break
            best, sig = scan(t_ir, spec, hours, baselines, telescope=telescope, detector=filter_detector)
            print(f"    {band}: sigma_s = {best.sigma_s:.4f} at B = {best.baseline_m:.0f} m "
                  f"(|V|^2 {best.vis2[0]:.2f}; rate {best.snr.total_rate_cps[0]:.2e} cps/tel)"
                  + (f"  <- paper: HD 17652 < {target_sigma:g} at ~100 m" if band == "H" else ""))
            rec["filters"][band] = _best_record(best, hours, target_sigma)
            rows.append([t_opt.name, f"{band} filter", f"{best.sigma_s:.4f}", f"{best.baseline_m:.0f}",
                         f"{hours * (best.sigma_s / target_sigma) ** 2:.2f}"])
        print(f"  Multiplexed optical backends (hours to sigma_s = {target_sigma:g}):")
        for name, spec, tel, det, pol in cases:
            best, sig = scan(t_opt, spec, hours, baselines, telescope=tel, detector=det, polarization_mode=pol)
            h_to = hours * (best.sigma_s / target_sigma) ** 2
            lim = f" READOUT-LIMITED x{best.snr.readout_scale:.2g}" if best.snr.readout_limited else ""
            print(f"    {name:52s} sigma_s = {best.sigma_s:.4f} at B = {best.baseline_m:.0f} m -> "
                  f"{h_to:7.2f} h; rate {best.snr.total_rate_cps[0]:.2e} cps/tel{lim}")
            rec["multiplexed"][name] = _best_record(best, hours, target_sigma)
            rows.append([t_opt.name, name, f"{best.sigma_s:.4f}", f"{best.baseline_m:.0f}", f"{h_to:.2f}"])

        if model is not None and chrom is not None:
            spec = chrom.spectrograph
            print(f"\n(3) Chromaticity: theta_UD(lambda) with {chrom.name} ({spec.n_channels} ch, "
                  f"{spec.lambda_min_nm:.0f}-{spec.lambda_max_nm:.0f} nm) at the optical-best baseline")
            best, _ = scan(t_opt, spec, hours, baselines, telescope=telescope, detector=chrom.detector,
                           polarization_mode=chrom.polarization_mode)
            p = prepare_single(t_opt, spec)
            nm = spec.channel_centers_nm
            v2 = single_star_vis2(p, best.baseline_m, nm, (pupil_m, pupil_m))[:, 0]
            th = ud_diameter_per_channel(v2, best.baseline_m, nm, (pupil_m, pupil_m))
            windows = {}
            for lo, hi in CHROMATICITY_WINDOWS_NM:
                m = (nm >= lo) & (nm < hi)
                if not m.any():
                    continue
                ratio = th[m] / t_opt.theta_ld_mas
                print(f"    {lo}-{hi} nm: theta_UD / theta_LD median {np.median(ratio):.4f}, "
                      f"range {ratio.min():.4f}-{ratio.max():.4f}")
                windows[f"{lo}-{hi}"] = {"median": _finite(np.median(ratio)), "min": _finite(ratio.min()),
                                         "max": _finite(ratio.max())}
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.median(best.sigma_vis2 / np.abs(best.dvis2_ds))
            print(f"    per-channel sigma(theta_UD)/theta in {hours:g} h: median {rel:.3g}; "
                  f"B = {best.baseline_m:.0f} m")
            rec["chromaticity"] = {"backend": chrom.name, "baseline_m": float(best.baseline_m),
                                   "windows": windows, "per_channel_sigma_rel_median": _finite(rel)}
            if opts.figures:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
                ax[0].plot(nm, v2, lw=0.6)
                ax[0].set_ylabel(rf"$|V|^2$ ({pupil_m:g} m pupils)")
                ax[1].plot(nm, th / t_opt.theta_ld_mas, lw=0.6)
                ax[1].set_ylabel(r"$\theta_{UD}/\theta_{LD}$")
                ax[1].set_xlabel("Vacuum wavelength [nm]")
                ax[0].set_title(f"{t_opt.name}: {label}, B = {best.baseline_m:.0f} m, {chrom.name}")
                fig.tight_layout()
                out = out_dir / f"redclump_{key}_chromatic.png"
                fig.savefig(out, dpi=140)
                plt.close(fig)
                print(f"    wrote {out}")

    print("\nModel request (Hamburg): NewEra HSR-RF at [M/H] = 0 for T_eff 4700, 4800, 4900 K x "
          "log g 2.0, 2.5, 3.0 (the red clump; 9 models), optionally 4800 K / 2.5 at [M/H] -0.5 and +0.5.")
    write_tables(out_dir, rows, ["target", "case", "sigma_s", "B [m]", f"hours to sigma_s = {target_sigma:g}"],
                 latex=opts.latex)
    return results
