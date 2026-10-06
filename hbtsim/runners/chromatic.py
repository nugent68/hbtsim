"""Runner `chromatic`: Balmer-core vs continuum (chromatic) diameters of
single stars (hbtsim.chromatic, hbtsim.single, model-atmosphere tables).

Two instrument modes, chosen by the campaign's `instrument` block:

* `instrument.array` (the EON-SII pair): for each target, backend and
  readout strategy the runner finds the first-lobe baseline that maximizes
  the Asimov significance of "theta_UD(lambda) is smooth"
  (chromatic.optimal_baseline, never below
  options.optimal_baseline.min_baseline_m) and prints the per-line signal,
  per-line and total significance per night, the continuum chromaticity,
  the tagged channels and the nights to 5 sigma.  Readout strategies:
  "link" (channel_selection "all" through the backend's time-tag
  detector), "subset" (only the line and reference channels that fit the
  link) and "correlator" (every channel through the backend's detector with
  its link ceiling removed; switched on by options.correlator_detector, the
  catalog's correlator readout of the QUASAR SPAD).
* `instrument.telescope` (the 2 x 1 m regression): the baseline is
  options.baseline_x lambda_ref / (pi theta_drawn), the integration
  options.t_int_h, the continuum fit of degree options.continuum_degree
  (chromatic.chromatic_signature with the backend's detector).

Options (campaign `options`): channel_selection (list of "all" / "subset"),
correlator_detector, nights, continuum_degree, optimal_baseline.min_baseline_m,
site, t_int_h, baseline_x, baseline_reference_nm.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from hbtsim.chromatic import (BALMER_VAC_NM, chromatic_signature, line_masks,
                              night_seconds, optimal_baseline)
from hbtsim.params import MAS
from hbtsim.runners import write_tables

READOUT_LABEL = {"all": "link", "subset": "subset"}     # channel_selection -> printed label


def readout_config(readout: str, detector, correlator_detector=None):
    """The readout strategy as (channel_selection, detector): "link" reads
    every channel through the detector's time-tag link, "subset" tags only
    the line and reference channels, "correlator" reads every channel with
    an on-board correlator (no link-rate cap): the catalog's correlator
    detector when given, else the backend's detector with its link ceiling
    removed."""
    if readout == "link":
        return "all", detector
    if readout == "subset":
        return "subset", detector
    if readout == "correlator":
        if correlator_detector is not None:
            return "all", correlator_detector
        return "all", replace(detector, readout="correlator", max_total_cps=None)
    raise ValueError(f"unknown readout {readout!r}")


def describe(r, nights):
    s = r.significance * np.sqrt(nights)
    lines = ", ".join(f"{k} {r.line_signal_pct.get(k, np.nan):+.2f} % / {v * np.sqrt(nights):.1f} sigma"
                      for k, v in r.line_significance.items())
    tag = f"{int(r.tagged.sum())} of {r.nm.size} ch tagged" if r.channel_selection == "subset" else f"{r.nm.size} ch"
    where = ""
    if r.channel_selection == "subset":
        m = line_masks(r.nm)
        on = [k for k in BALMER_VAC_NM if k in m and (r.tagged & (m[k]["core"] | m[k]["wing"])).any()]
        where = f" (lines covered: {', '.join(on) or 'none'}; continuum refs {int((r.tagged & m['continuum']).sum())})"
    return (f"B {r.baseline_m:5.1f} m, |V|^2 {np.median(r.vis2):.2f}; {tag}{where}; readout scale "
            f"{r.readout_scale:.3g}; total {s:.1f} sigma in {nights:g} night(s) "
            f"-> {r.nights_to(5.0) :.2g} nights to 5 sigma | {lines} | continuum "
            f"{r.continuum_chromaticity_pct:+.2f} % ({r.nm[0]:.0f}->{r.nm[-1]:.0f} nm)")


def plot(r, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(r.nm, r.vis2, lw=0.8)
    ax[0].set_ylabel(r"$|V|^2$ (pupil-averaged)")
    ok = r.tagged & np.isfinite(r.sigma_theta)
    ax[1].plot(r.nm, r.theta_ud, lw=0.8, color="C0", label=r"model $\theta_{UD}$")
    ax[1].errorbar(r.nm[ok], r.theta_ud[ok], yerr=r.sigma_theta[ok], fmt=".", ms=2, lw=0.5,
                   color="C1", label="tagged, 1-night error")
    ax[1].plot(r.nm, r.continuum, "k--", lw=0.8, label="continuum fit")
    for k, l in BALMER_VAC_NM.items():
        for a in ax:
            a.axvline(l, color="0.7", lw=0.5)
    ax[1].set_ylabel(r"$\theta_{UD}$ [mas]")
    ax[1].set_xlabel("Vacuum wavelength [nm]")
    ax[1].legend(fontsize=8)
    ax[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"    wrote {path}")


def _finite(x):
    """A float for JSON, None where it is not finite (inf nights to 5 sigma)."""
    x = float(x)
    return x if math.isfinite(x) else None


def _record(r, nights: float) -> dict:
    return {"baseline_m": float(r.baseline_m),
            "significance": _finite(r.significance * np.sqrt(nights)),
            "nights_to_5sigma": _finite(r.nights_to(5.0)),
            "line_significance": {k: _finite(v * np.sqrt(nights)) for k, v in r.line_significance.items()},
            "line_signal_pct": {k: _finite(v) for k, v in r.line_signal_pct.items()},
            "continuum_chromaticity_pct": _finite(r.continuum_chromaticity_pct),
            "n_tagged": int(r.tagged.sum()), "n_channels": int(r.nm.size),
            "readout_scale": _finite(r.readout_scale), "t_int_s": float(r.t_int_s),
            "vis2_median": _finite(np.median(r.vis2))}


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_").lower()


def target_names(campaign) -> list:
    """The catalog names of campaign.targets (file-name slugs for figures)."""
    spec = campaign.spec
    return ([spec["target"]] if "target" in spec else []) + list(spec.get("targets", ()))


def run(campaign, cat, opts, out_dir: Path) -> dict:
    warnings.filterwarnings("ignore", message=".*dead-time.*")
    nights = float(campaign.option("options.nights", 1.0))
    deg = int(campaign.option("options.continuum_degree", 1))
    selections = tuple(campaign.option("options.channel_selection", ["all"]))
    for sel in selections:
        if sel not in READOUT_LABEL:
            raise ValueError(f"options.channel_selection: {sel!r} is not 'all' or 'subset'")
    corr_name = campaign.option("options.correlator_detector")
    correlator = cat.load_detector(corr_name) if corr_name else None

    if campaign.array is not None:
        mode = "array"
        telescope = campaign.array.stations[0].telescope
        site = campaign.array.site
        gen_min = (cat.raw("array", campaign.spec["instrument"]["array"]).get("generator") or {}).get(
            "min_spacing_m", 0.0)
        min_baseline_m = float(campaign.option("options.optimal_baseline.min_baseline_m", gen_min or 0.0))
    elif campaign.telescope is not None:
        mode = "telescope"
        telescope = campaign.telescope
        site = cat.load_site(campaign.option("options.site", "calern"))
        x = float(campaign.option("options.baseline_x", 1.7))
        ref_nm = float(campaign.option("options.baseline_reference_nm", 486.27))
        t_int_s = float(campaign.option("options.t_int_h", 6.0)) * 3600.0
    else:
        raise ValueError(f"campaign {campaign.name}: needs instrument.array or instrument.telescope")

    results, rows = {}, []
    for key, tgt in zip(target_names(campaign), campaign.targets):
        print(f"\n=== {tgt.name}: theta_LD {tgt.theta_ld_mas} mas, drawn {tgt.drawn_diameter_mas:.4f} mas "
              f"(r_outer {tgt.star.radius_scale:.5f}); V {tgt.v_mag:+.2f}, model AB(550) - V = "
              f"{tgt.v_check():+.3f}")
        per_target = results.setdefault(tgt.name, {})
        if mode == "telescope":
            b = x * ref_nm * 1e-9 / (np.pi * tgt.drawn_diameter_mas * MAS)
            for be in campaign.backends:
                per_backend = per_target.setdefault(be.name, {})
                for sel in selections:
                    label = READOUT_LABEL[sel]
                    r = chromatic_signature(tgt, b, be.spectrograph, telescope=telescope, detector=be.detector,
                                            site=site, t_int_s=t_int_s, channel_selection=sel,
                                            polarization_mode=be.polarization_mode, deg=deg)
                    print(f"  2 x {telescope.diameter_m:g} m ({telescope.name or 'telescope'}), "
                          f"{be.spectrograph.name}, {be.name}, {t_int_s / 3600:g} h, {label}: "
                          f"{describe(r, nights)}")
                    per_backend[label] = _record(r, nights)
                    rows.append([tgt.name, be.name, label, f"{r.baseline_m:.1f}",
                                 f"{r.significance * np.sqrt(nights):.2f}", f"{r.nights_to(5.0):.2g}"])
                    if opts.figures:
                        plot(r, out_dir / f"chromatic_{_slug(key)}_{_slug(be.name)}_{label}.png",
                             f"{tgt.name}: {be.name}, {label} readout, B = {r.baseline_m:.1f} m")
            continue
        print(f"  {site.name} : {night_seconds(tgt, site) / 3600:.1f} h above 30 deg per night; "
              f"baselines >= {min_baseline_m:g} m ({telescope.diameter_m:g} m units)")
        for be in campaign.backends:
            per_backend = per_target.setdefault(be.name, {})
            cases = [(READOUT_LABEL[sel],) + readout_config(READOUT_LABEL[sel], be.detector) for sel in selections]
            if correlator is not None:
                # the backend's own detector behind a correlator (the script's readout_config);
                # the catalog entry is used where it is that detector (the QUASAR SPAD backends)
                derived = readout_config("correlator", be.detector)[1]
                same = replace(derived, name=correlator.name) == correlator
                cases.append(("correlator",) + readout_config("correlator", be.detector,
                                                              correlator if same else None))
                if not same and tgt is campaign.targets[0]:
                    print(f"  ({be.name}: options.correlator_detector {corr_name} is not this backend's "
                          f"detector; its own {be.detector.name} is read through a correlator)")
            for label, sel, det in cases:
                b, r = optimal_baseline(tgt, be.spectrograph, telescope=telescope, detector=det, site=site,
                                        channel_selection=sel, polarization_mode=be.polarization_mode,
                                        deg=deg, min_baseline_m=min_baseline_m)
                print(f"  {be.name:32s} {label:10s}: {describe(r, nights)}")
                per_backend[label] = _record(r, nights)
                rows.append([tgt.name, be.name, label, f"{b:.1f}",
                             f"{r.significance * np.sqrt(nights):.2f}", f"{r.nights_to(5.0):.2g}"])
                if opts.figures:
                    plot(r, out_dir / f"chromatic_{_slug(key)}_{_slug(be.name)}_{label}.png",
                         f"{tgt.name}: {be.name}, {label} readout, B = {b:.1f} m")

    write_tables(out_dir, rows, ["target", "backend", "readout", "B [m]",
                                 f"significance ({nights:g} night)", "nights to 5 sigma"],
                 latex=opts.latex)
    return results
