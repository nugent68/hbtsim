"""Night-by-night array configurations for the runners whose instrument is
a generator array (a pair or an equilateral triangle): a *schedule* entry
names the backend of the night and/or the array parameters of the night
(baseline_m, pa_deg, side_m), so a movable array such as the ISI trucks can
sweep |V|^2(B) through the null and the sidelobes over a run.

Entries may be
  "lpqi_mpd_halpha"                       a backend (catalog name or label)
  12.0                                    a baseline [m] on the campaign's first backend
  {"baseline_m": 12, "pa_deg": 90, "backend": "isi_id230_1550", "nights": 2}
Unknown keys are an error.  `array_for` rebuilds the campaign's array with
the entry's parameters on top of instrument.array_params.
"""

from __future__ import annotations

from dataclasses import replace

ARRAY_KEYS = ("baseline_m", "pa_deg", "side_m")
ENTRY_KEYS = ARRAY_KEYS + ("backend", "nights")


def backend_index(x, campaign, backends) -> int:
    """The index of a backend given by catalog name or label."""
    names = [b.name for b in backends]
    spec = campaign.spec.get("backends", [])
    catalog_names = [spec[i] if i < len(spec) and isinstance(spec[i], str) else names[i] for i in range(len(names))]
    if x in catalog_names:
        return catalog_names.index(x)
    if x in names:
        return names.index(x)
    raise SystemExit(f"schedule entry {x!r} is not one of the campaign's backends {catalog_names}")


def normalize_schedule(entries, campaign, backends) -> list[dict]:
    """[{backend_index, nights, [baseline_m], [pa_deg], [side_m]}, ...]."""
    out = []
    for x in entries:
        if isinstance(x, str):
            out.append({"backend_index": backend_index(x, campaign, backends), "nights": 1, "backend_given": True})
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            out.append({"backend_index": 0, "nights": 1, "baseline_m": float(x)})
        elif isinstance(x, dict):
            bad = set(x) - set(ENTRY_KEYS)
            if bad:
                raise SystemExit(f"schedule entry {x!r}: unknown key(s) {sorted(bad)}; allowed {ENTRY_KEYS}")
            e = {"backend_index": backend_index(x["backend"], campaign, backends) if "backend" in x else 0,
                 "nights": int(x.get("nights", 1)), "backend_given": "backend" in x}
            for k in ARRAY_KEYS:
                if k in x:
                    e[k] = float(x[k])
            out.append(e)
        else:
            raise SystemExit(f"schedule entry {x!r}: expected a backend name, a baseline in metres or a dict")
    return out


def array_params(entry: dict) -> dict:
    return {k: entry[k] for k in ARRAY_KEYS if k in entry}


def array_for(cat, campaign, entry: dict, detector=None):
    """The campaign's array rebuilt with the entry's parameters (a generator
    array is required when the entry carries any); with `detector` every
    station takes it, as the runners do."""
    from hbtsim.bispectrum import Array
    params = array_params(entry)
    if params:
        name = campaign.option("instrument.array")
        if name is None or "generator" not in cat.raw("array", name):
            raise SystemExit(f"schedule entries with {sorted(params)} need a generator array (instrument.array), not {name!r}")
        base = dict(campaign.option("instrument.array_params", {}) or {})
        base.update(params)
        arr = cat.load_array(name, **base)
    else:
        arr = campaign.array
    if detector is not None:
        arr = Array(tuple(replace(s, detector=detector) for s in arr.stations), arr.site)
    return arr


def describe(entry: dict, backends) -> str:
    bits = [backends[entry["backend_index"]].name]
    bits += [f"{k} {entry[k]:g}" for k in ARRAY_KEYS if k in entry]
    if entry.get("nights", 1) != 1:
        bits.append(f"x{entry['nights']} nights")
    return ", ".join(bits)


def with_ld_offset(target, d_u: float):
    """A single-star target whose linear limb-darkening coefficients are
    shifted by d_u at every tabulated wavelength (clipped to [0, 1]); for
    the finite-difference derivative of |V|^2 with respect to u.  None when
    the star carries a model I(mu) profile, which the linear law does not
    drive."""
    import numpy as np
    star = target.star
    if star.ld_profile is not None:
        return None
    table = tuple((float(l), float(np.clip(u + d_u, 0.0, 1.0))) for l, u in star.ld_table_nm)
    return replace(target, star=replace(star, ld_table_nm=table))
