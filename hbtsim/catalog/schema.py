"""Field specifications and cross-field checks for every catalog kind.

Pure stdlib: a Field table per kind plus a check function.  Errors are
SchemaError with the file, the key and what was expected.  Keys that
start with "_" are ignored everywhere (free comments).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NAME_RE = re.compile(r"^[a-z0-9_]+$")
KINDS = ("band", "ld_table", "target", "telescope", "detector", "spectrograph",
         "backend", "site", "array", "resource", "campaign")
SCHEMA_VERSION = 1

COMMON_KEYS = {"schema_version", "kind", "name", "label", "extends", "sources",
               "notes", "provenance", "assumptions", "summary"}


class CatalogError(Exception):
    pass


class SchemaError(CatalogError):
    pass


class UnknownNameError(CatalogError, KeyError):
    def __str__(self):
        return self.args[0] if self.args else ""


@dataclass(frozen=True)
class Field:
    key: str
    types: tuple            # accepted Python types (after JSON parsing)
    required: bool = False
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None
    nullable: bool = False
    one_of: str | None = None   # group name: exactly one key of the group must be present


NUM = (int, float)
STR = (str,)
BOOL = (bool,)
LIST = (list,)
DICT = (dict,)
REF = (str, dict)           # a name or an inline definition


def _check_value(where: str, f: Field, v) -> None:
    if v is None:
        if f.nullable:
            return
        raise SchemaError(f"{where}: '{f.key}' may not be null")
    if isinstance(v, bool) and bool not in f.types:
        raise SchemaError(f"{where}: '{f.key}' must be {_names(f.types)}, not a boolean")
    if not isinstance(v, f.types):
        raise SchemaError(f"{where}: '{f.key}' must be {_names(f.types)}, "
                          f"not {type(v).__name__}")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if f.lo is not None and v < f.lo:
            raise SchemaError(f"{where}: '{f.key}' = {v} is below {f.lo}")
        if f.hi is not None and v > f.hi:
            raise SchemaError(f"{where}: '{f.key}' = {v} is above {f.hi}")
    if f.choices is not None and v not in f.choices:
        raise SchemaError(f"{where}: '{f.key}' must be one of {list(f.choices)}, not {v!r}")


def _names(types) -> str:
    return "/".join(t.__name__ for t in types)


def check_fields(where: str, d: dict, fields: tuple, extra_ok: tuple = ()) -> None:
    """Required keys present, no unknown keys, types/ranges/choices OK,
    one_of groups satisfied."""
    spec = {f.key: f for f in fields}
    allowed = set(spec) | COMMON_KEYS | set(extra_ok)
    for k in d:
        if k.startswith("_"):
            continue
        if k not in allowed:
            raise SchemaError(f"{where}: unknown key '{k}' (allowed: {sorted(allowed)})")
    groups = {}
    for f in fields:
        if f.one_of:
            groups.setdefault(f.one_of, []).append(f.key)
        if f.key in d:
            _check_value(where, f, d[f.key])
        elif f.required:
            raise SchemaError(f"{where}: missing required key '{f.key}'")
    for g, keys in groups.items():
        present = [k for k in keys if k in d and d[k] is not None]
        if len(present) != 1:
            raise SchemaError(f"{where}: '{g}' needs exactly one of {keys}, got {present}")


def check_common(where: str, d: dict, kind: str, name: str) -> None:
    if d.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(f"{where}: schema_version must be {SCHEMA_VERSION}")
    if d.get("kind") != kind:
        raise SchemaError(f"{where}: kind is {d.get('kind')!r}, expected {kind!r}")
    if d.get("name") != name:
        raise SchemaError(f"{where}: name {d.get('name')!r} must equal the file stem {name!r}")
    if not NAME_RE.match(name):
        raise SchemaError(f"{where}: name {name!r} must match {NAME_RE.pattern}")
    for k in ("sources", "notes", "assumptions"):
        if k in d and not (isinstance(d[k], list) and all(isinstance(s, str) for s in d[k])):
            raise SchemaError(f"{where}: '{k}' must be a list of strings")
    if "provenance" in d and not (isinstance(d["provenance"], dict)
                                  and all(isinstance(v, str) for v in d["provenance"].values())):
        raise SchemaError(f"{where}: 'provenance' must map field names to strings")
    if "summary" in d and not (isinstance(d["summary"], dict)
                               and all(isinstance(v, str) for v in d["summary"].values())):
        raise SchemaError(f"{where}: 'summary' must map short keys to strings")
    if "label" in d and not isinstance(d["label"], str):
        raise SchemaError(f"{where}: 'label' must be a string")
    if "extends" in d and not isinstance(d["extends"], str):
        raise SchemaError(f"{where}: 'extends' must be the name of a {kind}")


# ---------------------------------------------------------------------------
# Per-kind specifications
# ---------------------------------------------------------------------------
FIELDS = {
    "band": (Field("bands", DICT, required=True),),
    "ld_table": (Field("law", STR, required=True, choices=("linear",)),
                 Field("table_nm", LIST, required=True)),
    "site": (Field("latitude_deg", NUM, required=True, lo=-90, hi=90),
             Field("longitude_deg", NUM, lo=-180, hi=360),
             Field("elevation_m", NUM, lo=-500, hi=9000)),
    "telescope": (Field("diameter_m", NUM, required=True, lo=0),
                  Field("throughput", NUM, lo=0, hi=1),
                  Field("collecting_area_m2", NUM, lo=0, nullable=True)),
    "detector": (Field("pde_table_nm", (list, dict), required=False),
                 Field("timing", DICT, required=False),
                 Field("dead_time_ns", NUM, lo=0),
                 Field("dark_cps_per_pixel", NUM, lo=0),
                 Field("n_pixels", (int,), lo=1),
                 Field("readout", STR, choices=("timetag", "correlator")),
                 Field("max_total_cps", NUM, lo=0, nullable=True),
                 Field("max_cps_per_pixel", NUM, lo=0, nullable=True)),
    "spectrograph": (Field("lambda_min_nm", NUM, lo=0),
                     Field("lambda_max_nm", NUM, lo=0),
                     Field("centre_nm", NUM, lo=0),
                     Field("n_channels", (int,), lo=1),
                     Field("resolving_power", NUM, lo=1, nullable=True),
                     Field("throughput", NUM, lo=0, hi=1),
                     Field("frame", STR, choices=("vacuum", "air"))),
    "backend": (Field("model", STR, required=True, choices=("counting", "analog")),
                # counting
                Field("spectrograph", REF), Field("detector", REF, nullable=True),
                Field("polarization_mode", STR, choices=("unpolarized", "pbs", "single_pol")),
                Field("kw", DICT),
                # analog (IACT)
                Field("lambda_nm", NUM, lo=0), Field("dlambda_nm", NUM, lo=0),
                Field("alpha", NUM, lo=0, hi=1), Field("q", NUM, lo=0, hi=1),
                Field("b_el_hz", NUM, lo=0), Field("noise_factor", NUM, lo=0),
                Field("sigma_spec", NUM, lo=0), Field("beta", NUM, lo=0),
                Field("time_resolution_ns", NUM, lo=0), Field("anchor", DICT, nullable=True)),
    "array": (Field("site", REF, nullable=True),
              Field("stations", LIST), Field("generator", DICT)),
    "resource": (Field("path", STR, required=True), Field("env_override", STR, nullable=True),
                 Field("pattern", STR), Field("contents", STR), Field("remote", DICT)),
    "target": (Field("type", STR, required=True, choices=("binary", "single", "uniform_disk")),
               # binary
               Field("primary", DICT), Field("secondary", DICT),
               Field("period_days", NUM, lo=0), Field("inclination_deg", NUM, lo=0, hi=180),
               Field("distance_pc", NUM, lo=0), Field("semimajor_au", NUM, lo=0),
               Field("eccentricity", NUM, lo=0, hi=0.999), Field("arg_periastron_deg", NUM),
               Field("mag_anchors", LIST), Field("node_pa_deg", NUM, nullable=True),
               Field("dec_deg", NUM, lo=-90, hi=90, nullable=True),
               Field("ra_hours", NUM, lo=0, hi=24, nullable=True),
               Field("doppler", BOOL), Field("a_v", NUM, lo=0),
               Field("atmosphere", DICT, nullable=True), Field("skip_omega_test", BOOL),
               # single
               Field("star", DICT), Field("theta_ld_mas", NUM, lo=0), Field("v_mag", NUM),
               Field("ellipse", DICT, nullable=True),
               # uniform disk
               Field("theta_mas", NUM, lo=0), Field("mag_ab", NUM), Field("ld_u", NUM, lo=0, hi=1),
               Field("reference_results", DICT)),
    "campaign": (Field("runner", STR, required=True,
                       choices=("g3", "g2", "g3_campaign", "chromatic", "montecarlo",
                                "scale", "suite", "nightmovie", "specmovie")),
                 Field("target", STR), Field("targets", LIST), Field("array", STR),
                 Field("array_params", DICT), Field("instrument", DICT),
                 Field("backends", LIST), Field("track_backends", LIST),
                 Field("phases", DICT), Field("night", DICT), Field("statistics", LIST),
                 Field("r_bin", NUM, lo=1), Field("precision_target", NUM, lo=0),
                 Field("vis_method", STR), Field("render_grid", STR), Field("pupils", (bool, list)),
                 Field("atmosphere", DICT, nullable=True), Field("outputs", DICT),
                 Field("cache_dir", STR, nullable=True), Field("options", DICT),
                 Field("members", LIST), Field("jobs", (int,), lo=1)),
}

STAR_FIELDS = (Field("label", STR, required=True), Field("mass_msun", NUM, required=True, lo=0),
               Field("radius_rsun", NUM, lo=0), Field("teff_k", NUM, required=True, lo=0),
               Field("ld_table", REF, required=True), Field("logg", NUM, nullable=True),
               Field("metallicity", NUM), Field("vsini_kms", NUM, lo=0, nullable=True),
               Field("radius_ref", STR, choices=("tau1", "outer")))
STATION_FIELDS = (Field("name", STR, required=True), Field("east_m", NUM, required=True),
                  Field("north_m", NUM, required=True), Field("up_m", NUM),
                  Field("telescope", REF, required=True), Field("detector", REF, required=True))
GENERATOR_FIELDS = (Field("type", STR, required=True, choices=("equilateral", "pair")),
                    Field("side_m", NUM, lo=0), Field("baseline_m", NUM, lo=0),
                    Field("pa_deg", NUM), Field("min_spacing_m", NUM, lo=0),
                    Field("station_names", LIST), Field("telescope", REF, required=True),
                    Field("detector", REF, required=True),
                    Field("telescope2", REF), Field("detector2", REF))
TIMING_KEYS = ("jitter_fwhm_ps", "jitter_sigma_ps", "pair_sigma_ps", "electronic_bandwidth_hz")
ANCHOR_FIELDS = (Field("band", STR), Field("wavelength_nm", NUM, lo=0),
                 Field("mag_ab", NUM), Field("mag_vega", NUM), Field("note", STR))
ATMOSPHERE_FIELDS = (Field("resource", STR, required=True), Field("model", STR),
                     Field("range", STR), Field("allow_extrapolation", BOOL),
                     Field("which", LIST))


def _check_star(where, s):
    check_fields(where, s, STAR_FIELDS, extra_ok=("sources", "notes", "provenance", "assumptions"))


def _check_one_anchor(where, a, extra=()):
    if not isinstance(a, dict):
        raise SchemaError(f"{where} must be an object")
    check_fields(where, a, ANCHOR_FIELDS + tuple(extra))
    if ("band" in a) == ("wavelength_nm" in a):
        raise SchemaError(f"{where} needs 'band' or 'wavelength_nm' (one of them)")
    if ("mag_ab" in a) == ("mag_vega" in a):
        raise SchemaError(f"{where} needs 'mag_ab' or 'mag_vega' (one of them)")
    if "mag_vega" in a and "band" not in a:
        raise SchemaError(f"{where}: mag_vega needs a 'band' for the AB offset")


def _check_anchors(where, anchors):
    for k, a in enumerate(anchors):
        _check_one_anchor(f"{where}: mag_anchors[{k}]", a)


def check_kind(where: str, kind: str, d: dict) -> None:
    """Full validation of a merged (extends-resolved) definition."""
    if kind not in FIELDS:
        raise SchemaError(f"{where}: unknown kind {kind!r}")
    check_fields(where, d, FIELDS[kind])
    if kind == "ld_table":
        t = d["table_nm"]
        if not all(isinstance(p, list) and len(p) == 2 and all(isinstance(x, NUM) for x in p) for p in t):
            raise SchemaError(f"{where}: table_nm must be [[nm, u], ...]")
        if any(t[i][0] >= t[i + 1][0] for i in range(len(t) - 1)):
            raise SchemaError(f"{where}: table_nm wavelengths must increase")
    elif kind == "band":
        for b, v in d["bands"].items():
            if not isinstance(v, dict) or "wavelength_nm" not in v:
                raise SchemaError(f"{where}: band {b!r} needs 'wavelength_nm'")
            check_fields(f"{where}: band {b!r}", v, (Field("wavelength_nm", NUM, required=True, lo=0),
                                                     Field("vega_to_ab", NUM), Field("fwhm_nm", NUM, lo=0),
                                                     Field("system", STR)))
    elif kind == "detector":
        for k in ("pde_table_nm", "timing", "dead_time_ns", "dark_cps_per_pixel"):
            if k not in d:
                raise SchemaError(f"{where}: missing required key '{k}'")
        pde = d["pde_table_nm"]
        if isinstance(pde, dict):
            check_fields(f"{where}: pde_table_nm", pde,
                         (Field("flat", NUM, required=True, lo=0, hi=1),
                          Field("range_nm", LIST, required=True)))
        elif not all(isinstance(p, list) and len(p) == 2 for p in pde):
            raise SchemaError(f"{where}: pde_table_nm must be [[nm, pde], ...] or {{flat, range_nm}}")
        present = [k for k in TIMING_KEYS if k in d["timing"]]
        unknown = [k for k in d["timing"] if k not in TIMING_KEYS and not k.startswith("_")]
        if unknown:
            raise SchemaError(f"{where}: timing has unknown key(s) {unknown}")
        if len(present) != 1:
            raise SchemaError(f"{where}: 'timing' needs exactly one of {list(TIMING_KEYS)}, got {present}")
    elif kind == "spectrograph":
        if "centre_nm" in d:
            if "lambda_min_nm" in d or "lambda_max_nm" in d:
                raise SchemaError(f"{where}: give centre_nm or lambda_min_nm/lambda_max_nm, not both")
            if d.get("n_channels") is None or d.get("resolving_power") is None:
                raise SchemaError(f"{where}: a centred window needs n_channels and resolving_power")
        else:
            if "lambda_min_nm" not in d or "lambda_max_nm" not in d:
                raise SchemaError(f"{where}: give lambda_min_nm and lambda_max_nm (or centre_nm)")
            if d["lambda_max_nm"] <= d["lambda_min_nm"]:
                raise SchemaError(f"{where}: lambda_max_nm must exceed lambda_min_nm")
            if d.get("n_channels") is None and d.get("resolving_power") is None:
                raise SchemaError(f"{where}: give n_channels and/or resolving_power")
    elif kind == "resource":
        if "remote" in d:
            check_fields(f"{where}: remote", d["remote"],
                         (Field("base_url", STR, required=True), Field("manifest", STR),
                          Field("default_range", STR)))
    elif kind == "backend":
        if d["model"] == "counting":
            if "spectrograph" not in d:
                raise SchemaError(f"{where}: counting backend needs 'spectrograph'")
        else:
            for k in ("lambda_nm", "dlambda_nm", "alpha", "q", "b_el_hz"):
                if k not in d:
                    raise SchemaError(f"{where}: analog backend needs '{k}'")
            if d.get("anchor") is not None:
                _check_one_anchor(f"{where}: anchor", d["anchor"],
                                  (Field("star", STR, required=True),
                                   Field("sigma_vis2", NUM, required=True, lo=0),
                                   Field("t_s", NUM, required=True, lo=0)))
    elif kind == "array":
        has_st, has_gen = "stations" in d, "generator" in d
        if has_st == has_gen:
            raise SchemaError(f"{where}: give either 'stations' or 'generator'")
        if has_st:
            if len(d["stations"]) < 2:
                raise SchemaError(f"{where}: an array needs at least two stations")
            for k, s in enumerate(d["stations"]):
                check_fields(f"{where}: stations[{k}]", s, STATION_FIELDS)
        else:
            g = d["generator"]
            check_fields(f"{where}: generator", g, GENERATOR_FIELDS)
            need = "side_m" if g["type"] == "equilateral" else "baseline_m"
            if need not in g:
                raise SchemaError(f"{where}: generator type {g['type']!r} needs '{need}'")
    elif kind == "target":
        t = d["type"]
        if t == "binary":
            for k in ("primary", "secondary", "period_days", "inclination_deg", "distance_pc",
                      "semimajor_au", "mag_anchors"):
                if k not in d:
                    raise SchemaError(f"{where}: binary target needs '{k}'")
            _check_star(f"{where}: primary", d["primary"])
            _check_star(f"{where}: secondary", d["secondary"])
            for s in (d["primary"], d["secondary"]):
                if "radius_rsun" not in s:
                    raise SchemaError(f"{where}: binary components need 'radius_rsun'")
        elif t == "single":
            for k in ("star", "theta_ld_mas", "v_mag", "dec_deg", "ra_hours", "distance_pc"):
                if k not in d:
                    raise SchemaError(f"{where}: single target needs '{k}'")
            _check_star(f"{where}: star", d["star"])
            if "radius_rsun" in d["star"]:
                raise SchemaError(f"{where}: a single star's radius is derived from theta_ld_mas "
                                  f"and distance_pc; do not give radius_rsun")
            if d.get("ellipse") is not None:
                check_fields(f"{where}: ellipse", d["ellipse"],
                             (Field("theta_minor_mas", NUM, lo=0), Field("theta_major_mas", NUM, lo=0),
                              Field("axis_ratio", NUM, required=True, lo=1),
                              Field("pa_deg", NUM, required=True)),
                             extra_ok=("sources", "notes"))
        else:
            for k in ("theta_mas", "mag_ab", "dec_deg"):
                if k not in d:
                    raise SchemaError(f"{where}: uniform_disk target needs '{k}'")
        if "mag_anchors" in d:
            _check_anchors(where, d["mag_anchors"])
        if d.get("atmosphere") is not None:
            check_fields(f"{where}: atmosphere", d["atmosphere"], ATMOSPHERE_FIELDS)
    elif kind == "campaign":
        r = d["runner"]
        if r == "suite":
            if not d.get("members"):
                raise SchemaError(f"{where}: a suite needs 'members'")
            for k, m in enumerate(d["members"]):
                if isinstance(m, dict):
                    check_fields(f"{where}: members[{k}]", m,
                                 (Field("campaign", STR, required=True), Field("suffix", STR),
                                  Field("set", DICT)))
                elif not isinstance(m, str):
                    raise SchemaError(f"{where}: members[{k}] must be a name or an object")
        elif r in ("g3", "g3_campaign"):
            for k in ("target", "array"):
                if k not in d:
                    raise SchemaError(f"{where}: runner {r!r} needs '{k}'")
            if not d.get("backends"):
                raise SchemaError(f"{where}: runner {r!r} needs 'backends'")
        elif r == "g2":
            if "instrument" not in d or not d.get("backends"):
                raise SchemaError(f"{where}: runner 'g2' needs 'instrument' and 'backends'")
            inst = d["instrument"]
            if ("array" in inst) == ("telescope" in inst):
                raise SchemaError(f"{where}: instrument needs 'array' or 'telescope' (one of them)")
        elif r in ("chromatic", "scale"):
            if not (d.get("target") or d.get("targets")):
                raise SchemaError(f"{where}: runner {r!r} needs 'target' or 'targets'")
            if "instrument" not in d:
                raise SchemaError(f"{where}: runner {r!r} needs 'instrument'")
        elif r in ("montecarlo", "nightmovie"):
            for k in ("target", "instrument"):
                if k not in d:
                    raise SchemaError(f"{where}: runner {r!r} needs '{k}'")
