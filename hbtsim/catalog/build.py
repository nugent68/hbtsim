"""Builders: validated definition dict -> hbtsim runtime object.

Every builder takes the Catalog (for references by name) and the merged,
validated definition.  Numbers are copied from the files verbatim; the
only arithmetic is the documented derivations (timing conversions, the
single-star radius from theta_LD and distance, generated array layouts,
Vega -> AB offsets), written so that they reproduce the former Python
presets bit for bit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import numpy as np

from ..geometry import Site
from ..params import MAS, PARSEC, R_SUN, BinarySystem, DiskTarget, Star
from ..snr import FWHM_TO_SIGMA, Detector, Spectrograph, Telescope
from .schema import CatalogError, SchemaError, check_kind

# ---------------------------------------------------------------------------
# Small runtime records that have no home in the physics modules
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Resource:
    """A directory of data files (NewEra tables) the catalog points at, with
    the remote it can be fetched from (hbtsim.data)."""
    name: str
    path: str
    env_override: str | None = None
    pattern: str = "*"
    contents: str = ""
    remote: tuple = ()               # (("base_url", ...), ("manifest", ...), ...)

    @property
    def remote_dict(self) -> dict:
        return dict(self.remote)

    def candidates(self) -> list:
        """Directories searched in order: the env override, the file's path
        (relative to the current directory), the user cache."""
        from ..data import data_dir
        out = []
        if self.env_override and os.environ.get(self.env_override):
            out.append(os.environ[self.env_override])
        out.append(self.path)
        out.append(str(data_dir() / self.name))
        return out

    @property
    def resolved_path(self) -> str:
        """The env override when set (authoritative, even if the directory
        does not exist yet); else the first existing candidate; else the
        cache directory (where a fetch would put the tables)."""
        if self.env_override and os.environ.get(self.env_override):
            return os.environ[self.env_override]
        cands = self.candidates()
        for c in cands:
            if os.path.isdir(c):
                return c
        return cands[-1]

    def exists(self) -> bool:
        return os.path.isdir(self.resolved_path)


@dataclass(frozen=True)
class Campaign:
    """A resolved campaign: runner name, built objects, the raw merged
    definition (runner-specific options live there)."""
    name: str
    label: str
    runner: str
    spec: dict
    targets: tuple = ()
    array: object = None
    telescope: object = None
    backends: tuple = ()
    track_backends: tuple = ()
    members: tuple = ()

    @property
    def target(self):
        return self.targets[0] if self.targets else None

    def option(self, key: str, default=None):
        """Dotted lookup into the raw definition: option("night.block_minutes")."""
        d = self.spec
        for part in key.split("."):
            if not isinstance(d, dict) or part not in d:
                return default
            d = d[part]
        return d


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _given(d: dict, keys, rename=None) -> dict:
    """{dataclass field: value} for the keys present in d."""
    rename = rename or {}
    return {rename.get(k, k): d[k] for k in keys if k in d}


def _ref(cat, kind: str, value, where: str):
    """A name -> cat.load(kind, name); an inline object -> built in place."""
    if isinstance(value, str):
        return cat.load(kind, value)
    if isinstance(value, dict):
        check_kind(f"{where}: inline {kind}", kind, value)
        return BUILDERS[kind](cat, value)
    raise SchemaError(f"{where}: a {kind} must be a name or an object, not {type(value).__name__}")


def _label(d: dict) -> str:
    return d.get("label") or d.get("name", "")


def _pairs(table) -> tuple:
    return tuple((float(a), float(b)) for a, b in table)


def _anchor(cat, a: dict, where: str) -> tuple:
    """(wavelength_nm, mag_ab) from {band|wavelength_nm, mag_ab|mag_vega}."""
    bands = cat.bands()
    if "wavelength_nm" in a:
        lam = float(a["wavelength_nm"])
    else:
        band = a["band"]
        if band not in bands:
            raise CatalogError(f"{where}: unknown band {band!r} (bands: {sorted(bands)})")
        lam = float(bands[band]["wavelength_nm"])
    if "mag_ab" in a:
        mag = float(a["mag_ab"])
    else:
        off = bands[a["band"]].get("vega_to_ab")
        if off is None:
            raise CatalogError(f"{where}: band {a['band']!r} has no vega_to_ab offset for mag_vega")
        mag = a["mag_vega"] + off
    return lam, float(mag)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def build_band(cat, d: dict) -> dict:
    return {b: dict(v) for b, v in d["bands"].items() if not b.startswith("_")}


def build_ld_table(cat, d: dict) -> tuple:
    return _pairs(d["table_nm"])


def build_site(cat, d: dict) -> Site:
    return Site(_label(d), **_given(d, ("latitude_deg", "longitude_deg", "elevation_m")))


def build_telescope(cat, d: dict) -> Telescope:
    return Telescope(name=_label(d),
                     **_given(d, ("diameter_m", "throughput", "collecting_area_m2")))


def jitter_fwhm_ps(timing: dict) -> float:
    """The detector's single-photon timing FWHM [ps] from whichever
    quantity the datasheet or paper gives."""
    if "jitter_fwhm_ps" in timing:
        return float(timing["jitter_fwhm_ps"])
    if "jitter_sigma_ps" in timing:
        return timing["jitter_sigma_ps"] * 2.0 * np.sqrt(2.0 * np.log(2.0))
    if "pair_sigma_ps" in timing:
        # per detector, so the pair (difference) sigma is the quoted value
        return timing["pair_sigma_ps"] / np.sqrt(2.0) / FWHM_TO_SIGMA
    if "electronic_bandwidth_hz" in timing:
        # analog photomultiplier + digitizer: the pair response of a
        # bandwidth-limited correlator (hbtsim.iact)
        from ..iact import analog_pair_fwhm_ps
        return analog_pair_fwhm_ps(timing["electronic_bandwidth_hz"])
    raise SchemaError(f"timing needs one of jitter_fwhm_ps / jitter_sigma_ps / "
                      f"pair_sigma_ps / electronic_bandwidth_hz, got {sorted(timing)}")


def build_detector(cat, d: dict) -> Detector:
    pde = d["pde_table_nm"]
    if isinstance(pde, dict):
        lo, hi = pde["range_nm"]
        table = ((float(lo), float(pde["flat"])), (float(hi), float(pde["flat"])))
    else:
        table = _pairs(pde)
    return Detector(name=_label(d), pde_table_nm=table,
                    jitter_fwhm_ps=jitter_fwhm_ps(d["timing"]),
                    **_given(d, ("dead_time_ns", "dark_cps_per_pixel", "n_pixels", "readout",
                                 "max_total_cps")))


def build_spectrograph(cat, d: dict) -> Spectrograph:
    name = _label(d)
    kw = _given(d, ("throughput", "frame"))
    if "centre_nm" in d:
        # n geometric channels at constant R centred on centre_nm
        n, R, lam0 = d["n_channels"], d["resolving_power"], d["centre_nm"]
        q = (2 * R + 1) / (2 * R - 1)
        lo = lam0 * q ** (-n / 2)
        return Spectrograph(lambda_min_nm=lo, lambda_max_nm=lo * q**n, n_channels=n,
                            resolving_power=R, name=name, **kw)
    if d.get("n_channels") is None:
        kw.pop("frame", None)
        return Spectrograph.from_resolving_power(d["resolving_power"], d["lambda_min_nm"],
                                                 d["lambda_max_nm"], name=name, **kw)
    return Spectrograph(lambda_min_nm=d["lambda_min_nm"], lambda_max_nm=d["lambda_max_nm"],
                        n_channels=d["n_channels"], resolving_power=d.get("resolving_power"),
                        name=name, **kw)


def build_backend(cat, d: dict):
    where = f"backends/{d['name']}" if "name" in d else "inline backend"
    label = _label(d)
    if d["model"] == "counting":
        from ..snr3 import Backend
        det = d.get("detector")
        return Backend(name=label,
                       spectrograph=_ref(cat, "spectrograph", d["spectrograph"], where),
                       detector=None if det is None else _ref(cat, "detector", det, where),
                       polarization_mode=d.get("polarization_mode", "unpolarized"),
                       kw=dict(d.get("kw", {})))
    from ..iact import IACTBackend, PrecisionAnchor
    anchor = None
    if d.get("anchor") is not None:
        a = d["anchor"]
        _, mag_ab = _anchor(cat, a, f"{where}: anchor")
        anchor = PrecisionAnchor(a["star"], mag_ab, float(a["sigma_vis2"]), float(a["t_s"]))
    return IACTBackend(name=label, anchor=anchor,
                       **_given(d, ("lambda_nm", "dlambda_nm", "alpha", "q", "b_el_hz",
                                    "noise_factor", "sigma_spec", "beta", "time_resolution_ns")))


def _star(cat, s: dict, where: str, radius_rsun=None) -> Star:
    kw = _given(s, ("mass_msun", "logg", "metallicity", "vsini_kms", "radius_ref"))
    if radius_rsun is None:
        radius_rsun = s["radius_rsun"]
    ld = s["ld_table"]
    ld_table = _pairs(ld) if isinstance(ld, list) else _ref(cat, "ld_table", ld, where)
    return Star(s["label"], radius_rsun=radius_rsun, teff=s["teff_k"], ld_table_nm=ld_table, **kw)


def build_target(cat, d: dict):
    where = f"targets/{d.get('name', '?')}"
    label = _label(d)
    anchors = tuple(_anchor(cat, a, where) for a in d.get("mag_anchors", ()))
    t = d["type"]
    if t == "binary":
        return BinarySystem(label, _star(cat, d["primary"], where), _star(cat, d["secondary"], where),
                            mag_anchors=anchors,
                            **_given(d, ("period_days", "inclination_deg", "distance_pc",
                                         "semimajor_au", "eccentricity", "arg_periastron_deg",
                                         "node_pa_deg", "dec_deg", "ra_hours", "doppler", "a_v")))
    if t == "single":
        from ..single import Ellipse, SingleStar
        theta, dist = d["theta_ld_mas"], d["distance_pc"]
        radius_rsun = 0.5 * theta * MAS * dist * PARSEC / R_SUN
        ellipse = None
        if d.get("ellipse") is not None:
            e = d["ellipse"]
            major = e["theta_major_mas"] if "theta_major_mas" in e else e["theta_minor_mas"] * e["axis_ratio"]
            ellipse = Ellipse(major, e["axis_ratio"], e["pa_deg"])
        return SingleStar(label, _star(cat, d["star"], where, radius_rsun=radius_rsun),
                          theta_ld_mas=theta, v_mag=d["v_mag"], dec_deg=d["dec_deg"],
                          ra_hours=d["ra_hours"], distance_pc=dist, mag_anchors=anchors,
                          ellipse=ellipse)
    return DiskTarget(label, **_given(d, ("theta_mas", "mag_ab", "dec_deg", "ld_u", "ra_hours")))


GENERATOR_KEYS = ("type", "side_m", "baseline_m", "pa_deg", "min_spacing_m", "station_names",
                  "telescope", "detector", "telescope2", "detector2")


def build_array(cat, d: dict, **overrides):
    from ..bispectrum import Array, Station
    where = f"arrays/{d.get('name', '?')}"
    site = None if d.get("site") is None else _ref(cat, "site", d["site"], where)
    if "stations" in d:
        if overrides:
            raise CatalogError(f"{where}: an explicit station list takes no generator "
                               f"parameters ({sorted(overrides)})")
        stations = tuple(Station(s["name"], s["east_m"], s["north_m"],
                                 _ref(cat, "telescope", s["telescope"], where),
                                 _ref(cat, "detector", s["detector"], where),
                                 up_m=s.get("up_m", 0.0)) for s in d["stations"])
        return Array(stations, site=site)
    from ..bispectrum import equilateral_array, pair_array
    g = dict(d["generator"])
    bad = sorted(set(overrides) - set(GENERATOR_KEYS) - {"type"})
    if bad:
        raise CatalogError(f"{where}: unknown generator parameter(s) {bad}; "
                           f"allowed: {[k for k in GENERATOR_KEYS if k != 'type']}")
    g.update(overrides)
    tel = g["telescope"] if isinstance(g["telescope"], Telescope) else _ref(cat, "telescope", g["telescope"], where)
    det = g["detector"] if isinstance(g["detector"], Detector) else _ref(cat, "detector", g["detector"], where)
    common = dict(site=site, pa_deg=g.get("pa_deg", 0.0), min_spacing_m=g.get("min_spacing_m"))
    if g["type"] == "equilateral":
        names = tuple(g.get("station_names", ("T1", "T2", "T3")))
        if len(names) != 3:
            raise CatalogError(f"{where}: an equilateral generator needs three station_names")
        return equilateral_array(float(g["side_m"]), tel, det, names=names, **common)
    names = tuple(g.get("station_names", ("T1", "T2")))
    if len(names) != 2:
        raise CatalogError(f"{where}: a pair generator needs two station_names")
    second = {}
    for key, kind in (("telescope2", "telescope"), ("detector2", "detector")):
        v = g.get(key)
        if v is not None:
            second[key] = v if isinstance(v, (Telescope, Detector)) else _ref(cat, kind, v, where)
    return pair_array(float(g["baseline_m"]), tel, det, names=names, **common, **second)


def build_resource(cat, d: dict) -> Resource:
    remote = tuple(sorted((k, v) for k, v in d.get("remote", {}).items() if not k.startswith("_")))
    return Resource(d["name"], remote=remote, **_given(d, ("path", "env_override", "pattern", "contents")))


def build_campaign(cat, d: dict) -> Campaign:
    where = f"campaigns/{d.get('name', '?')}"
    names = ([d["target"]] if "target" in d else []) + list(d.get("targets", ()))
    targets = tuple(cat.load_target(n) for n in names)
    array = None
    if "array" in d:
        array = cat.load_array(d["array"], **d.get("array_params", {}))
    telescope = None
    inst = d.get("instrument", {})
    if "array" in inst:
        array = cat.load_array(inst["array"], **inst.get("array_params", {}))
    if "telescope" in inst:
        telescope = _ref(cat, "telescope", inst["telescope"], where)
    backends = tuple(_ref(cat, "backend", b, where) for b in d.get("backends", ()))
    track = tuple(_ref(cat, "backend", b, where) for b in d.get("track_backends", ()))
    members = tuple(m if isinstance(m, str) else dict(m) for m in d.get("members", ()))
    return Campaign(d["name"], _label(d), d["runner"], d, targets, array,
                    telescope, backends, track, members)


BUILDERS = {
    "band": build_band, "ld_table": build_ld_table, "site": build_site,
    "telescope": build_telescope, "detector": build_detector, "spectrograph": build_spectrograph,
    "backend": build_backend, "target": build_target, "array": build_array,
    "resource": build_resource, "campaign": build_campaign,
}
