"""The hbtsim catalog: targets, telescopes, detectors, spectrographs,
backends, sites, arrays, data resources and campaigns as JSON files.

    from hbtsim.catalog import Catalog, load_target, load_array
    cat = Catalog()                       # shipped files + $HBTSIM_CONFIG_PATH
    spica = cat.load_target("spica")
    vlt = cat.load_array("vlt_ut")
    tri = cat.load_array("eonsii_triangle_paranal", side_m=12.0)

Search path (later entries override earlier ones by (kind, name)):
shipped `hbtsim/configs/<kind>s/*.json`, then the directories in
$HBTSIM_CONFIG_PATH (os.pathsep-separated), then `paths=`.  `hbtsim
catalog list|show|validate|dump|paths` is the command-line view.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .build import BUILDERS, Campaign, Resource
from .registry import ENV_VAR, Registry, kind_dir, shipped_root
from .schema import (KINDS, CatalogError, SchemaError, UnknownNameError, check_common,
                     check_kind)

__all__ = ["Catalog", "Campaign", "Resource", "CatalogError", "SchemaError", "UnknownNameError",
           "KINDS", "ENV_VAR", "default_catalog", "load", "load_target", "load_telescope",
           "load_detector", "load_spectrograph", "load_backend", "load_site", "load_array",
           "load_triangle", "load_ld_table", "load_resource", "load_campaign", "list_names"]


class Catalog:
    def __init__(self, paths=None, include_shipped: bool = True, env: bool = True):
        self.registry = Registry(paths, include_shipped=include_shipped, env=env)
        self._cache = {}

    # -- discovery -------------------------------------------------------------
    @property
    def roots(self) -> tuple:
        return self.registry.roots

    def list_names(self, kind: str) -> tuple:
        return self.registry.names(kind)

    names = list_names

    def raw(self, kind: str, name: str) -> dict:
        """The validated definition with `extends` resolved (a copy)."""
        return json.loads(json.dumps(self.registry.raw(kind, name)))

    def source_of(self, kind: str, name: str) -> Path:
        return self.registry.path(kind, name)

    # -- building ----------------------------------------------------------------
    def load(self, kind: str, name: str, **overrides):
        if kind not in KINDS:
            raise CatalogError(f"unknown kind {kind!r}; kinds are {list(KINDS)}")
        try:
            key = (kind, name, tuple(sorted(overrides.items())))
            hash(key)
        except TypeError:
            key = None
        if key is not None and key in self._cache:
            return self._cache[key]
        d = self.registry.raw(kind, name)
        if overrides and kind != "array":
            raise CatalogError(f"{kind} takes no load-time parameters ({sorted(overrides)})")
        obj = BUILDERS[kind](self, d, **overrides) if overrides else BUILDERS[kind](self, d)
        if key is not None:
            self._cache[key] = obj
        return obj

    def bands(self, name: str = "photometric") -> dict:
        return self.load("band", name)

    def load_ld_table(self, name: str) -> tuple:
        return self.load("ld_table", name)

    def load_site(self, name: str):
        return self.load("site", name)

    def load_telescope(self, name: str):
        return self.load("telescope", name)

    def load_detector(self, name: str):
        return self.load("detector", name)

    def load_spectrograph(self, name: str):
        return self.load("spectrograph", name)

    def load_backend(self, name: str):
        return self.load("backend", name)

    def load_resource(self, name: str) -> Resource:
        return self.load("resource", name)

    def load_array(self, name: str, **generator_params):
        return self.load("array", name, **generator_params)

    def load_triangle(self, name: str, **generator_params):
        from ..bispectrum import Triangle
        arr = self.load_array(name, **generator_params)
        if len(arr.stations) != 3:
            raise CatalogError(f"array {name!r} has {len(arr.stations)} stations; a triangle needs 3")
        return Triangle(arr.stations, arr.site)

    def load_target(self, name: str, *, atmosphere=False, newera_dir=None,
                    allow_extrapolation=None, verbose: bool = False):
        """The target object.  atmosphere=True attaches the model-atmosphere
        tables named by the target's `atmosphere` pointer when the resource
        directory exists (silently kept blackbody + linear law otherwise);
        "require" raises instead; newera_dir overrides the resource."""
        target = self.load("target", name)
        if not atmosphere:
            return target
        target, report = self.attach_atmosphere(target, name=name, newera_dir=newera_dir,
                                                allow_extrapolation=allow_extrapolation,
                                                require=(atmosphere == "require"))
        if verbose and report:
            print(f"{name}: {report}")
        return target

    def attach_atmosphere(self, target, name: str | None = None, newera_dir=None,
                          allow_extrapolation=None, require: bool = False):
        """(target with NewEra tables, report).  The target's `atmosphere`
        pointer (resource, optional explicit model, allow_extrapolation,
        which stars) says what to attach; newera_dir / allow_extrapolation
        override it.  Without a pointer and without newera_dir nothing is
        attached."""
        from .. import sed
        ptr = {}
        if name is not None:
            ptr = dict(self.registry.raw("target", name).get("atmosphere") or {})
        if newera_dir is None and not ptr:
            if require:
                raise CatalogError(f"{getattr(target, 'name', target)} has no `atmosphere` pointer "
                                   f"and no --newera-dir was given")
            return target, "no atmosphere pointer: blackbody + linear limb darkening"
        pattern = "newera_lte*.npz"
        if newera_dir is None:
            res = self.load_resource(ptr["resource"])
            newera_dir, pattern = res.resolved_path, res.pattern
        if allow_extrapolation is None:
            allow_extrapolation = bool(ptr.get("allow_extrapolation", False))
        if not os.path.isdir(newera_dir):
            msg = f"NewEra directory {newera_dir} not found: blackbody + linear limb darkening"
            if require:
                raise CatalogError(msg)
            return target, msg
        if ptr.get("model"):
            # an explicit stand-in model (no interpolation)
            path = os.path.join(newera_dir, f"newera_{ptr['model']}_{ptr.get('range', '380-1000nm_0.02nm')}.npz")
            if not os.path.exists(path):
                msg = f"stand-in table {path} not found"
                if require:
                    raise CatalogError(msg)
                return target, msg
            ft, ld = sed.load_star_tables(path)
            from dataclasses import replace
            if hasattr(target, "star"):
                return replace(target, star=sed.with_tables(target.star, ft, ld)), f"stand-in {ptr['model']}"
            raise CatalogError("explicit stand-in models are only supported for single stars")
        grid = sed.NewEraGrid.scan(newera_dir, pattern)
        if hasattr(target, "primary"):
            which = tuple(ptr.get("which", ("primary", "secondary")))
            return sed.with_newera(target, grid, which=which, allow_extrapolation=allow_extrapolation)
        if hasattr(target, "star"):
            from ..single import attach_newera_single
            return attach_newera_single(target, grid, allow_extrapolation)
        return target, "uniform-disk target: no atmosphere"

    def load_campaign(self, name_or_path: str) -> Campaign:
        if name_or_path.endswith(".json") or os.sep in name_or_path:
            p = Path(name_or_path)
            with open(p) as fh:
                d = json.load(fh)
            where = str(p)
            check_common(where, d, "campaign", p.stem)
            if d.get("extends"):
                from .registry import _merge
                d = _merge(self.registry.raw("campaign", d["extends"]), d)
            check_kind(where, "campaign", d)
            return BUILDERS["campaign"](self, d)
        return self.load("campaign", name_or_path)

    # -- checks --------------------------------------------------------------------
    def validate_all(self, build: bool = True) -> list:
        """Every definition parsed, validated and (build=True) built;
        returns the list of error messages (empty when clean)."""
        errors = []
        for kind in KINDS:
            for name in self.list_names(kind):
                try:
                    self.registry.raw(kind, name)
                    if build:
                        self.load(kind, name)
                except Exception as e:          # noqa: BLE001 -- collect everything
                    errors.append(f"{kind_dir(kind)}/{name}.json: {type(e).__name__}: {e}")
        return errors


_DEFAULT = None


def default_catalog() -> Catalog:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Catalog()
    return _DEFAULT


def load(kind, name, **kw):
    return default_catalog().load(kind, name, **kw)


def list_names(kind):
    return default_catalog().list_names(kind)


def load_target(name, **kw):
    return default_catalog().load_target(name, **kw)


def load_telescope(name):
    return default_catalog().load_telescope(name)


def load_detector(name):
    return default_catalog().load_detector(name)


def load_spectrograph(name):
    return default_catalog().load_spectrograph(name)


def load_backend(name):
    return default_catalog().load_backend(name)


def load_site(name):
    return default_catalog().load_site(name)


def load_array(name, **kw):
    return default_catalog().load_array(name, **kw)


def load_triangle(name, **kw):
    return default_catalog().load_triangle(name, **kw)


def load_ld_table(name):
    return default_catalog().load_ld_table(name)


def load_resource(name):
    return default_catalog().load_resource(name)


def load_campaign(name_or_path):
    return default_catalog().load_campaign(name_or_path)
