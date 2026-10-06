"""Discovery of definition files, search-path overrides, `extends`
resolution and the per-catalog cache."""

from __future__ import annotations

import difflib
import json
import os
from importlib import resources
from pathlib import Path

from .schema import (KINDS, CatalogError, SchemaError, UnknownNameError, check_common,
                     check_kind)

ENV_VAR = "HBTSIM_CONFIG_PATH"


def shipped_root() -> Path:
    """The package's own configs directory."""
    return Path(resources.files("hbtsim")) / "configs"


def kind_dir(kind: str) -> str:
    return kind + "s"


class Registry:
    """Maps (kind, name) -> definition file, with later search paths
    overriding earlier ones by name."""

    def __init__(self, paths=None, include_shipped: bool = True, env: bool = True):
        roots = []
        if include_shipped:
            roots.append(shipped_root())
        if env:
            for p in os.environ.get(ENV_VAR, "").split(os.pathsep):
                if p.strip():
                    roots.append(Path(p.strip()).expanduser())
        for p in (paths or ()):
            roots.append(Path(p).expanduser())
        self.roots = tuple(roots)
        self._files = {}           # (kind, name) -> Path
        self._scan()
        self._raw = {}             # (kind, name) -> merged dict

    def _scan(self) -> None:
        for root in self.roots:
            if not root.is_dir():
                raise CatalogError(f"config directory {root} does not exist")
            for kind in KINDS:
                d = root / kind_dir(kind)
                if not d.is_dir():
                    continue
                for f in sorted(d.glob("*.json")):
                    self._files[(kind, f.stem)] = f        # later roots win

    # -- names ---------------------------------------------------------------
    def names(self, kind: str) -> tuple:
        if kind not in KINDS:
            raise CatalogError(f"unknown kind {kind!r}; kinds are {list(KINDS)}")
        return tuple(sorted(n for (k, n) in self._files if k == kind))

    def path(self, kind: str, name: str) -> Path:
        try:
            return self._files[(kind, name)]
        except KeyError:
            close = difflib.get_close_matches(name, self.names(kind), n=3)
            hint = f"; did you mean {close}?" if close else ""
            roots = ", ".join(str(r) for r in self.roots)
            raise UnknownNameError(f"no {kind} named {name!r}{hint} (searched: {roots})") from None

    # -- raw definitions -----------------------------------------------------
    def raw(self, kind: str, name: str, _chain=()) -> dict:
        """The validated definition with `extends` resolved."""
        key = (kind, name)
        if key in self._raw:
            return self._raw[key]
        path = self.path(kind, name)
        where = f"{kind_dir(kind)}/{path.name}"
        try:
            with open(path) as fh:
                d = json.load(fh)
        except json.JSONDecodeError as e:
            raise SchemaError(f"{where}: invalid JSON ({e})") from None
        if not isinstance(d, dict):
            raise SchemaError(f"{where}: top level must be an object")
        check_common(where, d, kind, name)
        parent = d.get("extends")
        if parent is not None:
            if parent in _chain or parent == name:
                raise SchemaError(f"{where}: 'extends' cycle {list(_chain) + [name, parent]}")
            base = self.raw(kind, parent, _chain + (name,))
            d = _merge(base, d)
        check_kind(where, kind, d)
        self._raw[key] = d
        return d


def _merge(base: dict, child: dict) -> dict:
    """Shallow merge: child keys win; sources/notes/assumptions concatenate;
    provenance maps merge; `extends` is dropped from the result."""
    out = dict(base)
    for k, v in child.items():
        if k == "extends":
            continue
        if k in ("sources", "notes", "assumptions") and k in base:
            out[k] = list(base[k]) + [x for x in v if x not in base[k]]
        elif k == "provenance" and k in base:
            out[k] = {**base[k], **v}
        else:
            out[k] = v
    out.pop("extends", None)
    return out
