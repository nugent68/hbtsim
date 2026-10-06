"""Dataclass -> JSON-ready dict, canonical JSON text and content hashes.

Independent of hbtsim.catalog (so the physics modules may use it): the
campaign cache of hbtsim.snr3 keys its entries on content_hash() of the
system / array / backend objects instead of their repr(), and the
catalog's `dump` command checks that build(to_json(obj)) round-trips.

Two modes:
  mode="config"  -- a dict that mirrors the dataclass fields (tuples ->
                    lists, numpy scalars -> float, nested dataclasses
                    recursively).  Model-atmosphere tables (FluxTable /
                    LDProfile) are not configuration and raise.
  mode="key"     -- the same, but tables are replaced by a small record
                    {"source", "sha1"} hashing their arrays, so objects
                    carrying attached atmospheres still get a stable key.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np


def _is_table(obj) -> bool:
    return type(obj).__name__ in ("FluxTable", "LDProfile")


def _table_record(obj) -> dict:
    h = hashlib.sha1()
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if isinstance(v, np.ndarray):
            h.update(f.name.encode())
            h.update(np.ascontiguousarray(v).tobytes())
        else:
            h.update(f"{f.name}={v!r}".encode())
    return {"__table__": type(obj).__name__, "source": getattr(obj, "source", ""),
            "sha1": h.hexdigest()}


def to_json(obj, mode: str = "config"):
    """Recursively convert dataclasses / tuples / numpy values to plain
    JSON-compatible Python objects (dict / list / float / int / str /
    bool / None)."""
    if mode not in ("config", "key"):
        raise ValueError(f"mode must be 'config' or 'key', not {mode!r}")
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            raise ValueError("non-finite float cannot be serialized to JSON")
        return obj
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return to_json(obj.item(), mode)
    if isinstance(obj, np.ndarray):
        return to_json(obj.tolist(), mode)
    if _is_table(obj):
        if mode == "config":
            raise ValueError(f"{type(obj).__name__} tables are data, not configuration; "
                             f"use mode='key' or serialize a table pointer instead")
        return _table_record(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {"__class__": type(obj).__name__}
        for f in dataclasses.fields(obj):
            if not f.init or f.name.startswith("_"):
                continue
            out[f.name] = to_json(getattr(obj, f.name), mode)
        return out
    if isinstance(obj, dict):
        return {str(k): to_json(v, mode) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v, mode) for v in obj]
    raise TypeError(f"cannot serialize {type(obj).__name__} to JSON")


def canonical_json(obj, mode: str = "key") -> str:
    """Deterministic JSON text: sorted keys, no whitespace, no NaN."""
    return json.dumps(to_json(obj, mode), sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def content_hash(obj, mode: str = "key", length: int = 16) -> str:
    """Hex SHA-1 (truncated) of canonical_json(obj)."""
    return hashlib.sha1(canonical_json(obj, mode).encode()).hexdigest()[:length]
