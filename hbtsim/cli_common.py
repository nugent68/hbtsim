"""Shared command-line options of the hbtsim tools: which catalog
directories to read, which target, which instrument, and whether to
attach model atmospheres.

    p = argparse.ArgumentParser()
    add_catalog_options(p)                 # --config-dir, --target, --newera-dir, ...
    add_instrument_options(p)              # --instrument | --telescope/--detector/--spectrograph
    args = p.parse_args()
    cat = catalog_from(args)
    target = resolve_target(cat, args)     # with NewEra tables unless --no-newera
    inst = resolve_instrument(cat, args)   # .telescope, .detector, .baselines_m, .name
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace

import numpy as np

from .catalog import Catalog


def add_catalog_options(p: argparse.ArgumentParser, *, target: bool = True,
                        target_default: str = "betaaur", newera: bool = True) -> None:
    g = p.add_argument_group("catalog")
    g.add_argument("--config-dir", action="append", default=[], metavar="DIR",
                   help="extra catalog directory (repeatable; later ones override earlier "
                        "ones and $HBTSIM_CONFIG_PATH by name)")
    if target:
        g.add_argument("--target", "--system", dest="target", default=target_default,
                       help=f"catalog target name (`hbtsim catalog list target`; default {target_default})")
    if newera:
        g.add_argument("--newera-dir", default=None, metavar="DIR",
                       help="directory of binned NewEra tables (default: the target's atmosphere "
                            "resource, i.e. $HBTSIM_NEWERA_DIR or data/newera)")
        g.add_argument("--no-newera", action="store_true",
                       help="blackbody + linear limb darkening only")
        g.add_argument("--allow-extrapolation", action="store_true",
                       help="clamp a star just outside the NewEra grid onto its edge model")


def add_instrument_options(p: argparse.ArgumentParser, *, telescope_default: str = "c2pu_1m",
                           baselines: bool = True) -> None:
    g = p.add_argument_group("instrument",
                             "an array by name, or a telescope + detector (+ spectrograph) by name")
    g.add_argument("--instrument", default=None, metavar="ARRAY",
                   help="catalog array: telescope and detector of its first station, baselines "
                        "from its station pairs (`hbtsim catalog list array`)")
    g.add_argument("--telescope", default=telescope_default, metavar="NAME",
                   help=f"catalog telescope (default {telescope_default})")
    g.add_argument("--detector", default=None, metavar="NAME",
                   help="catalog detector (default: spad_lambda, or spad_lambda_ng with "
                        "--readout correlator)")
    g.add_argument("--spectrograph", default=None, metavar="NAME",
                   help="catalog spectrograph (default: built from --channels / --lambda-min / "
                        "--lambda-max / --resolving-power)")
    g.add_argument("--diameter", type=float, default=None, metavar="M",
                   help="override the telescope diameter [m]")
    g.add_argument("--throughput", type=float, default=None,
                   help="override the telescope throughput (optics + atmosphere, excl. PDE)")
    if baselines:
        g.add_argument("--baseline", type=float, nargs="+", default=None, metavar="M",
                       help="baseline(s) [m] (default: the array's pairs, else 50)")


def catalog_from(args) -> Catalog:
    return Catalog(paths=getattr(args, "config_dir", None) or None)


def resolve_target(cat: Catalog, args, verbose: bool = True):
    """The catalog target, with model atmospheres attached unless
    --no-newera (the target's `atmosphere` pointer, or --newera-dir)."""
    name = args.target
    target = cat.load_target(name)
    if getattr(args, "no_newera", False):
        return target
    newera_dir = getattr(args, "newera_dir", None)
    allow = getattr(args, "allow_extrapolation", False) or None
    target, report = cat.attach_atmosphere(target, name=name, newera_dir=newera_dir,
                                           allow_extrapolation=allow)
    if verbose and report:
        if isinstance(report, dict):
            for star, what in report.items():
                print(f"NewEra [{star}]: {what}")
        else:
            print(f"NewEra: {report}")
    return target


@dataclass(frozen=True)
class Instrument:
    name: str
    telescope: object
    detector: object
    baselines_m: tuple = ()          # from the array's station pairs ("" when a telescope)
    array: object = None


def resolve_instrument(cat: Catalog, args, readout: str | None = None) -> Instrument:
    """Telescope and detector from --instrument (an array) or --telescope /
    --detector, with --diameter / --throughput overrides; readout picks the
    default detector (spad_lambda or spad_lambda_ng) when none is named."""
    if getattr(args, "instrument", None):
        arr = cat.load_array(args.instrument)
        st = arr.stations[0]
        tel, det = st.telescope, st.detector
        baselines = tuple(float(np.hypot(*b)) for _, _, b in arr.pairs())
        name = args.instrument
    else:
        arr = None
        tel = cat.load_telescope(args.telescope)
        name = args.telescope
        baselines = ()
        det = None
    if getattr(args, "detector", None):
        det = cat.load_detector(args.detector)
    elif det is None:
        det = cat.load_detector("spad_lambda_ng" if readout == "correlator" else "spad_lambda")
    over = {}
    if getattr(args, "diameter", None) is not None:
        over["diameter_m"] = args.diameter
    if getattr(args, "throughput", None) is not None:
        over["throughput"] = args.throughput
    if over:
        tel = replace(tel, **over, name=f"{tel.name} (custom)")
        name = "custom"
    return Instrument(name, tel, det, baselines, arr)


def output_stem(target: str, instrument: str, backend: str | None = None) -> str:
    """`<target>_<instrument>[_<backend>]`, the common stem of output files."""
    parts = [target, instrument] + ([backend] if backend else [])
    return "_".join(p.replace(" ", "-").replace("/", "-") for p in parts)


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
