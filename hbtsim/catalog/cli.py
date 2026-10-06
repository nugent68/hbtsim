"""`hbtsim catalog` / `hbtsim-catalog`: list, show, validate, dump, paths."""

from __future__ import annotations

import argparse
import json
import sys

from . import Catalog, KINDS, kind_dir


def add_config_dir_option(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config-dir", action="append", default=[], metavar="DIR",
                   help="extra catalog directory (repeatable; later ones override earlier "
                        "ones and $HBTSIM_CONFIG_PATH by name)")


def catalog_from_args(args) -> Catalog:
    return Catalog(paths=args.config_dir)


def _kind_arg(p, nargs=None):
    p.add_argument("kind", choices=KINDS, nargs=nargs)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="hbtsim catalog", description=__doc__)
    add_config_dir_option(p)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list", help="names per kind"); _kind_arg(s, nargs="?")
    s.add_argument("--long", action="store_true", help="with labels and source files")
    s = sub.add_parser("show", help="the merged definition (notes, sources, provenance)")
    _kind_arg(s); s.add_argument("name")
    s = sub.add_parser("validate", help="parse, validate and build every definition")
    s.add_argument("--no-build", action="store_true", help="schema checks only")
    s = sub.add_parser("dump", help="the built object as JSON (what the code sees)")
    _kind_arg(s); s.add_argument("name")
    s.add_argument("--set", action="append", default=[], metavar="K=V",
                   help="generator parameter for arrays, e.g. side_m=12")
    sub.add_parser("paths", help="the search path")
    args = p.parse_args(argv)
    cat = catalog_from_args(args)

    if args.cmd == "paths":
        for r in cat.roots:
            print(r)
        return 0
    if args.cmd == "list":
        kinds = [args.kind] if args.kind else list(KINDS)
        for kind in kinds:
            names = cat.list_names(kind)
            if len(kinds) > 1:
                print(f"{kind_dir(kind)} ({len(names)})")
            for n in names:
                if args.long:
                    d = cat.registry.raw(kind, n)
                    print(f"  {n:32s} {d.get('label', ''):50s} {cat.source_of(kind, n)}")
                else:
                    print(("  " if len(kinds) > 1 else "") + n)
        return 0
    if args.cmd == "show":
        d = cat.raw(args.kind, args.name)
        print(f"# {cat.source_of(args.kind, args.name)}")
        print(json.dumps(d, indent=2))
        return 0
    if args.cmd == "validate":
        errs = cat.validate_all(build=not args.no_build)
        n = sum(len(cat.list_names(k)) for k in KINDS)
        for e in errs:
            print(e, file=sys.stderr)
        print(f"{n} definitions in {len(cat.roots)} director{'y' if len(cat.roots) == 1 else 'ies'}: "
              f"{'OK' if not errs else f'{len(errs)} error(s)'}")
        return 1 if errs else 0
    if args.cmd == "dump":
        from ..serialize import to_json
        kw = {}
        for item in args.set:
            k, _, v = item.partition("=")
            try:
                kw[k] = json.loads(v)
            except json.JSONDecodeError:
                kw[k] = v
        obj = cat.load(args.kind, args.name, **kw)
        print(json.dumps(to_json(obj, mode="key"), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
