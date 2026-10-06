"""`python -m hbtsim` / `hbtsim`: movie | snr | g2spec | catalog | run."""

from __future__ import annotations

import sys

COMMANDS = {
    "movie": ("hbtsim.cli", "main", "render the three-panel orbit / lightcurve / g2(B) movie"),
    "snr": ("hbtsim.snr_cli", "main", "photon budget and g2 SNR for a binary on a two-telescope instrument"),
    "g2spec": ("hbtsim.g2spec", "main", "channelized g2 spectrum movie"),
    "catalog": ("hbtsim.catalog.cli", "main", "list / show / validate / dump the JSON catalog"),
}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: hbtsim <command> [options]\n\ncommands:")
        for k, (_, _, doc) in COMMANDS.items():
            print(f"  {k:10s} {doc}")
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"hbtsim: unknown command {cmd!r}; one of {list(COMMANDS)}", file=sys.stderr)
        return 2
    import importlib
    mod, fn, _ = COMMANDS[cmd]
    out = getattr(importlib.import_module(mod), fn)(rest)
    return 0 if out is None else int(out)


if __name__ == "__main__":
    sys.exit(main())
