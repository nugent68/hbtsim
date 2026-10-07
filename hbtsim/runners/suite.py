"""Runner `suite`: run every member campaign (optionally in parallel
processes) and collect their results.json into one summary."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _member_args(member, opts) -> tuple:
    """(campaign name, suffix, extra argv) for one suite member."""
    if isinstance(member, str):
        return member, "", []
    extra = []
    for k, v in (member.get("set") or {}).items():
        extra += ["--set", f"{k}={json.dumps(v)}"]
    return member["campaign"], member.get("suffix", ""), extra


def run(campaign, cat, opts, out_dir: Path) -> dict:
    members = [_member_args(m, opts) for m in campaign.members]
    cmds = []
    for name, suffix, extra in members:
        argv = [sys.executable, "-m", "hbtsim.run_campaign", name, "--out", str(opts.out_dir), "--quiet"]
        if suffix:
            argv += ["--suffix", suffix]
        for d in cat.roots[1:]:          # user config dirs (the shipped root is implicit)
            argv += ["--config-dir", str(d)]
        if opts.newera_dir:
            argv += ["--newera-dir", opts.newera_dir]
        if opts.no_newera:
            argv.append("--no-newera")
        if opts.allow_extrapolation:
            argv.append("--allow-extrapolation")
        if opts.fetch:
            argv.append("--fetch")
        if not opts.figures:
            argv.append("--no-figures")
        if not opts.track:
            argv.append("--no-track")
        argv += extra
        cmds.append((name, suffix, argv))
    jobs = max(1, int(opts.jobs or 1))
    print(f"suite {campaign.name}: {len(cmds)} members, {jobs} parallel")
    results, failed = {}, []
    running = []
    pending = list(cmds)
    while pending or running:
        while pending and len(running) < jobs:
            name, suffix, argv = pending.pop(0)
            print(f"  start {name}{'_' + suffix if suffix else ''}")
            running.append((name, suffix, subprocess.Popen(argv)))
        name, suffix, proc = running.pop(0)
        rc = proc.wait()
        key = name + (f"_{suffix}" if suffix else "")
        if rc != 0:
            failed.append(key)
            print(f"  FAILED {key} (exit {rc})")
            continue
        res_path = opts.out_dir / key / "results.json"
        results[key] = json.loads(res_path.read_text()) if res_path.exists() else {}
        print(f"  done  {key} ({results[key].get('elapsed_s', '?')} s)")
    summary = {"members": sorted(results), "failed": failed,
               "elapsed_s": {k: v.get("elapsed_s") for k, v in results.items()}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if failed:
        raise SystemExit(f"suite {campaign.name}: {len(failed)} member(s) failed: {failed}")
    return summary
