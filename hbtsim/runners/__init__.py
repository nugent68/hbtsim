"""Campaign runners: `run(campaign, catalog, opts, out_dir) -> dict`.

RUNNERS maps the campaign's `runner` field to the module in this package.
Each runner reads the resolved objects from the Campaign (targets, array,
telescope, backends, track_backends) and its raw options
(campaign.option("night.block_minutes")), prints its report (captured to
log.txt by hbtsim.run_campaign), writes figures / tables into out_dir and
returns the numbers as a JSON-serializable dict.
"""

RUNNERS = {
    "g3": "g3",
    "g2": "g2",
    "g3_campaign": "g3_campaign",
    "chromatic": "chromatic",
    "montecarlo": "montecarlo",
    "scale": "scale",
    "suite": "suite",
}


def write_tables(out_dir, rows, header, name="table", latex=True) -> None:
    """rows: list of lists of strings; writes table.md (and table.tex)."""
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    (out_dir / f"{name}.md").write_text("\n".join(md) + "\n")
    if latex:
        tex = [" & ".join(header) + r" \\", r"\hline"]
        tex += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
        (out_dir / f"{name}.tex").write_text("\n".join(tex) + "\n")
