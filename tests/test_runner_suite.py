"""The suite runner executes its members in subprocesses and collects
their results.json files."""

import json
from pathlib import Path

from hbtsim.catalog import Catalog
from hbtsim.run_campaign import RunOptions, run

TINY_BACKEND = {"label": "tiny", "model": "counting", "detector": "spad_lambda_ng",
                "spectrograph": {"label": "t", "lambda_min_nm": 450.0, "lambda_max_nm": 480.0,
                                 "n_channels": 4, "throughput": 0.5}}


def test_suite_runs_members(tmp_path):
    cat0 = Catalog(env=False)
    g2 = cat0.raw("campaign", "g2_c2pu")
    g2.update(name="tiny_g2", targets=["spica"], backends=[TINY_BACKEND],
              atmosphere={"use": False},
              options={**g2["options"], "baseline_scan_m": {"start": 20.0, "stop": 30.0, "step": 10.0},
                       "section3_numbers": False})
    suite = {"schema_version": 1, "kind": "campaign", "name": "tiny_suite", "label": "two tiny runs",
             "runner": "suite", "jobs": 2,
             "members": ["tiny_g2", {"campaign": "tiny_g2", "suffix": "again",
                                     "set": {"options.baseline_scan_m.start": 25.0}}]}
    (tmp_path / "campaigns").mkdir()
    (tmp_path / "campaigns" / "tiny_g2.json").write_text(json.dumps(g2))
    (tmp_path / "campaigns" / "tiny_suite.json").write_text(json.dumps(suite))
    cat = Catalog(paths=[tmp_path], env=False)
    out = tmp_path / "out"
    res = run(cat, cat.load_campaign("tiny_suite"), RunOptions(out_dir=out, figures=False, jobs=2, quiet=True))
    assert res["failed"] == [] and sorted(res["members"]) == ["tiny_g2", "tiny_g2_again"]
    r1 = json.loads((out / "tiny_g2" / "results.json").read_text())
    r2 = json.loads((out / "tiny_g2_again" / "results.json").read_text())
    assert r1["rows"] and r2["rows"]
    assert r2["campaign_definition"]["options"]["baseline_scan_m"]["start"] == 25.0
    assert (out / "tiny_suite" / "summary.json").exists() and (out / "tiny_suite" / "log.txt").read_text()
