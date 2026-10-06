"""The g3 and g2 campaign runners end to end on tiny campaigns (4-channel
spectrograph, 60-min blocks, no model atmospheres, no figures)."""

import json
import math

import pytest
import subprocess
import sys
from pathlib import Path

from hbtsim.catalog import Catalog
from hbtsim.run_campaign import RunOptions, run

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = Catalog(env=False)

TINY_SPEC = {"label": "tiny", "lambda_min_nm": 450.0, "lambda_max_nm": 480.0,
             "n_channels": 4, "throughput": 0.5}
TINY_BACKEND = {"label": "tiny correlator", "model": "counting", "spectrograph": TINY_SPEC,
                "detector": "spad_lambda_ng"}
TINY_BACKEND_PBS = dict(TINY_BACKEND, label="tiny correlator + PBS", polarization_mode="pbs")


def _catalog_with(tmp_path, spec) -> tuple:
    d = tmp_path / "campaigns"
    d.mkdir(exist_ok=True)
    (d / f"{spec['name']}.json").write_text(json.dumps(spec, indent=1))
    cat = Catalog(paths=[tmp_path], env=False)
    return cat, cat.load_campaign(spec["name"])


def _finite_or_none(x):
    return x is None or (isinstance(x, (int, float)) and math.isfinite(x))


def _check_outputs(tmp_path, name, results):
    out = tmp_path / "out" / name
    assert (out / "results.json").exists() and (out / "table.md").exists()
    assert (out / "table.tex").exists()
    assert (out / "log.txt").read_text().strip()
    on_disk = json.loads((out / "results.json").read_text())
    assert on_disk["campaign"] == name and on_disk["runner"] == results["runner"]
    assert "campaign_hash" in on_disk and "elapsed_s" in on_disk
    return out


def test_shipped_catalog_valid_with_runner_options():
    assert SHIPPED.validate_all() == []
    assert SHIPPED.load_campaign("g2_c2pu").option("options.section3_numbers") is True
    assert SHIPPED.load_campaign("g2_keck").option("options.section3_numbers") is False
    assert SHIPPED.load_campaign("g2_eonsii").option("options.section3_numbers") is False


def _g3_spec(name, shipped, **edits):
    spec = json.loads(json.dumps(SHIPPED.raw("campaign", shipped)))
    spec.update(name=name, backends=[TINY_BACKEND, TINY_BACKEND_PBS], track_backends=[TINY_BACKEND],
                night={"block_minutes": 60.0}, atmosphere={"use": False},
                outputs={"figures": False, "track": True, "latex": True})
    spec.update(edits)
    return spec


def test_g3_runner_three_station_array_with_track(tmp_path):
    cat, camp = _catalog_with(tmp_path, _g3_spec("tiny_g3_maunakea", "g3_betaaur_maunakea",
                                                  phases={"snapshot": "max_separation",
                                                          "extra": [0.1]}))
    opts = RunOptions(out_dir=tmp_path / "out", figures=False, track=True, latex=True)
    res = run(cat, camp, opts, suffix="")
    out = _check_outputs(tmp_path, "tiny_g3_maunakea", res)
    assert set(res) >= {"phase", "snapshot", "extra_phases", "track", "smearing"}
    assert 0.0 <= res["phase"] < 1.0
    assert [r["backend"] for r in res["snapshot"]] == ["tiny correlator", "tiny correlator + PBS"]
    for r in res["snapshot"]:
        assert r["n_channels"] == 4
        for k in ("snr_amplitude", "snr_total", "t_amplitude_h", "t_binned_h", "t_channel_h",
                  "rate_cps", "dead_time_load"):
            assert _finite_or_none(r[k]), k
        assert r["snr_total"] > 0 and r["rate_cps"] > 0
        assert isinstance(r["readout_limited"], bool)
    assert list(res["extra_phases"]) == ["0.1"] and len(res["extra_phases"]["0.1"]) == 2
    assert len(res["track"]) == 1
    tr = res["track"][0]
    assert tr["backend"] == "tiny correlator" and tr["n_blocks"] > 0 and tr["t_total_h"] > 0
    for k in ("snr_amplitude", "nights_amplitude", "nights_binned", "nights_channel", "drift_max"):
        assert _finite_or_none(tr[k]), k
    assert 0 < res["smearing"]["retained_400nm"] <= 1.0
    table = (out / "table.md").read_text().splitlines()
    assert table[0].startswith("| target | backend | channels | SNR3_amp/sqrt(h) |")
    assert len(table) == 2 + 2
    assert not list(out.glob("*.png"))


def test_g3_runner_four_station_array_phase_override(tmp_path):
    cat, camp = _catalog_with(tmp_path, _g3_spec("tiny_g3_vlt", "g3_spica_vlt"))
    opts = RunOptions(out_dir=tmp_path / "out", figures=False, track=False, latex=False,
                      phase=0.3)
    res = run(cat, camp, opts, suffix="")
    out = tmp_path / "out" / "tiny_g3_vlt"
    assert res["phase"] == 0.3 and res["track"] == [] and res["extra_phases"] == {}
    assert len(res["snapshot"]) == 2 and res["snapshot"][0]["snr_total"] > 0
    assert (out / "results.json").exists() and (out / "table.md").exists()
    assert not (out / "table.tex").exists()
    assert (out / "log.txt").read_text().strip()


def test_g2_runner(tmp_path):
    spec = json.loads(json.dumps(SHIPPED.raw("campaign", "g2_c2pu")))
    spec.update(name="tiny_g2", targets=["betaaur", "algol"],
                backends=[TINY_BACKEND, TINY_BACKEND_PBS], atmosphere={"use": False})
    spec["options"].update(baseline_scan_m={"start": 20.0, "stop": 25.0, "step": 5.0},
                           section3_numbers=True)
    cat, camp = _catalog_with(tmp_path, spec)
    opts = RunOptions(out_dir=tmp_path / "out", figures=False, track=False, latex=True)
    res = run(cat, camp, opts, suffix="")
    out = _check_outputs(tmp_path, "tiny_g2", res)
    assert res["instrument"] == "c2pu_1m"
    assert set(res["best_baselines"]) == {"Beta", "Algol"}
    assert all(b in (20.0, 25.0) for b in res["best_baselines"].values())
    assert len(res["rows"]) == 4
    for r in res["rows"]:
        assert set(r) >= {"target", "instrument", "backend", "baseline_m", "snr_total", "rate_cps",
                          "readout_limited", "readout_scale", "dead_time_load", "vis2_median",
                          "tabled"}
        assert r["tabled"] == "blackbody" and r["snr_total"] > 0 and math.isfinite(r["snr_total"])
        assert 0.0 <= r["vis2_median"] <= 1.0
    log = (out / "log.txt").read_text()
    assert "Two-telescope g2 (Section 3)" in log and "Keck 85 m" in log
    table = (out / "table.md").read_text().splitlines()
    assert table[0] == "| target | SED | backend | B [m] | SNR2/sqrt(h) | rate cps/tel | limited |"
    assert len(table) == 2 + 4


def test_old_script_is_a_shim():
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "feasibility_g3.py")],
                       capture_output=True, text=True)
    assert p.returncode == 2
    assert "hbtsim run g3_spica_vlt" in p.stdout and "hbtsim run g2_c2pu" in p.stdout
    assert not (ROOT / "scripts" / "g2_logs_to_md.py").exists()


def test_g2_array_track_mode(tmp_path):
    """instrument.array: one night along the uv track of an unequal pair,
    one filter per night (nights add over the filter set)."""
    import json
    from hbtsim.catalog import Catalog
    from hbtsim.run_campaign import RunOptions, run
    cat0 = Catalog(env=False)
    d = cat0.raw("campaign", "g2_binaries_lpqi_pathfinder")
    d.update(name="tiny_lpqi", targets=["spica"], backends=["lpqi_500", "lpqi_550", "lpqi_nextgen_500"],
             atmosphere={"use": False}, night={"block_minutes": 120.0, "min_alt_deg": 30.0})
    (tmp_path / "campaigns").mkdir()
    (tmp_path / "campaigns" / "tiny_lpqi.json").write_text(json.dumps(d))
    cat = Catalog(paths=[tmp_path], env=False)
    out = tmp_path / "out"
    res = run(cat, cat.load_campaign("tiny_lpqi"), RunOptions(out_dir=out, figures=False, latex=True))
    assert res["mode"] == "array_track" and res["instrument"] == "lpqi_pathfinder"
    rows = res["rows"]
    assert len(rows) == 3 and all(r["pair"] == "NOT-TNG" for r in rows)
    for r in rows:
        assert 300 < r["baseline_min_m"] <= r["baseline_max_m"] <= 550.0 + 1e-6
        assert 0 <= r["vis2_min"] <= r["vis2_max"] < 0.05           # resolved out at 550 m
        assert r["snr_night"] > 0 and r["nights_detection"] > 1 and r["n_blocks"] >= 2
    by = {r["backend"]: r for r in rows}
    assert by["LPQI 1 nm 500 nm continuum, next-gen SPAD (assumed)"]["snr_night"] > \
        by["LPQI 1 nm 500 nm continuum, IMSE 64x64"]["snr_night"] * 5
    sets = res["filter_set_nights"]["Spica"]
    imse = [r for r in rows if r["detector"].startswith("LPQI-Pathfinder")]
    assert sets["LPQI-Pathfinder IMSE 64x64 SPAD array (I2CASS, LiDAR heritage)"] == \
        pytest.approx(sum(r["nights_detection"] for r in imse))          # one filter per night: nights add
    assert (out / "tiny_lpqi" / "table.md").read_text().count("NOT-TNG") == 3
    assert (out / "tiny_lpqi" / "results.json").exists()
