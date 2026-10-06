"""The g3_campaign and montecarlo campaign runners end to end on tiny
campaigns (hbtsim/runners/g3_campaign.py, hbtsim/runners/montecarlo.py)."""

import json
import math

import pytest

from hbtsim.catalog import Catalog
from hbtsim.run_campaign import RunOptions, run

SHIPPED = Catalog(env=False)
TINY_SPEC = {"label": "tiny", "lambda_min_nm": 450, "lambda_max_nm": 480, "n_channels": 4,
             "throughput": 0.5}
TINY_MCP = {"label": "tiny mcp", "model": "counting", "spectrograph": TINY_SPEC,
            "detector": "eonsii_mcp_pmt"}
TINY_SPAD_PBS = {"label": "tiny spad pbs", "model": "counting", "spectrograph": TINY_SPEC,
                 "detector": "eonsii_spad", "polarization_mode": "pbs"}


def _finite_numbers(obj):
    """Every number below obj (dicts / lists), for finiteness checks."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return
    if isinstance(obj, (int, float)):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _finite_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _finite_numbers(v)


def _catalog_with(tmp_path, spec: dict) -> Catalog:
    d = tmp_path / "campaigns"
    d.mkdir()
    (d / f"{spec['name']}.json").write_text(json.dumps(spec))
    return Catalog(paths=[tmp_path], env=False)


def _run(cat, name, tmp_path):
    camp = cat.load_campaign(name)
    out = tmp_path / "out"
    res = run(cat, camp, RunOptions(out_dir=out, figures=False, track=False, latex=True), suffix="")
    path = out / name / "results.json"
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert (out / name / "log.txt").read_text().strip()
    assert all(math.isfinite(x) for x in _finite_numbers(
        {k: v for k, v in on_disk.items() if k != "campaign_definition"}))
    return res, on_disk, out / name


def test_shipped_catalog_is_valid():
    assert SHIPPED.validate_all() == []
    g3 = SHIPPED.load_campaign("g3_deltavel_eonsii_paranal")
    assert g3.runner == "g3_campaign" and g3.option("options.layout_scan.backend") == "eonsii_spad_pbs"
    assert g3.option("options.side_m") is None and g3.option("options.compare_site") == "teide"
    mc = SHIPPED.load_campaign("mc_sirius_b_eonsii")
    assert mc.runner == "montecarlo" and len(mc.array.stations) == 2
    assert mc.option("options.paper.hours_to_10pct") == {"mcp": 1.5, "spad": 0.33}


def test_g3_campaign_runner(tmp_path):
    spec = SHIPPED.raw("campaign", "g3_deltavel_eonsii_paranal")
    spec.update(name="tiny_g3_campaign", backends=[TINY_MCP, TINY_SPAD_PBS], track_backends=[],
                atmosphere={"use": False}, cache_dir=None, night={"block_minutes": 60.0})
    spec["options"].update(n_uniform=2, side_m=None)
    spec["options"]["layout_scan"].update(sides_m=[12.0, 20.0], scan_phases=2, backend="tiny spad pbs")
    cat = _catalog_with(tmp_path, spec)
    res, on_disk, out = _run(cat, "tiny_g3_campaign", tmp_path)

    assert on_disk["runner"] == "g3_campaign"
    assert res["side_m"] in (12.0, 20.0) and res["n_uniform"] == 2
    assert res["layout_scan"]["backend"] == "tiny spad pbs"
    assert [r["side_m"] for r in res["layout_scan"]["sides"]] == [12.0, 20.0]
    assert len(res["layout_scan"]["scan_phases"]) == 2 + len(res["windows"])
    assert len(res["windows"]) == 2 and all(0 < a < b < 1 for a, b in res["windows"])
    assert res["n_nights"] == len(res["phases_mid"]) > 2
    for name in ("tiny mcp", "tiny spad pbs"):
        c = res["campaign"][name]
        assert c["template_snr"] > 0 and c["phase_snr"] > 0
        assert c["nights_amplitude"] > 0 and c["nights_phase"] > 0
        assert len(c["per_night_phase"]) == res["n_nights"]
        assert c["template_snr"] == pytest.approx(
            math.sqrt(c["amplitude_uniform"] ** 2 + c["amplitude_eclipse"] ** 2))
        bb = res["blackbody_vs_newera"]["backends"][name]
        assert len(bb["amplitude"]) == 2 and bb["amplitude"][0] > 0
        # no atmosphere attached: blackbody and "NewEra" are the same system
        assert bb["amplitude"][0] == pytest.approx(bb["amplitude"][1])
    # PBS doubles the usable photon statistics relative to the unpolarized MCP backend
    assert res["campaign"]["tiny spad pbs"]["template_snr"] > res["campaign"]["tiny mcp"]["template_snr"]
    assert (out / "table.md").exists() and (out / "table.tex").exists()
    assert "tiny mcp" in (out / "table.tex").read_text()
    assert not list(out.glob("*.png"))


def test_g3_campaign_fixed_side_skips_scan(tmp_path):
    spec = SHIPPED.raw("campaign", "g3_deltavel_eonsii_paranal")
    spec.update(name="tiny_g3_fixed", backends=[TINY_MCP], track_backends=[],
                atmosphere={"use": False}, cache_dir=None, night={"block_minutes": 60.0})
    spec["options"].update(n_uniform=2, side_m=30.0)
    cat = _catalog_with(tmp_path, spec)
    res, _, out = _run(cat, "tiny_g3_fixed", tmp_path)
    assert res["side_m"] == 30.0 and res["layout_scan"]["sides"] == []
    assert "(2) Layout scan" not in (out / "log.txt").read_text()


def test_montecarlo_runner(tmp_path):
    spec = SHIPPED.raw("campaign", "mc_sirius_b_eonsii")
    spec.update(name="tiny_mc", backends=[TINY_MCP, TINY_SPAD_PBS])
    spec["options"].update(n_realizations=2)
    cat = _catalog_with(tmp_path, spec)
    res, on_disk, out = _run(cat, "tiny_mc", tmp_path)

    assert on_disk["runner"] == "montecarlo"
    assert set(res) >= {"tiny mcp", "tiny spad pbs"}
    mcp, spad = res["tiny mcp"], res["tiny spad pbs"]
    assert mcp["paper_key"] == "mcp" and spad["paper_key"] == "spad"
    assert mcp["paper_hours_to_10pct"] == 1.5 and spad["paper_hours_to_10pct"] == 0.33
    for b in (mcp, spad):
        assert b["n_channels"] == 4 and b["baseline_m"] == pytest.approx(1750.0)
        assert len(b["projected_baselines_m"]) == 3
        assert b["analytic_precision_10h"] > 0
        assert b["hours_to_10pct"] == pytest.approx(10.0 * (b["analytic_precision_10h"] / 0.1) ** 2)
        assert b["hours_to_2pct"] > b["hours_to_5pct"] > b["hours_to_10pct"]
        assert set(b["mc"]) == {"matched", "box_opt", "box_sigma_raw", "box_tdc_raw", "matched_sideband"}
        assert set(b["mc"]["matched"]) == {"theta_mean", "theta_std", "bias_frac", "precision_frac",
                                           "sigma_pred", "pull_std"}
        assert len(b["ladder"]) == 9
        assert all(isinstance(lab, str) for lab, _, _ in b["ladder"])
        assert b["ladder"][0][0].startswith("our budget") and b["ladder"][0][1] == pytest.approx(
            b["analytic_precision_10h"])
        assert 1000.0 <= b["optimal_baseline_m"] <= 3600.0
    # the SPAD backend has the sharper timing, hence the better analytic precision
    assert spad["analytic_precision_10h"] < mcp["analytic_precision_10h"]
    assert (out / "ladder_tiny_mcp.md").exists() and (out / "ladder_tiny_spad_pbs.tex").exists()
