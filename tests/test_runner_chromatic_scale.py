"""The `chromatic` and `scale` campaign runners end to end on tiny
campaigns (a 4-6 channel inline spectrograph, one target, a 3-baseline
scan) built in tmp_path; and the shipped campaign JSON stays valid."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hbtsim.catalog import Catalog
from hbtsim.run_campaign import RunOptions, run

ROOT = Path(__file__).resolve().parents[1]
TINY_SPEC = {"label": "tiny", "lambda_min_nm": 480.0, "lambda_max_nm": 492.0, "n_channels": 6,
             "throughput": 0.5}
TINY_OPT = {"label": "tiny optical", "lambda_min_nm": 450.0, "lambda_max_nm": 480.0, "n_channels": 4,
            "throughput": 0.5}


def _campaign(tmp_path, base: str, name: str, **edits) -> tuple:
    """(Catalog, Campaign): the shipped campaign `base` copied into tmp_path
    as `name` with top-level edits applied (nested dicts merged one level)."""
    shipped = Catalog(env=False)
    d = shipped.raw("campaign", base)
    d["name"] = name
    for k, v in edits.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = {**d[k], **v}
        else:
            d[k] = v
    (tmp_path / "campaigns").mkdir(exist_ok=True)
    (tmp_path / "campaigns" / f"{name}.json").write_text(json.dumps(d, indent=1))
    cat = Catalog(paths=[tmp_path], env=False)
    return cat, cat.load_campaign(name)


def _walk_numbers(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_numbers(v)
    elif isinstance(obj, float):
        yield obj


def _run(cat, camp, tmp_path, **kw) -> tuple:
    opts = RunOptions(out_dir=tmp_path / "out", figures=False, track=False, latex=True, **kw)
    res = run(cat, camp, opts, suffix="")
    out = tmp_path / "out" / camp.name
    assert (out / "results.json").exists() and (out / "log.txt").stat().st_size > 0
    disk = json.loads((out / "results.json").read_text())
    assert all(math.isfinite(x) for x in _walk_numbers(disk))
    assert disk["runner"] == camp.runner and disk["campaign"] == camp.name
    assert (out / "table.md").read_text().count("\n") >= 3 and (out / "table.tex").exists()
    return res, disk, out


def test_shipped_catalog_is_valid():
    assert Catalog(env=False).validate_all() == []


def test_chromatic_array_mode(tmp_path):
    cat, camp = _campaign(
        tmp_path, "chromatic_sirius_vega_eonsii", "tiny_chromatic_array",
        targets=["sirius_a"], atmosphere={"use": False},
        backends=[{"label": "tiny spad", "model": "counting", "spectrograph": TINY_SPEC,
                   "detector": "eonsii_spad"}],
        options={"channel_selection": ["all", "subset"], "correlator_detector": "eonsii_spad_correlator",
                 "nights": 2.0})
    assert camp.array is not None and camp.telescope is None
    res, disk, out = _run(cat, camp, tmp_path)
    cases = disk["Sirius A"]["tiny spad"]
    assert set(cases) == {"link", "subset", "correlator"}
    for c in cases.values():
        # a blackbody has no Balmer line: zero signal, but every number is well defined
        assert c["n_channels"] == 6 and 0 <= c["n_tagged"] <= 6
        assert c["baseline_m"] >= 6.0                      # the pair's min_spacing_m
        assert set(c["line_significance"]) == {"Hbeta"} and c["significance"] >= 0.0
    assert cases["correlator"]["readout_scale"] == 1.0 and cases["link"]["readout_scale"] < 1.0
    assert cases["correlator"]["significance"] >= cases["link"]["significance"]
    log = (out / "log.txt").read_text()
    assert "tiny spad" in log and "correlator:" in log and "blackbody" in log
    assert "not this backend's detector" not in log       # eonsii_spad_correlator is the SPAD's correlator
    assert not list(out.glob("*.png"))


@pytest.mark.skipif(not Path(Catalog(env=False).load_resource("newera").resolved_path).is_dir(), reason="no NewEra tables")
def test_chromatic_array_mode_with_atmosphere(tmp_path):
    cat, camp = _campaign(
        tmp_path, "chromatic_sirius_vega_eonsii", "tiny_chromatic_newera",
        targets=["sirius_a"],
        backends=[{"label": "tiny spad", "model": "counting", "spectrograph": TINY_SPEC,
                   "detector": "eonsii_spad"}])
    res, disk, out = _run(cat, camp, tmp_path)
    cases = disk["Sirius A"]["tiny spad"]
    assert set(cases) == {"link", "subset", "correlator"}
    assert cases["link"]["significance"] > 1.0 and cases["correlator"]["significance"] > cases["link"]["significance"]
    assert cases["link"]["nights_to_5sigma"] == pytest.approx((5.0 / cases["link"]["significance"]) ** 2)
    assert 1 <= cases["subset"]["n_tagged"] <= 6
    assert "NewEra" in (out / "log.txt").read_text()


def test_chromatic_telescope_mode(tmp_path):
    cat, camp = _campaign(
        tmp_path, "chromatic_c2pu_regression", "tiny_chromatic_c2pu",
        targets=["vega"], atmosphere={"use": False},
        backends=[{"label": "tiny correlator", "model": "counting", "spectrograph": TINY_SPEC,
                   "detector": "spad_lambda_ng"}])
    assert camp.telescope is not None and camp.array is None
    res, disk, out = _run(cat, camp, tmp_path)
    (case,) = disk["Vega"]["tiny correlator"].values()
    assert list(disk["Vega"]["tiny correlator"]) == ["link"]
    # B = 1.7 lambda_ref / (pi theta_drawn) with theta_drawn = theta_LD (no atmosphere)
    b = 1.7 * 486.27e-9 / (math.pi * camp.target.theta_ld_mas * math.pi / 180 / 3.6e6)
    assert case["baseline_m"] == pytest.approx(b, rel=1e-6)
    assert case["t_int_s"] == pytest.approx(6 * 3600.0)
    assert "Calern" in (out / "log.txt").read_text()


def test_scale_linear_law(tmp_path):
    cat, camp = _campaign(
        tmp_path, "redclump_ii_dwarf", "tiny_scale",
        targets=["hd17652"], atmosphere={"use": False},
        backends=[{"label": "tiny", "model": "counting", "spectrograph": TINY_OPT, "detector": "spad_lambda_ng"}],
        options={"filters": ["filter_johnson_v", "filter_2mass_h"],
                 "baseline_scan_m": {"start": 50.0, "stop": 60.0, "step": 5.0}, "hours": 1.0,
                 "chromaticity_backend": None})
    res, disk, out = _run(cat, camp, tmp_path)
    assert disk["stand_in"] == "linear" and disk["hours"] == 1.0
    t = disk["HD 17652 (beta For, G9 IIIb)"]
    assert set(t["filters"]) == {"V", "H"} and t["theta_ud_over_ld"] == {} and "chromaticity" not in t
    assert set(t["multiplexed"]) == {"tiny", "EON-SII 1000 ch, QUASAR SPAD"}
    for c in list(t["filters"].values()) + list(t["multiplexed"].values()):
        assert c["sigma_s"] > 0 and 50.0 <= c["baseline_m"] <= 60.0 and c["rate_cps"] > 0
        assert c["hours_to_target"] == pytest.approx(1.0 * (c["sigma_s"] / 0.007) ** 2)
    assert t["multiplexed"]["EON-SII 1000 ch, QUASAR SPAD"]["n_channels"] == 1000
    log = (out / "log.txt").read_text()
    assert "Multiplexed optical backends" in log and "Model request" in log


@pytest.mark.skipif(not Path(Catalog(env=False).load_resource("newera_redclump").resolved_path).is_dir(), reason="no red-clump NewEra tables")
def test_scale_with_stand_in(tmp_path):
    cat, camp = _campaign(
        tmp_path, "redclump_ii_supergiant", "tiny_scale_standin",
        targets=["hd360"],
        backends=[{"label": "tiny", "model": "counting", "spectrograph": TINY_OPT, "detector": "spad_lambda_ng"}],
        options={"filters": ["filter_2mass_h"], "baseline_scan_m": {"start": 50.0, "stop": 60.0, "step": 5.0},
                 "chromaticity_backend": None})
    res, disk, out = _run(cat, camp, tmp_path)
    assert disk["stand_in"] == "lte05000-0.00-0.0"
    t = disk["HD 360 (HR 16, K1 II)"]
    assert 0.9 < t["theta_ud_over_ld"]["H"] < 1.0
    assert t["chromaticity"]["backend"] == "tiny" and set(t["chromaticity"]["windows"]) == {"450-500"}
    assert disk["radius_conventions"]["lte05000-0.00-0.0"]["r_outer_over_tau1"] > 1.1
    # --no-newera: the same campaign on the linear law
    res2, disk2, _ = _run(cat, camp, tmp_path / "linear", no_newera=True)
    assert disk2["stand_in"] == "linear" and "chromaticity" not in disk2["HD 360 (HR 16, K1 II)"]


@pytest.mark.parametrize("script", ["chromatic_diameters.py", "redclump_ii.py"])
def test_scripts_are_shims(script):
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], capture_output=True, text=True,
                       cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert p.returncode == 2 and "hbtsim run" in p.stdout
