"""The catalog loader: discovery, validation, overrides, errors, hashing."""

import json
import os
from pathlib import Path

import pytest

from hbtsim.catalog import (KINDS, Catalog, CatalogError, SchemaError, UnknownNameError,
                            default_catalog, load_target)
from hbtsim.catalog.registry import ENV_VAR
from hbtsim.serialize import canonical_json, content_hash, to_json

cat = Catalog(env=False)


def test_shipped_catalog_is_clean():
    assert cat.validate_all() == []
    assert set(KINDS) == {"band", "ld_table", "target", "telescope", "detector", "spectrograph",
                          "backend", "site", "array", "resource", "campaign"}
    for kind in KINDS:
        assert cat.list_names(kind), kind


def test_every_shipped_file_is_valid_json_with_the_common_keys():
    for kind in KINDS:
        for name in cat.list_names(kind):
            p = cat.source_of(kind, name)
            d = json.loads(p.read_text())
            assert d["name"] == p.stem == name and d["kind"] == kind and d["schema_version"] == 1
            assert d.get("label") or kind in ("band",), f"{p} lacks a label"


def test_arrays_resolve_and_triangles_need_three_stations():
    for name in cat.list_names("array"):
        arr = cat.load_array(name)
        assert len(arr.stations) >= 2
        for s in arr.stations:
            assert s.telescope.diameter_m > 0 and s.detector.pde(500.0) > 0
    cat.load_triangle("maunakea_subaru_keck")
    cat.load_triangle("eonsii_triangle_paranal", side_m=30.0)
    with pytest.raises(CatalogError, match="triangle needs 3"):
        cat.load_triangle("vlt_ut")
    with pytest.raises(CatalogError, match="takes no generator"):
        cat.load_array("vlt_ut", side_m=10.0)
    with pytest.raises(CatalogError, match="unknown generator"):
        cat.load_array("eonsii_triangle_paranal", sides=10.0)


def test_every_binary_has_an_orientation():
    for name in cat.list_names("target"):
        d = cat.raw("target", name)
        if d["type"] == "binary":
            assert d.get("node_pa_deg") is not None or d.get("skip_omega_test"), name
            assert cat.load_target(name).dec_deg is not None


def test_built_objects_are_hashable_and_serializable():
    # backends (kw dict) and campaigns (spec dict) become hashable in phase (b)
    for kind in KINDS:
        if kind in ("band", "backend", "campaign"):
            continue
        for name in cat.list_names(kind):
            obj = cat.load(kind, name)
            hash(obj)
            canonical_json(obj)
            to_json(obj, mode="config")
    for name in cat.list_names("backend"):
        canonical_json(cat.load_backend(name))


def test_content_hash_is_stable_and_order_independent(tmp_path):
    a = cat.load_target("spica")
    b = Catalog(env=False).load_target("spica")
    assert a == b and content_hash(a) == content_hash(b)
    # the same definition with its keys shuffled hashes the same
    d = cat.raw("target", "spica")
    shuffled = dict(reversed(list(d.items())))
    (tmp_path / "targets").mkdir()
    (tmp_path / "targets" / "spica.json").write_text(json.dumps(shuffled))
    c = Catalog(paths=[tmp_path], env=False).load_target("spica")
    assert content_hash(c) == content_hash(a)
    assert content_hash(cat.load_target("algol")) != content_hash(a)


def test_override_by_name_from_a_directory(tmp_path):
    d = cat.raw("target", "betaaur")
    d["name"] = "test_binary"
    d["label"] = "a copy of beta Aur"
    d["distance_pc"] = 30.0
    (tmp_path / "targets").mkdir()
    (tmp_path / "targets" / "test_binary.json").write_text(json.dumps(d))
    c = Catalog(paths=[tmp_path], env=False)
    assert "test_binary" in c.list_names("target") and "betaaur" in c.list_names("target")
    t = c.load_target("test_binary")
    assert t.distance_pc == 30.0 and t.name == "a copy of beta Aur"
    # shadowing a shipped name
    d["name"] = "betaaur"
    (tmp_path / "targets" / "betaaur.json").write_text(json.dumps(d))
    c2 = Catalog(paths=[tmp_path], env=False)
    assert c2.load_target("betaaur").distance_pc == 30.0
    assert c2.source_of("target", "betaaur") == tmp_path / "targets" / "betaaur.json"
    # a Catalog without the extra path is untouched
    assert Catalog(env=False).load_target("betaaur").distance_pc == 24.87
    with pytest.raises(UnknownNameError):
        Catalog(env=False).load_target("test_binary")


def test_env_search_path(tmp_path, monkeypatch):
    d = cat.raw("site", "teide")
    d["elevation_m"] = 2400.0
    (tmp_path / "sites").mkdir()
    (tmp_path / "sites" / "teide.json").write_text(json.dumps(d))
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert Catalog().load_site("teide").elevation_m == 2400.0
    assert Catalog(env=False).load_site("teide").elevation_m == 2390.0
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "missing"))
    with pytest.raises(CatalogError, match="does not exist"):
        Catalog()


def test_unknown_name_suggests():
    with pytest.raises(UnknownNameError, match="spad_lambda"):
        cat.load_detector("spad_lamda")
    with pytest.raises(CatalogError, match="unknown kind"):
        cat.load("telescopes", "keck_10m")


def _write(tmp_path, kind, name, body):
    (tmp_path / (kind + "s")).mkdir(exist_ok=True)
    body = {"schema_version": 1, "kind": kind, "name": name, **body}
    (tmp_path / (kind + "s") / f"{name}.json").write_text(json.dumps(body))
    return Catalog(paths=[tmp_path], env=False)


def test_schema_errors_name_file_and_field(tmp_path):
    c = _write(tmp_path, "detector", "bad_timing",
               {"label": "x", "pde_table_nm": [[400, 0.5], [900, 0.1]], "timing": {"jitter_ns": 1},
                "dead_time_ns": 10, "dark_cps_per_pixel": 1})
    with pytest.raises(SchemaError, match="detectors/bad_timing.json.*timing"):
        c.load_detector("bad_timing")
    c = _write(tmp_path, "spectrograph", "bad_order",
               {"label": "x", "lambda_min_nm": 900, "lambda_max_nm": 400, "n_channels": 10})
    with pytest.raises(SchemaError, match="bad_order.json.*lambda_max_nm"):
        c.load_spectrograph("bad_order")
    c = _write(tmp_path, "telescope", "bad_key", {"label": "x", "diameter_m": 1.0, "diametre_m": 1.0})
    with pytest.raises(SchemaError, match="unknown key 'diametre_m'"):
        c.load_telescope("bad_key")
    c = _write(tmp_path, "telescope", "bad_range", {"label": "x", "diameter_m": 1.0, "throughput": 1.5})
    with pytest.raises(SchemaError, match="throughput.*above 1"):
        c.load_telescope("bad_range")
    c = _write(tmp_path, "telescope", "wrong_kind", {"label": "x", "diameter_m": 1.0})
    (tmp_path / "telescopes" / "wrong_kind.json").write_text(
        json.dumps({"schema_version": 1, "kind": "detector", "name": "wrong_kind", "diameter_m": 1}))
    with pytest.raises(SchemaError, match="kind is 'detector'"):
        Catalog(paths=[tmp_path], env=False).load_telescope("wrong_kind")
    (tmp_path / "telescopes" / "wrong_kind.json").write_text("{not json")
    with pytest.raises(SchemaError, match="invalid JSON"):
        Catalog(paths=[tmp_path], env=False).load_telescope("wrong_kind")


def test_extends_merge_and_cycles(tmp_path):
    c = _write(tmp_path, "telescope", "keck_dirty",
               {"extends": "keck_10m", "label": "Keck with dirty mirrors", "throughput": 0.2,
                "notes": ["half the usual throughput"]})
    t = c.load_telescope("keck_dirty")
    assert t.diameter_m == 10.0 and t.throughput == 0.2 and t.name == "Keck with dirty mirrors"
    raw = c.raw("telescope", "keck_dirty")
    assert "extends" not in raw and raw["notes"][-1] == "half the usual throughput"
    assert raw["notes"][0] == cat.raw("telescope", "keck_10m")["notes"][0]
    _write(tmp_path, "telescope", "loop_a", {"extends": "loop_b", "label": "a"})
    c = _write(tmp_path, "telescope", "loop_b", {"extends": "loop_a", "label": "b"})
    with pytest.raises(SchemaError, match="cycle"):
        c.load_telescope("loop_a")
    c = _write(tmp_path, "telescope", "orphan", {"extends": "no_such_telescope", "label": "o"})
    with pytest.raises(UnknownNameError):
        c.load_telescope("orphan")


def test_inline_definitions_and_anchor_forms(tmp_path):
    c = _write(tmp_path, "array", "inline_pair", {
        "label": "two custom telescopes", "site": {"label": "somewhere", "latitude_deg": 10.0},
        "stations": [
            {"name": "A", "east_m": 0, "north_m": 0,
             "telescope": {"label": "small", "diameter_m": 0.5}, "detector": "spad_lambda"},
            {"name": "B", "east_m": 30, "north_m": 0, "telescope": "keck_10m", "detector": "spad_lambda_ng"}]})
    arr = c.load_array("inline_pair")
    assert arr.site.latitude_deg == 10.0 and arr.stations[0].telescope.diameter_m == 0.5
    assert arr.stations[1].detector.readout == "correlator"
    d = cat.raw("target", "sirius_a")
    d["name"] = "sirius_anchored"
    d["mag_anchors"] = [{"band": "V", "mag_vega": -1.46}, {"wavelength_nm": 1000.0, "mag_ab": -1.3}]
    c = _write(tmp_path, "target", "sirius_anchored", {k: v for k, v in d.items() if k not in ("schema_version", "kind", "name")})
    t = c.load_target("sirius_anchored")
    assert t.mag_anchors == ((551.0, -1.46 + 0.02), (1000.0, -1.3))
    d["mag_anchors"] = [{"band": "g", "mag_vega": 1.0}]
    c = _write(tmp_path, "target", "sirius_anchored", {k: v for k, v in d.items() if k not in ("schema_version", "kind", "name")})
    with pytest.raises(CatalogError, match="vega_to_ab"):
        c.load_target("sirius_anchored")


def test_single_star_radius_is_derived_not_given(tmp_path):
    d = cat.raw("target", "vega")
    d["star"]["radius_rsun"] = 2.5
    c = _write(tmp_path, "target", "vega", {k: v for k, v in d.items() if k not in ("schema_version", "kind", "name")})
    with pytest.raises(SchemaError, match="radius_rsun"):
        c.load_target("vega")


def test_atmosphere_attachment_without_tables(tmp_path, monkeypatch):
    monkeypatch.setenv("HBTSIM_NEWERA_DIR", str(tmp_path / "nowhere"))
    t = cat.load_target("betaaur", atmosphere=True)
    assert t.primary.flux_table is None
    _, report = cat.attach_atmosphere(cat.load_target("betaaur"), name="betaaur")
    assert "not found" in report
    with pytest.raises(CatalogError, match="not found"):
        cat.load_target("betaaur", atmosphere="require")
    sb, report = cat.attach_atmosphere(cat.load_target("sirius_b"), name="sirius_b")
    assert "no atmosphere pointer" in report
    res = cat.load_resource("newera")
    assert res.resolved_path == str(tmp_path / "nowhere") and not res.exists()
    monkeypatch.delenv("HBTSIM_NEWERA_DIR")
    assert cat.load_resource("newera").resolved_path == "data/newera"


def test_campaigns_resolve():
    for name in cat.list_names("campaign"):
        camp = cat.load_campaign(name)
        assert camp.runner in ("g3", "g2", "g3_campaign", "chromatic", "montecarlo", "scale", "suite")
        if camp.runner == "suite":
            members = [m if isinstance(m, str) else m["campaign"] for m in camp.members]
            assert all(m in cat.list_names("campaign") for m in members)
        else:
            assert camp.targets
            assert camp.backends
    g3 = cat.load_campaign("g3_spica_vlt")
    assert g3.target.name.startswith("Spica") and len(g3.array.stations) == 4
    assert g3.option("night.block_minutes") == 15.0 and g3.option("nothing.here", 7) == 7


def test_campaign_from_a_path(tmp_path):
    d = cat.raw("campaign", "g3_spica_vlt")
    d["name"] = "my_run"
    d["array"] = "maunakea_subaru_keck"
    p = tmp_path / "my_run.json"
    p.write_text(json.dumps(d))
    camp = cat.load_campaign(str(p))
    assert camp.name == "my_run" and len(camp.array.stations) == 3


def test_module_level_default_catalog():
    assert default_catalog() is default_catalog()
    assert load_target("algol").name.startswith("Algol")


def test_cli_smoke(capsys):
    from hbtsim.catalog.cli import main
    assert main(["validate"]) == 0
    assert "OK" in capsys.readouterr().out
    assert main(["list", "site"]) == 0
    assert "paranal" in capsys.readouterr().out
    assert main(["show", "target", "deltavel"]) == 0
    out = capsys.readouterr().out
    assert "CHECK against the SMEI" in out
    assert main(["dump", "array", "eonsii_triangle_paranal", "--set", "side_m=12"]) == 0
    assert '"east_m"' in capsys.readouterr().out
    assert main(["paths"]) == 0
    assert "configs" in capsys.readouterr().out
