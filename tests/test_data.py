"""hbtsim.data: cache location, manifests, fetching (from a file:// mirror),
corner selection for a target, and the resource resolution order."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hbtsim import data
from hbtsim.catalog import Catalog
from hbtsim.params import FluxTable, LDProfile
from hbtsim.sed import NewEraGrid, save_star_tables


def _tiny_table(path, teff):
    lam = np.linspace(380.0, 1000.0, 50)
    flux = np.full(lam.shape, 1e9 * (teff / 9000.0) ** 4)
    mu = np.linspace(0.0, 1.0, 9)
    inten = np.ones((lam.size, mu.size), dtype=np.float32) * (0.4 + 0.6 * mu)[None, :]
    save_star_tables(str(path), dict(wavelength_nm=lam, flux=flux, mu=mu, intensity=inten,
                                     source=f"tiny {teff}", teff=teff, logg=4.0, r0_cm=1e11))


@pytest.fixture
def mirror(tmp_path):
    """A directory of tiny tables with its manifest, as a file:// mirror."""
    d = tmp_path / "mirror"
    d.mkdir()
    models = ["lte09200-3.50-0.0", "lte09200-4.00-0.0", "lte09400-3.50-0.0", "lte09400-4.00-0.0",
              "lte12000-4.00-0.0", "lte04800-4.50-0.0"]
    for m in models:
        _tiny_table(d / data.table_name(m), float(m[3:8]))
    _tiny_table(d / "newera_lte09200-4.00-0.0_350-560nm_0.01nm.npz", 9200.0)     # a blue table
    man = data.build_manifest(d)
    (d / "manifest.json").write_text(json.dumps(man))
    return d


def test_data_dir_env_and_default(monkeypatch, tmp_path):
    monkeypatch.setenv(data.ENV_DATA_DIR, str(tmp_path / "x"))
    assert data.data_dir() == tmp_path / "x"
    monkeypatch.delenv(data.ENV_DATA_DIR)
    assert data.data_dir().name in ("hbtsim", "Cache") and data.data_dir().is_absolute()
    monkeypatch.setenv(data.ENV_AUTO_FETCH, "1")
    assert data.auto_fetch_enabled()
    monkeypatch.setenv(data.ENV_AUTO_FETCH, "0")
    assert not data.auto_fetch_enabled()


def test_parse_table_name():
    i = data.parse_table_name("newera_lte09200-4.00-0.0_380-1000nm_0.02nm.npz")
    assert (i["model"], i["teff"], i["logg"], i["z"], i["range"]) == ("lte09200-4.00-0.0", 9200.0, 4.0, 0.0, "380-1000nm_0.02nm")
    assert data.parse_table_name("newera_example.npz") is None
    assert data.table_name("lte04800-4.50-0.0") == "newera_lte04800-4.50-0.0_380-1000nm_0.02nm.npz"


def test_manifest(mirror):
    man = json.loads((mirror / "manifest.json").read_text())
    assert man["schema_version"] == 1 and len(man["files"]) == 7
    e = next(x for x in man["files"] if x["name"] == "newera_lte09200-4.00-0.0_380-1000nm_0.02nm.npz")
    assert e["sha256"] == hashlib.sha256((mirror / e["name"]).read_bytes()).hexdigest()
    assert e["size"] == (mirror / e["name"]).stat().st_size and e["teff"] == 9200.0
    ranges = {x["range"] for x in man["files"]}
    assert ranges == {"380-1000nm_0.02nm", "350-560nm_0.01nm"}


def test_fetch_from_file_mirror(mirror, tmp_path, monkeypatch):
    cat = Catalog(env=False)
    res = cat.load_resource("newera")
    dest = tmp_path / "cache" / "newera"
    url = mirror.as_uri() + "/"
    got = data.fetch(res, models=["lte09200-4.00-0.0", "lte09400-4.00-0.0"], base_url=url, dest=dest, progress=None)
    assert len(got) == 2 and all(p.exists() for p in got) and not list(dest.glob("*.part"))
    # a second call verifies and skips
    got2 = data.fetch(res, models=["lte09200-4.00-0.0"], base_url=url, dest=dest, progress=None)
    assert got2[0] == got[0]
    # the default range filter leaves the blue table out of --all
    allp = data.fetch(res, all_models=True, base_url=url, dest=dest, progress=None)
    assert len(allp) == 6 and not any("350-560" in p.name for p in allp)
    # $HBTSIM_DATA_URL is honoured when no base_url is given
    monkeypatch.setenv(data.ENV_DATA_URL, url)
    assert data.base_url_for(res) == url
    # unknown model, corrupt file
    with pytest.raises(data.DataError, match="not in the manifest"):
        data.fetch(res, models=["lte99999-9.99-0.0"], base_url=url, dest=dest, progress=None)
    bad = json.loads((mirror / "manifest.json").read_text())
    bad["files"][0]["sha256"] = "0" * 64
    (mirror / "manifest.json").write_text(json.dumps(bad))
    (dest / bad["files"][0]["name"]).unlink(missing_ok=True)
    with pytest.raises(data.DataError, match="sha256 mismatch"):
        data.fetch(res, models=[bad["files"][0]["model"]], base_url=url, dest=dest, progress=None)
    assert not list(dest.glob("*.part"))


def test_corners_for():
    entries = [dict(model=f"lte{t:05d}-{g:.2f}-0.0", teff=float(t), logg=g, z=0.0)
               for t in (9200, 9400, 12000) for g in (3.5, 4.0)]
    assert data.corners_for(9350.0, 3.9, 0.0, entries) == ["lte09200-3.50-0.0", "lte09200-4.00-0.0",
                                                           "lte09400-3.50-0.0", "lte09400-4.00-0.0"]
    assert data.corners_for(9200.0, 4.0, 0.0, entries) == ["lte09200-4.00-0.0"]        # on a node
    assert data.corners_for(12550.0, 4.07, 0.0, entries) == []                           # outside, no clamp
    assert data.corners_for(12550.0, 4.07, 0.0, entries, allow_extrapolation=True) == ["lte12000-4.00-0.0"]   # clamped to the edge node
    assert data.corners_for(25300.0, 3.7, 0.0, entries, allow_extrapolation=True) == []  # too far to clamp
    assert data.corners_for(9300.0, 4.0, 0.5, entries) == []                             # no such [M/H]


def test_fetch_for_target_and_auto_attach(mirror, tmp_path, monkeypatch):
    cat = Catalog(env=False)
    url = mirror.as_uri() + "/"
    monkeypatch.setenv(data.ENV_DATA_URL, url)
    monkeypatch.setenv(data.ENV_DATA_DIR, str(tmp_path / "cache"))
    monkeypatch.setenv("HBTSIM_NEWERA_DIR", str(tmp_path / "cache" / "newera"))      # not the repo's data/
    got = data.fetch_for_target(cat, "betaaur", progress=None)
    assert sorted(p.name for p in got) == sorted(data.table_name(m) for m in
                                                 ("lte09200-3.50-0.0", "lte09200-4.00-0.0", "lte09400-3.50-0.0", "lte09400-4.00-0.0"))
    # a stand-in pointer fetches the named model into the red-clump resource
    monkeypatch.setenv("HBTSIM_NEWERA_REDCLUMP_DIR", str(tmp_path / "cache" / "newera_redclump"))
    got = data.fetch_for_target(cat, "hd17652", progress=None)
    assert [p.name for p in got] == [data.table_name("lte04800-4.50-0.0")] and got[0].parent.name == "newera_redclump"
    # attach with fetch=True on a target whose tables are not there yet
    t, report = cat.attach_atmosphere(cat.load_target("betaaur"), name="betaaur", fetch=True)
    assert t.primary.flux_table is not None and all("fetched" in v for v in report.values())
    # the auto-fetch switch
    monkeypatch.setenv(data.ENV_AUTO_FETCH, "1")
    t2 = cat.load_target("betaaur", atmosphere=True)
    assert t2.primary.flux_table is not None
    # a dead URL fails softly (and loudly with require)
    monkeypatch.setenv(data.ENV_DATA_URL, (tmp_path / "nowhere").as_uri() + "/")
    monkeypatch.setenv("HBTSIM_NEWERA_DIR", str(tmp_path / "empty"))
    _, report = cat.attach_atmosphere(cat.load_target("spica"), name="spica", fetch=True)
    assert "fetch failed" in report
    with pytest.raises(Exception, match="manifest"):
        cat.attach_atmosphere(cat.load_target("spica"), name="spica", fetch=True, require=True)


def test_resource_resolution_order(tmp_path, monkeypatch):
    cat = Catalog(env=False)
    monkeypatch.setenv(data.ENV_DATA_DIR, str(tmp_path / "cache"))
    monkeypatch.delenv("HBTSIM_NEWERA_DIR", raising=False)
    res = cat.load_resource("newera")
    assert res.candidates() == ["data/newera", str(tmp_path / "cache" / "newera")]
    monkeypatch.chdir(tmp_path)                                # no data/ here
    assert res.resolved_path == str(tmp_path / "cache" / "newera") and not res.exists()
    (tmp_path / "cache" / "newera").mkdir(parents=True)
    assert res.exists()
    monkeypatch.setenv("HBTSIM_NEWERA_DIR", str(tmp_path / "mine"))
    assert res.candidates()[0] == str(tmp_path / "mine")
    assert res.remote_dict["base_url"].startswith("https://portal.nersc.gov/project/newera/")
    assert cat.load_resource("newera_redclump").env_override == "HBTSIM_NEWERA_REDCLUMP_DIR"


def test_scan_prefers_full_range_then_finest_step(mirror):
    grid = NewEraGrid.scan(str(mirror))
    chosen = os.path.basename(grid.tables[(9200.0, 4.0, 0.0)])
    assert chosen == "newera_lte09200-4.00-0.0_380-1000nm_0.02nm.npz"
    assert grid.coverage()["n"] == 6


def test_data_cli(mirror, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(data.ENV_DATA_DIR, str(tmp_path / "cache"))
    monkeypatch.setenv("HBTSIM_NEWERA_DIR", str(tmp_path / "cache" / "newera"))
    assert data.main(["path"]) == 0
    assert "newera:" in capsys.readouterr().out
    assert data.main(["fetch", "--model", "lte12000-4.00-0.0", "--base-url", mirror.as_uri() + "/"]) == 0
    assert "1 table(s) in place" in capsys.readouterr().out
    assert data.main(["list"]) == 0
    assert "lte12000-4.00-0.0" in capsys.readouterr().out
    assert data.main(["manifest", str(mirror)]) == 0
    assert '"schema_version": 1' in capsys.readouterr().out
    assert data.main(["fetch", "--model", "nope", "--base-url", mirror.as_uri() + "/"]) == 1
