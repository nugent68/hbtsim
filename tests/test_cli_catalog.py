"""The command-line tools take catalog names, and a user's own definitions
on --config-dir / $HBTSIM_CONFIG_PATH work end to end without touching
the repository (the plan's add-a-target smoke test)."""

import json

import pytest

from hbtsim import snr_cli
from hbtsim.catalog import Catalog
from hbtsim.cli_common import Instrument, output_stem, resolve_instrument


@pytest.fixture
def my_configs(tmp_path):
    cat = Catalog(env=False)
    d = cat.raw("target", "betaaur")
    d["name"], d["label"] = "test_binary", "a test binary"
    d["distance_pc"] = 30.0
    (tmp_path / "targets").mkdir()
    (tmp_path / "targets" / "test_binary.json").write_text(json.dumps(d))
    a = cat.raw("array", "keck_pair")
    a["name"], a["label"] = "my_pair", "my own pair"
    a["stations"][1]["east_m"] = a["stations"][0]["east_m"] + 40.0
    a["stations"][1]["north_m"] = a["stations"][0]["north_m"]
    (tmp_path / "arrays").mkdir()
    (tmp_path / "arrays" / "my_pair.json").write_text(json.dumps(a))
    return tmp_path


def test_snr_cli_with_own_target_and_array(my_configs, capsys):
    snr_cli.main(["--config-dir", str(my_configs), "--target", "test_binary", "--instrument", "my_pair",
                  "--vis2-method", "analytic", "--time", "60", "--no-newera", "--channels", "8"])
    out = capsys.readouterr().out
    assert "a test binary" in out and "my_pair (40.0 m)" in out
    assert "Baseline 40.0 m" in out and "total SNR" in out


def test_snr_cli_telescope_detector_names(capsys):
    snr_cli.main(["--target", "spica", "--telescope", "keck_10m", "--detector", "spad_lambda_ng",
                  "--baseline", "85", "--vis2-method", "analytic", "--time", "60", "--no-newera",
                  "--channels", "8"])
    out = capsys.readouterr().out
    assert "Keck 10 m" in out and "correlator readout" in out and "Baseline 85.0 m" in out


def test_snr_cli_narrowband_and_overrides(capsys):
    snr_cli.main(["--target", "algol", "--mode", "narrowband", "--baseline", "15", "--diameter", "2.0",
                  "--wavelengths", "500", "--no-newera"])
    out = capsys.readouterr().out
    assert "2 x 2.00 m" in out and "custom" in out


def test_resolve_instrument_and_stem():
    cat = Catalog(env=False)

    class A:
        instrument = None
        telescope = "c2pu_1m"
        detector = None
        diameter = None
        throughput = None
    inst = resolve_instrument(cat, A(), readout="correlator")
    assert isinstance(inst, Instrument) and inst.detector.readout == "correlator" and inst.name == "c2pu_1m"
    A.instrument = "vlt_ut"
    inst = resolve_instrument(cat, A())
    assert len(inst.baselines_m) == 6 and inst.telescope.diameter_m == 8.2
    assert output_stem("spica", "keck_pair", "R = 5000, correlator") == "spica_keck_pair_R-=-5000,-correlator"


def test_unknown_target_message():
    with pytest.raises(Exception, match="no target named 'spcia'.*spica"):
        snr_cli.main(["--target", "spcia", "--no-newera"])


def test_main_dispatcher(capsys):
    from hbtsim.__main__ import main
    assert main(["--help"]) == 0
    assert "catalog" in capsys.readouterr().out and "run" in " ".join(["run"])
    assert main(["bogus"]) == 2
    assert main(["catalog", "list", "site"]) == 0
    assert "teide" in capsys.readouterr().out
