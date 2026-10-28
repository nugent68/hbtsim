"""Runner `iact` (hbtsim/runners/iact.py): nights on a Cherenkov-telescope
array with the analog S/N, Fisher precision on the diameter / ellipse /
orbit with the zero-baseline normalization free."""

import json

import numpy as np
import pytest

from hbtsim.catalog import Catalog
from hbtsim.run_campaign import RunOptions, run


def _tiny(tmp_path, base, name, options, **extra):
    cat0 = Catalog(env=False)
    d = cat0.raw("campaign", base)
    d.update(name=name, night={"block_minutes": 60.0, "min_alt_deg": 30.0}, **extra)
    d["options"].update(options)
    (tmp_path / "campaigns").mkdir(exist_ok=True)
    (tmp_path / "campaigns" / f"{name}.json").write_text(json.dumps(d))
    return Catalog(paths=[tmp_path], env=False)


def _run(cat, name, tmp_path):
    out = tmp_path / "out"
    res = run(cat, cat.load_campaign(name), RunOptions(out_dir=out, figures=False, track=False, latex=False))
    return res, out / name


def test_iact_single_star_fit_u(tmp_path):
    """gamma Peg on the four LSTs, two nights, theta and u free: per-pair
    sigmas from the per-station constants, the Fisher precision tightening
    as 1/sqrt(2) for identical nights, theta alone better than joint, and
    the simulated circular fit near the truth."""
    cat = _tiny(tmp_path, "iact_gampeg_ctao_lst4", "tiny_iact_single", {"nights": 2, "seed": 1})
    res, out = _run(cat, "tiny_iact_single", tmp_path)
    assert res["single"] and res["parameters"] == ["theta", "u", "N0"] and res["n_nights"] == 2
    for name in ("CTAO-SII Stage I (425/26 nm, camera PMTs, ~400 MHz)",
                 "CTAO-SII Stage II (1 nm x 10 channels, dedicated photodetectors, > 2 GHz)"):
        b = res["backends"][name]
        assert len(b["pairs"]) == 6 and len(b["nights"]) == 2
        n1, n2 = b["nights"]
        assert n2["sigma_rel_cumulative"] == pytest.approx(n1["sigma_rel_cumulative"] / np.sqrt(2), rel=1e-6)
        assert b["sigma_rel_alone"] < b["sigma_rel"]
        assert b["nights_to_precision"] == pytest.approx(2 * (b["sigma_rel"] / 0.01) ** 2)
        fit = b["fit_first_night"]
        assert abs(fit["theta_mas"] - res["theta_ld_mas"]) < 3 * b["sigmas"]["theta"] * np.sqrt(2) + 1e-3
        assert abs(fit["N0"] - 1) < 0.05
    s1 = res["backends"]["CTAO-SII Stage I (425/26 nm, camera PMTs, ~400 MHz)"]
    s2 = res["backends"]["CTAO-SII Stage II (1 nm x 10 channels, dedicated photodetectors, > 2 GHz)"]
    # Stage II: 0.45/0.40 x sqrt(5 x 10) x 1.15/1.05 / 0.91 = 9.6 per pair
    assert s1["pairs"][0]["sigma_block"] / s2["pairs"][0]["sigma_block"] == pytest.approx(9.58, abs=0.05)
    assert (out / "table.md").exists() and (out / "table_nights.md").exists() and not (out / "tiny_iact_single.png").exists()


def test_iact_ellipse_and_binary(tmp_path):
    """gamma Cas's ellipse on MAGIC + LST-1 (LST-1 pairs more sensitive than
    the MAGIC pair) and Spica's orbit over two nights (ln a, Omega, omega)."""
    cat = _tiny(tmp_path, "iact_gammacas_magic_lst1", "tiny_iact_ellipse", {"nights": 1})
    res, _ = _run(cat, "tiny_iact_ellipse", tmp_path)
    assert res["parameters"] == ["theta_major", "axis_ratio", "pa", "N0"]
    b = res["backends"]["MAGIC SII (425/26 nm), per-station alpha and q"]
    sig = {p["pair"]: p["sigma_block"] for p in b["pairs"]}
    assert sig["MAGIC-I-LST-1"] == pytest.approx(sig["MAGIC-II-LST-1"]) and sig["MAGIC-I-LST-1"] < 0.5 * sig["MAGIC-I-MAGIC-II"]
    assert {round(p["baseline_max_m"]) for p in b["pairs"]} == {82, 55, 107}
    assert 0 < b["sigmas"]["axis_ratio"] < 0.05 and 0 < b["sigmas"]["pa"] < 5
    assert b["fit_first_night"] is None

    cat = _tiny(tmp_path, "iact_spica_magic_lst1", "tiny_iact_binary", {"nights": 2})
    res, _ = _run(cat, "tiny_iact_binary", tmp_path)
    assert not res["single"] and res["parameters"] == ["lna", "Omega", "omega", "N0"] and res["a_mas"] == pytest.approx(1.086, abs=0.002)
    b = res["backends"]["MAGIC SII (425/26 nm), per-station alpha and q"]
    n1, n2 = b["nights"]
    assert n2["phase0"] == pytest.approx((n1["phase0"] + 1.0 / 4.0145) % 1.0)
    assert n2["sigma_rel_cumulative"] < n1["sigma_rel_cumulative"] and b["sigma_rel"] < 0.01
    assert b["sigmas"]["Omega"] < 2.0 and b["sigmas"]["omega"] < 5.0


def test_iact_hyades_profiles(tmp_path):
    """Two Hyades giants with NewEra stand-in profiles on MAGIC + LST-1:
    several targets in one campaign, every baseline past the first null,
    and compare_models fitting the compact profile to the extended one."""
    import os
    cat0 = Catalog(env=False)
    d = cat0.load_resource("newera_redclump").resolved_path
    if not all(os.path.exists(os.path.join(d, f"newera_{m}_380-1000nm_0.02nm.npz"))
               for m in ("lte05000-4.50-0.0", "lte05000-0.00-0.0")):
        pytest.skip("Hyades stand-in tables not present")
    cat = _tiny(tmp_path, "iact_hyades_magic_lst1", "tiny_iact_hyades",
                {"nights": 1, "compare_models": ["lte05000-0.00-0.0"]}, targets=["eps_tau", "tet1_tau"])
    res, out = _run(cat, "tiny_iact_hyades", tmp_path)
    assert res["target_names"] == ["eps_tau", "tet1_tau"] and set(res["targets"]) == {"eps_tau", "tet1_tau"}
    e = res["targets"]["eps_tau"]
    assert e["profile"].startswith("lte05000-4.50-0.0") and e["theta_ld_mas"] == 2.733     # CHARA, Boyajian et al. 2009
    b = e["backends"]["MAGIC SII (425/26 nm), per-station alpha and q"]
    assert b["lobes"]["null_m"] == pytest.approx(44.2, abs=0.3) and b["lobes"]["lobe2_vis2"] == pytest.approx(0.0078, abs=0.0005)
    by = {p["pair"]: p for p in b["pairs"]}
    for k in ("MAGIC-I-MAGIC-II", "MAGIC-I-LST-1"):                   # past the null all night
        assert by[k]["baseline_min_m"] > 55 and by[k]["vis2_max"] < 0.01
    assert by["MAGIC-II-LST-1"]["baseline_min_m"] < 44 and by["MAGIC-II-LST-1"]["vis2_max"] < 0.3   # crosses the null at low altitude
    assert b["nights"][0]["snr"] < 3                                   # sidelobes only: ~1 sigma per night
    c = b["compare_models"]["lte05000-0.00-0.0"]
    assert c["drawn_diameter_mas"] == pytest.approx(2.733 * 1.165, rel=0.01)   # the supergiant's outer radius
    assert c["sigma_per_night"] < 0.5 and c["nights_to_3sigma"] > 100
    assert (out / "table_eps_tau.md").exists() and (out / "table_nights_tet1_tau.md").exists()
