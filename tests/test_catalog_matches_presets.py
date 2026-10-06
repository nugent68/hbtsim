"""Phase (a) of the catalog migration: every shipped JSON definition builds
the same object as the Python preset it replaces.  Documented exceptions:
Telescope / Spectrograph labels (the presets had none), the gamma Cas
radius (derived from theta_LD and distance instead of the preset's
inconsistent 10 Rsun), the ellipse and precision anchor now carried on
the objects, and binary anchors keyed by wavelength instead of band.
This file is deleted when the presets go (phase b) and replaced by
test_catalog_values.py."""

from dataclasses import replace

import numpy as np
import pytest

from hbtsim import bispectrum as B
from hbtsim import diameter as D
from hbtsim import geometry as G
from hbtsim import iact as I
from hbtsim import params as P
from hbtsim import single as SG
from hbtsim import snr as S
from hbtsim import snr3
from hbtsim.catalog import Catalog
from hbtsim.serialize import to_json

cat = Catalog(env=False)


def norm(obj):
    """to_json with the labels of Telescope / Spectrograph removed and the
    binaries' band-named anchors converted to wavelengths."""
    j = to_json(obj, mode="key")

    def walk(x):
        if isinstance(x, dict):
            cls = x.get("__class__")
            if cls in ("Telescope", "Spectrograph"):
                x = {k: v for k, v in x.items() if k != "name"}
            if cls in ("BinarySystem",) and "mag_anchors" in x:
                x["mag_anchors"] = [[S.anchor_wavelength_nm(k), m] for k, m in x["mag_anchors"]]
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x
    return walk(j)


def same(a, b):
    assert norm(a) == norm(b)


@pytest.mark.parametrize("name,preset", [("betaaur", P.BETA_AUR), ("algol", P.ALGOL),
                                         ("spica", P.SPICA), ("deltavel", P.DELTA_VEL)])
def test_binaries(name, preset):
    same(cat.load_target(name), preset)
    assert cat.load_target(name).mag_anchors == tuple(
        (S.anchor_wavelength_nm(b), m) for b, m in preset.mag_anchors)


@pytest.mark.parametrize("name,preset", [("sirius_a", SG.SIRIUS_A), ("vega", SG.VEGA),
                                         ("hd17652", SG.HD_17652), ("hd360", SG.HD_360)])
def test_singles(name, preset):
    same(cat.load_target(name), preset)


def test_gamma_cas():
    built = cat.load_target("gammacas")
    e = SG.GAMMA_CAS_ELLIPSE
    radius = 0.5 * SG.GAMMA_CAS.theta_ld_mas * P.MAS * SG.GAMMA_CAS.distance_pc * P.PARSEC / P.R_SUN
    expect = replace(SG.GAMMA_CAS, star=replace(SG.GAMMA_CAS.star, radius_rsun=radius),
                     ellipse=SG.Ellipse(e["theta_major_mas"], e["axis_ratio"], e["pa_deg"]))
    same(built, expect)
    assert built.star.radius_rsun == pytest.approx(9.63, abs=0.01)
    assert built.star.log_g == 3.5


def test_uniform_disks():
    sb = cat.load_target("sirius_b")
    assert (sb.theta_mas, sb.mag_ab) == (0.0285, 8.44)
    assert sb.dec_deg == -16.716


def test_ld_tables():
    assert cat.load_ld_table("claret11_betaaur") == P.LD_BETA_AUR
    assert cat.load_ld_table("claret11_algol_a") == P.LD_ALGOL_A
    assert cat.load_ld_table("claret11_algol_b") == P.LD_ALGOL_B
    assert cat.load_ld_table("claret11_spica") == P.LD_SPICA
    assert cat.load_ld_table("k_giant_rough") == SG.LD_K_GIANT


@pytest.mark.parametrize("name,preset", [
    ("c2pu_1m", S.C2PU), ("keck_10m", S.KECK), ("subaru_8p2m", S.SUBARU),
    ("eonsii_4m", S.EON_SII_TELESCOPE), ("kk_4m", D.KK_TELESCOPE),
    ("veritas_12m", I.VERITAS_TELESCOPE), ("magic_17m", I.MAGIC_TELESCOPE), ("lst1_23m", I.LST1_TELESCOPE),
    ("vlt_ut_8p2m", B.VLT_UT.stations[0].telescope)])
def test_telescopes(name, preset):
    same(cat.load_telescope(name), preset)
    assert cat.load_telescope(name).name


@pytest.mark.parametrize("name,preset", [
    ("spad_lambda", S.SPAD_LAMBDA), ("spad_lambda_ng", S.SPAD_LAMBDA_NG),
    ("eonsii_mcp_pmt", S.EONSII_MCP_PMT), ("eonsii_spad", S.EONSII_SPAD), ("kk_ideal", D.KK_DETECTOR),
    ("veritas_pmt", I.VERITAS_PMT), ("magic_pmt", I.MAGIC_PMT)])
def test_detectors(name, preset):
    assert cat.load_detector(name) == preset
    assert cat.load_detector(name).jitter_fwhm_ps == preset.jitter_fwhm_ps     # bitwise


def test_detector_extends():
    d = cat.load_detector("eonsii_spad_correlator")
    assert d.readout == "correlator" and d.max_total_cps is None
    assert d.pde_table_nm == S.EONSII_SPAD.pde_table_nm and d.jitter_fwhm_ps == 20.0


@pytest.mark.parametrize("name,preset", [
    ("spad_lambda_320", S.Spectrograph(n_channels=320)),
    ("r5000_400_950", S.Spectrograph.from_resolving_power(5000.0)),
    ("eonsii_1000ch", S.EONSII_SPECTROGRAPH), ("eonsii_r7500", S.EONSII_SPECTROGRAPH_R7500),
    ("filter_johnson_b", D.BANDS["B"]), ("filter_johnson_v", D.BANDS["V"]), ("filter_cousins_r", D.BANDS["R"]),
    ("filter_cousins_i", D.BANDS["I"]), ("filter_2mass_h", D.BANDS["H"]), ("filter_2mass_k", D.BANDS["K"])])
def test_spectrographs(name, preset):
    same(cat.load_spectrograph(name), preset)


def test_hbeta_window_matches_script():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "chromatic_diameters", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "chromatic_diameters.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    same(cat.load_spectrograph("hbeta_window_r5000"), mod.hbeta_window_spectrograph())


def test_counting_backends_match_deltavel_script():
    expect = (snr3.Backend("1000 ch, MCP-PMT", S.EONSII_SPECTROGRAPH, S.EONSII_MCP_PMT),
              snr3.Backend("1000 ch, QUASAR SPAD", S.EONSII_SPECTROGRAPH, S.EONSII_SPAD),
              snr3.Backend("1000 ch, QUASAR SPAD + PBS", S.EONSII_SPECTROGRAPH, S.EONSII_SPAD, "pbs"),
              snr3.Backend("R = 7500 (2388 ch), QUASAR SPAD", S.EONSII_SPECTROGRAPH_R7500, S.EONSII_SPAD))
    for name, b in zip(("eonsii_mcp", "eonsii_spad", "eonsii_spad_pbs", "eonsii_r7500_spad"), expect):
        same(cat.load_backend(name), b)
    g3 = [("spad320_timetag", S.Spectrograph(n_channels=320), S.SPAD_LAMBDA, "unpolarized"),
          ("spad320_correlator", S.Spectrograph(n_channels=320), S.SPAD_LAMBDA_NG, "unpolarized"),
          ("r5000_correlator", S.Spectrograph.from_resolving_power(5000.0), S.SPAD_LAMBDA_NG, "unpolarized"),
          ("r5000_correlator_pbs", S.Spectrograph.from_resolving_power(5000.0), S.SPAD_LAMBDA_NG, "pbs")]
    for name, spec, det, pol in g3:
        b = cat.load_backend(name)
        same(b.spectrograph, spec)
        assert b.detector == det and b.polarization_mode == pol


def test_analog_backends():
    anchor = I.PrecisionAnchor("eps Ori", I.VERITAS_ANCHOR["mag_b"] + I.VEGA_TO_AB_B,
                               I.VERITAS_ANCHOR["sigma_vis2"], I.VERITAS_ANCHOR["t_s"])
    assert cat.load_backend("veritas_sii") == replace(I.VERITAS_SII, anchor=anchor)
    assert cat.load_backend("magic_sii") == I.MAGIC_SII
    v = I.calibrated(cat.load_backend("veritas_sii"), cat.load_telescope("veritas_12m").area_m2)
    assert v.q == pytest.approx(I.veritas_calibrated().q, rel=1e-12)


@pytest.mark.parametrize("name,preset", [("maunakea", G.MAUNAKEA), ("paranal", G.PARANAL), ("teide", G.TEIDE),
                                         ("flwo", G.FLWO), ("orm", G.ORM)])
def test_sites(name, preset):
    assert cat.load_site(name) == preset


def test_arrays():
    same(cat.load_triangle("maunakea_subaru_keck"), B.MAUNAKEA_SUBARU_KECK)
    same(cat.load_array("vlt_ut"), B.VLT_UT)
    same(cat.load_array("veritas"), I.VERITAS)
    same(cat.load_array("magic_lst1"), I.MAGIC_LST1)
    same(cat.load_array("eonsii_triangle_paranal"), B.eonsii_triangle(20.0))
    same(cat.load_array("eonsii_triangle_paranal", side_m=12.0, pa_deg=30.0),
         B.eonsii_triangle(12.0, pa_deg=30.0))
    same(cat.load_array("eonsii_pair_teide", baseline_m=120.0, pa_deg=90.0), B.eonsii_pair(120.0, pa_deg=90.0))
    same(cat.load_array("eonsii_triangle_paranal", detector=cat.load_detector("eonsii_spad")),
         B.eonsii_triangle(20.0, detector=S.EONSII_SPAD))
    kp = cat.load_array("keck_pair")
    (_, _, bvec), = kp.pairs()
    assert np.hypot(*bvec) == pytest.approx(84.9, abs=0.05)
    with pytest.raises(ValueError):
        cat.load_array("eonsii_triangle_paranal", side_m=B.EONSII_MIN_SPACING_M - 0.1)


def test_mc_config_inputs():
    from hbtsim.montecarlo import MCConfig
    c = MCConfig()
    camp = cat.load_campaign("mc_sirius_b_eonsii")
    sb = camp.target
    arr = camp.array
    assert (sb.mag_ab, sb.dec_deg) == (c.mag_ab, c.dec_deg)
    assert arr.site == c.site
    (_, _, bvec), = arr.pairs()
    assert np.hypot(*bvec) == pytest.approx(c.ground_baseline_m)
    assert np.degrees(np.arctan2(bvec[0], bvec[1])) == pytest.approx(c.ground_pa_deg)
    assert camp.option("options.theta_true_mas") == 0.0295
    assert camp.option("options.zenith_angles_deg") == list(c.zenith_angles_deg)
    assert camp.backends[0].spectrograph == c.spectrograph
    assert camp.backends[0].detector == c.detector and camp.backends[1].detector == S.EONSII_SPAD
    assert arr.stations[0].telescope.diameter_m == c.telescope.diameter_m
