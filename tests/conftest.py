"""Shared test configuration: enable float64 in JAX once, before any
module is imported, so array dtypes do not depend on collection order
(the renderer and DFT kernels force float32 internally regardless); and
the catalog objects every test module used to import as Python presets.

Tests that need an object at module level (parametrize) build their own
`Catalog(env=False)`; functions take the fixtures below."""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from hbtsim.catalog import Catalog  # noqa: E402

CAT = Catalog(env=False)


@pytest.fixture(scope="session")
def catalog():
    return CAT


def _target(name):
    return pytest.fixture(scope="session")(lambda: CAT.load_target(name))


def _kind(kind, name, **kw):
    return pytest.fixture(scope="session")(lambda: CAT.load(kind, name, **kw))


# targets
beta_aur = _target("betaaur")
algol = _target("algol")
spica = _target("spica")
delta_vel = _target("deltavel")
sirius_a = _target("sirius_a")
vega = _target("vega")
hd17652 = _target("hd17652")
hd360 = _target("hd360")
gamma_cas = _target("gammacas")
sirius_b = _target("sirius_b")


@pytest.fixture(scope="session")
def systems():
    return {k: CAT.load_target(k) for k in ("betaaur", "algol", "spica", "deltavel")}


# telescopes
c2pu = _kind("telescope", "c2pu_1m")
keck = _kind("telescope", "keck_10m")
subaru = _kind("telescope", "subaru_8p2m")
vlt_tel = _kind("telescope", "vlt_ut_8p2m")
eonsii_tel = _kind("telescope", "eonsii_4m")
kk_tel = _kind("telescope", "kk_4m")
veritas_tel = _kind("telescope", "veritas_12m")

# detectors
spad_lambda = _kind("detector", "spad_lambda")
spad_lambda_ng = _kind("detector", "spad_lambda_ng")
eonsii_mcp = _kind("detector", "eonsii_mcp_pmt")
eonsii_spad = _kind("detector", "eonsii_spad")
eonsii_spad_correlator = _kind("detector", "eonsii_spad_correlator")
kk_det = _kind("detector", "kk_ideal")

# spectrographs
spec320 = _kind("spectrograph", "spad_lambda_320")
spec_r5000 = _kind("spectrograph", "r5000_400_950")
eonsii_spec = _kind("spectrograph", "eonsii_1000ch")
eonsii_spec_r7500 = _kind("spectrograph", "eonsii_r7500")


@pytest.fixture(scope="session")
def kk_filters():
    """{band: one-channel Spectrograph} for the Kim & Kaiser broad bands."""
    return {b: CAT.load_spectrograph(f"filter_{s}_{b.lower()}")
            for b, s in (("B", "johnson"), ("V", "johnson"), ("R", "cousins"), ("I", "cousins"),
                         ("H", "2mass"), ("K", "2mass"))}


# sites
maunakea = _kind("site", "maunakea")
paranal = _kind("site", "paranal")
teide = _kind("site", "teide")
flwo = _kind("site", "flwo")
orm = _kind("site", "orm")

# arrays
maunakea_arr = _kind("array", "maunakea_subaru_keck")
vlt_ut = _kind("array", "vlt_ut")
keck_pair = _kind("array", "keck_pair")
veritas = _kind("array", "veritas")
magic_lst1 = _kind("array", "magic_lst1")
eonsii_pair = _kind("array", "eonsii_pair_teide")          # 100 m, Teide
eonsii_tri = _kind("array", "eonsii_triangle_paranal")     # 20 m, Paranal


@pytest.fixture(scope="session")
def maunakea_tri():
    return CAT.load_triangle("maunakea_subaru_keck")


# backends
veritas_sii = _kind("backend", "veritas_sii")
magic_sii = _kind("backend", "magic_sii")


@pytest.fixture(scope="session")
def g3_backends():
    """The four closure-phase backends of the feasibility study."""
    return tuple(CAT.load_backend(n) for n in ("spad320_timetag", "spad320_correlator",
                                                "r5000_correlator", "r5000_correlator_pbs"))


@pytest.fixture(scope="session")
def eonsii_backends():
    return tuple(CAT.load_backend(n) for n in ("eonsii_mcp", "eonsii_spad", "eonsii_spad_pbs",
                                                "eonsii_r7500_spad"))
