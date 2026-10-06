"""Imaging atmospheric Cherenkov telescope (IACT) arrays as intensity
interferometers: VERITAS (four 12 m dishes, Arizona) and MAGIC + LST-1
(17 + 17 + 23 m, La Palma), with the groups' own analog-photomultiplier
sensitivity formula, and per-pair |V|^2 tracks over a night.

These systems digitize photomultiplier currents (250-500 MS/s) and
correlate the streams; they are not photon counters, so their S/N is
taken from their published form (MAGIC, Abe et al. 2024, MNRAS 529,
4387, Eq. 4; the same expression in Raiola et al. 2025, PoS ICRC2025 957):

    S/N = A alpha q n_nu |V|^2 sqrt(b_el) sigma_spec / (sqrt(2) F (1 + beta)) sqrt(T)

with A the mirror area (sqrt(A1 A2) for unequal dishes), alpha the
quantum efficiency, q the optical efficiency, n_nu the photon spectral
flux [photons m^-2 s^-1 Hz^-1], b_el the electronic bandwidth, sigma_spec
the normalized spectral factor of the passband, F the excess noise
factor and beta the background-to-starlight ratio.  hbtsim supplies only
the visibility model (binary fringe, limb-darkened disks, pupil
averaging); the sensitivity constants are theirs.  The VERITAS optical
efficiency is not published here, so calibrate_q solves for it from a
published precision (sigma(|V|^2) = 0.016 per pair in 4.25 h on epsilon
Ori, Abeysekara et al. 2020).

Assumptions are flagged in the preset comments; see docs/iact_targets.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .aperture import PupilQuadrature, fringe_period_m, pupil_pair_quadrature, pupil_pair_quadrature_for
from .bispectrum import Array, Station, binary_vis_complex_analytic
from .geometry import FLWO, ORM, hour_angle_blocks, hour_angle_window
from .orbit import positions_at
from .params import AB_ZERO_FNU, C_LIGHT, DAY, H_PLANCK, MAS
from .snr import Detector, Telescope

# ---------------------------------------------------------------------------
# Telescopes.  diameter_m sets the pupil smearing, collecting_area_m2 the
# photon rate; throughput is not used by the classic S/N (q is), so it is
# set to the optical efficiency for consistency with the photon-counting path.
# ---------------------------------------------------------------------------
VERITAS_TELESCOPE = Telescope(diameter_m=12.0, throughput=0.3, collecting_area_m2=110.0)   # 345 facets; area ASSUMED
MAGIC_TELESCOPE = Telescope(diameter_m=17.0, throughput=0.304, collecting_area_m2=236.0)   # Abe et al. 2024
LST1_TELESCOPE = Telescope(diameter_m=23.0, throughput=0.304, collecting_area_m2=390.0)    # area and efficiency ASSUMED as MAGIC's

# Analog PMT "detectors" for the photon-counting cross-check path only: the
# electronic bandwidth as an equivalent Gaussian pair kernel (estimators.
# matched_filter_equivalents: sigma_pair = 1/(sqrt(pi) b_el)), no dead
# time, no link ceiling.
def _analog_pmt(name, qe, b_el_hz):
    sigma_pair = 1.0 / (np.sqrt(np.pi) * b_el_hz)
    fwhm_ps = sigma_pair / np.sqrt(2.0) * 2.3548200450309493 * 1e12
    return Detector(name=name, pde_table_nm=((300.0, qe), (700.0, qe)), jitter_fwhm_ps=fwhm_ps,
                    dead_time_ns=0.0, dark_cps_per_pixel=0.0, readout="correlator", max_total_cps=None)


VERITAS_PMT = _analog_pmt("VERITAS R10560 PMT + 250 MS/s (analog, 125 MHz)", 0.30, 125e6)
MAGIC_PMT = _analog_pmt("MAGIC PMT + 500 MS/s (analog, 110 MHz)", 0.295, 110e6)


@dataclass(frozen=True)
class PrecisionAnchor:
    """A published precision used to calibrate the optical efficiency q:
    sigma(|V|^2) per pair after t_s seconds on a star of AB magnitude
    mag_ab in the backend's passband."""
    star: str
    mag_ab: float
    sigma_vis2: float
    t_s: float


@dataclass(frozen=True)
class IACTBackend:
    """One group's filter and sensitivity constants (their Eq. 4)."""
    name: str
    lambda_nm: float
    dlambda_nm: float
    alpha: float              # quantum efficiency at the passband
    q: float                  # optical efficiency of the rest of the system
    b_el_hz: float            # electronic bandwidth
    noise_factor: float = 1.0 # F
    sigma_spec: float = 1.0   # normalized spectral factor of the passband
    beta: float = 0.0         # background-to-starlight ratio
    time_resolution_ns: float = 4.0
    anchor: PrecisionAnchor | None = None   # for calibrated()


# VERITAS: 416 nm / 13 nm effective passband (Abeysekara et al. 2020; 10 nm in
# the 2025 gamma Cas paper), Hamamatsu R10560 QE ~0.30, 250 MS/s, 4 ns time
# resolution.  q, F and sigma_spec are not published: q is calibrated from
# their quoted precision (calibrate_q) with F = sigma_spec = 1 absorbed.
VERITAS_SII = IACTBackend("VERITAS SII (416/13 nm)", 416.0, 13.0, alpha=0.30, q=0.25, b_el_hz=125e6,
                          noise_factor=1.0, sigma_spec=1.0, time_resolution_ns=4.0)
# MAGIC: Semrock 425/26, QE 0.295, q 0.304, b_el 110 MHz effective (125 MHz
# anti-aliasing), F 1.15, sigma 0.87 (Abe et al. 2024, Table 4 and Eq. 4).
MAGIC_SII = IACTBackend("MAGIC SII (425/26 nm)", 425.0, 26.0, alpha=0.295, q=0.304, b_el_hz=110e6,
                        noise_factor=1.15, sigma_spec=0.87, time_resolution_ns=2.0)

# Published VERITAS precision anchor: sigma(|g|^2) = 2e-8 on N0 = 1.26e-6,
# i.e. sigma(|V|^2) = 0.016 per telescope pair over the full 4.25 h epsilon
# Ori data set (B = 1.50; Abeysekara et al. 2020).
VERITAS_ANCHOR = dict(star="eps Ori", mag_b=1.50, sigma_vis2=0.016, t_s=4.25 * 3600.0)

# ---------------------------------------------------------------------------
# Arrays.  VERITAS positions (E, N in m) are read from Fig. 1 of Abeysekara
# et al. 2020 and adjusted so the separations match the published baselines
# (81.5, 99.4, 108.8, 126.4, 172.5 m and 106 m): T1 front-centre, T2 left,
# T3 right, T4 back-centre.  MAGIC-I to MAGIC-II is 86 m; LST-1 adds two
# baselines of ~100 m (Raiola et al. 2025).  The ORIENTATION of the MAGIC
# triangle on the ground is ASSUMED (MAGIC-I -> MAGIC-II due east, LST-1 to
# the north); it affects which position angles each pair samples.
# ---------------------------------------------------------------------------
VERITAS_POSITIONS_M = {"T1": (134.8, -8.0), "T2": (43.4, -47.0), "T3": (28.8, 60.9), "T4": (-36.6, 12.1)}
# -> T3T4 81.5, T1T2 99.4, T2T4 99.4, T2T3 108.8, T1T3 126.4, T1T4 172.5 m
VERITAS = Array(tuple(Station(n, e, nn, VERITAS_TELESCOPE, VERITAS_PMT) for n, (e, nn) in VERITAS_POSITIONS_M.items()),
                site=FLWO)
MAGIC_LST1_POSITIONS_M = {"MAGIC-I": (0.0, 0.0), "MAGIC-II": (86.0, 0.0), "LST-1": (43.0, 90.3)}
MAGIC_LST1 = Array((Station("MAGIC-I", 0.0, 0.0, MAGIC_TELESCOPE, MAGIC_PMT),
                    Station("MAGIC-II", 86.0, 0.0, MAGIC_TELESCOPE, MAGIC_PMT),
                    Station("LST-1", 43.0, 90.3, LST1_TELESCOPE, MAGIC_PMT)), site=ORM)


# ---------------------------------------------------------------------------
# Their sensitivity formula
# ---------------------------------------------------------------------------
VEGA_TO_AB_B = -0.09      # Vega -> AB in the B band (Blanton & Roweis 2007)


def photon_spectral_flux(mag_ab, wavelength_nm):
    """n_nu [photons m^-2 s^-1 Hz^-1] from an AB magnitude."""
    nu = C_LIGHT / (np.asarray(wavelength_nm, dtype=float) * 1e-9)
    return AB_ZERO_FNU * 10.0 ** (-0.4 * np.asarray(mag_ab, dtype=float)) / (H_PLANCK * nu)


def classic_snr(vis2, mag_ab, area1_m2, area2_m2, backend: IACTBackend, t_s, q=None):
    """Their S/N (module docstring) for |V|^2 (array-capable) on a pair of
    dishes; q overrides the backend's optical efficiency."""
    q = backend.q if q is None else q
    n_nu = photon_spectral_flux(mag_ab, backend.lambda_nm)
    a = np.sqrt(area1_m2 * area2_m2)
    return (a * backend.alpha * q * n_nu * np.asarray(vis2, dtype=float) * np.sqrt(backend.b_el_hz)
            * backend.sigma_spec / (np.sqrt(2.0) * backend.noise_factor * (1.0 + backend.beta)) * np.sqrt(t_s))


def vis2_sigma(mag_ab, area1_m2, area2_m2, backend: IACTBackend, t_s, q=None):
    """sigma(|V|^2) in t_s: 1 / S/N at |V|^2 = 1 (noise is independent of |V|^2)."""
    return 1.0 / classic_snr(1.0, mag_ab, area1_m2, area2_m2, backend, t_s, q)


def calibrate_q(backend: IACTBackend, area_m2, mag_ab, sigma_vis2, t_s) -> float:
    """The optical efficiency q for which vis2_sigma reproduces a published
    precision (S/N is linear in q)."""
    return float(backend.q * vis2_sigma(mag_ab, area_m2, area_m2, backend, t_s) / sigma_vis2)


def calibrated(backend: IACTBackend, area_m2: float) -> IACTBackend:
    """backend with q set so that vis2_sigma reproduces backend.anchor on
    telescopes of the given area; a backend without an anchor is returned
    unchanged."""
    from dataclasses import replace
    a = backend.anchor
    if a is None:
        return backend
    return replace(backend, q=calibrate_q(backend, area_m2, a.mag_ab, a.sigma_vis2, a.t_s))


def veritas_calibrated() -> IACTBackend:
    """VERITAS_SII with q set by the epsilon Ori anchor."""
    from dataclasses import replace
    q = calibrate_q(VERITAS_SII, VERITAS_TELESCOPE.area_m2, VERITAS_ANCHOR["mag_b"] + VEGA_TO_AB_B,
                    VERITAS_ANCHOR["sigma_vis2"], VERITAS_ANCHOR["t_s"])
    return replace(VERITAS_SII, q=q)


# ---------------------------------------------------------------------------
# Per-pair |V|^2 along a night
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PairTrack:
    pair_names: tuple
    diameters_m: tuple            # (d_i, d_j) per pair
    areas_m2: tuple               # (A_i, A_j) per pair
    hour_angle_h: np.ndarray      # (n_blk,)
    phase: np.ndarray             # (n_blk,) orbital phase (0 for single stars)
    baseline_vec_m: np.ndarray    # (n_pair, n_blk, 2) projected (u, v) [m]
    baseline_len_m: np.ndarray    # (n_pair, n_blk)
    vis2: np.ndarray              # (n_pair, n_blk)
    block_s: float

    @property
    def position_angle_deg(self) -> np.ndarray:
        """PA of each projected baseline, east of north, (n_pair, n_blk)."""
        return np.degrees(np.arctan2(self.baseline_vec_m[..., 0], self.baseline_vec_m[..., 1]))


def pair_track(array: Array, dec_deg: float, lambda_nm: float, vis2_fn, *, block_minutes: float = 17.0,
               hour_angle_window_h=None, min_alt_deg: float = 30.0, phase0: float = 0.0,
               period_days: float | None = None) -> PairTrack:
    """|V|^2 of every telescope pair at every block of a night.  vis2_fn(bvecs
    (K, 2) [m], lambda_nm, phase, diameters (d_i, d_j)) -> (K,) |V|^2.  With
    period_days the orbital phase advances from phase0 through the night.
    Fringe drift within a block is irrelevant at nanosecond timing and is
    not applied."""
    h0, h1 = (hour_angle_window(dec_deg, array.site.latitude_deg, min_alt_deg)
              if hour_angle_window_h is None else hour_angle_window_h)
    mids = hour_angle_blocks(h0, h1, block_minutes)
    block_s = (h1 - h0) / max(mids.size, 1) * 3600.0
    pairs0 = array.pairs()
    names = tuple(f"{array.stations[i].name}-{array.stations[j].name}" for i, j, _ in pairs0)
    diam = tuple((array.stations[i].telescope.diameter_m, array.stations[j].telescope.diameter_m) for i, j, _ in pairs0)
    areas = tuple((array.stations[i].telescope.area_m2, array.stations[j].telescope.area_m2) for i, j, _ in pairs0)
    bv = np.zeros((len(pairs0), mids.size, 2)); v2 = np.zeros((len(pairs0), mids.size)); ph = np.zeros(mids.size)
    for k, H in enumerate(mids):
        phase = phase0 + ((H - h0) * 3600.0 / (period_days * DAY) if period_days else 0.0)
        ph[k] = phase
        proj = array.projected(float(H), dec_deg).pairs()
        for p, (i, j, b) in enumerate(proj):
            bv[p, k] = b
            v2[p, k] = float(np.asarray(vis2_fn(b[None, :], lambda_nm, phase, diam[p]))[0])
    return PairTrack(names, diam, areas, mids, ph, bv, np.hypot(bv[..., 0], bv[..., 1]), v2, block_s)


def binary_vis2_fn(system, pupils: bool = True):
    """vis2_fn for pair_track: the analytic two-disk |V|^2 of `system` at the
    orbital phase, averaged over the pair's pupils when pupils=True."""
    def fn(bvecs, lambda_nm, phase, diameters):
        pos = positions_at(system, phase)
        b = np.asarray(bvecs, dtype=float)
        if not pupils:
            return np.abs(binary_vis_complex_analytic(b, lambda_nm, system, pos)) ** 2
        quad = pupil_pair_quadrature_for(diameters[0], diameters[1], fringe_period_m(float(pos.rho), lambda_nm))
        pts = quad.points(b).reshape(-1, 2)
        v = binary_vis_complex_analytic(pts, lambda_nm, system, pos)
        return quad.reduce(np.abs(v).reshape(b.shape[0], -1) ** 2)
    return fn


def single_vis2_fn(vis_fn, pupils: bool = True):
    """vis2_fn for pair_track from a complex visibility function
    vis_fn(bvecs (K, 2), lambda_nm) -> (K,) (e.g. single.composite_vis)."""
    def fn(bvecs, lambda_nm, phase, diameters):
        b = np.asarray(bvecs, dtype=float)
        if not pupils:
            return np.abs(vis_fn(b, lambda_nm)) ** 2
        quad = pupil_pair_quadrature(diameters[0], diameters[1], n_r=7, n_theta=16)
        pts = quad.points(b).reshape(-1, 2)
        return quad.reduce(np.abs(vis_fn(pts, lambda_nm)).reshape(b.shape[0], -1) ** 2)
    return fn


def track_sigma(track: PairTrack, mag_ab, backend: IACTBackend, t_s=None, q=None) -> np.ndarray:
    """sigma(|V|^2) per pair (n_pair,) in t_s (default one block)."""
    t = track.block_s if t_s is None else t_s
    return np.array([vis2_sigma(mag_ab, a1, a2, backend, t, q) for a1, a2 in track.areas_m2])
