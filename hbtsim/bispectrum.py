"""Complex visibilities, the triple product, and the closure phase.

Two-telescope intensity interferometry measures only |V|^2 -- the Fourier
phase of the source is lost.  Three telescopes measure the triple
correlation of thermal light (Wick's theorem for Gaussian fields):

    g3 = 1 + |g12|^2 + |g23|^2 + |g31|^2 + 2 Re[g12 g23 g31],

whose last term is the bispectrum: amplitude 2|g12 g23 g31| and phase
phi_c = arg(g12 g23 g31), the CLOSURE PHASE (Malvimat, Wucknitz & Saha
2013; Nunez & Domiciano de Souza 2015; Zmija et al. 2025).  Because the
intensity correlation is real, only cos(phi_c) is observable.  The
closure phase is immune to any per-telescope phase (atmosphere,
geometry) and to translations of the source -- the vector baselines
close, B12 + B23 + B31 = 0, so linear phase ramps cancel -- making it
the phase observable usable for image reconstruction.

The complex degree of coherence gamma_ij is sampled exactly from the
rendered image by hbtsim.hbt (K-point DFT; the same call that gives
|V|^2 for the pair case), or from the analytic two-disk model out of
eclipse.  This module also defines the telescope-triangle geometry, with
the Maunakea Subaru + Keck I + Keck II triangle built in (site
coordinates give pairwise distances 152.1 / 84.9 / 225.9 m, confirming
the nominal 150 / 85 / 225 m) and the four VLT Unit Telescopes.
Baselines are projected assuming a flat layout with the source at zenith
-- a documented simplification (summit elevations agree to ~20 m;
hour-angle projection is planned).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .hbt import vis_of_baselines
from .orbit import SkyPositions, sky_positions
from .params import MAS, BinarySystem, GridConfig, planck, require_out_of_eclipse
from .snr import KECK, SPAD_LAMBDA, SUBARU, Detector, Telescope
from .spectral import spectral_vis


# ---------------------------------------------------------------------------
# Analytic complex visibility of the binary (validation / fast path)
# ---------------------------------------------------------------------------
def binary_vis_complex_analytic(bvecs_m, wavelength_nm, system: BinarySystem,
                                pos: SkyPositions) -> np.ndarray:
    """Complex V at baseline vectors (K, 2) [m, East/North] for two
    non-overlapping limb-darkened disks at the epoch's sky positions:

        V(u) = [f1 V1 e^{-2 pi i u.theta1} + f2 V2 e^{-2 pi i u.theta2}]
               / (f1 + f2),

    with per-star fluxes f_s = B_lambda(T_s) theta_s^2 (1 - u_s/3) and
    LD disk visibilities V_s.  wavelength_nm scalar -> (K,); array ->
    (n_lambda, K).  Raises ValueError in (or near) eclipse."""
    from .limbdark import visibility_ld_disk

    require_out_of_eclipse(system, float(pos.rho),
                           "the analytic complex visibility",
                           "method='render' / spectral_bispectrum")
    scalar_lam = np.ndim(wavelength_nm) == 0
    lam_nm = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))[:, None]
    lam = lam_nm * 1e-9                                   # (n_l, 1)
    b = np.atleast_2d(np.asarray(bvecs_m, dtype=float))   # (K, 2)
    u = b[None, :, :] / lam[:, :, None]                   # (n_l, K, 2)
    b_len = np.hypot(b[:, 0], b[:, 1])[None, :]

    stars = (system.primary, system.secondary)
    positions = (np.array([float(pos.x1), float(pos.y1)]) * MAS,
                 np.array([float(pos.x2), float(pos.y2)]) * MAS)
    out = np.zeros((lam.shape[0], b.shape[0]), dtype=complex)
    norm = np.zeros((lam.shape[0], 1))
    for s, th_pos in zip(stars, positions):
        theta_d = 2.0 * system.angular_radius_mas(s) * MAS
        u_s = np.atleast_1d(s.ld_coeff(lam_nm[:, 0]))[:, None]
        f_s = planck(lam, s.teff) * theta_d**2 * (1.0 - u_s / 3.0)
        v_s = visibility_ld_disk(np.pi * theta_d * b_len / lam, u_s)
        phase = np.exp(-2j * np.pi * (u[..., 0] * th_pos[0] + u[..., 1] * th_pos[1]))
        out += f_s * v_s * phase
        norm += f_s
    out /= norm
    return out[0] if scalar_lam else out


# ---------------------------------------------------------------------------
# Telescope triangle geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Station:
    name: str
    east_m: float
    north_m: float
    telescope: Telescope
    detector: Detector = SPAD_LAMBDA


@dataclass(frozen=True)
class Triangle:
    """Three stations; baselines are the closed cycle B12, B23, B31
    (vector sum identically zero).  Flat layout, source at zenith."""
    stations: tuple

    def baseline_vectors(self) -> np.ndarray:
        """(3, 2) [m]: B12 = p2 - p1, B23 = p3 - p2, B31 = p1 - p3."""
        p = np.array([[s.east_m, s.north_m] for s in self.stations])
        return np.array([p[1] - p[0], p[2] - p[1], p[0] - p[2]])

    def baseline_lengths(self) -> np.ndarray:
        b = self.baseline_vectors()
        return np.hypot(b[:, 0], b[:, 1])


# Maunakea: ENU positions relative to Subaru, from site coordinates
# (Subaru 19d49m32s N 155d28m34s W; Keck I 19.8259465 N 155.474719 W;
# Keck II 19.8265606 N 155.474234 W).  Pairwise: Subaru-Keck I 152.1 m,
# Keck I-Keck II 84.9 m, Keck II-Subaru 225.9 m.
MAUNAKEA_SUBARU_KECK = Triangle((
    Station("Subaru", 0.0, 0.0, SUBARU),
    Station("Keck I", 145.8, 43.3, KECK),
    Station("Keck II", 196.6, 111.3, KECK),
))


@dataclass(frozen=True)
class Array:
    """N stations: all pairwise baselines and all telescope triangles.
    Of the C(N,3) triangles, only (N-1)(N-2)/2 closure phases are
    independent, but every triangle's triple-coincidence stream carries
    (largely) independent accidental noise, so all contribute to the
    detection sensitivity."""
    stations: tuple

    def pairs(self):
        """[(i, j, baseline_vector), ...] for i < j."""
        out = []
        for i in range(len(self.stations)):
            for j in range(i + 1, len(self.stations)):
                si, sj = self.stations[i], self.stations[j]
                out.append((i, j, np.array([sj.east_m - si.east_m,
                                            sj.north_m - si.north_m])))
        return out

    def triangles(self):
        """All C(N,3) Triangle objects."""
        from itertools import combinations
        return [Triangle((self.stations[i], self.stations[j],
                          self.stations[k]))
                for i, j, k in combinations(range(len(self.stations)), 3)]


# The four VLT Unit Telescopes (8.2 m) at Paranal, published VLTI station
# (E, N) coordinates [m]; pairwise separations 46.6 (UT2-UT3) to 130.2 m
# (UT1-UT4).  NOTE Paranal is at latitude -24.6 deg: Algol and Beta Aur
# (dec ~ +41/+45 deg) culminate below ~25 deg altitude and are not useful
# targets from there; Spica (dec -11 deg) transits at ~77 deg.
VLT_UT = Array(tuple(
    Station(name, e, n, Telescope(diameter_m=8.2, throughput=0.3))
    for name, (e, n) in (("UT1", (-9.925, -20.335)),
                         ("UT2", (14.887, 30.502)),
                         ("UT3", (44.915, 66.183)),
                         ("UT4", (103.306, 43.999)))))


def equilateral_triangle(side_m: float, telescope: Telescope = KECK,
                         detector: Detector = SPAD_LAMBDA) -> Triangle:
    """Hypothetical compact comparison array: three identical telescopes
    on an equilateral triangle of the given side."""
    h = side_m * np.sqrt(3.0) / 2.0
    return Triangle((
        Station("T1", 0.0, 0.0, telescope, detector),
        Station("T2", side_m, 0.0, telescope, detector),
        Station("T3", side_m / 2.0, h, telescope, detector),
    ))


# ---------------------------------------------------------------------------
# Closure phase of the model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BispectrumResult:
    gammas: np.ndarray       # complex (3,): gamma_12, gamma_23, gamma_31
    bispectrum: complex      # gamma_12 * gamma_23 * gamma_31
    triple_amp: float        # |gamma_12 gamma_23 gamma_31|
    phi_c: float             # closure phase [rad]
    cos_phi_c: float
    wavelength_nm: float
    orbital_phase: float


def _resolve_method(method: str) -> str:
    if method == "fft":
        warnings.warn("method='fft' is now method='render' (exact DFT of the "
                      "rendered image)", DeprecationWarning, stacklevel=3)
        return "render"
    if method not in ("analytic", "render"):
        raise ValueError(f"unknown method {method!r} "
                         f"(expected 'analytic' or 'render')")
    return method


def closure_phase(system: BinarySystem, triangle: Triangle,
                  wavelength_nm: float, orbital_phase: float = 0.0,
                  method: str = "analytic",
                  grid: GridConfig = GridConfig()) -> BispectrumResult:
    """Model gammas and closure phase on the triangle at one epoch.

    method="analytic" (instant; out of eclipse only) or "render" (renders
    the binary and samples its exact DFT; valid at all phases)."""
    method = _resolve_method(method)
    pos = SkyPositions(*(np.asarray(v) for v in
                         sky_positions(2 * np.pi * orbital_phase, system)))
    bvecs = triangle.baseline_vectors()

    if method == "analytic":
        gam = binary_vis_complex_analytic(bvecs, wavelength_nm, system, pos)
    else:
        from .render import render_image
        img = render_image(pos, system, wavelength_nm, grid)
        gam = np.asarray(vis_of_baselines(img, bvecs, wavelength_nm * 1e-9, grid))

    bis = complex(gam[0] * gam[1] * gam[2])
    return BispectrumResult(gammas=np.asarray(gam), bispectrum=bis,
                            triple_amp=abs(bis), phi_c=float(np.angle(bis)),
                            cos_phi_c=float(np.cos(np.angle(bis))),
                            wavelength_nm=wavelength_nm,
                            orbital_phase=orbital_phase)


# ---------------------------------------------------------------------------
# Batched multi-wavelength gammas (GPU; eclipse-capable)
# ---------------------------------------------------------------------------
def spectral_bispectrum(pos: SkyPositions, triangle: Triangle,
                        wavelengths_nm, system: BinarySystem,
                        grid: GridConfig, *,
                        chunk_size: int | None = None):
    """Complex (gamma_12, gamma_23, gamma_31) for every wavelength channel
    at one epoch via the batched render + DFT pipeline -- (n_lambda, 3)
    complex64.  Valid at all orbital phases including eclipses."""
    return spectral_vis(pos, triangle.baseline_vectors(), wavelengths_nm,
                        system, grid, chunk_size=chunk_size)
