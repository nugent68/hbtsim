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
Triangle.baseline_vectors() gives the flat-layout, source-at-zenith
baselines; Triangle.projected(hour_angle, dec) rotates the station
positions into the (u, v) plane for a real pointing (hbtsim.geometry).

Finite apertures: with pupils=True the triple product is the exact
three-pupil average of hbtsim.aperture (the correlator does not sample
gamma at a point), and the pair-smeared |gamma_ij|^2 that make up the
pair ridges of the triple correlation come from the same samples.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np

from .aperture import TripleQuadrature, fringe_period_m, triple_quadrature_for
from .geometry import HOUR, Site, enu_to_uv
from .hbt import vis_of_baselines
from .orbit import SkyPositions, positions_at
from .params import MAS, BinarySystem, GridConfig, require_out_of_eclipse
from .snr import Detector, Telescope
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

    with per-star fluxes f_s = F_s(lambda) theta_s^2 (model SED or pi
    B_lambda(T_s)) and LD disk visibilities V_s (linear-law series or
    the numeric transform of a tabulated profile).  wavelength_nm scalar
    -> (K,); array -> (n_lambda, K).  Raises ValueError in (or near)
    eclipse."""
    from .limbdark import star_disk_visibility

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
        theta_d = 2.0 * system.drawn_radius_mas(s) * MAS
        f_s = np.atleast_1d(s.surface_flux(lam_nm[:, 0]))[:, None] * theta_d**2
        v_s = star_disk_visibility(s, np.pi * theta_d * b_len / lam, lam_nm[:, 0])
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
    detector: Detector
    up_m: float = 0.0

    @property
    def enu_m(self) -> np.ndarray:
        return np.array([self.east_m, self.north_m, self.up_m])


def _project_stations(stations, site, hour_angle_h, dec_deg):
    """Stations at their (u, v) positions [m] for the given pointing."""
    if site is None or dec_deg is None:
        raise ValueError("projection needs a Site (Triangle/Array.site) "
                         "and the source declination")
    enu = np.array([s.enu_m for s in stations])
    uv = enu_to_uv(enu, hour_angle_h * HOUR, np.radians(dec_deg),
                   site.latitude_rad)
    return tuple(replace(s, east_m=float(u), north_m=float(v), up_m=0.0)
                 for s, (u, v) in zip(stations, uv))


@dataclass(frozen=True)
class Triangle:
    """Three stations; baselines are the closed cycle B12, B23, B31
    (vector sum identically zero).  baseline_vectors() is the flat
    layout with the source at zenith; projected() gives the triangle
    for a real pointing."""
    stations: tuple
    site: Site | None = None

    def baseline_vectors(self) -> np.ndarray:
        """(3, 2) [m]: B12 = p2 - p1, B23 = p3 - p2, B31 = p1 - p3."""
        p = np.array([[s.east_m, s.north_m] for s in self.stations])
        return np.array([p[1] - p[0], p[2] - p[1], p[0] - p[2]])

    def baseline_lengths(self) -> np.ndarray:
        b = self.baseline_vectors()
        return np.hypot(b[:, 0], b[:, 1])

    @property
    def diameters_m(self) -> tuple:
        return tuple(s.telescope.diameter_m for s in self.stations)

    @property
    def name(self) -> str:
        return "-".join(s.name for s in self.stations)

    def projected(self, hour_angle_h: float, dec_deg: float) -> "Triangle":
        """The same telescopes at their projected (u, v) positions for a
        source at the given hour angle [h] and declination."""
        return Triangle(_project_stations(self.stations, self.site,
                                          hour_angle_h, dec_deg), self.site)


@dataclass(frozen=True)
class Array:
    """N stations: all pairwise baselines and all telescope triangles.
    Of the C(N,3) triangles, only (N-1)(N-2)/2 closure phases are
    independent, but every triangle's triple-coincidence stream carries
    (largely) independent accidental noise, so all contribute to the
    detection sensitivity."""
    stations: tuple
    site: Site | None = None

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
                          self.stations[k]), self.site)
                for i, j, k in combinations(range(len(self.stations)), 3)]

    def projected(self, hour_angle_h: float, dec_deg: float) -> "Array":
        return Array(_project_stations(self.stations, self.site,
                                       hour_angle_h, dec_deg), self.site)

    def as_triangle(self) -> "Triangle":
        """The Triangle of a three-station array."""
        if len(self.stations) != 3:
            raise ValueError(f"an Array of {len(self.stations)} stations is not a triangle")
        return Triangle(self.stations, self.site)


# The named arrays (Maunakea Subaru + Keck, the VLT UTs, VERITAS, MAGIC +
# LST-1, the EON-SII pair and triangle) are hbtsim/configs/arrays/*.json,
# built by hbtsim.catalog.load_array; the generators below are what the
# catalog's "generator" arrays call.
def pair_array(baseline_m: float, telescope: Telescope, detector: Detector, *,
               site: Site | None = None, pa_deg: float = 0.0,
               names: tuple = ("T1", "T2"), min_spacing_m: float | None = None) -> Array:
    """Two identical telescopes at the given baseline and position angle
    (E of N): the first at the origin, as an Array (g2 only, no triangles)."""
    if min_spacing_m is not None and baseline_m < min_spacing_m:
        raise ValueError(f"baseline {baseline_m} m is below the {min_spacing_m} m minimum "
                         f"spacing of two {telescope.diameter_m:g} m telescopes")
    pa = np.radians(pa_deg)
    stations = (Station(names[0], 0.0, 0.0, telescope, detector),
                Station(names[1], baseline_m * np.sin(pa), baseline_m * np.cos(pa),
                        telescope, detector))
    return Array(stations, site=site)


def equilateral_array(side_m: float, telescope: Telescope, detector: Detector, *,
                      site: Site | None = None, pa_deg: float = 0.0,
                      names: tuple = ("T1", "T2", "T3"),
                      min_spacing_m: float | None = None) -> Array:
    """Three identical telescopes on an equilateral triangle of the given
    side: the first at the origin, the second at position angle pa_deg
    (E of N), the third at pa_deg + 60."""
    if min_spacing_m is not None and side_m < min_spacing_m:
        raise ValueError(f"side {side_m} m is below the {min_spacing_m} m minimum "
                         f"spacing of two {telescope.diameter_m:g} m telescopes")
    a, b = np.radians(pa_deg), np.radians(pa_deg + 60.0)
    stations = (Station(names[0], 0.0, 0.0, telescope, detector),
                Station(names[1], side_m * np.sin(a), side_m * np.cos(a), telescope, detector),
                Station(names[2], side_m * np.sin(b), side_m * np.cos(b), telescope, detector))
    return Array(stations, site=site)


def equilateral_triangle(side_m: float, telescope: Telescope,
                         detector: Detector) -> Triangle:
    """Hypothetical compact comparison array: three identical telescopes
    on an equilateral triangle of the given side (T1 at the origin, T2 due
    east, T3 to the north-east), as a Triangle without a site."""
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
    gammas: np.ndarray       # complex (3,): point gamma_12, gamma_23, gamma_31
    bispectrum: complex      # <gamma_12 gamma_23 gamma_31> (pupil-averaged if smeared)
    triple_amp: float        # |bispectrum|
    phi_c: float             # closure phase [rad]
    cos_phi_c: float
    wavelength_nm: float
    orbital_phase: float
    vis2_pairs: np.ndarray = None   # (3,) pair-smeared |gamma_ij|^2
    smeared: bool = False


def resolve_triple_pupils(pupils, triangle,
                          fringe_period_m: float | None = None) -> TripleQuadrature | None:
    """None/False -> point sampling; True -> the triangle's telescope
    diameters; or a ready TripleQuadrature."""
    if pupils is None or pupils is False:
        return None
    if pupils is True:
        return triple_quadrature_for(triangle, fringe_period_m=fringe_period_m)
    if isinstance(pupils, TripleQuadrature):
        return pupils
    raise TypeError("pupils must be None, True or a TripleQuadrature")


def reduce_triple(gam_points, quad: TripleQuadrature | None):
    """From point gammas (..., K): (bispectrum (...,), vis2_pairs (..., 3)).
    Without a quadrature K = 3 and the bispectrum is the plain product."""
    g = np.asarray(gam_points)
    if quad is None:
        return g[..., 0] * g[..., 1] * g[..., 2], np.abs(g) ** 2
    return quad.reduce(g)


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
                  grid: GridConfig | None = None, *,
                  pupils=None) -> BispectrumResult:
    """Model gammas and closure phase on the triangle at one epoch.

    method="analytic" (instant; out of eclipse only) or "render" (renders
    the binary and samples its exact DFT; valid at all phases).  With
    pupils=True the bispectrum, triple_amp, phi_c and cos_phi_c are the
    exact three-pupil averages for the triangle's telescope diameters
    (gammas stay the point values)."""
    method = _resolve_method(method)
    pos = positions_at(system, orbital_phase)
    if grid is None:
        grid = GridConfig().fit_orbit(system)
    quad = resolve_triple_pupils(pupils, triangle,
                                 fringe_period_m(float(pos.rho), wavelength_nm))
    bvecs = triangle.baseline_vectors()
    pts = bvecs if quad is None else np.vstack([bvecs, quad.flat_points(bvecs)])

    if method == "analytic":
        gam = binary_vis_complex_analytic(pts, wavelength_nm, system, pos)
    else:
        from .render import render_image
        img = render_image(pos, system, wavelength_nm, grid)
        gam = np.asarray(vis_of_baselines(img, pts, wavelength_nm * 1e-9, grid))

    bis, v2 = reduce_triple(gam[3:] if quad is not None else gam, quad)
    bis = complex(bis)
    return BispectrumResult(gammas=np.asarray(gam[:3]), bispectrum=bis,
                            triple_amp=abs(bis), phi_c=float(np.angle(bis)),
                            cos_phi_c=float(np.cos(np.angle(bis))),
                            wavelength_nm=wavelength_nm,
                            orbital_phase=orbital_phase,
                            vis2_pairs=np.asarray(v2, dtype=float),
                            smeared=quad is not None)


# ---------------------------------------------------------------------------
# Batched multi-wavelength gammas (GPU; eclipse-capable)
# ---------------------------------------------------------------------------
def spectral_bispectrum(pos: SkyPositions, triangle: Triangle,
                        wavelengths_nm, system: BinarySystem,
                        grid: GridConfig, *,
                        chunk_size: int | None = None):
    """Point-sampled complex (gamma_12, gamma_23, gamma_31) for every
    wavelength channel at one epoch via the batched render + DFT pipeline
    -- (n_lambda, 3) complex64.  Valid at all orbital phases including
    eclipses.  See spectral_triple for the pupil-averaged bispectrum."""
    return spectral_vis(pos, triangle.baseline_vectors(), wavelengths_nm,
                        system, grid, chunk_size=chunk_size)


@dataclass(frozen=True)
class TripleSamples:
    """Per-channel triple-correlation model on one triangle."""
    gammas: np.ndarray       # (n_lambda, 3) point gammas
    bispectrum: np.ndarray   # (n_lambda,) complex, pupil-averaged if smeared
    vis2_pairs: np.ndarray   # (n_lambda, 3) pair-smeared |gamma_ij|^2
    smeared: bool
    flux: np.ndarray = None  # (n_lambda,) rendered image flux (render method only)

    @property
    def triple_amp(self) -> np.ndarray:
        return np.abs(self.bispectrum)

    @property
    def cos_phi_c(self) -> np.ndarray:
        return np.cos(np.angle(self.bispectrum))


def spectral_triple(pos: SkyPositions, triangle: Triangle, wavelengths_nm,
                    system: BinarySystem, grid: GridConfig | None = None, *,
                    method: str = "analytic", pupils=True,
                    chunk_size: int | None = None) -> TripleSamples:
    """The triple-correlation model for every channel at one epoch: the
    (optionally three-pupil-averaged) bispectrum and the pair-smeared
    |gamma_ij|^2, by the analytic two-disk model (out of eclipse) or the
    batched render + DFT pipeline (any phase)."""
    method = _resolve_method(method)
    nm = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
    if grid is None:
        grid = GridConfig().fit_orbit(system)
    quad = resolve_triple_pupils(pupils, triangle,
                                 fringe_period_m(float(pos.rho), nm))
    bvecs = triangle.baseline_vectors()
    pts = bvecs if quad is None else np.vstack([bvecs, quad.flat_points(bvecs)])
    flux = None
    if method == "analytic":
        gam = binary_vis_complex_analytic(pts, nm, system, pos)
    else:
        gam, flux = spectral_vis(pos, pts, nm, system, grid,
                                 chunk_size=chunk_size, return_flux=True)
        gam, flux = np.asarray(gam), np.asarray(flux, dtype=float)
    bis, v2 = reduce_triple(gam[:, 3:] if quad is not None else gam, quad)
    return TripleSamples(gammas=np.asarray(gam[:, :3]),
                         bispectrum=np.asarray(bis),
                         vis2_pairs=np.asarray(v2, dtype=float),
                         smeared=quad is not None, flux=flux)
