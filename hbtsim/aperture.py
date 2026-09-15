"""Finite-aperture averaging of the intensity correlations.

A telescope of diameter D does not sample V at one point of the (u, v)
plane.  The intensity it collects is the integral of |E|^2 over its
pupil, so the cross-correlation of two telescopes at vector separation B
is the average of |gamma|^2 over every pair of pupil points,

    <|V|^2>(B) = int |V(B + s)|^2 W(s) d^2s / (A1 A2),

with W(s) the cross-correlation of the two pupil functions -- for two
circles, the overlap area of circles of diameters D1 and D2 whose
centres are |s| apart (the "Chinese hat", support |s| < (D1 + D2)/2).
For a binary this is what suppresses the fringe: a fringe of period
P = lambda/rho sampled with pupils D1, D2 keeps only the fraction
A(pi D1/P) A(pi D2/P) of its contrast, A(x) = 2 J1(x)/x (the Airy
amplitude), which is 0.66 for Keck (10 m) on Beta Aur at 400 nm
(P ~ 25 m) and 0.46 for the VLT UTs on delta Vel at maximum separation.

The triple correlation couples all three pupils,

    <g12 g23 g31> = sum_{i,j,k} w_i w_j w_k
                    gamma(B12 + r_j - r_i) gamma(B23 + r_k - r_j)
                    gamma(B31 + r_i - r_k),

with r_i, r_j, r_k running over quadrature points of pupils 1, 2, 3.
Smearing each gamma separately with its pair kernel is NOT equivalent
(it would apply every pupil twice) and over-attenuates; the exact
three-pupil average needs 3 m^2 visibility samples and an m^3
contraction, which the exact DFT core makes cheap (m = 14 -> 588
points per channel).  The pair-smeared |gamma_ij|^2 that sets the pair
"ridges" of the triple correlation falls out of the same samples.

Quadrature.  Pair kernel: Gauss-Legendre in |s| on [0, (D1 + D2)/2]
weighted by W(s)|s|, times equally spaced angles (product rule; weights
renormalized to sum to one so a constant |V|^2 is smeared exactly).
Pupil points: Gauss-Legendre in (r/R)^2 (uniform in area) times
equally spaced angles, rings staggered.  Accuracy against the closed
forms above is checked in tests/test_aperture.py: the pair rule (60
points) is good to 3e-4 and the three-pupil rule (14 points per pupil)
to 1e-5 out to D/P = 0.7-0.8, beyond any configuration in this package
(VLT on delta Vel at maximum separation: D/P ~ 0.55).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import j1


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------
def circle_overlap_area(r1: float, r2: float, d) -> np.ndarray:
    """Lens area common to two circles of radii r1, r2 with centres d
    apart (vectorized in d; 0 beyond contact, the smaller disk inside
    the larger when d <= |r1 - r2|)."""
    d = np.asarray(d, dtype=float)
    rmin, rmax = min(r1, r2), max(r1, r2)
    out = np.zeros_like(d)
    inside = d <= rmax - rmin
    out[inside] = np.pi * rmin**2
    lens = (~inside) & (d < r1 + r2)
    dl = d[lens]
    a1 = r1**2 * np.arccos(np.clip((dl**2 + r1**2 - r2**2) / (2 * dl * r1), -1, 1))
    a2 = r2**2 * np.arccos(np.clip((dl**2 + r2**2 - r1**2) / (2 * dl * r2), -1, 1))
    a3 = 0.5 * np.sqrt(np.maximum(
        (-dl + r1 + r2) * (dl + r1 - r2) * (dl - r1 + r2) * (dl + r1 + r2), 0.0))
    out[lens] = a1 + a2 - a3
    return out


def airy_amplitude(x) -> np.ndarray:
    """2 J1(x) / x: the Fourier transform of a uniform circular pupil,
    normalized to 1 at x = 0."""
    x = np.asarray(x, dtype=float)
    xs = np.where(x == 0.0, 1.0, x)
    return np.where(x == 0.0, 1.0, 2.0 * j1(xs) / xs)


def fringe_smearing_factor(d1_m: float, d2_m: float, rho_rad: float,
                           wavelength_m: float) -> float:
    """Contrast retained by a pure fringe of period lambda/rho when
    sampled with pupils D1, D2: A(pi D1 rho/lambda) A(pi D2 rho/lambda)."""
    x = np.pi * rho_rad / wavelength_m
    return float(airy_amplitude(x * d1_m) * airy_amplitude(x * d2_m))


# ---------------------------------------------------------------------------
# Quadratures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PupilQuadrature:
    """Offsets (M, 2) [m] and weights (M,) (sum 1) that turn point samples
    |V(B + offsets)|^2 into the pupil-pair average."""
    offsets_m: np.ndarray
    weights: np.ndarray
    diameters_m: tuple

    @property
    def n_points(self) -> int:
        return self.weights.size

    def points(self, bvecs_m) -> np.ndarray:
        """(K, M, 2): every baseline vector plus every offset."""
        b = np.atleast_2d(np.asarray(bvecs_m, dtype=float))
        return b[:, None, :] + self.offsets_m[None, :, :]

    def reduce(self, values):
        """Weighted average over the offset axis (the last axis of
        `values`, e.g. |V|^2 of shape (..., K, M) -> (..., K))."""
        return np.tensordot(np.asarray(values), self.weights, axes=([-1], [0]))


def pupil_pair_quadrature(d1_m: float, d2_m: float, n_r: int = 5,
                          n_theta: int = 12) -> PupilQuadrature:
    """Quadrature of the pupil cross-correlation W(s) of two circular
    apertures (see the module docstring)."""
    r1, r2 = d1_m / 2.0, d2_m / 2.0
    s_max = r1 + r2
    t, wt = np.polynomial.legendre.leggauss(n_r)
    s = 0.5 * s_max * (t + 1.0)
    ws = 0.5 * s_max * wt * circle_overlap_area(r1, r2, s) * s
    th = 2.0 * np.pi * (np.arange(n_theta) + 0.5) / n_theta
    # stagger successive rings by half a step
    ang = th[None, :] + (np.arange(n_r)[:, None] % 2) * np.pi / n_theta
    off = np.stack([s[:, None] * np.cos(ang), s[:, None] * np.sin(ang)],
                   axis=-1).reshape(-1, 2)
    w = np.repeat(ws, n_theta)
    return PupilQuadrature(offsets_m=off, weights=w / w.sum(),
                           diameters_m=(d1_m, d2_m))


def point_quadrature() -> PupilQuadrature:
    """The no-smearing limit: one offset of zero (point apertures)."""
    return PupilQuadrature(offsets_m=np.zeros((1, 2)), weights=np.ones(1),
                           diameters_m=(0.0, 0.0))


def disk_quadrature(diameter_m: float, n_r: int = 2, n_theta: int = 7):
    """Points (m, 2) [m] and weights (m,) (sum 1) integrating a uniform
    circular pupil of the given diameter: Gauss-Legendre in (r/R)^2 times
    equally spaced angles (m = n_r n_theta; the defaults, m = 14, hold
    the three-pupil bispectrum to ~1e-6 at D/P = 0.7)."""
    R = diameter_m / 2.0
    t, wt = np.polynomial.legendre.leggauss(n_r)
    q = 0.5 * (t + 1.0)              # (r/R)^2 in (0, 1), uniform-area
    r = R * np.sqrt(q)
    wr = 0.5 * wt
    th = 2.0 * np.pi * (np.arange(n_theta) + 0.5) / n_theta
    ang = th[None, :] + (np.arange(n_r)[:, None] % 2) * np.pi / n_theta
    pts = np.stack([r[:, None] * np.cos(ang), r[:, None] * np.sin(ang)],
                   axis=-1).reshape(-1, 2)
    w = np.repeat(wr / n_theta, n_theta)
    return pts, w / w.sum()


@dataclass(frozen=True)
class TripleQuadrature:
    """Pupil points of three apertures for the exact three-pupil average
    of the triple product (and the pair-smeared |gamma_ij|^2)."""
    points_m: tuple      # three (m_i, 2) arrays
    weights: tuple       # three (m_i,) arrays, each summing to 1
    diameters_m: tuple

    @classmethod
    def from_diameters(cls, d1_m: float, d2_m: float, d3_m: float,
                       n_r: int = 2, n_theta: int = 7) -> "TripleQuadrature":
        pw = [disk_quadrature(d, n_r, n_theta) for d in (d1_m, d2_m, d3_m)]
        return cls(points_m=tuple(p for p, _ in pw),
                   weights=tuple(w for _, w in pw),
                   diameters_m=(d1_m, d2_m, d3_m))

    @property
    def shape(self) -> tuple:
        return tuple(w.size for w in self.weights)

    def offsets(self, bvecs_m) -> list:
        """For the closed cycle B12, B23, B31 (3, 2): the three sample
        grids [B12 + r_j - r_i (m1, m2, 2), B23 + r_k - r_j (m2, m3, 2),
        B31 + r_i - r_k (m3, m1, 2)]."""
        b = np.asarray(bvecs_m, dtype=float)
        p1, p2, p3 = self.points_m
        return [b[0] + p2[None, :, :] - p1[:, None, :],
                b[1] + p3[None, :, :] - p2[:, None, :],
                b[2] + p1[None, :, :] - p3[:, None, :]]

    def flat_points(self, bvecs_m) -> np.ndarray:
        """All sample points as one (K, 2) array (K = m1 m2 + m2 m3 + m3 m1),
        in the order split_values expects."""
        return np.concatenate([g.reshape(-1, 2) for g in self.offsets(bvecs_m)])

    def split_values(self, values):
        """Inverse of flat_points on the last axis: (..., K) -> the three
        (..., m_i, m_j) grids."""
        v = np.asarray(values)
        m1, m2, m3 = self.shape
        a = v[..., :m1 * m2].reshape(*v.shape[:-1], m1, m2)
        b = v[..., m1 * m2:m1 * m2 + m2 * m3].reshape(*v.shape[:-1], m2, m3)
        c = v[..., m1 * m2 + m2 * m3:].reshape(*v.shape[:-1], m3, m1)
        return a, b, c

    def reduce(self, values):
        """From point samples of gamma at flat_points (..., K): the
        three-pupil-averaged bispectrum (...,) and the pair-smeared
        |gamma_12|^2, |gamma_23|^2, |gamma_31|^2 (..., 3)."""
        g12, g23, g31 = self.split_values(values)
        w1, w2, w3 = self.weights
        bis = np.einsum("i,j,k,...ij,...jk,...ki->...", w1, w2, w3, g12, g23, g31)
        v2 = np.stack([np.einsum("i,j,...ij->...", w1, w2, np.abs(g12) ** 2),
                       np.einsum("j,k,...jk->...", w2, w3, np.abs(g23) ** 2),
                       np.einsum("k,i,...ki->...", w3, w1, np.abs(g31) ** 2)],
                      axis=-1)
        return bis, v2


def triple_quadrature_for(triangle, n_r: int = 2, n_theta: int = 7) -> TripleQuadrature:
    """TripleQuadrature from a bispectrum.Triangle's telescope diameters."""
    return TripleQuadrature.from_diameters(
        *(s.telescope.diameter_m for s in triangle.stations), n_r, n_theta)


def resolve_pupils(pupils, diameters):
    """Normalize the `pupils` argument of the sampling functions:
    None/False -> point sampling; True -> the given diameters; a
    (d1, d2) pair; or a ready PupilQuadrature."""
    if pupils is None or pupils is False:
        return None
    if isinstance(pupils, PupilQuadrature):
        return pupils
    if pupils is True:
        pupils = diameters
    d1, d2 = pupils
    return pupil_pair_quadrature(float(d1), float(d2))
