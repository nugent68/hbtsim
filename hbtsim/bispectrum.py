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

This module samples the COMPLEX degree of coherence gamma_ij from the
same zero-padded FFT used for |V|^2 (hbt.vis2_map keeps only the
modulus), provides the analytic complex visibility of the binary for
validation, and defines the telescope-triangle geometry, with the
Maunakea Subaru + Keck I + Keck II triangle built in (site coordinates
give pairwise distances 152.1 / 84.9 / 225.9 m, confirming the nominal
150 / 85 / 225 m).  Baselines are projected assuming a flat layout with
the source at zenith -- a documented simplification (summit elevations
agree to ~20 m; hour-angle projection can be added at need).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .hbt import CROP_HALF
from .orbit import SkyPositions, sky_positions
from .params import MAS, BinarySystem, GridConfig, planck
from .render import _render_kernel, spectral_weights
from .snr import KECK, SPAD_LAMBDA, SUBARU, Detector, Telescope


# ---------------------------------------------------------------------------
# Complex visibility from the FFT
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("n", "pad", "crop_half"))
def vis_complex_map(img: jax.Array, n: int, pad: int,
                    crop_half: int = CROP_HALF) -> jax.Array:
    """Complex V(u, v) on the same central (2*crop_half, crop_half) region
    as hbt.vis2_map (zero frequency at [crop_half, 0]); |result|^2 equals
    vis2_map exactly.

    The phase origin is moved from FFT pixel [0, 0] to the image-grid
    center ((n-1)/2, where render.py puts the binary's center of mass) by
    demodulating the whole map BEFORE any interpolation -- the raw map's
    phase ramps by ~2 pi (n/2)/pad per bin, which would wreck bilinear
    interpolation of Re/Im; the centered map varies only on the source's
    structure scale."""
    F = jnp.fft.rfft2(img, s=(pad, pad)) / jnp.sum(img)
    c = pad // 2
    F = jnp.fft.fftshift(F, axes=0)[c - crop_half:c + crop_half, :crop_half]
    ctr = (n - 1) / 2.0
    ky = jnp.arange(-crop_half, crop_half, dtype=jnp.float32)[:, None]
    kx = jnp.arange(0, crop_half, dtype=jnp.float32)[None, :]
    return F * jnp.exp(2j * jnp.pi * ctr * (kx + ky) / pad)


def uv_bins_of_baseline(bvecs_m, wavelength_m, grid: GridConfig):
    """FFT-bin offsets (fx, fy) for baseline vectors (..., 2) in meters:
    f = (B / lambda) * dtheta * pad, per axis (x = East <-> FFT axis 1,
    y = North <-> FFT axis 0)."""
    b = jnp.asarray(bvecs_m)
    return b / wavelength_m * grid.pixel_scale_rad * grid.pad


def vis_complex_of_uv(vmap: jax.Array, fx, fy,
                      crop_half: int = CROP_HALF) -> jax.Array:
    """Sample the (centered) complex map at FFT-bin offsets (fx, fy) by
    bilinear interpolation of Re and Im separately.  Points with fx < 0
    use the Hermitian symmetry of the real image, V(-u) = V*(u) (which
    the centered map preserves: recentering is a real-image shift)."""
    fx = jnp.asarray(fx)
    fy = jnp.asarray(fy)
    sign = jnp.where(fx < 0, -1.0, 1.0)
    row = crop_half + sign * fy
    col = sign * fx
    interp = lambda a: jax.scipy.ndimage.map_coordinates(a, [row, col], order=1)
    v = interp(jnp.real(vmap)) + 1j * interp(jnp.imag(vmap))
    return jnp.where(sign < 0, jnp.conj(v), v)


# ---------------------------------------------------------------------------
# Analytic complex visibility of the binary (validation / fast path)
# ---------------------------------------------------------------------------
def binary_vis_complex_analytic(bvecs_m, wavelength_nm: float,
                                system: BinarySystem,
                                pos: SkyPositions) -> np.ndarray:
    """Complex V at baseline vectors (n, 2) [m, East/North] for two
    non-overlapping limb-darkened disks at the epoch's sky positions:

        V(u) = [f1 V1 e^{-2 pi i u.theta1} + f2 V2 e^{-2 pi i u.theta2}]
               / (f1 + f2),

    with per-star fluxes f_s = B_lambda(T_s) theta_s^2 (1 - u_s/3) and
    LD disk visibilities V_s.  NOT valid during eclipses (overlapping
    disks)."""
    from .limbdark import visibility_ld_disk

    lam = wavelength_nm * 1e-9
    b = np.atleast_2d(np.asarray(bvecs_m, dtype=float))  # (n, 2)
    u = b / lam  # cycles/rad, (ux, uy)

    stars = (system.primary, system.secondary)
    positions = (np.array([float(pos.x1), float(pos.y1)]) * MAS,
                 np.array([float(pos.x2), float(pos.y2)]) * MAS)
    out = np.zeros(b.shape[0], dtype=complex)
    norm = 0.0
    for s, th_pos in zip(stars, positions):
        theta_d = 2.0 * system.angular_radius_mas(s) * MAS
        u_s = s.ld_coeff(wavelength_nm)
        f_s = planck(lam, s.teff) * theta_d**2 * (1.0 - u_s / 3.0)
        b_len = np.hypot(b[:, 0], b[:, 1])
        v_s = visibility_ld_disk(np.pi * theta_d * b_len / lam, u_s)
        phase = np.exp(-2j * np.pi * (u[:, 0] * th_pos[0] + u[:, 1] * th_pos[1]))
        out += f_s * v_s * phase
        norm += f_s
    return out / norm


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


def closure_phase(system: BinarySystem, triangle: Triangle,
                  wavelength_nm: float, orbital_phase: float = 0.0,
                  method: str = "analytic",
                  grid: GridConfig = GridConfig()) -> BispectrumResult:
    """Model gammas and closure phase on the triangle at one epoch.

    method="analytic" (instant; out of eclipse only) or "fft" (renders
    the binary and samples the complex FFT; valid at all phases)."""
    pos = SkyPositions(*(np.asarray(v) for v in
                         sky_positions(2 * np.pi * orbital_phase, system)))
    bvecs = triangle.baseline_vectors()

    if method == "analytic":
        sum_r = (system.angular_radius_mas(system.primary)
                 + system.angular_radius_mas(system.secondary))
        if float(pos.rho) < 1.05 * sum_r:
            raise ValueError("in (or near) eclipse: the analytic complex "
                             "visibility is invalid; use method='fft'")
        gam = binary_vis_complex_analytic(bvecs, wavelength_nm, system, pos)
    elif method == "fft":
        from .render import render_image
        img = render_image(pos, system, wavelength_nm, grid)
        vmap = vis_complex_map(img, grid.n, grid.pad)
        f = np.asarray(uv_bins_of_baseline(bvecs, wavelength_nm * 1e-9, grid))
        gam = np.asarray(vis_complex_of_uv(vmap, f[:, 0], f[:, 1]))
    else:
        raise ValueError(f"unknown method {method!r}")

    bis = complex(gam[0] * gam[1] * gam[2])
    return BispectrumResult(gammas=np.asarray(gam), bispectrum=bis,
                            triple_amp=abs(bis), phi_c=float(np.angle(bis)),
                            cos_phi_c=float(np.cos(np.angle(bis))),
                            wavelength_nm=wavelength_nm,
                            orbital_phase=orbital_phase)


# ---------------------------------------------------------------------------
# Batched multi-wavelength gammas (GPU; eclipse-capable)
# ---------------------------------------------------------------------------
def _auto_chunk() -> int:
    return 16 if jax.default_backend() == "gpu" else 4


@partial(jax.jit, static_argnames=("grid", "chunk", "crop_half"))
def _spectral_bispec_jit(x1, y1, x2, y2, front2, r1, r2,
                         u1, u2, w1, lam_m,      # (n_lambda,)
                         bvecs_m,                # (3, 2)
                         grid: GridConfig, chunk: int, crop_half: int):
    def one_channel(ch):
        u1_k, u2_k, w1_k, lam_k = ch
        img = _render_kernel(x1, y1, x2, y2, front2, r1, r2,
                             w1_k, jnp.float32(1.0), u1_k, u2_k, grid.n)
        vmap = vis_complex_map(img, grid.n, grid.pad, crop_half)
        f = bvecs_m / lam_k * grid.pixel_scale_rad * grid.pad
        return vis_complex_of_uv(vmap, f[:, 0], f[:, 1], crop_half)

    return jax.lax.map(one_channel, (u1, u2, w1, lam_m), batch_size=chunk)


def spectral_bispectrum(pos: SkyPositions, triangle: Triangle,
                        wavelengths_nm, system: BinarySystem,
                        grid: GridConfig, *,
                        chunk_size: int | None = None) -> jax.Array:
    """Complex (gamma_12, gamma_23, gamma_31) for every wavelength channel
    at one epoch via the batched FFT pipeline -- (n_lambda, 3) complex.
    Valid at all orbital phases including eclipses."""
    scale = grid.pixel_scale_mas
    u1, u2, w1 = spectral_weights(wavelengths_nm, system)
    lam_m = jnp.asarray(wavelengths_nm, dtype=jnp.float32) * 1e-9
    chunk = _auto_chunk() if chunk_size is None else int(chunk_size)
    return _spectral_bispec_jit(
        jnp.float32(pos.x1 / scale), jnp.float32(pos.y1 / scale),
        jnp.float32(pos.x2 / scale), jnp.float32(pos.y2 / scale),
        jnp.bool_(pos.front2),
        jnp.float32(system.angular_radius_mas(system.primary) / scale),
        jnp.float32(system.angular_radius_mas(system.secondary) / scale),
        u1, u2, w1, lam_m,
        jnp.asarray(triangle.baseline_vectors(), dtype=jnp.float32),
        grid, chunk, CROP_HALF)
