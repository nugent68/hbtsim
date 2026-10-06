"""Single stars: |V|^2 of one limb-darkened disk, its uniform-disk
inversion, and the multiplexed g2 budget -- for chromatic-diameter
studies of bright single stars (Sirius A, Vega) with model-atmosphere
limb profiles (NewEra tables via sed.with_newera_star).

The diameter convention follows the binary path: theta_ld_mas is the
literature (tau = 1, limb-darkened) diameter; with a spherical NewEra
profile the disk is drawn at theta_ld x r_outer (Star.radius_scale),
because the profile's mu = 0 is the model's outer boundary.

Pupil smearing is not optional here: a 4 m pupil on a ~100 m baseline
(D/B ~ 0.04; for EON-SII on Sirius at 4 mas scales D/B ~ 0.4 on short
baselines) averages |V|^2 over a range of baselines, so the inversion
to a UD diameter must invert the SMEARED uniform-disk model
(ud_diameter_per_channel(..., pupils=...)); inverting the point model
biases the diameter.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.special import j1

from .aperture import PupilQuadrature, pupil_pair_quadrature
from .limbdark import star_disk_visibility
from .params import AB_ZERO_FNU, C_LIGHT, MAS, PARSEC, R_SUN, Star
from .snr import Detector, Spectrograph, SpectralSNRResult, Telescope, _g2_budget


@dataclass(frozen=True)
class Ellipse:
    """An elliptical (e.g. rotationally flattened or disk-bearing) outline
    for ellipse_vis / composite_vis: major axis [mas], axis ratio >= 1 and
    the position angle of the major axis [deg E of N]."""
    theta_major_mas: float
    axis_ratio: float = 1.0
    pa_deg: float = 0.0

    @property
    def theta_minor_mas(self) -> float:
        return self.theta_major_mas / self.axis_ratio


@dataclass(frozen=True)
class SingleStar:
    name: str
    star: Star
    theta_ld_mas: float           # literature limb-darkened (tau = 1) diameter
    v_mag: float                  # observed V (a check on the model photometry)
    dec_deg: float
    ra_hours: float
    distance_pc: float
    # observed AB magnitudes ((wavelength_nm, mag), ...): when given, the
    # model spectrum is scaled by the offset interpolated in log(lambda)
    # between the anchors (held constant beyond them), as
    # snr.system_ab_mag does for the binaries; empty = unanchored model
    mag_anchors: tuple = ()
    # optional resolved non-circular outline (gamma Cas's decretion disk)
    ellipse: Ellipse | None = None

    @property
    def drawn_diameter_mas(self) -> float:
        """The diameter the disk is drawn with (outer boundary for NewEra)."""
        return self.theta_ld_mas * self.star.radius_scale

    def model_ab_mag(self, wavelength_nm):
        """Unanchored model AB magnitude: surface flux x solid angle of the
        drawn disk."""
        lam_nm = np.asarray(wavelength_nm, dtype=float)
        theta_r = 0.5 * self.drawn_diameter_mas * MAS
        f_nu = self.star.surface_flux(lam_nm) * theta_r**2 * (lam_nm * 1e-9) ** 2 / C_LIGHT
        out = -2.5 * np.log10(f_nu / AB_ZERO_FNU)
        return float(out) if np.ndim(out) == 0 else out

    def anchor_offset(self, wavelength_nm):
        """Observed minus model magnitude, interpolated between the anchors."""
        lam_nm = np.asarray(wavelength_nm, dtype=float)
        if not self.mag_anchors:
            return np.zeros(lam_nm.shape) if np.ndim(lam_nm) else 0.0
        pts = sorted(self.mag_anchors)
        lam_a = np.array([p[0] for p in pts], dtype=float)
        off = np.array([p[1] - float(self.model_ab_mag(p[0])) for p in pts])
        out = np.interp(np.log10(lam_nm), np.log10(lam_a), off)
        return float(out) if np.ndim(out) == 0 else out

    def ab_mag(self, wavelength_nm):
        """AB magnitude of the model, anchored to the observed magnitudes
        when mag_anchors is set."""
        return self.model_ab_mag(wavelength_nm) + self.anchor_offset(wavelength_nm)

    def v_check(self) -> float:
        """Unanchored model AB mag at 550 nm minus the observed V (Vega-system
        V and AB agree to ~0.02 mag there)."""
        return float(self.model_ab_mag(550.0) - self.v_mag)


# The single-star targets (Sirius A, Vega, HD 17652, HD 360, gamma Cas) are
# hbtsim/configs/targets/*.json: load_target("vega").  Their radii follow
# from theta_LD and the distance (0.5 theta d).


# ---------------------------------------------------------------------------
# Oblate photosphere and circumstellar disk: complex visibilities on (u, v)
# baseline vectors (E, N components in metres), for rapid rotators and Be stars
# ---------------------------------------------------------------------------
def _axis_components(bvecs_m, pa_deg):
    """Baseline components along the major axis (position angle pa_deg, east
    of north) and the minor axis, each (K,)."""
    b = np.atleast_2d(np.asarray(bvecs_m, dtype=float))
    pa = np.radians(pa_deg)
    e_maj = np.array([np.sin(pa), np.cos(pa)])
    e_min = np.array([np.cos(pa), -np.sin(pa)])
    return b @ e_maj, b @ e_min


def ellipse_vis(bvecs_m, wavelength_nm, star: Star, theta_major_mas: float, axis_ratio: float = 1.0,
                pa_deg: float = 0.0) -> np.ndarray:
    """Visibility (n_lambda, K) of a limb-darkened elliptical disk: the
    affine image of the circular disk of diameter theta_major, so V(u, v)
    is the circular visibility at the effective baseline
    sqrt(b_maj^2 + (b_min / axis_ratio)^2).  axis_ratio = major / minor
    >= 1; pa_deg is the major axis's position angle east of north."""
    nm = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
    b_maj, b_min = _axis_components(bvecs_m, pa_deg)
    b_eff = np.hypot(b_maj, b_min / axis_ratio)
    x = np.pi * theta_major_mas * MAS * b_eff[None, :] / (nm[:, None] * 1e-9)
    return star_disk_visibility(star, x, nm)


def gaussian_disk_vis(bvecs_m, wavelength_nm, fwhm_major_mas: float, axis_ratio: float = 1.0,
                      pa_deg: float = 0.0) -> np.ndarray:
    """Visibility (n_lambda, K) of an elliptical Gaussian of the given FWHM
    along its major axis (FWHM / axis_ratio along the minor one)."""
    nm = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
    b_maj, b_min = _axis_components(bvecs_m, pa_deg)
    th_maj = fwhm_major_mas * MAS
    th_min = th_maj / axis_ratio
    arg = (np.pi**2 / (4.0 * np.log(2.0))) * ((th_maj * b_maj[None, :])**2 + (th_min * b_min[None, :])**2) \
        / (nm[:, None] * 1e-9) ** 2
    return np.exp(-arg)


def composite_vis(bvecs_m, wavelength_nm, star: Star, theta_major_mas: float, *, axis_ratio: float = 1.0,
                  pa_deg: float = 0.0, disk_fraction: float = 0.0, disk_fwhm_mas: float = 2.9,
                  disk_axis_ratio: float = 1.0, disk_pa_deg: float = 0.0) -> np.ndarray:
    """Photosphere plus an extended Gaussian component carrying a fraction
    disk_fraction of the flux at this wavelength: V = (1 - f) V_star + f V_disk
    (both centred, so the sum is real)."""
    v = (1.0 - disk_fraction) * ellipse_vis(bvecs_m, wavelength_nm, star, theta_major_mas, axis_ratio, pa_deg)
    if disk_fraction > 0.0:
        v = v + disk_fraction * gaussian_disk_vis(bvecs_m, wavelength_nm, disk_fwhm_mas, disk_axis_ratio, disk_pa_deg)
    return v


def attach_newera_single(target: SingleStar, grid, allow_extrapolation: bool = False):
    """(target with NewEra tables, report); grid a sed.NewEraGrid or a directory."""
    from .sed import NewEraGrid, with_newera_star
    if not isinstance(grid, NewEraGrid):
        grid = NewEraGrid.scan(grid)
    star, report = with_newera_star(target.star, grid, allow_extrapolation)
    return replace(target, star=star), report


def prepare_single(target: SingleStar, spectrograph: Spectrograph | None) -> SingleStar:
    """Channel-averaged tables (sed.rebin_to_channels), as a channelized
    measurement sees the star."""
    if spectrograph is None:
        return target
    from .sed import air_to_vacuum, rebin_to_channels
    edges = np.asarray(spectrograph.channel_edges_nm, dtype=float)
    if getattr(spectrograph, "frame", "vacuum") == "air":
        edges = air_to_vacuum(edges)
    return replace(target, star=rebin_to_channels(target.star, edges))


def _pupil_quad(pupils) -> PupilQuadrature | None:
    if pupils is None or pupils is False:
        return None
    if isinstance(pupils, PupilQuadrature):
        return pupils
    d1, d2 = pupils
    # |V|^2 of a single disk varies smoothly across the pupils: a denser
    # base rule than the binary default is ample
    return pupil_pair_quadrature(d1, d2, n_r=7, n_theta=16)


def _baseline_samples(baseline_m, quad):
    """|B| at every pupil sample: (K,) for points, (K, M) with pupils."""
    b = np.atleast_1d(np.asarray(baseline_m, dtype=float))
    if quad is None:
        return b
    pts = quad.points(np.stack([b, np.zeros_like(b)], axis=-1))      # (K, M, 2)
    return np.hypot(pts[..., 0], pts[..., 1])


def single_star_vis2(target: SingleStar, baseline_m, wavelength_nm, pupils=None) -> np.ndarray:
    """|V|^2 of the drawn disk (n_lambda, K baselines), averaged over the
    pupil cross-correlation when pupils is a (d1, d2) pair or quadrature."""
    nm = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
    quad = _pupil_quad(pupils)
    bs = _baseline_samples(baseline_m, quad)
    x = np.pi * target.drawn_diameter_mas * MAS * bs[None, ...] / (nm * 1e-9).reshape(
        (-1,) + (1,) * bs.ndim)
    v = star_disk_visibility(target.star, x.reshape(nm.size, -1), nm).reshape(x.shape)
    v2 = np.abs(v) ** 2
    return v2 if quad is None else quad.reduce(v2)


def ud_vis2(theta_mas, baseline_m, wavelength_nm, pupils=None) -> np.ndarray:
    """Uniform-disk |V|^2 (n_lambda, K); theta may be (n_lambda,) per channel."""
    nm = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
    quad = _pupil_quad(pupils)
    bs = _baseline_samples(baseline_m, quad)
    th = np.broadcast_to(np.atleast_1d(np.asarray(theta_mas, dtype=float)), nm.shape)
    shape = (-1,) + (1,) * bs.ndim
    x = np.pi * th.reshape(shape) * MAS * bs[None, ...] / (nm * 1e-9).reshape(shape)
    xs = np.where(x == 0, 1.0, x)
    v2 = np.where(x == 0, 1.0, (2.0 * j1(xs) / xs) ** 2)
    return v2 if quad is None else quad.reduce(v2)


def ud_diameter_per_channel(vis2, baseline_m: float, wavelength_nm, pupils=None,
                            theta_max_mas: float | None = None, n_iter: int = 60) -> np.ndarray:
    """Invert |V|^2 per channel to a uniform-disk diameter on the first
    lobe (bisection; |V|^2 decreases monotonically there) with the same
    pupil smearing as the data.  vis2 (n_lambda,) at one baseline."""
    nm = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
    v2 = np.asarray(vis2, dtype=float).reshape(nm.size)
    # first null of the point UD at x = 3.8317
    hi = (3.8317 * nm * 1e-9 / (np.pi * baseline_m * MAS) if theta_max_mas is None
          else np.full(nm.size, float(theta_max_mas)))
    lo = np.zeros(nm.size)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        m = ud_vis2(mid, baseline_m, nm, pupils)[:, 0]
        lo = np.where(m > v2, mid, lo)
        hi = np.where(m > v2, hi, mid)
    return 0.5 * (lo + hi)


def ud_dtheta_dvis2(theta_mas, baseline_m: float, wavelength_nm, pupils=None,
                    rel_step: float = 1e-4) -> np.ndarray:
    """d theta_UD / d|V|^2 per channel (error propagation)."""
    th = np.asarray(theta_mas, dtype=float)
    up = ud_vis2(th * (1 + rel_step), baseline_m, wavelength_nm, pupils)[:, 0]
    dn = ud_vis2(th * (1 - rel_step), baseline_m, wavelength_nm, pupils)[:, 0]
    return 2.0 * rel_step * th / (up - dn)


def spectral_g2_snr_single(target: SingleStar, baseline_m: float,
                           spectrograph: Spectrograph, *,
                           t_int_s: float = 3600.0,
                           telescope1: Telescope,
                           telescope2: Telescope | None = None,
                           detector1: Detector,
                           detector2: Detector | None = None,
                           polarization_mode: str = "unpolarized",
                           sky_cps_per_channel: float = 0.0,
                           pupils=True, channel_mask=None,
                           coherence_broadening: bool = True,
                           enforce_readout: bool = True) -> SpectralSNRResult:
    """Multiplexed g2 of a single star: pupil-averaged |V|^2 of the
    channel-averaged model, unanchored model magnitudes, and the photon
    budget shared with the binary path (snr._g2_budget; channel_mask
    tags only the selected channels)."""
    telescope2 = telescope1 if telescope2 is None else telescope2
    detector2 = detector1 if detector2 is None else detector2
    det1 = replace(detector1, n_pixels=1)
    det2 = replace(detector2, n_pixels=1)
    tgt = prepare_single(target, spectrograph)
    nm = spectrograph.channel_centers_nm
    pup = (telescope1.diameter_m, telescope2.diameter_m) if pupils is True else pupils
    vis2 = single_star_vis2(tgt, baseline_m, nm, pup)[:, 0]
    mag = tgt.ab_mag(nm)
    return _g2_budget(vis2, mag, spectrograph, baseline_m, t_int_s=t_int_s,
                      telescope1=telescope1, telescope2=telescope2, det1=det1, det2=det2,
                      polarization_mode=polarization_mode,
                      sky_cps_per_channel=sky_cps_per_channel,
                      coherence_broadening=coherence_broadening,
                      enforce_readout=enforce_readout, vis2_method="single",
                      smeared=pup is not None, channel_mask=channel_mask,
                      caller="spectral_g2_snr_single")
