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
from .params import AB_ZERO_FNU, C_LIGHT, LD_BETA_AUR, MAS, PARSEC, R_SUN, Star
from .snr import (C2PU, SPAD_LAMBDA, Detector, Spectrograph, SpectralSNRResult,
                  Telescope, _g2_budget)


@dataclass(frozen=True)
class SingleStar:
    name: str
    star: Star
    theta_ld_mas: float           # literature limb-darkened (tau = 1) diameter
    v_mag: float                  # observed V (a check on the model photometry)
    dec_deg: float
    ra_hours: float
    distance_pc: float

    @property
    def drawn_diameter_mas(self) -> float:
        """The diameter the disk is drawn with (outer boundary for NewEra)."""
        return self.theta_ld_mas * self.star.radius_scale

    def ab_mag(self, wavelength_nm):
        """Model AB magnitude (unanchored): surface flux x solid angle of
        the drawn disk."""
        lam_nm = np.asarray(wavelength_nm, dtype=float)
        theta_r = 0.5 * self.drawn_diameter_mas * MAS
        f_nu = self.star.surface_flux(lam_nm) * theta_r**2 * (lam_nm * 1e-9) ** 2 / C_LIGHT
        out = -2.5 * np.log10(f_nu / AB_ZERO_FNU)
        return float(out) if np.ndim(out) == 0 else out

    def v_check(self) -> float:
        """Model AB mag at 550 nm minus the observed V (Vega-system V and AB
        agree to ~0.02 mag there)."""
        return float(self.ab_mag(550.0) - self.v_mag)


def _star(name, teff, logg, mass, theta_ld_mas, d_pc):
    radius_rsun = 0.5 * theta_ld_mas * MAS * d_pc * PARSEC / R_SUN
    # A0-A2 V linear law as the fallback (the beta Aur table, 9350 K);
    # replaced by the NewEra profile when attached
    return Star(name, mass_msun=mass, radius_rsun=radius_rsun, teff=teff,
                ld_table_nm=LD_BETA_AUR, logg=logg)


# Sirius A: T_eff 9940 K, log g 4.33 (Adelman 2004); theta_LD 6.039 mas
# (Kervella et al. 2003, VINCI); d = 2.64 pc (Hipparcos); V = -1.46.
SIRIUS_A = SingleStar("Sirius A", _star("Sirius A", 9940.0, 4.33, 2.06, 6.039, 2.637),
                      theta_ld_mas=6.039, v_mag=-1.46, dec_deg=-16.716, ra_hours=6.7525,
                      distance_pc=2.637)
# Vega: pole-on rapid rotator; mean T_eff ~9550 K, log g 4.0; theta_LD 3.329
# mas (Aufdenberg et al. 2006); d = 7.68 pc; V = 0.03.  The spherical model
# ignores the 2000 K pole-to-equator gradient (docs/chromatic_diameters_eonsii.md).
VEGA = SingleStar("Vega", _star("Vega", 9550.0, 4.0, 2.15, 3.329, 7.68),
                  theta_ld_mas=3.329, v_mag=0.03, dec_deg=38.7837, ra_hours=18.6156,
                  distance_pc=7.68)
# Red-clump validation targets of Kim & Kaiser (2026, PASP 138, 044202):
# theta_LD from Gallenne et al. (VLTI/PIONIER), T_eff / log g as adopted
# there; coordinates from SIMBAD.  The linear-law fallback is a rough K-giant
# table (replace by a NewEra profile: no 4800 K / log g 2.5 model exists yet,
# see scripts/redclump_ii.py for the stand-ins and the model request).
LD_K_GIANT = ((400.0, 0.85), (450.0, 0.78), (550.0, 0.68), (650.0, 0.60), (800.0, 0.50),
              (1000.0, 0.42), (1650.0, 0.32), (2200.0, 0.28))


def _giant(name, teff, logg, mass, theta_ld_mas, d_pc):
    radius_rsun = 0.5 * theta_ld_mas * MAS * d_pc * PARSEC / R_SUN
    return Star(name, mass_msun=mass, radius_rsun=radius_rsun, teff=teff,
                ld_table_nm=LD_K_GIANT, logg=logg)


HD_17652 = SingleStar("HD 17652 (beta For, G9 IIIb)", _giant("HD 17652", 4786.0, 2.5, 1.5, 1.835, 54.17),
                      theta_ld_mas=1.835, v_mag=4.456, dec_deg=-32.4059, ra_hours=2.8182,
                      distance_pc=54.17)
HD_360 = SingleStar("HD 360 (HR 16, K1 II)", _giant("HD 360", 4764.0, 2.5, 1.5, 0.906, 110.97),
                    theta_ld_mas=0.906, v_mag=5.986, dec_deg=-8.8241, ra_hours=0.1382,
                    distance_pc=110.97)
SINGLE_STARS = {"sirius": SIRIUS_A, "vega": VEGA, "hd17652": HD_17652, "hd360": HD_360}


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
                           spectrograph: Spectrograph = Spectrograph(),
                           t_int_s: float = 3600.0,
                           telescope1: Telescope = C2PU,
                           telescope2: Telescope | None = None,
                           detector1: Detector = SPAD_LAMBDA,
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
