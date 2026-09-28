"""System parameters, grid configuration, and physical constants.

All system values are taken from the literature.

Beta Aurigae:
  [S07] Southworth, Bruntt & Buzasi 2007, A&A 467, 1215 (WIRE photometry,
        astro-ph/0703634): P, i, masses, radii, Teff.
  [vL07] van Leeuwen 2007, A&A 474, 653 (Hipparcos re-reduction): parallax.
  [H95] Hummel et al. 1995, AJ 110, 376 (Mark III interferometric orbit).

Algol (beta Persei) A-B (component C, ~70 mas away, is excluded; its
~10% third light is removed from the photometric anchors):
  [B12] Baron et al. 2012, ApJ 752, 20 (CHARA/MIRC imaging): P, i, a,
        masses, radii.
  [Z10] Zavala et al. 2010, ApJ 715, L44: parallax 34.7 mas (28.82 pc).
  [K15] Kolbas et al. 2015, MNRAS 451, 4150 (spectral disentangling):
        Teff_A = 12550 K, Teff_B = 4900 K.

Spica (alpha Virginis), the recommended bright target for three-telescope
closure-phase work (docs/three_telescope_feasibility.md):
  [HE71] Herbison-Evans, Hanbury Brown, Davis & Allen 1971, MNRAS 151,
         161: the Narrabri INTENSITY-INTERFEROMETER orbit -- theta_A =
         0.90 +/- 0.04 mas, angular semi-major axis, i.  Spica's orbit
         was itself measured by intensity interferometry.
  [T16] Tkachenko et al. 2016, MNRAS 458, 1964 (disentangling +
        asteroseismology): masses, radii, Teffs, i = 63.1 deg.
  Note: the true orbit has e = 0.108 with apsidal motion; this package
  implements circular orbits only, so Spica is approximated as circular
  (fine for visibility/SNR forecasts; not for timing work).  The primary
  is a beta Cep pulsator and tidally distorted (ellipsoidal variable,
  non-eclipsing at i = 63 deg) -- rendered here as a static sphere.

delta Velorum Aa-Ab, the second VLT closure-phase target (eccentric
orbit, unresolved disks, both components rapid rotators):
  [M11] Merand et al. 2011, A&A 532, A50 (VLTI/AMBER + spectroscopy
        with Pribulla et al. 2011): P = 45.1503 d, e = 0.290,
        omega = 109.7 deg, i = 89.0 deg, masses, radii, Teffs, orbital
        parallax 39.8 +/- 0.4 mas = 25.1 pc (a purely geometric distance;
        Hipparcos gives 40.5 mas).  An earlier revision of this file
        carried 80.6 pc, which put the blackbody photometry 2.5 mag off
        the anchors and shrank every delta Vel angular scale threefold.  The B component (F dwarf, ~0.6 arcsec away,
        ~4% third light) is excluded; its dilution is folded into the
        photometric anchors.  The components' rapid rotation
        (v sin i ~ 145 km/s; oblate, gravity-darkened) is NOT modeled
        -- spherical disks here -- which is precisely the signal a real
        closure-phase campaign would target.

Limb darkening (per star):
  [C11] Claret & Bloemen 2011, A&A 529, A75 (VizieR J/A+A/529/A75):
        linear limb-darkening coefficients u(lambda), ATLAS models,
        [M/H] = 0, interpolated to each star's (Teff, log g).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
H_PLANCK = 6.62607015e-34   # J s
C_LIGHT = 2.99792458e8      # m / s
K_BOLTZ = 1.380649e-23      # J / K

R_SUN = 6.957e8             # m
AU = 1.495978707e11         # m
PARSEC = 3.0856775814913673e16  # m
DAY = 86400.0               # s

MAS = np.pi / (180.0 * 3600.0 * 1000.0)  # 1 milliarcsecond in radians

AB_ZERO_FNU = 3.631e-23     # AB zero point, 3631 Jy in W m^-2 Hz^-1


def planck(wavelength_m: float, teff: float) -> float:
    """Planck spectral radiance B_lambda(T); only relative values matter here."""
    x = H_PLANCK * C_LIGHT / (wavelength_m * K_BOLTZ * teff)
    return 2.0 * H_PLANCK * C_LIGHT**2 / wavelength_m**5 / np.expm1(x)


# ---------------------------------------------------------------------------
# Limb-darkening tables: linear u vs wavelength [nm], [C11], interpolated
# linearly between points (np.interp clamps beyond the table ends).
# ---------------------------------------------------------------------------
LD_BETA_AUR = ((400.0, 0.52), (477.0, 0.50), (763.0, 0.30), (800.0, 0.29),
               (913.0, 0.26))                                    # 9250 K, log g 3.9
LD_ALGOL_A = ((400.0, 0.42), (445.0, 0.40), (477.0, 0.38), (551.0, 0.34),
              (623.0, 0.30), (763.0, 0.24), (806.0, 0.23), (913.0, 0.21))  # 12500 K, log g 4.0
LD_ALGOL_B = ((400.0, 0.89), (445.0, 0.87), (477.0, 0.81), (551.0, 0.73),
              (623.0, 0.65), (763.0, 0.55), (806.0, 0.52), (913.0, 0.46))  # 4900 K, log g 3.2
LD_SPICA = ((400.0, 0.32), (477.0, 0.29), (551.0, 0.26), (623.0, 0.24),
            (763.0, 0.20), (913.0, 0.17))                        # ~21-25 kK, log g 3.7-4.2


# ---------------------------------------------------------------------------
# System description
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class FluxTable:
    """Surface flux F_lambda(lambda) of a model atmosphere [W m^-2 m^-1,
    per unit area of the stellar surface] on a wavelength grid [nm];
    F = pi B_lambda for a blackbody.  Compared by identity (hashable)."""
    wavelength_nm: np.ndarray
    flux: np.ndarray
    source: str = ""

    def __call__(self, wavelength_nm):
        out = np.interp(wavelength_nm, self.wavelength_nm, self.flux)
        return float(out) if np.ndim(out) == 0 else out

    @property
    def lambda_range_nm(self) -> tuple:
        return (float(self.wavelength_nm[0]), float(self.wavelength_nm[-1]))


@dataclass(frozen=True, eq=False)
class LDProfile:
    """Centre-to-limb intensity I(mu, lambda)/I(1, lambda) on a mu grid
    (increasing, mu = 0 at the limb) and a wavelength grid [nm]:
    intensity has shape (n_lambda, n_mu).  Compared by identity."""
    mu: np.ndarray
    wavelength_nm: np.ndarray
    intensity: np.ndarray
    source: str = ""

    def rows(self, wavelength_nm) -> np.ndarray:
        """I(mu)/I(1) interpolated in wavelength: (..., n_mu)."""
        lam = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
        k = np.clip(np.searchsorted(self.wavelength_nm, lam) - 1, 0,
                    self.wavelength_nm.size - 2)
        w = np.clip((lam - self.wavelength_nm[k])
                    / (self.wavelength_nm[k + 1] - self.wavelength_nm[k]), 0.0, 1.0)
        return ((1.0 - w)[:, None] * self.intensity[k]
                + w[:, None] * self.intensity[k + 1])

    def on_grid(self, mu_grid, wavelength_nm) -> np.ndarray:
        """rows() resampled onto another mu grid (linear), (..., n_grid)."""
        r = self.rows(wavelength_nm)
        return np.stack([np.interp(mu_grid, self.mu, row) for row in r])


def integrate_profile_times_mu(mu, rows) -> np.ndarray:
    """int f(mu) mu dmu over the node range for rows (..., n_mu) taken
    as piecewise linear in mu (n_mu,); exact for a linear law."""
    mu = np.asarray(mu, dtype=float)
    f = np.asarray(rows, dtype=float)
    m0, m1 = mu[:-1], mu[1:]
    f0, f1 = f[..., :-1], f[..., 1:]
    h = m1 - m0
    seg = (f0 * (m1**2 - m0**2) / 2.0
           + (f1 - f0) / h * ((m1**3 - m0**3) / 3.0 - m0 * (m1**2 - m0**2) / 2.0))
    return seg.sum(axis=-1)


def linear_ld_rows(u, mu_grid) -> np.ndarray:
    """1 - u (1 - mu) for u (...,) on mu_grid (n_mu,): (..., n_mu)
    (a scalar u gives one row of shape (n_mu,))."""
    u = np.asarray(u, dtype=float)[..., None]
    return 1.0 - u * (1.0 - np.asarray(mu_grid, dtype=float))


@dataclass(frozen=True)
class Star:
    name: str
    mass_msun: float
    radius_rsun: float
    teff: float  # K
    # per-star linear limb-darkening u(lambda) table, ((nm, u), ...) [C11];
    # required, so that every star's law is an explicit choice
    ld_table_nm: tuple
    # optional model-atmosphere hooks (hbtsim.sed): when present they
    # replace the blackbody SED and/or the linear limb-darkening law
    flux_table: FluxTable | None = None
    ld_profile: LDProfile | None = None

    def ld_coeff(self, wavelength_nm):
        """Linear LD coefficient at one wavelength (float) or an array of
        wavelengths (ndarray)."""
        lam, u = zip(*self.ld_table_nm)
        out = np.interp(wavelength_nm, lam, u)
        return float(out) if np.ndim(out) == 0 else out

    @property
    def ld_mode(self) -> str:
        return "table" if self.ld_profile is not None else "linear"

    def ld_rows(self, wavelength_nm, mu_grid) -> np.ndarray:
        """I(mu)/I(1) on mu_grid for each wavelength: (n_lambda, n_mu)."""
        if self.ld_profile is not None:
            return self.ld_profile.on_grid(mu_grid, wavelength_nm)
        return linear_ld_rows(self.ld_coeff(np.atleast_1d(wavelength_nm)), mu_grid)

    def disk_flux_factor(self, wavelength_nm):
        """F / (pi I(1)) = 2 int I(mu)/I(1) mu dmu: 1 - u/3 for the linear
        law; for a tabulated profile the exact integral of its
        piecewise-linear interpolant (what the renderer draws), so a
        tabulated linear law reproduces 1 - u/3 on any mu grid."""
        if self.ld_profile is None:
            out = 1.0 - np.asarray(self.ld_coeff(wavelength_nm)) / 3.0
        else:
            out = 2.0 * integrate_profile_times_mu(
                self.ld_profile.mu, self.ld_profile.rows(wavelength_nm))
        return float(out) if np.ndim(out) == 0 else out

    def surface_flux(self, wavelength_nm):
        """F_lambda at the surface [W m^-2 m^-1]: the model table or pi
        B_lambda(T_eff)."""
        if self.flux_table is not None:
            return self.flux_table(wavelength_nm)
        return np.pi * planck(np.asarray(wavelength_nm, dtype=float) * 1e-9, self.teff)

    def central_intensity(self, wavelength_nm):
        """I(mu = 1) = F / (pi x disk_flux_factor) [W m^-2 m^-1 sr^-1]."""
        return self.surface_flux(wavelength_nm) / (np.pi * self.disk_flux_factor(wavelength_nm))


@dataclass(frozen=True)
class BinarySystem:
    name: str
    primary: Star
    secondary: Star
    period_days: float
    inclination_deg: float
    distance_pc: float
    semimajor_au: float          # relative orbit a = a1 + a2
    eccentricity: float = 0.0
    # omega of the SECONDARY's relative orbit (visual-binary convention);
    # ignored when e = 0 (no periastron: phase 0 is then quadrature)
    arg_periastron_deg: float = 0.0
    # Observed out-of-eclipse (maximum light) apparent magnitudes per band,
    # used to anchor the synthetic lightcurves (Jester et al. 2005
    # Johnson->SDSS transformations of the literature photometry).
    mag_anchors: tuple = (("g", 1.80), ("i", 2.10))
    # Sky orientation.  node_pa_deg is Omega, the position angle of the
    # ASCENDING node (where the secondary crosses the sky plane receding
    # from the observer), measured from North through East, in the
    # visual/interferometric convention with the mirror ambiguity
    # resolved by radial velocities or closure phases where the
    # literature does so.  None keeps the legacy frame (line of nodes
    # along +x), which is a mirror image and has no definite Omega.
    node_pa_deg: float | None = None
    dec_deg: float | None = None     # ICRS declination (for uv projection)
    ra_hours: float | None = None    # ICRS right ascension

    # ---- derived angular quantities (sky plane) ----
    @property
    def angular_semimajor_mas(self) -> float:
        return (self.semimajor_au * AU) / (self.distance_pc * PARSEC) / MAS

    def angular_radius_mas(self, star: Star) -> float:
        return (star.radius_rsun * R_SUN) / (self.distance_pc * PARSEC) / MAS

    @property
    def sum_of_radii_mas(self) -> float:
        return (self.angular_radius_mas(self.primary)
                + self.angular_radius_mas(self.secondary))


# ---------------------------------------------------------------------------
# Eclipse test shared by every analytic (non-overlapping disks) path
# ---------------------------------------------------------------------------
ECLIPSE_MARGIN = 1.05  # "near eclipse" safety factor on the sum of the radii


def in_eclipse_rho(system: BinarySystem, rho_mas, margin: float = ECLIPSE_MARGIN):
    """True where the projected separation rho (mas; scalar or array) is
    below margin x (theta_1 + theta_2), i.e. the disks overlap or nearly
    do and the analytic two-disk visibility is invalid."""
    return np.asarray(rho_mas, dtype=float) < margin * system.sum_of_radii_mas


def in_eclipse(system: BinarySystem, pos, margin: float = ECLIPSE_MARGIN):
    """in_eclipse_rho for an orbit.SkyPositions (scalar epoch or array)."""
    return in_eclipse_rho(system, pos.rho, margin)


def require_out_of_eclipse(system: BinarySystem, rho_mas,
                           what: str = "the analytic binary visibility",
                           alternative: str = "the rendered-image path",
                           margin: float = ECLIPSE_MARGIN) -> None:
    """Raise ValueError if any epoch is in (or near) eclipse."""
    rho = np.asarray(rho_mas, dtype=float)
    if np.any(in_eclipse_rho(system, rho, margin)):
        limit = margin * system.sum_of_radii_mas
        raise ValueError(
            f"in (or near) eclipse (rho = {float(rho.min()):.3f} mas; the "
            f"disks overlap below {limit:.3f} mas): {what} is invalid "
            f"there; use {alternative}")


BETA_AUR = BinarySystem(
    name="Beta Aurigae (Menkalinan)",
    primary=Star("beta Aur Aa", mass_msun=2.376, radius_rsun=2.762,
                 teff=9350.0, ld_table_nm=LD_BETA_AUR),   # [S07]
    secondary=Star("beta Aur Ab", mass_msun=2.291, radius_rsun=2.568,
                   teff=9200.0, ld_table_nm=LD_BETA_AUR),  # [S07]
    period_days=3.96004,        # [S07]
    inclination_deg=76.8,       # [S07] (H95: 76.0 +/- 0.4)
    distance_pc=24.87,          # [vL07], parallax 40.21 mas
    # Kepler's third law with the [S07] masses (4.667 Msun) and period:
    # a^3 = M P^2 -> 0.08186 AU = 3.29 mas at 24.87 pc (H95 measured
    # 3.3 +/- 0.1 mas; Jonak et al. 2026: 3.365 mas at 24.30 pc)
    semimajor_au=0.08186,
    eccentricity=0.0,           # [S07]
    # from V = 1.90, B-V = 0.03 (Bright Star Catalogue): g ~ 1.80, i ~ 2.10
    mag_anchors=(("g", 1.80), ("i", 2.10)),
    # Omega: Jonak et al. 2026 (arXiv:2609.03886; CHARA + RVs + closure
    # phases resolve the mirror: 295.15 deg, vs H95's 115.4 = the other
    # node); position: SIMBAD ICRS
    node_pa_deg=295.15,
    dec_deg=44.94743,
    ra_hours=5.99214,
)

ALGOL = BinarySystem(
    name="Algol (beta Persei) A-B",
    primary=Star("Algol A (B8V)", mass_msun=3.17, radius_rsun=2.73,
                 teff=12550.0, ld_table_nm=LD_ALGOL_A),    # [B12], Teff [K15]
    secondary=Star("Algol B (K0IV)", mass_msun=0.70, radius_rsun=3.48,
                   teff=4900.0, ld_table_nm=LD_ALGOL_B),   # [B12], Teff [K15]
    period_days=2.867328,       # [B12]
    inclination_deg=98.70,      # [B12]
    distance_pc=28.82,          # [Z10], parallax 34.7 mas
    semimajor_au=0.0620,        # [B12] (2.15 mas measured; Kepler-consistent)
    eccentricity=0.0,           # [B12]
    # V_max = 2.12, B-V = -0.05 include Algol C (~10% third light, ~0.10
    # mag); C-corrected A-B-only anchors via Jester et al. 2005:
    mag_anchors=(("g", 2.07), ("i", 2.58)),
    node_pa_deg=43.43,          # [B12] Table 6 inner orbit (+/- 0.32)
    dec_deg=40.955647,
    ra_hours=3.13615,
)

SPICA = BinarySystem(
    name="Spica (alpha Virginis)",
    primary=Star("Spica A (B1 III-IV)", mass_msun=11.43, radius_rsun=7.47,
                 teff=25300.0, ld_table_nm=LD_SPICA),   # [T16]
    # the same [C11] table for B (20.9 kK, log g 4.2): u differs from A's
    # by ~0.02, below the other uncertainties of this hot pair
    secondary=Star("Spica B (B2 V)", mass_msun=7.21, radius_rsun=3.74,
                   teff=20900.0, ld_table_nm=LD_SPICA),  # [T16]
    period_days=4.0145,         # [HE71]/[T16]
    inclination_deg=63.1,       # [T16]; non-eclipsing
    distance_pc=76.6,           # van Leeuwen 2007, parallax 13.06 mas
    semimajor_au=0.1311,        # Kepler's third law with [T16] masses
    eccentricity=0.0,           # TRUE e = 0.108 approximated as circular
    # V = 0.97, B-V = -0.235 -> g ~ 0.71 (Jester et al. 2005); the i
    # anchor is the anchored-blackbody color of a ~25 kK photosphere
    # (+/- ~0.1 mag; only shifts the red-channel zero point)
    mag_anchors=(("g", 0.71), ("i", 1.06)),
    # Omega: [HE71] 131.6 +/- 2.1 (retrograde sense confirmed by the
    # polarimetric orbit of Bailey et al. 2019, 130.4 +/- 6.8)
    node_pa_deg=131.6,
    dec_deg=-11.161319,
    ra_hours=13.41989,
)

DELTA_VEL = BinarySystem(
    name="delta Velorum Aa-Ab",
    # both components use the Beta Aur (9250 K, log g 3.9) [C11] table:
    # within ~0.02 in u of their own (9450 / 9830 K, log g 3.9 / 4.0)
    primary=Star("delta Vel Aa (A2 IV)", mass_msun=2.43, radius_rsun=2.97,
                 teff=9450.0, ld_table_nm=LD_BETA_AUR),   # [M11]
    secondary=Star("delta Vel Ab (A4 V)", mass_msun=2.27, radius_rsun=2.52,
                   teff=9830.0, ld_table_nm=LD_BETA_AUR),  # [M11]
    period_days=45.1503,        # [M11]
    inclination_deg=89.0,       # [M11]; grazing eclipses
    distance_pc=25.13,          # [M11] orbital parallax 39.8 +/- 0.4 mas
    semimajor_au=0.4156,        # Kepler's third law with [M11] masses
    eccentricity=0.290,         # [M11]
    # omega = 109.7 deg [M11], which agrees to 0.1 deg with the ROCHE
    # spectroscopic value of Pribulla et al. 2011 and is therefore the
    # PRIMARY's argument of periastron (RV convention).  This code applies
    # omega to the secondary's relative orbit (orbit.py), where the
    # visual-binary convention would call it omega + 180 deg.  The two
    # choices swap which conjunction falls near periastron and hence the
    # eclipse durations and the phase gap between the minima -- CHECK
    # against the SMEI light curve (primary minimum deeper, secondary
    # ~0.43 in phase later) before using delta Vel eclipse timing.
    arg_periastron_deg=109.7,
    # V = 1.95 for the unresolved A pair + B (F dwarf, V ~ 5.5, ~3.7%
    # third light, excluded): A-only V ~ 1.99, B-V ~ 0.04 -> g ~ 1.90;
    # i anchor from the anchored-blackbody color (+- ~0.1 mag)
    mag_anchors=(("g", 1.90), ("i", 2.25)),
    # Omega: [M11] Table 2, 65.0 +/- 0.6 (the interferometry-only
    # element; NOTE ORB6 lists 155.0 with i -> 180 - i, which is not a
    # valid mirror transformation -- the published value is used)
    node_pa_deg=65.0,
    dec_deg=-54.708821,
    ra_hours=8.74506,
)

SYSTEMS = {"betaaur": BETA_AUR, "algol": ALGOL, "spica": SPICA,
           "deltavel": DELTA_VEL}


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GridConfig:
    n: int = 1024                 # source image grid (pixels)
    pixel_scale_mas: float = 0.01  # mas / pixel
    # zero-padded FFT size for the 2-D |V|^2 maps of hbtsim.fftmap only
    # (dB ~ 1 m at 400 nm); the science path samples V(u, v) by an exact
    # DFT (hbtsim.hbt) and does not use it
    pad: int = 8192

    @property
    def pixel_scale_rad(self) -> float:
        return self.pixel_scale_mas * MAS

    @property
    def half_extent_mas(self) -> float:
        """Largest |x| or |y| (mas) a pixel centre can have on the grid."""
        return (self.n - 1) / 2.0 * self.pixel_scale_mas

    # renderer accuracy: each pixel is the mean of supersample^2 sub-pixel
    # soft-rim renders (1 = the plain kernel); the limb-darkening profile
    # is tabulated on n_mu points uniform in mu
    supersample: int = 1
    n_mu: int = 128

    @property
    def mu_grid(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.n_mu)

    def baseline_step_m(self, wavelength_m: float) -> float:
        """Baseline sampling of the padded FFT map (hbtsim.fftmap only):
        dB = lambda / (pad * dtheta)."""
        return wavelength_m / (self.pad * self.pixel_scale_rad)

    def fit_orbit(self, system: "BinarySystem", n_max: int = 4096,
                  margin: float = 1.1) -> "GridConfig":
        """This grid, with n grown (power of two <= n_max) until the whole
        orbit fits with a margin; the pixel scale is kept.  Cheap enough
        to be the default for every renderer entry point."""
        from .orbit import sky_positions

        r_max = max(system.angular_radius_mas(system.primary),
                    system.angular_radius_mas(system.secondary))
        pos = sky_positions(np.linspace(0.0, 2.0 * np.pi, 4001), system)
        reach = max(float(np.max(np.abs(np.asarray(v)))) for v in
                    (pos.x1, pos.y1, pos.x2, pos.y2)) + r_max
        need = 2.0 * margin * reach / self.pixel_scale_mas + 2.0
        if need <= self.n:
            return self
        n = int(2 ** np.ceil(np.log2(need)))
        if n > n_max:
            raise ValueError(f"{system.name} needs a {n}-pixel grid at "
                             f"{self.pixel_scale_mas:.4f} mas/px (> n_max = {n_max})")
        return GridConfig(n=n, pixel_scale_mas=self.pixel_scale_mas, pad=self.pad,
                          supersample=self.supersample, n_mu=self.n_mu)

    def for_system(self, system: "BinarySystem", min_radius_px: float = 50.0,
                   n_max: int = 4096, margin: float = 1.1) -> "GridConfig":
        """A grid whose pixel scale resolves the smaller star with at
        least min_radius_px pixels (never coarser than this one) and
        whose extent holds the whole orbit with a margin, n a power of
        two <= n_max (raises if the orbit does not fit)."""
        from .orbit import sky_positions

        r_min = min(system.angular_radius_mas(system.primary),
                    system.angular_radius_mas(system.secondary))
        scale = min(self.pixel_scale_mas, r_min / min_radius_px)
        pos = sky_positions(np.linspace(0.0, 2.0 * np.pi, 4001), system)
        reach = max(float(np.max(np.abs(np.asarray(v)))) for v in
                    (pos.x1, pos.y1, pos.x2, pos.y2)) + r_min
        need = 2.0 * margin * reach / scale + 2.0
        n = int(2 ** np.ceil(np.log2(max(need, self.n))))
        if n > n_max:
            raise ValueError(f"{system.name} needs a {n}-pixel grid at "
                             f"{scale:.4f} mas/px (> n_max = {n_max}); relax "
                             f"min_radius_px or raise n_max")
        return GridConfig(n=n, pixel_scale_mas=scale, pad=self.pad,
                          supersample=self.supersample, n_mu=self.n_mu)


@dataclass(frozen=True)
class MovieConfig:
    n_frames: int = 240
    fps: int = 24
    # g2(B) panel
    wavelengths_nm: tuple = (400.0, 800.0)
    baselines_m: tuple = tuple(float(b) for b in range(10, 151, 10))
    fine_baseline_max_m: float = 160.0
    fine_baseline_step_m: float = 1.0
    baseline_pa: str | float = "follow"  # "follow" = along projected separation
    # telescope diameter [m] for aperture-averaged g2 (None: point
    # sampling -- the didactic curve)
    aperture_m: float | None = None
    # lightcurve panel: SDSS effective wavelengths
    bands: tuple = (("g", 477.0), ("i", 763.0))

    @property
    def fine_baselines_m(self) -> np.ndarray:
        return np.arange(0.0, self.fine_baseline_max_m + 1e-9, self.fine_baseline_step_m)
