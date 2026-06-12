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
        parallax 80.6 pc.  The B component (F dwarf, ~0.6 arcsec away,
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
@dataclass(frozen=True)
class Star:
    name: str
    mass_msun: float
    radius_rsun: float
    teff: float  # K
    # per-star linear limb-darkening u(lambda) table, ((nm, u), ...) [C11]
    ld_table_nm: tuple = LD_BETA_AUR

    def ld_coeff(self, wavelength_nm: float) -> float:
        lam, u = zip(*self.ld_table_nm)
        return float(np.interp(wavelength_nm, lam, u))


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
    arg_periastron_deg: float = 0.0  # omega; ignored when e = 0
    # Observed out-of-eclipse (maximum light) apparent magnitudes per band,
    # used to anchor the synthetic lightcurves (Jester et al. 2005
    # Johnson->SDSS transformations of the literature photometry).
    mag_anchors: tuple = (("g", 1.80), ("i", 2.10))

    # ---- derived angular quantities (sky plane) ----
    @property
    def angular_semimajor_mas(self) -> float:
        return (self.semimajor_au * AU) / (self.distance_pc * PARSEC) / MAS

    def angular_radius_mas(self, star: Star) -> float:
        return (star.radius_rsun * R_SUN) / (self.distance_pc * PARSEC) / MAS


BETA_AUR = BinarySystem(
    name="Beta Aurigae (Menkalinan)",
    primary=Star("beta Aur Aa", mass_msun=2.376, radius_rsun=2.762,
                 teff=9350.0, ld_table_nm=LD_BETA_AUR),   # [S07]
    secondary=Star("beta Aur Ab", mass_msun=2.291, radius_rsun=2.568,
                   teff=9200.0, ld_table_nm=LD_BETA_AUR),  # [S07]
    period_days=3.96004,        # [S07]
    inclination_deg=76.8,       # [S07] (H95: 76.0 +/- 0.4)
    distance_pc=24.87,          # [vL07], parallax 40.21 mas
    semimajor_au=0.08214,       # Kepler's third law with [S07] masses; cf. H95
    eccentricity=0.0,           # [S07]
    # from V = 1.90, B-V = 0.03 (Bright Star Catalogue): g ~ 1.80, i ~ 2.10
    mag_anchors=(("g", 1.80), ("i", 2.10)),
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
)

SPICA = BinarySystem(
    name="Spica (alpha Virginis)",
    primary=Star("Spica A (B1 III-IV)", mass_msun=11.43, radius_rsun=7.47,
                 teff=25300.0, ld_table_nm=LD_SPICA),   # [T16]
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
)

DELTA_VEL = BinarySystem(
    name="delta Velorum Aa-Ab",
    primary=Star("delta Vel Aa (A2 IV)", mass_msun=2.43, radius_rsun=2.97,
                 teff=9450.0, ld_table_nm=LD_BETA_AUR),   # [M11]
    secondary=Star("delta Vel Ab (A4 V)", mass_msun=2.27, radius_rsun=2.52,
                   teff=9830.0, ld_table_nm=LD_BETA_AUR),  # [M11]
    period_days=45.1503,        # [M11]
    inclination_deg=89.0,       # [M11]; grazing eclipses
    distance_pc=80.6,           # [M11] orbital parallax
    semimajor_au=0.4156,        # Kepler's third law with [M11] masses
    eccentricity=0.290,         # [M11]
    arg_periastron_deg=109.7,   # [M11]
    # V = 1.95 for the unresolved A pair + B (F dwarf, V ~ 5.5, ~3.7%
    # third light, excluded): A-only V ~ 1.99, B-V ~ 0.04 -> g ~ 1.90;
    # i anchor from the anchored-blackbody color (+- ~0.1 mag)
    mag_anchors=(("g", 1.90), ("i", 2.25)),
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
    pad: int = 8192               # zero-padded FFT size (dB ~ 1 m at 400 nm)

    @property
    def pixel_scale_rad(self) -> float:
        return self.pixel_scale_mas * MAS

    def baseline_step_m(self, wavelength_m: float) -> float:
        """Baseline sampling of the padded FFT: dB = lambda / (pad * dtheta)."""
        return wavelength_m / (self.pad * self.pixel_scale_rad)


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
    # lightcurve panel: SDSS effective wavelengths
    bands: tuple = (("g", 477.0), ("i", 763.0))

    @property
    def fine_baselines_m(self) -> np.ndarray:
        return np.arange(0.0, self.fine_baseline_max_m + 1e-9, self.fine_baseline_step_m)
