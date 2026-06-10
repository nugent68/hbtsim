"""System parameters, grid configuration, and physical constants.

All Beta Aurigae values are taken from the literature:

  [S07] Southworth, Bruntt & Buzasi 2007, A&A 467, 1215 (WIRE photometry,
        astro-ph/0703634): P, i, masses, radii, Teff.
  [vL07] van Leeuwen 2007, A&A 474, 653 (Hipparcos re-reduction): parallax.
  [H95] Hummel et al. 1995, AJ 110, 376 (Mark III interferometric orbit).
  [C11] Claret & Bloemen 2011, A&A 529, A75 (VizieR J/A+A/529/A75):
        linear limb-darkening coefficients, ATLAS models, interpolated to
        Teff ~ 9250 K, log g ~ 3.9, [M/H] = 0.
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
# System description
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Star:
    name: str
    mass_msun: float
    radius_rsun: float
    teff: float  # K


@dataclass(frozen=True)
class BinarySystem:
    name: str
    primary: Star
    secondary: Star
    period_days: float
    inclination_deg: float
    distance_pc: float
    semimajor_au: float          # relative orbit a = a1 + a2
    eccentricity: float = 0.0    # only e = 0 is implemented
    # Linear limb-darkening coefficient u vs wavelength [nm], interpolated
    # linearly between table points (same law assumed for both stars).
    ld_table_nm: tuple = ((400.0, 0.52), (477.0, 0.50), (763.0, 0.30), (800.0, 0.29))
    # Observed out-of-eclipse (maximum light) apparent magnitudes per band,
    # used to anchor the synthetic lightcurves.  For Beta Aurigae these are
    # derived from V = 1.90, B-V = 0.03 (Bright Star Catalogue) with the
    # Jester et al. 2005 Johnson->SDSS transformations: g ~ 1.80, i ~ 2.10.
    mag_anchors: tuple = (("g", 1.80), ("i", 2.10))

    # ---- derived angular quantities (sky plane) ----
    @property
    def angular_semimajor_mas(self) -> float:
        return (self.semimajor_au * AU) / (self.distance_pc * PARSEC) / MAS

    def angular_radius_mas(self, star: Star) -> float:
        return (star.radius_rsun * R_SUN) / (self.distance_pc * PARSEC) / MAS

    def ld_coeff(self, wavelength_nm: float) -> float:
        lam, u = zip(*self.ld_table_nm)
        return float(np.interp(wavelength_nm, lam, u))


BETA_AUR = BinarySystem(
    name="Beta Aurigae (Menkalinan)",
    primary=Star("beta Aur Aa", mass_msun=2.376, radius_rsun=2.762, teff=9350.0),   # [S07]
    secondary=Star("beta Aur Ab", mass_msun=2.291, radius_rsun=2.568, teff=9200.0),  # [S07]
    period_days=3.96004,        # [S07]
    inclination_deg=76.8,       # [S07] (H95: 76.0 +/- 0.4)
    distance_pc=24.87,          # [vL07], parallax 40.21 mas
    semimajor_au=0.08214,       # Kepler's third law with [S07] masses; cf. H95
    eccentricity=0.0,           # [S07]
)


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
