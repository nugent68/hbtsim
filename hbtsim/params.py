"""Target dataclasses (Star, BinarySystem, DiskTarget), grid configuration
and physical constants.

The systems themselves (beta Aur, Algol, Spica, delta Vel, ...) and their
literature sources live in hbtsim/configs/targets/*.json and are built by
hbtsim.catalog (load_target("spica")); the Claret & Bloemen 2011 linear
limb-darkening tables are hbtsim/configs/ld_tables/*.json.
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
ANCHOR_CHECK_MAG = 0.2      # tolerated |model - observed| anchor miss with SED tables


def planck(wavelength_m: float, teff: float) -> float:
    """Planck spectral radiance B_lambda(T); only relative values matter here."""
    x = H_PLANCK * C_LIGHT / (wavelength_m * K_BOLTZ * teff)
    return 2.0 * H_PLANCK * C_LIGHT**2 / wavelength_m**5 / np.expm1(x)


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
    # spherical models: the table's mu = 0 is the model's OUTER boundary,
    # R_outer = r_outer x R_edge where R_edge is the continuum limb (the
    # tau = 1 radius, what a literature radius means) at mu_edge
    r_outer: float = 1.0
    mu_edge: float = 0.0

    @property
    def nodes(self) -> np.ndarray:
        """The mu grid padded with 0 at the limb (I = 0 there), the grid
        the renderer interpolates on."""
        mu = np.asarray(self.mu, dtype=float)
        return mu if mu[0] <= 0.0 else np.concatenate([[0.0], mu])

    def rows_on_nodes(self, wavelength_nm) -> np.ndarray:
        """rows() on `nodes` (a zero column prepended when padded)."""
        r = self.rows(wavelength_nm)
        if float(np.asarray(self.mu)[0]) <= 0.0:
            return r
        return np.concatenate([np.zeros((r.shape[0], 1)), r], axis=1)

    def rows(self, wavelength_nm) -> np.ndarray:
        """I(mu)/I(1) interpolated in wavelength: (..., n_mu)."""
        lam = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
        if self.wavelength_nm.size == 1:
            return np.broadcast_to(self.intensity[0], (lam.size, self.mu.size)).copy()
        k = np.clip(np.searchsorted(self.wavelength_nm, lam) - 1, 0,
                    self.wavelength_nm.size - 2)
        w = np.clip((lam - self.wavelength_nm[k])
                    / (self.wavelength_nm[k + 1] - self.wavelength_nm[k]), 0.0, 1.0)
        return ((1.0 - w)[:, None] * self.intensity[k]
                + w[:, None] * self.intensity[k + 1])

    def on_grid(self, mu_grid, wavelength_nm) -> np.ndarray:
        """rows() resampled onto another mu grid (linear), (..., n_grid)."""
        r = self.rows(wavelength_nm)
        return np.stack([np.interp(mu_grid, self.mu, row, left=0.0) for row in r])


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
    # model-atmosphere selection and physics hooks (hbtsim.sed): log g
    # (from mass and radius when None), metallicity, projected rotation
    # (rotational broadening of the tables when set), and which radius
    # radius_rsun is: "tau1" (the continuum limb, literature radii) or
    # "outer" (the model's outer boundary)
    logg: float | None = None
    metallicity: float = 0.0
    vsini_kms: float | None = None
    radius_ref: str = "tau1"
    # TODO (gravity darkening): delta Vel's components rotate at ~145 km/s
    # (v/v_crit ~ 0.45), beta Aur at ~34, Algol A at ~50 km/s; oblateness
    # and the von Zeipel T_eff / log g gradient over the surface are not
    # modeled (spherical disks).  sed.NewEraGrid.interpolate(teff, logg) is
    # the per-tile lookup a Roche-surface renderer would call.

    @property
    def log_g(self) -> float:
        """log g [cgs]: the logg field, else from mass and radius."""
        if self.logg is not None:
            return float(self.logg)
        return float(4.438 + np.log10(self.mass_msun) - 2.0 * np.log10(self.radius_rsun))

    @property
    def radius_scale(self) -> float:
        """Drawn (outer) radius over radius_rsun: the profile's
        r_outer when radius_rsun is the tau = 1 radius, else 1."""
        if self.ld_profile is not None and self.radius_ref == "tau1":
            return float(self.ld_profile.r_outer)
        return 1.0

    def ld_nodes(self, grid) -> np.ndarray:
        """The mu grid the renderer uses for this star: the table's own
        (padded) nodes, or the uniform grid for the linear law."""
        if self.ld_profile is not None:
            return self.ld_profile.nodes
        return np.asarray(grid.mu_grid, dtype=float)

    def ld_rows_native(self, wavelength_nm, grid) -> np.ndarray:
        """I(mu)/I(1) on ld_nodes(grid) for each wavelength."""
        lam = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
        if self.ld_profile is not None:
            return self.ld_profile.rows_on_nodes(lam)
        return linear_ld_rows(self.ld_coeff(lam), grid.mu_grid)

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
    # Observed out-of-eclipse (maximum light) apparent AB magnitudes,
    # ((wavelength_nm, mag), ...), used to anchor the synthetic lightcurves
    # and the blackbody photometry (snr.system_ab_mag); empty = the model's
    # own magnitudes
    mag_anchors: tuple = ()
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
    # physics hooks (defaults off): shift each star's model tables by its
    # orbital radial velocity per epoch (sed.prepare_system), and redden
    # the model SED by A_V (Cardelli et al. 1989, R_V = 3.1; snr.model_ab_mag)
    doppler: bool = False
    a_v: float = 0.0

    # ---- derived angular quantities (sky plane) ----
    @property
    def has_sed_tables(self) -> bool:
        """Both stars carry a model-atmosphere flux table: magnitudes come
        from the model and the observed anchors are only a check."""
        return (self.primary.flux_table is not None
                and self.secondary.flux_table is not None)

    @property
    def angular_semimajor_mas(self) -> float:
        return (self.semimajor_au * AU) / (self.distance_pc * PARSEC) / MAS

    def angular_radius_mas(self, star: Star) -> float:
        return (star.radius_rsun * R_SUN) / (self.distance_pc * PARSEC) / MAS

    def drawn_radius_mas(self, star: Star) -> float:
        """The radius the disk is drawn with: angular_radius_mas scaled by
        the star's radius_scale (the model's outer boundary for a
        spherical profile whose radius_rsun is the tau = 1 radius)."""
        return self.angular_radius_mas(star) * star.radius_scale

    @property
    def sum_of_radii_mas(self) -> float:
        return (self.drawn_radius_mas(self.primary)
                + self.drawn_radius_mas(self.secondary))


@dataclass(frozen=True)
class DiskTarget:
    """A compact single source modelled as a (linearly limb-darkened)
    uniform disk of angular diameter theta_mas and a flat AB spectrum
    (the EON-SII white-dwarf targets, hbtsim.montecarlo)."""
    name: str
    theta_mas: float
    mag_ab: float
    dec_deg: float
    ld_u: float = 0.0
    ra_hours: float | None = None


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

        r_max = max(system.drawn_radius_mas(system.primary),
                    system.drawn_radius_mas(system.secondary))
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

    def fit_epoch(self, system: "BinarySystem", pos, margin: float = 1.2,
                  n_min: int = 256, n_max: int = 4096) -> "GridConfig":
        """This grid, with n the smallest power of two (n_min <= n <= n_max)
        holding both disks at ONE epoch (an orbit.SkyPositions) with a
        margin; the pixel scale is kept.  Near conjunction both stars sit
        close to the centre of mass, so this is 16-64x cheaper than the
        whole-orbit grid of fit_orbit for a wide binary."""
        r_max = max(system.drawn_radius_mas(system.primary),
                    system.drawn_radius_mas(system.secondary))
        reach = max(abs(float(np.asarray(v))) for v in (pos.x1, pos.y1, pos.x2, pos.y2)) + r_max
        need = 2.0 * margin * reach / self.pixel_scale_mas + 2.0
        n = int(2 ** np.ceil(np.log2(max(need, n_min))))
        if n > n_max:
            raise ValueError(f"{system.name} needs a {n}-pixel grid at "
                             f"{self.pixel_scale_mas:.4f} mas/px at this epoch (> n_max = {n_max})")
        return GridConfig(n=n, pixel_scale_mas=self.pixel_scale_mas, pad=self.pad,
                          supersample=self.supersample, n_mu=self.n_mu)

    def for_system(self, system: "BinarySystem", min_radius_px: float = 50.0,
                   n_max: int = 4096, margin: float = 1.1) -> "GridConfig":
        """A grid whose pixel scale resolves the smaller star with at
        least min_radius_px pixels (never coarser than this one) and
        whose extent holds the whole orbit with a margin, n a power of
        two <= n_max (raises if the orbit does not fit)."""
        from .orbit import sky_positions

        r_min = min(system.drawn_radius_mas(system.primary),
                    system.drawn_radius_mas(system.secondary))
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
