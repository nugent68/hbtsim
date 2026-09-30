"""Model-atmosphere hooks: flux tables and centre-to-limb profiles.

A Star may carry a FluxTable (surface flux F_lambda) and an LDProfile
(I(mu, lambda)/I(1, lambda)) in place of the blackbody + linear
limb-darkening defaults (hbtsim.params).  This module builds them:

  * from NewEra PHOENIX HSR-RF files (Hauschildt et al. 2025, A&A 698,
    A47; the angle-resolved "RF" variant provided on request by the
    authors): read_newera_hsr_rf() reads /PHOENIX_RF/{wl, flux, mu,
    Intensities}, restricts to a wavelength range and bins to a common
    step; scripts/prepare_newera.py writes the result to a small .npz
    that load_star_tables() reads back (no h5py needed at run time).
  * synthetic tables for tests and cross-checks: planck_flux_table()
    (F = pi B_lambda), linear_ld_profile() (the linear law tabulated).

NewEra conventions (checked on the example HSR-RF file): wavelengths in
vacuum Angstrom; flux is log10 of F_lambda in erg s^-1 cm^-2 cm^-1 at
the surface (1 erg s^-1 cm^-2 cm^-1 = 0.1 W m^-2 m^-1); the 127 mu
values are 64 core rays plus 63 tangent rays of the spherical model,
with Intensities (n_wl, 127) in the same units per steradian.  The
intensity table is normalized here to I(mu = 1) so that the renderer's
central-intensity weights come from the flux table alone
(Star.central_intensity = F / (pi x 2 int I mu dmu)).
"""

from __future__ import annotations

import glob
import os
import re
import warnings
from dataclasses import dataclass, replace

import numpy as np

from .params import (C_LIGHT, FluxTable, LDProfile, integrate_profile_times_mu,
                     linear_ld_rows, planck)

CGS_FLAM_TO_SI = 0.1     # erg s^-1 cm^-2 cm^-1 -> W m^-2 m^-1


# ---------------------------------------------------------------------------
# Synthetic tables
# ---------------------------------------------------------------------------
def planck_flux_table(teff: float, wavelength_nm) -> FluxTable:
    """F_lambda = pi B_lambda(T_eff) on the given grid [nm]."""
    lam = np.asarray(wavelength_nm, dtype=float)
    return FluxTable(wavelength_nm=lam, flux=np.pi * planck(lam * 1e-9, teff),
                     source=f"blackbody {teff:g} K")


def linear_ld_profile(ld_table_nm, wavelength_nm, n_mu: int = 64) -> LDProfile:
    """The linear law 1 - u(lambda)(1 - mu) tabulated on a mu grid."""
    lam = np.asarray(wavelength_nm, dtype=float)
    tab_lam, tab_u = zip(*ld_table_nm)
    u = np.interp(lam, tab_lam, tab_u)
    mu = np.linspace(0.0, 1.0, n_mu)
    return LDProfile(mu=mu, wavelength_nm=lam, intensity=linear_ld_rows(u, mu),
                     source="linear law")


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------
def bin_to_step(wavelength_nm, values, lam_min_nm, lam_max_nm, step_nm):
    """Mean of `values` (n_wl, ...) in bins of step_nm over
    [lam_min, lam_max]; returns (bin_centres, binned).  Empty bins are
    filled by interpolation from their neighbours."""
    lam = np.asarray(wavelength_nm, dtype=float)
    vals = np.asarray(values)
    edges = np.arange(lam_min_nm, lam_max_nm + step_nm / 2, step_nm)
    idx = np.searchsorted(edges, lam, side="right") - 1
    ok = (idx >= 0) & (idx < edges.size - 1)
    idx, v = idx[ok], vals[ok]
    n = edges.size - 1
    counts = np.bincount(idx, minlength=n)
    flat = v.reshape(v.shape[0], -1)
    sums = np.stack([np.bincount(idx, weights=flat[:, j], minlength=n)
                     for j in range(flat.shape[1])], axis=1)
    out = np.full(sums.shape, np.nan)
    good = counts > 0
    out[good] = sums[good] / counts[good, None]
    centres = 0.5 * (edges[:-1] + edges[1:])
    if not good.all():
        for j in range(out.shape[1]):
            out[~good, j] = np.interp(centres[~good], centres[good], out[good, j])
    return centres, out.reshape((n,) + v.shape[1:])


# ---------------------------------------------------------------------------
# NewEra HSR-RF files
# ---------------------------------------------------------------------------
def read_newera_hsr_rf(path: str, lam_min_nm: float = 380.0,
                       lam_max_nm: float = 1000.0, step_nm: float = 0.02,
                       chunk: int = 200_000) -> dict:
    """Read a NewEra HSR-RF HDF5 file (requires h5py) and bin the flux
    and the angle-resolved intensities to step_nm over the range.
    Returns dict(wavelength_nm, flux [W m^-2 m^-1], mu, intensity
    [normalized to mu = 1], teff/logg/meta when present)."""
    import h5py

    with h5py.File(path, "r") as h:
        g = h["PHOENIX_RF"]
        wl = np.asarray(g["wl"][()], dtype=float) / 10.0     # vacuum A -> nm
        mu = np.asarray(g["mu"][()], dtype=float)
        lo = int(np.searchsorted(wl, lam_min_nm))
        hi = int(np.searchsorted(wl, lam_max_nm, side="right"))
        cen = None
        flux_acc, int_acc, cnt_acc = None, None, None
        for a in range(lo, hi, chunk):
            b = min(a + chunk, hi)
            w = wl[a:b]
            f = 10.0 ** np.asarray(g["flux"][a:b], dtype=float) * CGS_FLAM_TO_SI
            inten = np.asarray(g["Intensities"][a:b, :], dtype=float)
            edges = np.arange(lam_min_nm, lam_max_nm + step_nm / 2, step_nm)
            idx = np.searchsorted(edges, w, side="right") - 1
            n = edges.size - 1
            ok = (idx >= 0) & (idx < n)
            idx = idx[ok]
            if cen is None:
                cen = 0.5 * (edges[:-1] + edges[1:])
                flux_acc = np.zeros(n)
                int_acc = np.zeros((n, mu.size))
                cnt_acc = np.zeros(n)
            flux_acc += np.bincount(idx, weights=f[ok], minlength=n)
            for j in range(mu.size):
                int_acc[:, j] += np.bincount(idx, weights=inten[ok, j], minlength=n)
            cnt_acc += np.bincount(idx, minlength=n)
        meta = {k: g.attrs[k] for k in g.attrs} if hasattr(g, "attrs") else {}
        extra = _newera_file_extras(g, wl, mu)
    good = cnt_acc > 0
    flux = np.full(cen.size, np.nan)
    flux[good] = flux_acc[good] / cnt_acc[good]
    inten = np.full((cen.size, mu.size), np.nan)
    inten[good] = int_acc[good] / cnt_acc[good, None]
    if not good.all():
        flux[~good] = np.interp(cen[~good], cen[good], flux[good])
        for j in range(mu.size):
            inten[~good, j] = np.interp(cen[~good], cen[good], inten[good, j])
    order = np.argsort(mu)
    mu = mu[order]
    inten = inten[:, order]
    centre = inten[:, np.argmax(mu)]
    norm = inten / np.maximum(centre[:, None], 1e-300)
    return dict(wavelength_nm=cen, flux=flux, mu=mu, intensity=norm,
                meta=meta, path=path, **extra)


def _newera_file_extras(g, wl_nm, mu) -> dict:
    """r0 (tau = 1 radius at wltau), T_eff, log g, mass, and the limb of
    the spherical model at wltau: mu_tau1 from the tangent-ray optical
    depths (dtau), mu_edge from the half-intensity drop.  All optional."""
    out = {}
    scal = lambda k: float(np.ravel(g[k][()])[0])
    for k, name in (("r0", "r0_cm"), ("teff", "teff"), ("logg", "logg"),
                    ("m_sun", "m_sun"), ("wltau", "wltau_A")):
        if k in g:
            out[name] = scal(k)
    wltau_nm = out.get("wltau_A", 5000.0) / 10.0
    k = int(np.searchsorted(wl_nm, wltau_nm))
    mu = np.asarray(mu, dtype=float)
    order = np.argsort(mu)
    if "dtau" in g and "ntang" in g:
        ntang = int(scal("ntang"))
        tau = np.asarray(g["dtau"][k, :], dtype=float)
        mu_t = mu[order][:ntang]
        lt = np.log(np.maximum(tau, 1e-300))
        j = int(np.searchsorted(lt, 0.0))
        if 0 < j < ntang:
            f = (0.0 - lt[j - 1]) / (lt[j] - lt[j - 1])
            out["mu_tau1"] = float(mu_t[j - 1] + f * (mu_t[j] - mu_t[j - 1]))
            out["r_outer_over_tau1"] = 1.0 / np.sqrt(1.0 - out["mu_tau1"]**2)
    row = np.asarray(g["Intensities"][k, :], dtype=float)[order]
    row = row / max(row[-1], 1e-300)
    out["mu_edge"] = limb_edge(mu[order], row)
    out["r_outer_over_edge"] = 1.0 / np.sqrt(1.0 - out["mu_edge"]**2)
    return out


def save_star_tables(path_npz: str, tables: dict) -> None:
    extra = {k: tables[k] for k in ("r0_cm", "teff", "logg", "m_sun", "wltau_A",
                                    "mu_tau1", "mu_edge", "r_outer_over_tau1",
                                    "r_outer_over_edge") if k in tables}
    np.savez_compressed(path_npz, wavelength_nm=tables["wavelength_nm"],
                        flux=tables["flux"], mu=tables["mu"],
                        intensity=tables["intensity"].astype(np.float32),
                        source=np.str_(tables.get("path", "")), **extra)


def load_star_tables(path_npz: str) -> tuple:
    """(FluxTable, LDProfile) from a prepare_newera.py output."""
    d = np.load(path_npz, allow_pickle=False)
    src = str(d["source"]) if "source" in d else path_npz
    ft = FluxTable(wavelength_nm=d["wavelength_nm"], flux=d["flux"], source=src)
    r_outer = float(d["r_outer_over_tau1"]) if "r_outer_over_tau1" in d else None
    mu_edge = float(d["mu_tau1"]) if "mu_tau1" in d else None
    ld = _profile_with_edge(d["mu"], d["wavelength_nm"],
                            np.asarray(d["intensity"], dtype=float), src,
                            r_outer, mu_edge, ft)
    return ft, ld


def with_tables(star, flux_table: FluxTable | None = None,
                ld_profile: LDProfile | None = None):
    """A copy of a Star carrying the given tables."""
    from dataclasses import replace
    return replace(star, flux_table=flux_table, ld_profile=ld_profile)


# ---------------------------------------------------------------------------
# Spherical models: where is the limb?
# ---------------------------------------------------------------------------
def limb_edge(mu, row, threshold: float = 0.5, plateau_mu: float = 0.2) -> float:
    """mu at which a (continuum) profile row drops to `threshold` times its
    value at plateau_mu, going outward: the apparent limb of a spherical
    model whose intensity falls to ~0 at mu > 0 (the tangent rays).  Plane-
    parallel profiles (no drop) return 0."""
    mu = np.asarray(mu, dtype=float)
    row = np.asarray(row, dtype=float)
    level = threshold * float(np.interp(plateau_mu, mu, row))
    below = np.where(row < level)[0]
    if below.size == 0 or below[-1] + 1 >= mu.size:
        return 0.0
    j = int(below[-1])
    f = (level - row[j]) / max(row[j + 1] - row[j], 1e-300)
    return float(mu[j] + f * (mu[j + 1] - mu[j]))


def spherical_extension(ld_profile, flux_table=None, wavelength_nm: float = 500.0,
                        window_nm: float = 10.0) -> dict:
    """The limb of a tabulated spherical profile at a continuum wavelength:
    the brightest table point within +/- window_nm of wavelength_nm (the
    continuum, when a flux table is given), its limb_edge mu_edge, and
    R_outer / R_edge = 1 / sqrt(1 - mu_edge^2)."""
    lam = np.asarray(ld_profile.wavelength_nm, dtype=float)
    sel = np.where(np.abs(lam - wavelength_nm) <= window_nm)[0]
    if sel.size == 0:
        sel = np.array([int(np.argmin(np.abs(lam - wavelength_nm)))])
    if flux_table is not None and np.array_equal(flux_table.wavelength_nm, lam):
        k = int(sel[np.argmax(np.asarray(flux_table.flux)[sel])])
    else:
        k = int(sel[sel.size // 2])
    mu_edge = limb_edge(ld_profile.mu, ld_profile.intensity[k])
    return dict(mu_edge=mu_edge, r_outer_over_edge=1.0 / np.sqrt(1.0 - mu_edge**2),
                wavelength_used_nm=float(lam[k]))


def _profile_with_edge(mu, lam, inten, source, r_outer=None, mu_edge=None,
                       flux_table=None):
    """LDProfile with r_outer / mu_edge taken from the file when stored,
    otherwise measured from the 500 nm continuum drop."""
    prof = LDProfile(mu=mu, wavelength_nm=lam, intensity=inten, source=source)
    if r_outer is None or not np.isfinite(r_outer) or r_outer <= 0:
        ext = spherical_extension(prof, flux_table)
        r_outer, mu_edge = ext["r_outer_over_edge"], ext["mu_edge"]
    if mu_edge is None or not np.isfinite(mu_edge):
        mu_edge = float(np.sqrt(max(0.0, 1.0 - 1.0 / r_outer**2)))
    return replace(prof, r_outer=float(r_outer), mu_edge=float(mu_edge))


# ---------------------------------------------------------------------------
# Channel averaging
# ---------------------------------------------------------------------------
def band_average_flux(table: FluxTable, edges_nm) -> np.ndarray:
    """Mean of the table over each [e_k, e_k+1]: the cumulative trapezoid
    integral of the piecewise-linear table, differenced at the edges
    (exact when the edges fall on table points; channels narrower than
    a table step fall back to the point value at the centre)."""
    lam = np.asarray(table.wavelength_nm, dtype=float)
    f = np.asarray(table.flux, dtype=float)
    e = np.asarray(edges_nm, dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(lam))])
    ce = np.interp(e, lam, cum)
    width = np.diff(e)
    out = np.diff(ce) / width
    step = float(np.median(np.diff(lam)))
    narrow = width < step
    if narrow.any():
        out[narrow] = np.interp(0.5 * (e[:-1] + e[1:])[narrow], lam, f)
    return out


def band_average_profile(ld: LDProfile, flux: FluxTable, edges_nm) -> np.ndarray:
    """Channel-averaged I(mu)/I(1) rows (n_ch, n_mu): the table points in
    each channel weighted by their central intensity F / (pi dff) -- the
    normalized profile of the channel-integrated image.  Channels with no
    table point take the row at their centre."""
    lam = np.asarray(ld.wavelength_nm, dtype=float)
    if not np.array_equal(lam, np.asarray(flux.wavelength_nm, dtype=float)):
        raise ValueError("band_average_profile needs the flux table and the "
                         "profile on the same wavelength grid")
    e = np.asarray(edges_nm, dtype=float)
    n = e.size - 1
    dff = 2.0 * integrate_profile_times_mu(ld.mu, ld.intensity)
    w = np.asarray(flux.flux, dtype=float) / np.maximum(dff, 1e-300)
    idx = np.searchsorted(e, lam, side="right") - 1
    ok = (idx >= 0) & (idx < n)
    wsum = np.bincount(idx[ok], weights=w[ok], minlength=n)
    rows = np.zeros((n, ld.mu.size))
    inten = np.asarray(ld.intensity, dtype=float)
    for j in range(ld.mu.size):
        rows[:, j] = np.bincount(idx[ok], weights=(w * inten[:, j])[ok], minlength=n)
    good = wsum > 0
    rows[good] /= wsum[good, None]
    if not good.all():
        rows[~good] = ld.rows(0.5 * (e[:-1] + e[1:])[~good])
    return rows


_REBIN_CACHE: dict = {}
_REBIN_CACHE_MAX = 64


def rebin_to_channels(star, edges_nm):
    """The star with its tables replaced by channel-averaged ones on the
    channel centres (flux: band mean; profile: intensity-weighted mean);
    unchanged when it carries no tables.  Memoized on the source tables
    and the channel edges, so repeated calls (every SNR evaluation of a
    feasibility scan) return the SAME table objects and the disk-
    visibility cache keyed on them keeps hitting."""
    if star.flux_table is None and star.ld_profile is None:
        return star
    e = np.asarray(edges_nm, dtype=float)
    centres_req = 0.5 * (e[:-1] + e[1:])
    # idempotent: tables already on these channel centres are returned as
    # they are (a second pass would smooth them by ~[1/8, 3/4, 1/8])
    done = True
    for t in (star.flux_table, star.ld_profile):
        if t is not None:
            w = np.asarray(t.wavelength_nm, dtype=float)
            done = done and w.shape == centres_req.shape and np.allclose(w, centres_req, rtol=0, atol=1e-9)
    if done:
        return star
    key = (id(star.flux_table), id(star.ld_profile), e.size, float(e[0]), float(e[-1]),
           hash(e.tobytes()))
    hit = _REBIN_CACHE.get(key)
    if hit is not None and hit[0] is star.flux_table and hit[1] is star.ld_profile:
        return with_tables(star, hit[2], hit[3])
    centres = 0.5 * (e[:-1] + e[1:])
    ft = star.flux_table
    ld = star.ld_profile
    if ft is not None:
        ft = FluxTable(wavelength_nm=centres, flux=band_average_flux(ft, e),
                       source=ft.source)
    if ld is not None:
        if star.flux_table is not None and np.array_equal(
                ld.wavelength_nm, star.flux_table.wavelength_nm):
            rows = band_average_profile(ld, star.flux_table, e)
        else:
            rows = ld.rows(centres)
        ld = replace(ld, wavelength_nm=centres, intensity=rows)
    if len(_REBIN_CACHE) >= _REBIN_CACHE_MAX:
        _REBIN_CACHE.clear()
    # keep the source tables referenced so their ids stay unique
    _REBIN_CACHE[key] = (star.flux_table, star.ld_profile, ft, ld)
    return with_tables(star, ft, ld)


def prepare_system(system, spectrograph=None, pos=None):
    """The system as a channelized measurement sees it: each star's tables
    Doppler-shifted for the epoch (when BinarySystem.doppler is set and
    pos carries radial velocities) and averaged over the spectrograph's
    channels (so that point-sampling at the channel centres returns the
    channel means).  A no-op for stars without tables."""
    if not (system.primary.flux_table is not None or system.primary.ld_profile is not None
            or system.secondary.flux_table is not None
            or system.secondary.ld_profile is not None):
        return system
    stars = {"primary": system.primary, "secondary": system.secondary}
    if getattr(system, "doppler", False) and pos is not None:
        for key, vr in (("primary", getattr(pos, "vr1_kms", None)),
                        ("secondary", getattr(pos, "vr2_kms", None))):
            if vr is not None:
                stars[key] = doppler_shift(stars[key], float(np.asarray(vr)))
    if spectrograph is not None:
        edges = np.asarray(spectrograph.channel_edges_nm, dtype=float)
        if getattr(spectrograph, "frame", "vacuum") == "air":
            edges = air_to_vacuum(edges)
        stars = {k: rebin_to_channels(s, edges) for k, s in stars.items()}
    return replace(system, primary=stars["primary"], secondary=stars["secondary"])


def doppler_shift(star, v_kms: float):
    """The star's tables shifted to the observer's frame for a radial
    velocity v (positive = receding): lambda -> lambda (1 + v/c)."""
    if abs(v_kms) < 1e-9 or (star.flux_table is None and star.ld_profile is None):
        return star
    f = 1.0 + v_kms * 1e3 / C_LIGHT
    ft = star.flux_table
    ld = star.ld_profile
    if ft is not None:
        ft = replace(ft, wavelength_nm=np.asarray(ft.wavelength_nm) * f)
    if ld is not None:
        ld = replace(ld, wavelength_nm=np.asarray(ld.wavelength_nm) * f)
    return with_tables(star, ft, ld)


def cardelli_extinction(lam_nm, a_v: float, r_v: float = 3.1):
    """A_lambda [mag] from Cardelli, Clayton & Mathis 1989 for A_V and
    R_V, over the IR (0.3 <= x < 1.1 um^-1) and optical/NUV
    (1.1 <= x <= 3.3) branches; x = 1/lambda[um]."""
    lam_um = np.asarray(lam_nm, dtype=float) * 1e-3
    x = 1.0 / lam_um
    if np.any(x < 0.3) or np.any(x > 3.3):
        raise ValueError("cardelli_extinction: wavelengths must lie in 303-3333 nm")
    a = np.where(x < 1.1, 0.574 * x**1.61, 0.0)
    b = np.where(x < 1.1, -0.527 * x**1.61, 0.0)
    y = x - 1.82
    opt = x >= 1.1
    a_opt = (1.0 + 0.17699 * y - 0.50447 * y**2 - 0.02427 * y**3 + 0.72085 * y**4
             + 0.01979 * y**5 - 0.77530 * y**6 + 0.32999 * y**7)
    b_opt = (1.41338 * y + 2.28305 * y**2 + 1.07233 * y**3 - 5.38434 * y**4
             - 0.62251 * y**5 + 5.30260 * y**6 - 2.09002 * y**7)
    a = np.where(opt, a_opt, a)
    b = np.where(opt, b_opt, b)
    out = a_v * (a + b / r_v)
    return float(out) if np.ndim(out) == 0 else out


def air_to_vacuum(lam_nm):
    """Air -> vacuum wavelengths [nm] (Morton 1991 / IAU standard)."""
    lam = np.asarray(lam_nm, dtype=float)
    s2 = (1e4 / (lam * 10.0)) ** 2          # (1/lambda[um])^2
    n = 1.0 + 6.4328e-5 + 2.94981e-2 / (146.0 - s2) + 2.5540e-4 / (41.0 - s2)
    return lam * n


def vacuum_to_air(lam_nm):
    """Vacuum -> air wavelengths [nm] (inverse of air_to_vacuum, iterated)."""
    lam = np.asarray(lam_nm, dtype=float)
    air = lam.copy()
    for _ in range(3):
        air = lam / (air_to_vacuum(air) / air)
    return air


# ---------------------------------------------------------------------------
# The NewEra grid
# ---------------------------------------------------------------------------
_NEWERA_RE = re.compile(r"lte(\d{5})-(\d\.\d\d)([-+]\d\.\d)")


@dataclass(frozen=True)
class NewEraTables:
    flux_table: FluxTable
    ld_profile: LDProfile
    corners: tuple            # ((teff, logg, weight), ...)
    extrapolated: bool = False
    clamped: dict = None      # {"teff": (requested, used), ...}


class NewEraGrid:
    """Binned NewEra tables (prepare_newera / bin_rf_standalone .npz) keyed
    by (T_eff, log g, [M/H]), with bilinear interpolation in T_eff and
    log g of log10 F(lambda) and of the centre-to-limb profile.

    Profiles are interpolated on a common radial grid r' = r / R_edge
    (R_edge the continuum limb of each corner model, LDProfile.r_outer)
    so that the limb of the interpolated model sits at r' = 1: the
    tangent-ray tails of the corners, which differ in extent, are kept
    and the result is expressed on a mu grid relative to the largest
    corner R_outer.  This is also the (T_eff, log g) lookup a surface-
    tiling (gravity-darkened) renderer would call per tile."""

    def __init__(self, tables: dict):
        self.tables = dict(tables)      # (teff, logg, z) -> path or loaded tuple

    @classmethod
    def scan(cls, directory: str, pattern: str = "newera_lte*.npz") -> "NewEraGrid":
        found = {}
        for path in sorted(glob.glob(os.path.join(directory, pattern))):
            m = _NEWERA_RE.search(os.path.basename(path))
            if not m:
                continue
            key = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            found.setdefault(key, path)      # first match wins (sorted: finest step first)
        if not found:
            raise FileNotFoundError(f"no NewEra tables matching {pattern} in {directory}")
        return cls(found)

    def coverage(self, z: float = 0.0) -> dict:
        keys = [k for k in self.tables if k[2] == z]
        return dict(teff=sorted({k[0] for k in keys}), logg=sorted({k[1] for k in keys}),
                    n=len(keys))

    def load(self, key):
        v = self.tables[key]
        if isinstance(v, str):
            v = load_star_tables(v)
            self.tables[key] = v
        return v

    @staticmethod
    def _bracket(values, x):
        """Neighbouring grid values (lo, hi) around x; (x, x) when x sits on
        a grid value, so no spurious upper neighbour is demanded."""
        values = sorted(values)
        for v in values:
            if abs(x - v) < 1e-9:
                return v, v
        if x <= values[0]:
            return values[0], values[0]
        if x >= values[-1]:
            return values[-1], values[-1]
        k = int(np.searchsorted(values, x, side="right"))
        return values[k - 1], values[k]

    MAX_CLAMP_TEFF = 1000.0    # K beyond the grid edge that allow_extrapolation may clamp
    MAX_CLAMP_LOGG = 0.5       # dex

    def interpolate(self, teff: float, logg: float, z: float = 0.0,
                    allow_extrapolation: bool = False) -> NewEraTables:
        """Tables at (teff, logg): bilinear inside the grid; outside it a
        ValueError, or -- with allow_extrapolation and within
        MAX_CLAMP_TEFF / MAX_CLAMP_LOGG of the edge -- the edge model,
        flagged and warned about (Algol A at 12 550 K -> 12 000 K).  A
        25 000 K star is never clamped onto a 12 000 K model."""
        keys = [k for k in self.tables if k[2] == z]
        if not keys:
            raise ValueError(f"no NewEra tables at [M/H] = {z}")
        teffs = sorted({k[0] for k in keys})
        loggs = sorted({k[1] for k in keys})
        clamped = {}
        t_use, g_use = float(teff), float(logg)
        if not (teffs[0] <= teff <= teffs[-1]):
            t_use = float(np.clip(teff, teffs[0], teffs[-1]))
            clamped["teff"] = (float(teff), t_use)
        if not (loggs[0] <= logg <= loggs[-1]):
            g_use = float(np.clip(logg, loggs[0], loggs[-1]))
            clamped["logg"] = (float(logg), g_use)
        t0, t1 = self._bracket(teffs, t_use)
        g0, g1 = self._bracket(loggs, g_use)
        wt = 0.0 if t1 == t0 else (t_use - t0) / (t1 - t0)
        wg = 0.0 if g1 == g0 else (g_use - g0) / (g1 - g0)
        corners = []
        for t, w_t in ((t0, 1.0 - wt), (t1, wt)):
            for g, w_g in ((g0, 1.0 - wg), (g1, wg)):
                w = w_t * w_g
                if w > 0.0:
                    corners.append((t, g, w))
        # only the corners that actually carry weight must exist
        missing = [(t, g, z) for t, g, _ in corners if (t, g, z) not in self.tables]
        if missing:
            near = sorted(keys, key=lambda k: ((k[0] - teff) / 100.0)**2 + ((k[1] - logg) / 0.25)**2)[:4]
            raise ValueError(f"NewEra grid does not bracket T_eff = {teff:.0f} K, "
                             f"log g = {logg:.2f}: missing corner(s) {sorted(missing)}; "
                             f"nearest models {[(k[0], k[1]) for k in near]}")
        if clamped:
            too_far = (abs(teff - t_use) > self.MAX_CLAMP_TEFF
                       or abs(logg - g_use) > self.MAX_CLAMP_LOGG)
            if not allow_extrapolation or too_far:
                raise ValueError(f"({teff:.0f} K, log g {logg:.2f}) lies outside the NewEra "
                                 f"grid (T_eff {teffs[0]:.0f}-{teffs[-1]:.0f} K, log g "
                                 f"{loggs[0]:.1f}-{loggs[-1]:.1f})"
                                 + (f" by more than {self.MAX_CLAMP_TEFF:.0f} K / "
                                    f"{self.MAX_CLAMP_LOGG} dex: not clamped" if too_far else
                                    "; pass allow_extrapolation=True to clamp to the edge"))
            warnings.warn(f"NewEra: ({teff:.0f} K, log g {logg:.2f}) clamped to the grid "
                          f"edge {clamped}", stacklevel=2)
        loaded = [(self.load((t, g, z)), w) for t, g, w in corners]
        lam = np.asarray(loaded[0][0][0].wavelength_nm, dtype=float)
        for (ft, ld), _ in loaded:
            if not (np.shape(ft.wavelength_nm) == lam.shape and np.shape(ld.wavelength_nm) == lam.shape
                    and np.allclose(ft.wavelength_nm, lam, rtol=0, atol=1e-6)
                    and np.allclose(ld.wavelength_nm, lam, rtol=0, atol=1e-6)):
                raise ValueError("NewEra corner tables must share one wavelength grid")
        logf = sum(w * np.log10(np.maximum(np.asarray(ft.flux, dtype=float), 1e-300))
                   for (ft, _), w in loaded)
        flux = FluxTable(wavelength_nm=lam, flux=10.0**logf,
                         source=f"NewEra bilinear {[(t, g, round(w, 4)) for t, g, w in corners]}")
        # profiles on a common r' = r / R_edge grid
        r_outer_new = max(ld.r_outer for (_, ld), _ in loaded)
        rp_nodes = np.unique(np.concatenate(
            [np.sqrt(np.clip(1.0 - np.asarray(ld.mu, dtype=float)**2, 0.0, 1.0)) * ld.r_outer
             for (_, ld), _ in loaded] + [np.array([0.0, r_outer_new])]))
        rows = np.zeros((lam.size, rp_nodes.size))
        for (_, ld), w in loaded:
            rp = np.sqrt(np.clip(1.0 - np.asarray(ld.mu, dtype=float)**2, 0.0, 1.0)) * ld.r_outer
            order = np.argsort(rp)
            rp_s, inten = rp[order], np.asarray(ld.intensity, dtype=float)[:, order]
            for i in range(lam.size):
                rows[i] += w * np.interp(rp_nodes, rp_s, inten[i], right=0.0)
        mu_new = np.sqrt(np.clip(1.0 - (rp_nodes / r_outer_new)**2, 0.0, 1.0))
        order = np.argsort(mu_new)
        mu_new, rows = mu_new[order], rows[:, order]
        centre = rows[:, -1]
        rows = rows / np.maximum(centre[:, None], 1e-300)
        mu_edge = float(np.sqrt(max(0.0, 1.0 - 1.0 / r_outer_new**2)))
        prof = LDProfile(mu=mu_new, wavelength_nm=lam, intensity=rows,
                         source=flux.source, r_outer=float(r_outer_new), mu_edge=mu_edge)
        return NewEraTables(flux_table=flux, ld_profile=prof, corners=tuple(corners),
                            extrapolated=bool(clamped), clamped=clamped or None)


def with_newera_star(star, grid: NewEraGrid, allow_extrapolation: bool = False):
    """One star with NewEra tables attached (bilinear in T_eff and its
    log_g; rotationally broadened when Star.vsini_kms is set), or the star
    unchanged when the grid does not cover it.  Returns (star, report)."""
    try:
        t = grid.interpolate(star.teff, star.log_g, star.metallicity,
                             allow_extrapolation=allow_extrapolation)
    except ValueError as e:
        return star, (f"no NewEra coverage (T_eff {star.teff:.0f} K, log g "
                      f"{star.log_g:.2f}): kept blackbody + linear law [{e}]")
    ft, ld = t.flux_table, t.ld_profile
    if star.vsini_kms:
        ft, ld = rotational_broaden(ft, ld, star.vsini_kms)
    how = "clamped to the grid edge " + str(t.clamped) if t.extrapolated else "interpolated"
    return with_tables(star, ft, ld), (f"NewEra {how} from {[(c[0], c[1]) for c in t.corners]} "
                                       f"(weights {[round(c[2], 3) for c in t.corners]}); "
                                       f"R_outer/R_edge = {ld.r_outer:.5f}")


def with_newera(system, grid: NewEraGrid, which=("primary", "secondary"),
                allow_extrapolation: bool = False):
    """The system with NewEra tables attached to the stars the grid covers
    (with_newera_star per star).  Returns (system, report) where report
    maps each star's name to what happened; stars outside the grid keep
    their blackbody + linear-law defaults."""
    report = {}
    stars = {"primary": system.primary, "secondary": system.secondary}
    for key in which:
        stars[key], report[stars[key].name] = with_newera_star(stars[key], grid, allow_extrapolation)
    return replace(system, primary=stars["primary"], secondary=stars["secondary"]), report


def attach_from_cli(system, newera_dir, allow_extrapolation: bool = False,
                    verbose: bool = True):
    """CLI helper: attach NewEra tables from a directory (None -> the
    system unchanged) and print the per-star report."""
    if not newera_dir:
        return system
    grid = NewEraGrid.scan(newera_dir)
    system, report = with_newera(system, grid, allow_extrapolation=allow_extrapolation)
    if verbose:
        cov = grid.coverage()
        print(f"NewEra tables from {newera_dir}: {cov['n']} models, T_eff "
              f"{min(cov['teff']):.0f}-{max(cov['teff']):.0f} K, log g "
              f"{min(cov['logg']):.1f}-{max(cov['logg']):.1f}")
        for name, what in report.items():
            print(f"  {name}: {what}")
    return system


def rotational_broaden(flux_table: FluxTable, ld_profile: LDProfile | None,
                       vsini_kms: float, epsilon: float = 0.6):
    """Rotationally broaden a flux table (and, as an azimuthal average,
    each row of the profile) with the Gray (1992) kernel for projected
    equatorial velocity v sin i and linear limb-darkening epsilon, on a
    uniform log-lambda grid.  The photocentre shift inside a line (the
    spectro-astrometric signal of a rotating star) is NOT modeled: the
    broadened profile is the disk-averaged one.  Equivalent widths are
    conserved."""
    if not vsini_kms or vsini_kms <= 0.0:
        return flux_table, ld_profile
    lam = np.asarray(flux_table.wavelength_nm, dtype=float)
    beta = vsini_kms * 1e3 / C_LIGHT
    dlog = float(np.min(np.diff(np.log(lam))))
    dlog = min(dlog, beta / 8.0)
    lg = np.arange(np.log(lam[0]), np.log(lam[-1]) + dlog / 2, dlog)
    half = int(np.ceil(beta / dlog))
    x = np.arange(-half, half + 1) * dlog / beta
    k = np.where(np.abs(x) < 1.0,
                 (2.0 * (1.0 - epsilon) * np.sqrt(np.clip(1.0 - x**2, 0.0, 1.0))
                  + 0.5 * np.pi * epsilon * (1.0 - x**2)), 0.0)
    k = k / k.sum()

    def conv(values):
        v = np.interp(lg, np.log(lam), values)
        pad = np.pad(v, half, mode="edge")
        c = np.convolve(pad, k, mode="valid")
        return np.interp(np.log(lam), lg, c)

    ft = FluxTable(wavelength_nm=lam, flux=conv(np.asarray(flux_table.flux, dtype=float)),
                   source=flux_table.source + f" vsini={vsini_kms:g}")
    ld = ld_profile
    if ld is not None:
        if not np.array_equal(np.asarray(ld.wavelength_nm, dtype=float), lam):
            raise ValueError("rotational_broaden needs flux and profile on one wavelength grid")
        inten = np.asarray(ld.intensity, dtype=float)
        rows = np.stack([conv(inten[:, j]) for j in range(inten.shape[1])], axis=1)
        ld = replace(ld, intensity=rows, source=ld.source + f" vsini={vsini_kms:g}")
    return ft, ld
