"""Linear limb-darkening law and its analytic visibility (for tests).

I(mu)/I(1) = 1 - u (1 - mu), mu = cos(gamma) = sqrt(1 - (r/R)^2).

The analytic visibility of a single linearly limb-darkened disk
(Hanbury Brown et al. 1974; Rai, Basak & Saha 2021 eqs. 17-19) is

    V(x) = [ (1-u) J1(x)/x + u sqrt(pi/2) J_{3/2}(x) / x^{3/2} ]
           / [ (1-u)/2 + u/3 ],

with x = pi * theta_d * B / lambda (theta_d the angular diameter).
"""

from __future__ import annotations

import numpy as np


def ld_profile(mu: np.ndarray, u: float) -> np.ndarray:
    return 1.0 - u * (1.0 - mu)


def disk_flux_factor(u: float) -> float:
    """Integral of the LD profile over the disk, relative to a uniform disk:
    2 int_0^1 [1 - u(1-mu)] mu dmu / ... = (1 - u/3); flux = pi R^2 I(1) (1 - u/3)."""
    return 1.0 - u / 3.0


def visibility_ld_disk(x: np.ndarray, u: float) -> np.ndarray:
    """Analytic V(x) for a linearly limb-darkened disk (test reference only).

    x = pi * theta_diameter * B / lambda.  Uses scipy for Bessel functions.
    """
    from scipy.special import j1, jv

    x = np.asarray(x, dtype=float)
    xs = np.where(x == 0.0, 1.0, x)  # avoid 0/0; patched below
    term_ud = (1.0 - u) * j1(xs) / xs
    term_ld = u * np.sqrt(np.pi / 2.0) * jv(1.5, xs) / xs**1.5
    norm = (1.0 - u) / 2.0 + u / 3.0
    v = (term_ud + term_ld) / norm
    return np.where(x == 0.0, 1.0, v)


def visibility_profile(x, mu, intensity, n_r: int = 2000):
    """Numeric visibility of a circularly symmetric disk with a tabulated
    centre-to-limb profile I(mu)/I(1) (n_mu,) at x = pi theta_d B/lambda
    (any shape): V(x) = int I(r) J0(x r) r dr / int I(r) r dr with
    r = sqrt(1 - mu^2) the fractional radius.  Reference for the
    tabulated-LD renderer."""
    from scipy.special import j0

    x = np.asarray(x, dtype=float)
    r = np.linspace(0.0, 1.0, n_r)
    mu_r = np.sqrt(np.clip(1.0 - r**2, 0.0, 1.0))
    prof = np.interp(mu_r, np.asarray(mu), np.asarray(intensity)) * r
    norm = np.trapezoid(prof, r)
    v = np.trapezoid(prof * j0(x[..., None] * r), r, axis=-1)
    return v / norm


def star_disk_visibility(star, x, wavelength_nm):
    """V(x) of one star's disk at x = pi theta_d B/lambda, shaped
    (n_lambda, K) for wavelength_nm (n_lambda,) and x (n_lambda, K):
    the analytic linear-law series, or the vectorized Gauss-Legendre
    Hankel transform of the star's tabulated profile
    (visibility_profile_batch; cached on a fine x grid for large K).
    theta_d must be the DRAWN diameter (BinarySystem.drawn_radius_mas):
    the profile's mu = 0 is the table's outer boundary."""
    x = np.asarray(x, dtype=float)
    lam = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
    if star.ld_profile is None:
        u = np.atleast_1d(star.ld_coeff(lam))[:, None]
        return visibility_ld_disk(x, u)
    prof = star.ld_profile
    if x.size > 4096:
        # many pupil/baseline points per channel: interpolate a cached
        # V(x) table per channel (< 1e-5 error)
        return DISK_CACHE.lookup(prof, lam, x)
    return visibility_profile_batch(x, prof.mu, prof.rows(lam))


# ---------------------------------------------------------------------------
# Vectorized Hankel transform of tabulated profiles
# ---------------------------------------------------------------------------
def hankel_nodes(mu):
    """Two-point Gauss-Legendre nodes and weights on every segment of the
    native mu grid (ascending, mu[0] >= 0), plus the interpolation
    matrix that evaluates a piecewise-linear profile at the nodes:
    (nodes (n_q,), weights (n_q,), matrix (n_q, n_mu)) with
    I(nodes) = rows @ matrix.T.  With r dr = -mu dmu the disk integral
    int I(r) J0(x r) r dr becomes sum_q w_q I(mu_q) mu_q J0(x r_q),
    r_q = sqrt(1 - mu_q^2); exact for a linear profile times a
    polynomial of degree <= 2 per segment."""
    mu = np.asarray(mu, dtype=float)
    if mu[0] > 0.0:                       # the limb: I = 0 below mu_min
        mu = np.concatenate([[0.0], mu])
        pad = True
    else:
        pad = False
    m0, m1 = mu[:-1], mu[1:]
    g = 1.0 / np.sqrt(3.0)
    nodes = np.concatenate([0.5 * (m0 + m1) - 0.5 * (m1 - m0) * g,
                            0.5 * (m0 + m1) + 0.5 * (m1 - m0) * g])
    weights = np.concatenate([0.5 * (m1 - m0), 0.5 * (m1 - m0)])
    order = np.argsort(nodes)
    nodes, weights = nodes[order], weights[order]
    k = np.clip(np.searchsorted(mu, nodes) - 1, 0, mu.size - 2)
    t = (nodes - mu[k]) / (mu[k + 1] - mu[k])
    n_mu = mu.size - (1 if pad else 0)
    mat = np.zeros((nodes.size, n_mu))
    for q in range(nodes.size):
        j = k[q]
        if pad:
            j -= 1            # column index in the unpadded rows; -1 = the padded zero
            if j >= 0:
                mat[q, j] += 1.0 - t[q]
            mat[q, j + 1] += t[q]
        else:
            mat[q, j] += 1.0 - t[q]
            mat[q, j + 1] += t[q]
    return nodes, weights, mat


def visibility_profile_batch(x, mu, rows, chunk_elems: float = 2e7) -> np.ndarray:
    """V(x) for rows (n_lam, n_mu) of I(mu)/I(1) on the native mu grid,
    at x (n_lam, K) (or (K,), broadcast to every row), by the
    Gauss-Legendre rule of hankel_nodes; J0 evaluated on (chunk, K, n_q)
    blocks of at most chunk_elems elements.  Agrees with
    visibility_profile to ~1e-5 and is vectorized over channels."""
    from scipy.special import j0

    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = np.broadcast_to(x, (rows.shape[0], x.size))
    nodes, weights, mat = hankel_nodes(mu)
    iq = rows @ mat.T                                  # (n_lam, n_q)
    wq = iq * (weights * nodes)[None, :]               # (n_lam, n_q)
    norm = wq.sum(axis=1)                              # int I mu dmu
    rq = np.sqrt(np.clip(1.0 - nodes**2, 0.0, 1.0))
    n_lam, K = x.shape
    out = np.empty((n_lam, K))
    step = max(1, int(chunk_elems // max(1, K * nodes.size)))
    for a in range(0, n_lam, step):
        b = min(a + step, n_lam)
        j = j0(x[a:b, :, None] * rq[None, None, :])    # (chunk, K, n_q)
        out[a:b] = np.einsum("lkq,lq->lk", j, wq[a:b]) / norm[a:b, None]
    return out


class DiskVisibilityCache:
    """V(x) of a tabulated profile on a uniform x grid per channel,
    computed once and linearly interpolated afterwards (error < 1e-4 at
    dx = 0.02; the build costs ~10 s per 1000 channels and 16 units of
    x).  Keyed by the profile object, the channel wavelengths and the x
    range rounded up to a multiple of 8, so a uv track or a baseline
    scan (x_max varying call by call) reuses one table; the feasibility
    drivers evaluate the same channels at thousands of pupil points per
    call."""
    DX = 0.02
    X_ROUND = 8.0

    def __init__(self):
        self._store = {}

    def _key(self, profile, lam, x_max):
        lam = np.asarray(lam, dtype=float)
        return (id(profile), lam.size, float(lam[0]), float(lam[-1]),
                hash(lam.tobytes()), int(np.ceil(x_max / self.X_ROUND)) * int(self.X_ROUND))

    def lookup(self, profile, lam, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x_max = float(np.max(x)) if x.size else 0.0
        key = self._key(profile, lam, x_max)
        entry = self._store.get(key)
        if entry is None:
            grid_max = key[-1]
            xg = np.arange(0.0, grid_max + self.DX, self.DX)
            rows = profile.rows(lam)
            vg = visibility_profile_batch(xg, profile.mu, rows)
            entry = (xg, vg)
            if len(self._store) > 64:
                self._store.clear()
            self._store[key] = entry
        xg, vg = entry
        pos = np.clip(x / self.DX, 0.0, xg.size - 1.0001)
        k = pos.astype(int)
        t = pos - k
        rows_idx = np.arange(vg.shape[0])[:, None]
        return (1.0 - t) * vg[rows_idx, k] + t * vg[rows_idx, k + 1]


DISK_CACHE = DiskVisibilityCache()
