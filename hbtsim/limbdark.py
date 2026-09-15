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
    the analytic linear-law series, or the numeric Hankel transform of
    the star's tabulated profile (chunked over wavelength)."""
    x = np.asarray(x, dtype=float)
    lam = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
    if star.ld_profile is None:
        u = np.atleast_1d(star.ld_coeff(lam))[:, None]
        return visibility_ld_disk(x, u)
    out = np.empty(x.shape)
    mu = star.ld_profile.mu
    step = 64
    for k in range(0, lam.size, step):
        rows = star.ld_profile.rows(lam[k:k + step])
        for j, row in enumerate(rows):
            out[k + j] = visibility_profile(x[k + j], mu, row)
    return out
