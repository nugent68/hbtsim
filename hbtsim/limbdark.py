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
