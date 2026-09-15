"""Keplerian sky geometry for the binary, in the center-of-mass frame.

Conventions: the line of nodes lies along the x axis, and the phase
angle psi = 2 pi t / P is the MEAN anomaly measured from periastron.
Kepler's equation E - e sin E = psi is solved by Newton iteration, the
true anomaly nu and separation r follow, and the relative orbit is

    dx = r cos(omega + nu),
    dy = r cos(i) sin(omega + nu),
    dz = r sin(i) sin(omega + nu),

with omega the argument of periastron and dz > 0 meaning the secondary
lies in front of (closer to the observer than) the primary.

For a circular orbit (e = 0, omega = 0) this reduces exactly to the
original convention: psi = 0 places the stars at greatest projected
separation (quadrature) and the eclipses occur at psi = 90 deg (the
secondary transiting the primary -> primary minimum) and 270 deg.  For
eccentric systems psi = 0 is periastron passage and the eclipse phases
depend on omega.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .params import BinarySystem


class SkyPositions(NamedTuple):
    """All angles in mas, in the center-of-mass frame."""
    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    front2: np.ndarray  # True where the secondary is in front of the primary
    rho: np.ndarray     # projected separation
    pa: np.ndarray      # position angle of the separation vector, atan2(dy, dx)


def solve_kepler(mean_anomaly: np.ndarray, e: float,
                 n_iter: int = 25) -> np.ndarray:
    """Eccentric anomaly E from Kepler's equation E - e sin E = M
    (Newton iteration; converges to machine precision for e < 0.95)."""
    M = np.asarray(mean_anomaly, dtype=float)
    if not 0.0 <= e < 1.0:
        raise ValueError(f"eccentricity {e} outside [0, 1)")
    E = M + e * np.sin(M)
    for _ in range(n_iter):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    resid = np.max(np.abs(E - e * np.sin(E) - M)) if M.size else 0.0
    if not resid < 1e-10:
        raise RuntimeError(f"Kepler solver did not converge (e = {e}, "
                           f"max residual {resid:.2e})")
    return E


def sky_positions(psi: np.ndarray, system: BinarySystem) -> SkyPositions:
    psi = np.asarray(psi, dtype=float)
    a = system.angular_semimajor_mas
    inc = np.radians(system.inclination_deg)
    e = system.eccentricity
    w = np.radians(system.arg_periastron_deg)

    if e == 0.0:
        r = a
        u = w + psi  # nu = M = psi for a circular orbit
    else:
        E = solve_kepler(psi, e)
        nu = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0),
                              np.sqrt(1.0 - e) * np.cos(E / 2.0))
        r = a * (1.0 - e * np.cos(E))
        u = w + nu

    dx = r * np.cos(u)
    dy = r * np.cos(inc) * np.sin(u)
    dz = r * np.sin(inc) * np.sin(u)

    m1 = system.primary.mass_msun
    m2 = system.secondary.mass_msun
    f1 = m2 / (m1 + m2)
    f2 = m1 / (m1 + m2)

    return SkyPositions(
        x1=-f1 * dx, y1=-f1 * dy,
        x2=f2 * dx, y2=f2 * dy,
        front2=dz > 0,
        rho=np.hypot(dx, dy),
        pa=np.arctan2(dy, dx),
    )
