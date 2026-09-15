"""Keplerian sky geometry for the binary, in the center-of-mass frame.

The phase angle psi = 2 pi t / P is the MEAN anomaly measured from
periastron.  Kepler's equation E - e sin E = psi is solved by Newton
iteration, the true anomaly nu and separation r follow, and in the
orbit's node frame (p along the line of nodes, q perpendicular) the
relative position of the secondary is

    p  = r cos(omega + nu),
    q  = r cos(i) sin(omega + nu),
    dz = r sin(i) sin(omega + nu),

with omega the argument of periastron and dz > 0 meaning the secondary
lies in front of (closer to the observer than) the primary.  The node
at u = omega + nu = 0 (+p) is therefore the DESCENDING node (the
secondary starts approaching after it) and the ascending node, from
which the position angle Omega is measured, lies along -p.

Sky frame: x = East, y = North.  When the system carries node_pa_deg
(Omega, N through E) the node frame is placed on the sky by

    E = -p sin(Omega) - q cos(Omega),
    N = -p cos(Omega) + q sin(Omega),

which puts the ascending node at position angle Omega and, for i < 90
deg, makes the position angle increase with time (direct motion;
i > 90 deg is retrograde), the standard visual-binary convention.  This
map is a reflection of the legacy frame (x = p, y = q), which no Omega
reproduces exactly; systems without node_pa_deg keep the legacy frame.
Projected separation rho, front/behind and hence the eclipse geometry
and lightcurves do not depend on Omega.

For a circular orbit (e = 0, omega = 0) psi = 0 places the stars at
greatest projected separation (quadrature) and the eclipses occur at
psi = 90 deg (the secondary transiting the primary -> primary minimum)
and 270 deg.  For eccentric systems psi = 0 is periastron passage and
the eclipse phases depend on omega.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .params import BinarySystem


class SkyPositions(NamedTuple):
    """All angles in mas, in the center-of-mass frame; x = East, y = North."""
    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    front2: np.ndarray  # True where the secondary is in front of the primary
    rho: np.ndarray     # projected separation
    pa: np.ndarray      # math angle of the separation vector, atan2(dy, dx)

    @property
    def position_angle_deg(self):
        """Astronomical position angle of the secondary relative to the
        primary, North through East, in [0, 360)."""
        return np.degrees(np.arctan2(self.x2 - self.x1, self.y2 - self.y1)) % 360.0

    def take(self, k: int) -> "SkyPositions":
        """The scalar epoch k of an array-valued SkyPositions."""
        return SkyPositions(*(np.asarray(v)[k] for v in self))


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

    p = r * np.cos(u)
    q = r * np.cos(inc) * np.sin(u)
    dz = r * np.sin(inc) * np.sin(u)
    if system.node_pa_deg is None:      # legacy frame
        dx, dy = p, q
    else:
        node = np.radians(system.node_pa_deg)
        dx = -p * np.sin(node) - q * np.cos(node)   # East
        dy = -p * np.cos(node) + q * np.sin(node)   # North

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
