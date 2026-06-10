"""Circular-orbit sky geometry for the binary, in the center-of-mass frame.

Conventions: the line of nodes lies along the x axis; the orbital phase angle
psi = 2 pi t / P is measured so that psi = 0 places the stars at greatest
projected separation (quadrature).  The relative orbit is

    dx = a cos(psi),  dy = a cos(i) sin(psi),  dz = a sin(i) sin(psi),

with dz > 0 meaning the secondary lies in front of (closer to the observer
than) the primary, so the eclipses occur at psi = 90 deg (secondary transits
the hotter primary -> primary minimum) and psi = 270 deg.
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


def sky_positions(psi: np.ndarray, system: BinarySystem) -> SkyPositions:
    psi = np.asarray(psi, dtype=float)
    a = system.angular_semimajor_mas
    inc = np.radians(system.inclination_deg)

    dx = a * np.cos(psi)
    dy = a * np.cos(inc) * np.sin(psi)
    dz = a * np.sin(inc) * np.sin(psi)

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
