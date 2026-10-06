"""Observing geometry: sites, hour angle, and the projection of ground
baselines onto the (u, v) plane.

Station positions are local East-North-Up (E, N, U) in metres.  For a
source at hour angle H and declination delta seen from latitude phi
the projected baseline components conjugate to the sky's East (u) and
North (v) directions are (Thompson, Moran & Swenson, eq. 4.1 via the
ENU -> XYZ rotation)

    u =  E cos H - N sin phi sin H + U cos phi sin H
    v =  E sin delta sin H + N (sin phi sin delta cos H + cos phi cos delta)
                           - U (cos phi sin delta cos H - sin phi cos delta)
    w = -E cos delta sin H + N (cos phi sin delta - sin phi cos delta cos H)
                           + U (sin phi sin delta + cos phi cos delta cos H),

in metres (divide by lambda for cycles per radian).  For a source at
the zenith (H = 0, delta = phi) this is the identity (u, v, w) =
(E, N, U), the legacy flat-layout assumption.  The (u, v) point of a
baseline moves along an ellipse over the night; a binary's fringe on a
baseline B drifts by (B(t2) - B(t1)) . rho / lambda cycles between two
epochs, which limits how long a correlation block can be integrated
coherently (fringe_drift_cycles).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOUR = np.pi / 12.0   # radians of hour angle per hour


@dataclass(frozen=True)
class Site:
    name: str
    latitude_deg: float
    longitude_deg: float = 0.0   # East positive
    elevation_m: float = 0.0

    @property
    def latitude_rad(self) -> float:
        return np.radians(self.latitude_deg)


# Sites (Maunakea, Paranal, Teide, FLWO, ORM, Calern) are hbtsim/configs/sites/*.json.


def enu_to_uvw(enu_m, hour_angle_rad, dec_rad, lat_rad) -> np.ndarray:
    """Project ENU vectors (..., 3) [m] to (u, v, w) (..., 3) [m]."""
    e, n, up = np.moveaxis(np.asarray(enu_m, dtype=float), -1, 0)
    H, d, phi = hour_angle_rad, dec_rad, lat_rad
    sH, cH = np.sin(H), np.cos(H)
    sd, cd = np.sin(d), np.cos(d)
    sp, cp = np.sin(phi), np.cos(phi)
    u = e * cH - n * sp * sH + up * cp * sH
    v = e * sd * sH + n * (sp * sd * cH + cp * cd) - up * (cp * sd * cH - sp * cd)
    w = -e * cd * sH + n * (cp * sd - sp * cd * cH) + up * (sp * sd + cp * cd * cH)
    return np.stack([u, v, w], axis=-1)


def enu_to_uv(enu_m, hour_angle_rad, dec_rad, lat_rad) -> np.ndarray:
    """The (u, v) part of enu_to_uvw, (..., 2) [m]."""
    return enu_to_uvw(enu_m, hour_angle_rad, dec_rad, lat_rad)[..., :2]


def altitude_rad(hour_angle_rad, dec_rad, lat_rad):
    """Source altitude above the horizon."""
    return np.arcsin(np.sin(lat_rad) * np.sin(dec_rad)
                     + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_angle_rad))


def hour_angle_window(dec_deg: float, lat_deg: float,
                      min_alt_deg: float = 30.0) -> tuple:
    """(H_min, H_max) in hours over which the source is above min_alt;
    (0, 0) if it never is, (-12, 12) if it always is."""
    d, phi, a = (np.radians(x) for x in (dec_deg, lat_deg, min_alt_deg))
    c = (np.sin(a) - np.sin(phi) * np.sin(d)) / (np.cos(phi) * np.cos(d))
    if c >= 1.0:
        return (0.0, 0.0)
    if c <= -1.0:
        return (-12.0, 12.0)
    h = np.degrees(np.arccos(c)) / 15.0
    return (-h, h)


def hour_angle_blocks(h_start: float, h_end: float, block_minutes: float) -> np.ndarray:
    """Block mid-points [h] covering [h_start, h_end] in equal blocks of
    (at most) block_minutes."""
    span = h_end - h_start
    if span <= 0.0:
        return np.zeros(0)
    n = max(1, int(np.ceil(span * 60.0 / block_minutes)))
    edges = np.linspace(h_start, h_end, n + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def fringe_drift_cycles(bvec_start_m, bvec_end_m, separation_rad,
                        wavelength_m: float) -> np.ndarray:
    """Fringe cycles a binary of projected separation vector
    `separation_rad` (2,) [rad, (E, N)] sweeps on baseline(s) (..., 2)
    between two epochs: |(B_end - B_start) . rho| / lambda."""
    db = np.asarray(bvec_end_m, dtype=float) - np.asarray(bvec_start_m, dtype=float)
    return np.abs(db @ np.asarray(separation_rad, dtype=float)) / wavelength_m


def drift_loss(cycles) -> np.ndarray:
    """Contrast retained by a fringe averaged over a linear drift of the
    given number of cycles: sinc(pi c)."""
    c = np.asarray(cycles, dtype=float)
    return np.sinc(c)  # numpy sinc(x) = sin(pi x)/(pi x)
