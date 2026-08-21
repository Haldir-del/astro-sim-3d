"""Pure-numpy orbital mechanics utilities.

All functions operate on state vectors of the form [x, y, z, vx, vy, vz]
in a Cartesian inertial frame centered on the primary body (meters, m/s).
This module has no dependency on the native engine and is fully unit-testable.

References
----------
[1] NGA.STND.0036_1.0.0_WGS84, "Department of Defense World Geodetic System
    1984", National Geospatial-Intelligence Agency, 2014.
    https://earth-info.nga.mil/index.php?action=wgs84&dir=wgs84
[2] U.S. Standard Atmosphere, 1976, NOAA/NASA/USAF, Washington D.C., 1976.
    https://ntrs.nasa.gov/citations/19770009539
[3] NASA/GSFC, "GDC Orbit Primer" (J2 secular nodal regression and apsidal
    precession rates).
    https://science.nasa.gov/wp-content/uploads/2023/05/GDC_OrbitPrimer.pdf
"""
from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# WGS-84 defining parameters [1].
WGS84_A = 6378137.0                 # semi-major axis (m)
WGS84_INV_F = 298.257223563         # inverse flattening (-)
WGS84_F = 1.0 / WGS84_INV_F         # flattening (-)
WGS84_GM = 3.986004418e14           # geocentric gravitational constant (m^3/s^2)
WGS84_OMEGA_EARTH = 7.292115e-5     # nominal mean angular velocity (rad/s)

# Earth zonal harmonic J2 (EGM96 / WGS-84 EGM value).
EARTH_J2 = 1.08262668e-3            # dimensionless

EARTH_MU = WGS84_GM                 # alias kept for backwards compatibility
EARTH_RADIUS_M = WGS84_A            # equatorial radius (m)
EARTH_RADIUS_MEAN_M = 6371000.0     # mean radius (m), MSL reference for USSA76


@dataclass(frozen=True)
class OrbitalElements:
    """Classical orbital elements computed from a Cartesian state."""

    specific_energy: float          # epsilon = v^2/2 - mu/r   (J/kg)
    semi_major_axis: Optional[float]  # a (m); None if hyperbolic/parabolic
    eccentricity: float             # e (>= 0)
    inclination_deg: float          # i (0..180 deg)
    raan_deg: float                 # right ascension of ascending node (deg)
    arg_periapsis_deg: float        # argument of periapsis (deg)
    periapsis_radius: float         # r_p (m); valid also for e >= 1
    apoapsis_radius: Optional[float]  # r_a (m); None if e >= 1
    period: Optional[float]         # T (s); None if e >= 1

    @property
    def is_elliptic(self) -> bool:
        return self.eccentricity < 1.0


def speed(state: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(state, dtype=float)[3:6]))


def radius(state: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(state, dtype=float)[0:3]))


def specific_energy(state: np.ndarray, mu: float) -> float:
    s = np.asarray(state, dtype=float)
    return 0.5 * float(s[3:6] @ s[3:6]) - mu / np.linalg.norm(s[0:3])


def orbital_elements(state: np.ndarray, mu: float) -> OrbitalElements:
    """Compute classical orbital elements from a Cartesian state vector."""
    s = np.asarray(state, dtype=float)
    r_vec = s[0:3]
    v_vec = s[3:6]

    r = np.linalg.norm(r_vec)
    v2 = float(v_vec @ v_vec)

    eps = 0.5 * v2 - mu / r
    h_vec = np.cross(r_vec, v_vec)
    h_norm = np.linalg.norm(h_vec)

    n_vec = np.cross(np.array([0.0, 0.0, 1.0]), h_vec)  # node line
    n_norm = np.linalg.norm(n_vec)

    # Eccentricity vector: e = (v x h)/mu - r_hat
    e_vec = (np.cross(v_vec, h_vec) / mu) - r_vec / r
    ecc = float(np.linalg.norm(e_vec))

    inc = np.degrees(np.arccos(np.clip(h_vec[2] / h_norm, -1.0, 1.0)))

    # RAAN
    if n_norm > 1e-12:
        raan = np.degrees(np.arctan2(n_vec[1], n_vec[0])) % 360.0
    else:
        raan = 0.0  # undefined for equatorial orbits

    # Argument of periapsis
    if n_norm > 1e-12 and ecc > 1e-12:
        arg_p = np.degrees(
            np.arccos(np.clip(float(n_vec @ e_vec) / (n_norm * ecc), -1.0, 1.0))
        )
        if e_vec[2] < 0.0:
            arg_p = 360.0 - arg_p
    else:
        arg_p = 0.0  # undefined for circular/equatorial orbits

    semi_major = None if eps >= 0.0 else -mu / (2.0 * eps)
    semi_latus = h_norm * h_norm / mu
    periapsis = semi_latus / (1.0 + ecc)

    apoapsis = None
    period = None
    if ecc < 1.0 and semi_major is not None:
        apoapsis = semi_major * (1.0 + ecc)
        period = 2.0 * pi * sqrt(semi_major**3 / mu)

    return OrbitalElements(
        specific_energy=float(eps),
        semi_major_axis=None if semi_major is None else float(semi_major),
        eccentricity=ecc,
        inclination_deg=float(inc),
        raan_deg=float(raan),
        arg_periapsis_deg=float(arg_p),
        periapsis_radius=float(periapsis),
        apoapsis_radius=None if apoapsis is None else float(apoapsis),
        period=None if period is None else float(period),
    )


def launch_state(altitude_km: float, velocity_ms: float,
                 inclination_deg: float, raan_deg: float = 0.0) -> np.ndarray:
    """Build an initial state at the ascending node.

    The spacecraft starts on the node line at the given altitude with purely
    tangential velocity; both vectors are rotated about the X axis by the
    inclination angle (orbital plane tilt) and about Z by the RAAN.
    """
    r = EARTH_RADIUS_MEAN_M + altitude_km * 1000.0
    i = np.deg2rad(inclination_deg)
    raan = np.deg2rad(raan_deg)

    pos = np.array([r, 0.0, 0.0])
    vel = np.array([0.0, velocity_ms, 0.0])

    rot_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(i), -np.sin(i)],
        [0.0, np.sin(i), np.cos(i)],
    ])
    rot_z = np.array([
        [np.cos(raan), -np.sin(raan), 0.0],
        [np.sin(raan), np.cos(raan), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rot = rot_z @ rot_x
    return np.concatenate([rot @ pos, rot @ vel])


def circular_velocity(radius_m: float, mu: float = EARTH_MU) -> float:
    return float(sqrt(mu / radius_m))


def escape_velocity(radius_m: float, mu: float = EARTH_MU) -> float:
    return float(sqrt(2.0 * mu / radius_m))


# ---------------------------------------------------------------------------
# Analytic J2 secular rates (first order in J2) — NASA/GSFC GDC Orbit Primer [3]
# ---------------------------------------------------------------------------

def mean_motion(semi_major_axis_m: float, mu: float = EARTH_MU) -> float:
    """Keplerian mean motion n = sqrt(mu/a^3) (rad/s)."""
    return sqrt(mu / semi_major_axis_m**3)


def j2_nodal_rate_rad_s(semi_major_axis_m: float, ecc: float, inc_deg: float,
                        mu: float = EARTH_MU, req_m: float = WGS84_A,
                        j2: float = EARTH_J2) -> float:
    """Secular nodal regression rate d(Omega)/dt (rad/s).

    dOmega/dt = -(3/2) n J2 (Re/p)^2 cos i,   p = a (1-e^2)
    """
    p = semi_major_axis_m * (1.0 - ecc * ecc)
    n = mean_motion(semi_major_axis_m, mu)
    i = np.deg2rad(inc_deg)
    return -1.5 * n * j2 * (req_m / p) ** 2 * np.cos(i)


def j2_apsidal_rate_rad_s(semi_major_axis_m: float, ecc: float, inc_deg: float,
                          mu: float = EARTH_MU, req_m: float = WGS84_A,
                          j2: float = EARTH_J2) -> float:
    """Secular apsidal precession rate d(omega)/dt (rad/s).

    domega/dt = (3/4) n J2 (Re/p)^2 (4 - 5 sin^2 i),   p = a (1-e^2)
    """
    p = semi_major_axis_m * (1.0 - ecc * ecc)
    n = mean_motion(semi_major_axis_m, mu)
    i = np.deg2rad(inc_deg)
    return 0.75 * n * j2 * (req_m / p) ** 2 * (4.0 - 5.0 * np.sin(i) ** 2)


def sun_sync_inclination_deg(semi_major_axis_m: float, ecc: float,
                             mu: float = EARTH_MU, req_m: float = WGS84_A,
                             j2: float = EARTH_J2) -> float:
    """Inclination that makes the orbit sun-synchronous.

    Required nodal rate equals the Sun's apparent motion, 0.9856 deg/day
    (360 deg per 365.2422 days) [3]. Returns the inclination in degrees.
    """
    omega_req = np.deg2rad(360.0 / 365.2422) / 86400.0  # rad/s
    p = semi_major_axis_m * (1.0 - ecc * ecc)
    n = mean_motion(semi_major_axis_m, mu)
    cos_i = -omega_req / (1.5 * n * j2 * (req_m / p) ** 2)
    cos_i = float(np.clip(cos_i, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_i)))


# ---------------------------------------------------------------------------
# U.S. Standard Atmosphere 1976 density (Python mirror of the C++ model) [2]
# ---------------------------------------------------------------------------

_G0 = 9.80665
_RSPEC = 287.05287
_USSA_RE = 6356766.0

_LAYER_H = (0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0)
_LAYER_T = (288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65)
_LAYER_L = (-6.5e-3, 0.0, 1.0e-3, 2.8e-3, 0.0, -2.8e-3, -2.0e-3)
_LAYER_P = (101325.0, 22632.06, 5474.889, 868.0187, 110.9063, 66.93887,
            3.9564200)
_TOP_H = 84852.0

_HIGH_ALT_RHO = (
    (86.0e3, 6.9579e-6), (90.0e3, 3.4400e-6), (95.0e3, 1.3873e-6),
    (100.0e3, 5.6044e-7), (110.0e3, 9.6734e-8), (120.0e3, 2.2199e-8),
    (130.0e3, 8.1494e-9), (150.0e3, 2.0752e-9), (180.0e3, 5.1940e-10),
    (200.0e3, 2.5407e-10), (250.0e3, 6.0706e-11), (300.0e3, 1.9159e-11),
    (350.0e3, 7.0011e-12), (400.0e3, 2.8028e-12), (450.0e3, 1.1843e-12),
    (500.0e3, 5.2148e-13), (550.0e3, 2.3832e-13), (600.0e3, 1.1367e-13),
    (650.0e3, 5.7114e-14), (700.0e3, 3.0698e-14), (750.0e3, 1.7900e-14),
    (800.0e3, 1.1358e-14), (850.0e3, 7.8223e-15), (900.0e3, 5.7587e-15),
    (950.0e3, 4.4525e-15), (1000.0e3, 3.5611e-15),
)


def ussa76_density(altitude_m: float) -> float:
    """Density (kg/m^3) at geometric altitude above mean sea level (m).

    Exact barometric layer equations up to 86 km; log-linear interpolation of
    the USSA76 Part 4 tables above. Mirrors core/orbital_engine.cpp.
    """
    z = float(altitude_m)
    if z < 0.0:
        z = 0.0

    if z <= 86000.0:
        H = _USSA_RE * z / (_USSA_RE + z)
        i = len(_LAYER_H) - 1
        while i > 0 and H < _LAYER_H[i]:
            i -= 1
        Tb, Lb, pb = _LAYER_T[i], _LAYER_L[i], _LAYER_P[i]
        dH = H - _LAYER_H[i]
        if Lb != 0.0:
            T = Tb + Lb * dH
            p = pb * (T / Tb) ** (-_G0 / (Lb * _RSPEC))
        else:
            T = Tb
            p = pb * np.exp(-_G0 * dH / (_RSPEC * Tb))
        return float(p / (_RSPEC * T))

    anchors = _HIGH_ALT_RHO
    if z >= anchors[-1][0]:
        return anchors[-1][1]
    j = 0
    while j < len(anchors) - 1 and anchors[j + 1][0] < z:
        j += 1
    z0, r0 = anchors[j]
    z1, r1 = anchors[j + 1]
    w = (z - z0) / (z1 - z0)
    return float(np.exp(np.log(r0) + w * (np.log(r1) - np.log(r0))))
