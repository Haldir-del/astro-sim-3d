"""Tests for app/orbital.py — analytic utilities and the USSA76 model.

USSA76 reference points come from the official Part 4 tables
(NOAA/NASA/USAF, U.S. Standard Atmosphere 1976, NTRS 19770009539).
"""
import numpy as np
import pytest

from app.orbital import (
    EARTH_J2,
    EARTH_MU,
    WGS84_A,
    WGS84_OMEGA_EARTH,
    circular_velocity,
    escape_velocity,
    j2_apsidal_rate_rad_s,
    j2_nodal_rate_rad_s,
    launch_state,
    mean_motion,
    orbital_elements,
    specific_energy,
    sun_sync_inclination_deg,
    ussa76_density,
)

# ----------------------------------------------------------------------
# Constants (WGS-84 defining parameters)
# ----------------------------------------------------------------------


def test_wgs84_constants():
    assert WGS84_A == 6378137.0
    assert EARTH_MU == pytest.approx(3.986004418e14)
    assert WGS84_OMEGA_EARTH == pytest.approx(7.292115e-5)


# ----------------------------------------------------------------------
# Keplerian utilities
# ----------------------------------------------------------------------


def test_circular_velocity_matches_vis_viva():
    r = 6371000.0 + 400_000.0
    v = circular_velocity(r)
    # vis-viva for a circular orbit: v^2 = mu/r
    assert v**2 == pytest.approx(EARTH_MU / r, rel=1e-12)
    assert v == pytest.approx(7672.0, abs=3.0)  # ISS-like


def test_escape_velocity_is_sqrt2_times_circular():
    r = 6371000.0 + 400_000.0
    assert escape_velocity(r) == pytest.approx(
        np.sqrt(2.0) * circular_velocity(r), rel=1e-14)


def test_orbital_elements_circular_equatorial():
    r = 7000e3
    state = np.array([r, 0.0, 0.0, 0.0, circular_velocity(r), 0.0])
    el = orbital_elements(state, EARTH_MU)
    assert el.eccentricity == pytest.approx(0.0, abs=1e-12)
    assert el.inclination_deg == pytest.approx(0.0, abs=1e-9)
    assert el.semi_major_axis == pytest.approx(r, rel=1e-12)
    assert el.period == pytest.approx(2 * np.pi * np.sqrt(r**3 / EARTH_MU),
                                      rel=1e-12)


def test_launch_state_recovers_elements():
    alt_km, vel, inc = 550.0, 7590.0, 53.0
    s = launch_state(alt_km, vel, inc, raan_deg=40.0)
    el = orbital_elements(s, EARTH_MU)
    assert np.linalg.norm(s[0:3]) == pytest.approx(6371e3 + alt_km * 1e3,
                                                   rel=1e-12)
    assert el.inclination_deg == pytest.approx(inc, abs=1e-9)
    assert el.raan_deg == pytest.approx(40.0, abs=1e-6)
    # Energy consistency: eps < 0 for a bound orbit.
    assert el.specific_energy == pytest.approx(specific_energy(s, EARTH_MU))


# ----------------------------------------------------------------------
# Analytic J2 secular rates (GDC Orbit Primer formulas)
# ----------------------------------------------------------------------


def test_j2_nodal_rate_sign_and_magnitude():
    # Prograde LEO orbits regress: dOmega/dt < 0.
    a = (6371e3 + 550e3)
    rate = j2_nodal_rate_rad_s(a, 0.0, 53.0)
    assert rate < 0.0
    # Typical magnitude for ISS-class orbit ~ -5 deg/day.
    deg_per_day = np.degrees(rate) * 86400.0
    assert deg_per_day == pytest.approx(-5.0, abs=1.0)


def test_j2_polar_orbit_no_nodal_precession():
    rate = j2_nodal_rate_rad_s(7000e3, 0.001, 90.0)
    assert rate == pytest.approx(0.0, abs=1e-18)


def test_j2_critical_inclination_zero_apsidal_drift():
    # cos²i = 1/5 -> i ≈ 63.4349° freezes apsidal precession.
    i_crit = np.degrees(np.arccos(np.sqrt(1.0 / 5.0)))
    rate = j2_apsidal_rate_rad_s(7000e3, 0.01, i_crit)
    assert rate == pytest.approx(0.0, abs=1e-16)


def test_sun_sync_inclination_reasonable():
    # SSO at ~700 km altitude is classically ~98.2 deg.
    a = 6371e3 + 700e3
    inc = sun_sync_inclination_deg(a, 0.0)
    assert 96.0 < inc < 100.0
    # And it must reproduce exactly the required nodal rate.
    rate = j2_nodal_rate_rad_s(a, 0.0, inc)
    required = np.deg2rad(360.0 / 365.2422) / 86400.0
    assert rate == pytest.approx(required, rel=1e-9)


# ----------------------------------------------------------------------
# U.S. Standard Atmosphere 1976
# ----------------------------------------------------------------------

@pytest.mark.parametrize("z_m,rho_ref", [
    (0.0, 1.2250),               # sea level
    (5000.0, 7.3643e-1),         # Part 4 table @5 km
    (11000.0, 3.6392e-1),        # tropopause base
    (20000.0, 8.8910e-2),
    (32000.0, 1.3270e-2),
    (47000.0, 1.4275e-3),
    (71000.0, 6.4200e-5),
    (86000.0, 6.9579e-6),        # top of barometric layers
    (100000.0, 5.6044e-7),       # tabulated anchors above 86 km
    (200000.0, 2.5407e-10),
    (400000.0, 2.8028e-12),
    (1000000.0, 3.5611e-15),
])
def test_ussa76_density_table_points(z_m, rho_ref):
    assert ussa76_density(z_m) == pytest.approx(rho_ref, rel=2e-3)


def test_ussa76_monotonic_below_86km():
    zs = np.linspace(0.0, 86000.0, 500)
    rhos = [ussa76_density(float(z)) for z in zs]
    assert all(b <= a + 1e-12 for a, b in zip(rhos, rhos[1:]))


def test_ussa76_continuous_across_86km_boundary():
    lo = ussa76_density(85999.0)
    hi = ussa76_density(86001.0)
    assert hi == pytest.approx(lo, rel=2e-2)
