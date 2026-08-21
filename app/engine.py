"""ctypes binding for the native orbital engine (orbital_engine.dll).

Declares explicit argument/return types for every foreign call, owns all
buffer allocation, and converts raw C arrays into numpy views. The GUI and
tests should only interact with :class:`Forces` and :func:`simulate`.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from app.orbital import (
    EARTH_J2,
    EARTH_MU,
    EARTH_RADIUS_MEAN_M,
    WGS84_A,
    WGS84_OMEGA_EARTH,
)

STATE_DIM = 6
PARAM_LEN = 12

STATUS_COMPLETED = 0
STATUS_IMPACT = 1
STATUS_ESCAPE = 2
STATUS_BUFFER_FULL = 3

_STATUS_NAMES: Dict[int, str] = {
    STATUS_COMPLETED: "completed",
    STATUS_IMPACT: "impact",
    STATUS_ESCAPE: "escape",
    STATUS_BUFFER_FULL: "buffer_full",
}

_LIB_NAME = "orbital_engine.dll"


class EngineNotAvailable(RuntimeError):
    """Raised when the native engine library cannot be loaded."""


@dataclass(frozen=True)
class Forces:
    """Physical model configuration passed to the native engine.

    Defaults describe Earth (WGS-84) with J2 enabled and drag disabled.
    """

    mu: float = EARTH_MU                  # gravitational parameter (m^3/s^2)
    body_radius_m: float = EARTH_RADIUS_MEAN_M  # impact sphere radius (m)
    escape_radius_m: float = 0.0          # escape sphere radius (m); 0 = off
    j2: float = EARTH_J2                  # zonal harmonic (-)
    req_m: float = WGS84_A                # equatorial radius for J2 (m)
    cd: float = 2.2                       # drag coefficient (-)
    area_m2: float = 20.0                 # frontal area (m^2)
    mass_kg: float = 1000.0               # spacecraft mass (kg)
    omega_earth: float = WGS84_OMEGA_EARTH  # atmosphere rotation (rad/s)
    enable_j2: bool = True
    enable_drag: bool = False

    def to_params(self) -> np.ndarray:
        params = np.zeros(PARAM_LEN, dtype=np.float64)
        params[0] = self.mu
        params[1] = self.body_radius_m
        params[2] = self.escape_radius_m
        params[3] = self.j2 if self.enable_j2 else 0.0
        params[4] = self.req_m
        if self.enable_drag:
            params[5] = self.cd
            params[6] = self.area_m2
            params[7] = self.mass_kg
        params[8] = self.omega_earth
        return params


@dataclass(frozen=True)
class SimulationResult:
    """Trajectory history produced by one engine run."""

    t: np.ndarray        # (n,) elapsed time per sample (s)
    state: np.ndarray    # (n, 6) [x,y,z,vx,vy,vz] (m, m/s)
    status: str          # completed | impact | escape | buffer_full
    acc_pert: np.ndarray  # (n,) |a_J2 + a_drag| per sample (m/s^2)

    @property
    def positions(self) -> np.ndarray:
        return self.state[:, 0:3]

    @property
    def velocities(self) -> np.ndarray:
        return self.state[:, 3:6]

    @property
    def radii(self) -> np.ndarray:
        return np.linalg.norm(self.positions, axis=1)

    @property
    def speeds(self) -> np.ndarray:
        return np.linalg.norm(self.velocities, axis=1)


_lib: Optional[ctypes.CDLL] = None


def _candidate_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    return [root / _LIB_NAME, Path.cwd() / _LIB_NAME]


def _load_library() -> ctypes.CDLL:
    errors = []
    for path in _candidate_paths():
        try:
            lib = ctypes.CDLL(str(path))
            break
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    else:
        raise EngineNotAvailable(
            "No se pudo cargar el motor nativo. Intentos:\n" + "\n".join(errors)
        )

    lib.simulate_trajectory.argtypes = [
        ctypes.POINTER(ctypes.c_double),   # state0[6]
        ctypes.c_double,                   # duration (s)
        ctypes.c_double,                   # dt_max (s)
        ctypes.c_double,                   # rel_tol
        ctypes.POINTER(ctypes.c_double),   # params[12]
        ctypes.c_int,                      # max_steps
        ctypes.POINTER(ctypes.c_double),   # out_t[max_steps]
        ctypes.POINTER(ctypes.c_double),   # out_state[max_steps*6]
        ctypes.POINTER(ctypes.c_double),   # out_acc[max_steps] (nullable)
        ctypes.POINTER(ctypes.c_int),      # out_status
    ]
    lib.simulate_trajectory.restype = ctypes.c_int
    return lib


def get_engine() -> ctypes.CDLL:
    """Load (once) and return the native engine handle."""
    global _lib
    if _lib is None:
        _lib = _load_library()
    return _lib


def simulate(
    initial_state: np.ndarray,
    duration_s: float,
    *,
    dt_max: float = 1.0,
    rel_tol: float = 1e-10,
    forces: Forces | None = None,
    max_steps: int = 500_000,
) -> SimulationResult:
    """Propagate a trajectory with the adaptive RKF45 native core.

    Parameters mirror the C ABI; see core/orbital_engine.cpp for details.
    Raises ValueError on malformed inputs and RuntimeError on engine errors.
    """
    f = forces if forces is not None else Forces()
    state0 = np.ascontiguousarray(initial_state, dtype=np.float64).reshape(-1)
    if state0.size != STATE_DIM:
        raise ValueError(f"initial_state must have {STATE_DIM} components")
    if not np.all(np.isfinite(state0)):
        raise ValueError("initial_state contains non-finite values")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if dt_max <= 0.0 or rel_tol <= 0.0:
        raise ValueError("dt_max and rel_tol must be positive")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    params = np.ascontiguousarray(f.to_params(), dtype=np.float64)
    out_t = np.empty(max_steps, dtype=np.float64)
    out_state = np.empty((max_steps, STATE_DIM), dtype=np.float64)
    out_acc = np.empty(max_steps, dtype=np.float64)
    status = ctypes.c_int(-1)

    n = get_engine().simulate_trajectory(
        state0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(float(duration_s)),
        ctypes.c_double(float(dt_max)),
        ctypes.c_double(float(rel_tol)),
        params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(max_steps)),
        out_t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_acc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(status),
    )
    if n < 0:
        raise RuntimeError(f"native engine rejected the call (code {n})")

    return SimulationResult(
        t=out_t[:n].copy(),
        state=out_state[:n].copy(),
        status=_STATUS_NAMES.get(status.value, f"unknown({status.value})"),
        acc_pert=out_acc[:n].copy(),
    )
