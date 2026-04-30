"""Single-axle Kinematic Bicycle Model (S-KBM).

State: (x, y, theta, v) in (m, m, rad, m/s).
Control: (a, delta) in (m/s^2, rad).

Discrete-time update with step h:

    x'     = x + v cos(theta) h
    y'     = y + v sin(theta) h
    theta' = theta + (v / L) tan(delta) h
    v'     = v + a h

L is the wheelbase (default 5 m for a small vessel proxy; 2.7 m for a
sedan; 0.5 m for a UAV body length). The model is the same prior used
by Pi-DPM (Sharma et al., GeoAnomalies '25).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SkbmConfig:
    wheelbase_m: float = 5.0
    v_max_mps: float = 12.86  # 25 kts
    v_min_mps: float = 0.0
    a_max_mps2: float = 0.5   # vessels rarely accel hard
    delta_max_rad: float = math.radians(35.0)


@dataclass(frozen=True)
class TrajectoryViolations:
    speed_max_pct: float        # fraction of steps with v > v_max
    accel_max_pct: float        # fraction of steps with |a| > a_max
    steer_max_pct: float        # fraction of steps with |delta| > delta_max
    curvature_p95: float        # 95th percentile curvature
    jerk_p95: float             # 95th percentile jerk magnitude


def step(state: np.ndarray, control: np.ndarray, *, cfg: SkbmConfig, h: float = 1.0) -> np.ndarray:
    """One Euler step. `state` is (4,), `control` is (2,)."""

    x, y, th, v = state
    a, delta = control
    nx = x + v * math.cos(th) * h
    ny = y + v * math.sin(th) * h
    nth = th + (v / cfg.wheelbase_m) * math.tan(delta) * h
    nv = v + a * h
    return np.array([nx, ny, nth, nv], dtype=np.float64)


def rollout(states: np.ndarray, controls: np.ndarray, *, cfg: SkbmConfig, h: float = 1.0) -> np.ndarray:
    """Roll out from a single initial state under a control sequence."""

    out = np.empty((controls.shape[0] + 1, 4))
    out[0] = states[0]
    for i in range(controls.shape[0]):
        out[i + 1] = step(out[i], controls[i], cfg=cfg, h=h)
    return out


def evaluate(states: np.ndarray, *, cfg: SkbmConfig, h: float = 1.0) -> TrajectoryViolations:
    """Audit a state sequence against S-KBM bounds.

    `states` is (T, 4). Returns the fraction of steps that violate
    each bound plus the 95th percentiles of jerk and curvature.
    """

    if states.shape[0] < 3:
        return TrajectoryViolations(0.0, 0.0, 0.0, 0.0, 0.0)
    v = states[:, 3]
    th = states[:, 2]
    a = np.diff(v) / h
    th_dot = np.diff(th) / h
    jerk = np.diff(a) / h
    # implied steering: from theta_dot = v/L tan(delta)  => delta = atan(theta_dot * L / v)
    safe_v = np.where(np.abs(v[:-1]) > 1e-3, v[:-1], 1e-3)
    delta = np.arctan(th_dot * cfg.wheelbase_m / safe_v)
    speed_max = float(np.mean(v > cfg.v_max_mps + 1e-6))
    accel_max = float(np.mean(np.abs(a) > cfg.a_max_mps2 + 1e-6))
    steer_max = float(np.mean(np.abs(delta) > cfg.delta_max_rad + 1e-6))
    curvature_p95 = float(np.percentile(np.abs(th_dot), 95))
    jerk_p95 = float(np.percentile(np.abs(jerk), 95)) if jerk.size else 0.0
    return TrajectoryViolations(speed_max, accel_max, steer_max, curvature_p95, jerk_p95)
