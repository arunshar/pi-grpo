"""Numerical invariants of the S-KBM."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.components.kinematic_bicycle import SkbmConfig, rollout, evaluate, step


def test_constant_velocity_straight_line() -> None:
    cfg = SkbmConfig()
    s0 = np.array([[0.0, 0.0, 0.0, 5.0]])  # heading east at 5 m/s
    controls = np.zeros((10, 2))
    states = rollout(s0, controls, cfg=cfg, h=1.0)
    assert math.isclose(states[-1, 0], 50.0, abs_tol=1e-6)
    assert math.isclose(states[-1, 1], 0.0, abs_tol=1e-6)
    assert math.isclose(states[-1, 3], 5.0, abs_tol=1e-6)


def test_evaluate_flags_speed_violation() -> None:
    cfg = SkbmConfig(v_max_mps=10.0)
    states = np.zeros((5, 4))
    states[:, 3] = [0, 5, 10, 15, 20]  # accelerating beyond v_max
    v = evaluate(states, cfg=cfg)
    assert v.speed_max_pct > 0.0


def test_step_pure_function() -> None:
    cfg = SkbmConfig()
    s = np.array([0.0, 0.0, 0.0, 5.0])
    s2 = step(s, np.array([0.0, 0.0]), cfg=cfg, h=1.0)
    assert s2[0] == 5.0
