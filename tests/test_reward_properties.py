"""Property-based tests for the physics reward.

The hard term must be unbounded above (in the sense that growing the
violation grows the penalty without limit). Clean trajectories must
score zero on the hard floor regardless of length. These properties
should hold for any reasonable bounds and any reasonable trajectory.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.components.kinematic_bicycle import SkbmConfig, evaluate
from app.components.physics_reward import (
    EmpiricalEnvelope,
    PhysicsReward,
    RewardWeights,
)


_v_max = 12.86  # 25 kt
_cfg = SkbmConfig(v_max_mps=_v_max)
_reward = PhysicsReward(
    _cfg,
    weights=RewardWeights(hard=5.0, soft=1.0, data=1.0, pref=1.0),
    envelope=EmpiricalEnvelope(),
)


def _clean_trajectory(n: int, v: float) -> np.ndarray:
    s = np.zeros((n, 4))
    s[:, 0] = np.arange(n) * v
    s[:, 3] = v
    return s


def _speeding_trajectory(n: int, v: float) -> np.ndarray:
    s = np.zeros((n, 4))
    s[:, 0] = np.arange(n) * v
    s[:, 3] = v  # all steps over v_max
    return s


# ----------------------------------------------------------------- tests


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.integers(min_value=4, max_value=200), st.floats(min_value=0.5, max_value=12.0))
def test_clean_trajectory_has_zero_hard_penalty(n: int, v: float) -> None:
    traj = _clean_trajectory(n, v)
    rb = _reward.score(traj)
    assert rb.hard == 0.0


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.integers(min_value=4, max_value=200), st.floats(min_value=15.0, max_value=80.0))
def test_speeding_trajectory_has_negative_hard(n: int, v: float) -> None:
    traj = _speeding_trajectory(n, v)
    rb = _reward.score(traj)
    assert rb.hard < 0.0


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.floats(min_value=15.0, max_value=20.0),
    st.floats(min_value=25.0, max_value=80.0),
)
def test_more_speeding_means_more_negative_hard(v_low: float, v_high: float) -> None:
    """Penalty must be monotone non-decreasing in violation magnitude."""

    if v_high <= v_low:
        return
    a = _speeding_trajectory(20, v_low)
    b = _speeding_trajectory(20, v_high)
    rb_a = _reward.score(a)
    rb_b = _reward.score(b)
    assert rb_b.hard <= rb_a.hard + 1e-9


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.floats(min_value=20.0, max_value=80.0), st.floats(min_value=-20.0, max_value=20.0))
def test_hard_dominates_pref_at_extreme_pref(v_speed: float, pref_logit: float) -> None:
    """Even with a maxed-out preference signal, a sustained hard violation
    must keep the total reward negative."""

    traj = _speeding_trajectory(30, v_speed)
    rb = _reward.score(traj, pref_logit=pref_logit)
    assert rb.hard < 0
    # If hard is strongly negative, the total must be too
    if rb.hard < -1.0:
        assert rb.total < 0


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.integers(min_value=3, max_value=50), st.floats(min_value=0.0, max_value=12.0))
def test_evaluate_violation_fractions_are_valid(n: int, v: float) -> None:
    traj = _clean_trajectory(n, v)
    info = evaluate(traj, cfg=_cfg)
    for frac in (info.speed_max_pct, info.accel_max_pct, info.steer_max_pct):
        assert 0.0 <= frac <= 1.0
        assert math.isfinite(frac)
    assert info.curvature_p95 >= 0
    assert info.jerk_p95 >= 0
