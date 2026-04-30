"""Physics-aware reward for trajectory and reasoning policies.

R(trajectory)  =   w_hard * R_hard
                 + w_soft * R_soft
                 + w_data * R_data
                 + w_pref * R_pref

R_hard penalizes any S-KBM bound violation. Implemented as

    R_hard = - sum_i max(0, c_i)         with c_i in {speed/V_max - 1,
                                                       |a|/A_max - 1,
                                                       |delta|/D_max - 1}

so a single bad step dominates the run and makes the policy avoid hard
violations. R_soft penalizes the 95th-percentile curvature and jerk
relative to the empirical envelope (calibrated on Porto / Harbin /
MarineCadastre AIS). R_data is a calibration of Pi-DPM reconstruction-
error tail probability. R_pref is the preference classifier output
(used only when a preference reward model is configured).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.components.kinematic_bicycle import SkbmConfig, TrajectoryViolations, evaluate


@dataclass(frozen=True)
class RewardWeights:
    hard: float = 5.0
    soft: float = 1.0
    data: float = 1.0
    pref: float = 1.0


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    hard: float
    soft: float
    data: float
    pref: float

    def to_panel(self) -> dict[str, float]:
        return {
            "reward/total": self.total,
            "reward/hard": self.hard,
            "reward/soft": self.soft,
            "reward/data": self.data,
            "reward/pref": self.pref,
        }


@dataclass(frozen=True)
class EmpiricalEnvelope:
    """95th-percentile bounds for soft penalties, fit on the training corpus."""

    curvature_p95: float = 0.05
    jerk_p95: float = 0.5


class PhysicsReward:
    def __init__(
        self,
        cfg: SkbmConfig,
        weights: RewardWeights = RewardWeights(),
        envelope: EmpiricalEnvelope = EmpiricalEnvelope(),
    ) -> None:
        self.cfg = cfg
        self.weights = weights
        self.envelope = envelope

    def score(
        self,
        states: np.ndarray,
        *,
        pi_dpm_log_prob: Optional[float] = None,
        pref_logit: Optional[float] = None,
    ) -> RewardBreakdown:
        v = evaluate(states, cfg=self.cfg)
        r_hard = -self._hard_penalty(v, states)
        r_soft = -self._soft_penalty(v)
        r_data = float(pi_dpm_log_prob) if pi_dpm_log_prob is not None else 0.0
        r_pref = float(pref_logit) if pref_logit is not None else 0.0
        total = (
            self.weights.hard * r_hard
            + self.weights.soft * r_soft
            + self.weights.data * r_data
            + self.weights.pref * r_pref
        )
        return RewardBreakdown(total=total, hard=r_hard, soft=r_soft, data=r_data, pref=r_pref)

    # ------------------------------------------------------------ hard

    def _hard_penalty(self, v: TrajectoryViolations, states: np.ndarray) -> float:
        """Sum of relative excess across all bounds. Unbounded above."""

        # speed
        v_seq = states[:, 3]
        speed_excess = np.maximum(0.0, np.abs(v_seq) / max(self.cfg.v_max_mps, 1e-6) - 1.0)
        # accel
        a_seq = np.diff(v_seq)
        accel_excess = np.maximum(0.0, np.abs(a_seq) / max(self.cfg.a_max_mps2, 1e-6) - 1.0)
        # steer (proxied by curvature normalized by max-curvature ~ tan(delta_max)/L)
        kappa_max = abs(np.tan(self.cfg.delta_max_rad)) / max(self.cfg.wheelbase_m, 1e-6)
        if states.shape[0] > 1:
            th_dot = np.diff(states[:, 2])
            kappa_excess = np.maximum(0.0, np.abs(th_dot) / max(kappa_max, 1e-6) - 1.0)
        else:
            kappa_excess = np.zeros(1)
        return float(speed_excess.sum() + accel_excess.sum() + kappa_excess.sum())

    # ------------------------------------------------------------ soft

    def _soft_penalty(self, v: TrajectoryViolations) -> float:
        return float(
            max(0.0, v.curvature_p95 - self.envelope.curvature_p95)
            + max(0.0, v.jerk_p95 - self.envelope.jerk_p95)
            + 0.5 * v.speed_max_pct
            + 0.5 * v.accel_max_pct
            + 0.5 * v.steer_max_pct
        )
