"""Motion-primitive codebook: token ids <-> S-KBM controls.

The policy emits discrete tokens; each token indexes a control pair (a, delta).
A rollout of tokens is decoded into a control sequence and integrated through the
single-axle Kinematic Bicycle Model into a state trajectory (x, y, theta, v) that
the `PhysicsReward` scores. The grid deliberately spans **beyond** the S-KBM
bounds (factor `span`), so some tokens are kinematically infeasible: that gives
the reward real variance and a feasible region for the policy to learn toward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.components.kinematic_bicycle import SkbmConfig, rollout


@dataclass(frozen=True)
class CodebookConfig:
    n_accel: int = 5      # accel bins
    n_steer: int = 5      # steer bins
    span: float = 1.6     # grid reaches span * bound, so edges violate
    init_speed: float = 5.0
    h: float = 1.0


class MotionCodebook:
    """Discrete (a, delta) grid over the S-KBM control space."""

    def __init__(self, skbm: SkbmConfig, cfg: CodebookConfig = CodebookConfig()) -> None:
        self.skbm = skbm
        self.cfg = cfg
        a_lim = cfg.span * skbm.a_max_mps2
        d_lim = cfg.span * skbm.delta_max_rad
        accels = np.linspace(-a_lim, a_lim, cfg.n_accel)
        steers = np.linspace(-d_lim, d_lim, cfg.n_steer)
        # row-major: token = ai * n_steer + di
        grid = np.array([[a, d] for a in accels for d in steers], dtype=np.float64)
        self._controls = grid                       # (V, 2)

    @property
    def vocab_size(self) -> int:
        return int(self._controls.shape[0])

    def controls(self, token_ids: np.ndarray) -> np.ndarray:
        """(T,) token ids -> (T, 2) (a, delta) controls."""
        ids = np.clip(np.asarray(token_ids, dtype=np.int64), 0, self.vocab_size - 1)
        return self._controls[ids]

    def tokens_to_states(self, token_ids: np.ndarray) -> np.ndarray:
        """(T,) token ids -> (T+1, 4) state trajectory from a fixed initial state."""
        controls = self.controls(token_ids)
        init = np.array([0.0, 0.0, 0.0, self.cfg.init_speed], dtype=np.float64)
        return rollout(init[None, :], controls, cfg=self.skbm, h=self.cfg.h)


__all__ = ["CodebookConfig", "MotionCodebook"]
