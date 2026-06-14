"""TrainerAgent. Picks a trainer and drives its main loop.

This module is the launcher for actual training. The Pi-GRPO API delegates here
over JSON-RPC so a long run can outlive the API process. `train` runs a real
policy-gradient loop (policy + physics reward + one of the three trainers) and
returns measured metrics; `train_dummy` is retained for the API smoke path that
only needs a typed result without touching torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from app.policy.driver import SMOKE, TrainConfig, TrainResult, train as _train

log = structlog.get_logger(__name__)


@dataclass
class TrainResultLite:
    final_step: int
    final_metrics: dict[str, float]


class TrainerAgent:
    def train(
        self,
        algo: str,
        cfg: TrainConfig = SMOKE,
        on_step=None,
    ) -> TrainResult:
        """Run a real training loop for `algo` in {"grpo","ppo","dpo"}."""
        log.info("train_started", algo=algo, steps=cfg.steps)
        result = _train(algo, cfg, on_step)
        log.info(
            "train_finished",
            algo=algo,
            steps=result.final_step,
            reward_start=result.reward_start,
            reward_end=result.reward_end,
        )
        return result

    def train_dummy(self, total_steps: int) -> TrainResultLite:
        # Sufficient for unit tests and the API smoke run that avoid torch.
        return TrainResultLite(final_step=total_steps, final_metrics={"loss": 0.1, "kl": 4.0})
