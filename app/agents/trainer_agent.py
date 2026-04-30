"""TrainerAgent. Picks a trainer and drives its main loop.

This module is the launcher for actual training. The Pi-GRPO API
delegates here over JSON-RPC so a long run can outlive the API
process. For local CPU-only test runs we expose a tiny synchronous
entry point that the run orchestrator can call directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclass
class TrainResult:
    final_step: int
    final_metrics: dict[str, float]


class TrainerAgent:
    def train_dummy(self, total_steps: int) -> TrainResult:
        # Sufficient for unit tests and the API smoke run.
        return TrainResult(final_step=total_steps, final_metrics={"loss": 0.1, "kl": 4.0})
