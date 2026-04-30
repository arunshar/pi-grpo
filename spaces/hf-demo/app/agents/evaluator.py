"""EvaluatorAgent. Runs golden-dataset evaluation on a checkpoint."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalResult:
    pass_rate: float


class EvaluatorAgent:
    def evaluate_dummy(self) -> EvalResult:
        return EvalResult(pass_rate=1.0)
