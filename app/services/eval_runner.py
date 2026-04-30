"""Run the golden-dataset evaluation against a checkpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalReport:
    pass_rate: float
    mean_reward: float
    physics_violation_rate: float
    n_items: int


def run(report_path: str = "evaluation/eval_results/report.json") -> EvalReport:
    """Stub for the offline evaluator. Production fills this in."""

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    rep = EvalReport(pass_rate=0.0, mean_reward=0.0, physics_violation_rate=0.0, n_items=0)
    Path(report_path).write_text(json.dumps(rep.__dict__))
    return rep
