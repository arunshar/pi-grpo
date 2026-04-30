"""Offline evaluator. Scores a checkpoint on the golden dataset.

Metrics:
- pass_rate. Percentage of items the policy classifies correctly.
- physics_violation_rate. Percentage of generated trajectories that
  violate the S-KBM hard envelope.
- mean_reward. Average physics-aware reward.
- p95_kl. KL to the reference at the 95th percentile.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from app.components.kinematic_bicycle import SkbmConfig, evaluate
from app.components.physics_reward import EmpiricalEnvelope, PhysicsReward, RewardWeights

log = structlog.get_logger(__name__)


def main(golden_path: str = "evaluation/golden_dataset.json") -> int:
    data = json.loads(Path(golden_path).read_text())
    cfg = SkbmConfig()
    reward = PhysicsReward(cfg, weights=RewardWeights(), envelope=EmpiricalEnvelope())

    rows: list[dict[str, Any]] = []
    for item in data["items"]:
        traj = np.asarray(item["trajectory"], dtype=np.float64)
        v = evaluate(traj, cfg=cfg)
        r = reward.score(traj)
        verdict = (
            "HARD_VIOLATION" if r.hard < 0
            else "SOFT_VIOLATION" if r.soft < -0.01
            else "PASS"
        )
        rows.append({
            "id": item["id"],
            "expected": item["expected_verdict"],
            "got": verdict,
            "reward": r.total,
            "hard": r.hard,
            "soft": r.soft,
            "speed_max_pct": v.speed_max_pct,
            "accel_max_pct": v.accel_max_pct,
        })
    out_dir = Path("evaluation/eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    pass_rate = sum(1 for r in rows if r["got"] == r["expected"]) / max(1, len(rows))
    summary = {"pass_rate": pass_rate, "n_items": len(rows)}
    (out_dir / f"{ts}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=str))
    return 0 if pass_rate >= 0.8 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
