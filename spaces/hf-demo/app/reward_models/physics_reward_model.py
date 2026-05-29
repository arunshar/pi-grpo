"""Wrapper that loads a `PhysicsReward` from a yaml config and exposes it as a callable."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from app.components.kinematic_bicycle import SkbmConfig
from app.components.physics_reward import (
    EmpiricalEnvelope,
    PhysicsReward,
    RewardWeights,
)
from app.errors import RewardConfigInvalid


def load(yaml_path: str | Path) -> PhysicsReward:
    p = Path(yaml_path)
    if not p.exists():
        raise RewardConfigInvalid(f"reward config not found: {p}")
    raw: dict[str, Any] = yaml.safe_load(p.read_text())
    try:
        cfg = SkbmConfig(
            wheelbase_m=float(raw["skbm"]["wheelbase_m"]),
            v_max_mps=float(raw["skbm"]["v_max_mps"]),
            v_min_mps=float(raw["skbm"].get("v_min_mps", 0.0)),
            a_max_mps2=float(raw["skbm"]["a_max_mps2"]),
            delta_max_rad=float(raw["skbm"]["delta_max_rad"]),
        )
        weights = RewardWeights(**raw["weights"])
        envelope = EmpiricalEnvelope(**raw["envelope"])
    except KeyError as exc:
        raise RewardConfigInvalid(f"missing key: {exc}") from exc
    return PhysicsReward(cfg, weights=weights, envelope=envelope)
