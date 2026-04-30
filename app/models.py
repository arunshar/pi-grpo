"""Pi-GRPO typed schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Mut(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthOut(_Frozen):
    status: Literal["ok", "degraded"] = "ok"
    version: str


class RunIn(_Mut):
    algo: Literal["ppo", "dpo", "grpo"]
    base_model: str = "Qwen/Qwen2-7B-Instruct"
    preferences_path: str | None = None         # required for dpo
    reward_config: str | None = None            # path to physics_reward yaml
    total_steps: int = Field(2_000, ge=10, le=50_000)
    seed: int = 42
    extra: dict[str, Any] = {}


class RunOut(_Frozen):
    run_id: str
    submitted_at: datetime


class RunStatus(_Frozen):
    run_id: str
    state: Literal["pending", "running", "succeeded", "failed", "cancelled"]
    step: int
    metrics: dict[str, float] = {}
    error: str | None = None


class InferIn(_Mut):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    return_logprobs: bool = False


class InferOut(_Frozen):
    text: str
    tokens: int
    cost_usd: float = 0.0


class RewardConfigYaml(_Frozen):
    """Schema mirror of `configs/physics_reward.yaml`. We validate on load."""

    weights_hard: float = 5.0
    weights_soft: float = 1.0
    weights_data: float = 1.0
    weights_pref: float = 1.0
    skbm_wheelbase_m: float = 5.0
    skbm_v_max_mps: float = 12.86
    skbm_a_max_mps2: float = 0.5
    envelope_curvature_p95: float = 0.05
    envelope_jerk_p95: float = 0.5
