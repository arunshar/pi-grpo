"""Pi-GRPO settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="PG_", extra="ignore")

    env: Literal["dev", "staging", "prod"] = "dev"
    version: str = "0.1.0"

    pg_dsn: str = "postgresql+asyncpg://pigrpo:pigrpo@postgres:5432/pigrpo"
    redis_url: str = "redis://redis:6380/0"

    vllm_url: str = "http://vllm:8000/v1"
    base_model: str = "Qwen/Qwen2-7B-Instruct"
    primary_inference_model: str = "Qwen/Qwen2-7B-Instruct"

    pidpm_checkpoint: str | None = None

    wandb_project: str = "pi-grpo"
    wandb_api_key: SecretStr | None = None
    otel_endpoint: str = "http://otel-collector:4318"

    runs_root: str = "runs"
    preferences_root: str = "data/preferences"

    # default training caps
    max_train_steps: int = 5_000
    max_run_seconds: float = 60 * 60 * 6.0   # 6 h


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
