"""Stable error codes for the Pi-GRPO API surface."""

from __future__ import annotations

from typing import Any


class PiGrpoError(Exception):
    code: str = "pigrpo.unknown"
    http_status: int = 500
    message: str = "internal error"

    def __init__(self, message: str | None = None, **context: Any) -> None:
        super().__init__(message or self.message)
        if message:
            self.message = message
        self.context = context


class RewardConfigInvalid(PiGrpoError):
    code = "pigrpo.reward_config_invalid"
    http_status = 400


class RunNotFound(PiGrpoError):
    code = "pigrpo.run_not_found"
    http_status = 404


class TrainerCrashed(PiGrpoError):
    code = "pigrpo.trainer_crashed"
    http_status = 500


class UnsafeRange(PiGrpoError):
    code = "pigrpo.unsafe_range"
    http_status = 400
