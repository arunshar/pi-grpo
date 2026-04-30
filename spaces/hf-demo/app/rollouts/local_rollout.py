"""Local Hugging Face Transformers rollout. Used for tests and CPU-only dev."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LocalRollout:
    text: str


class LocalRolloutBackend:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    async def sample(self, prompt: str, *, n: int = 1, max_tokens: int = 64, **kw: Any) -> list[LocalRollout]:
        # Deterministic stub: returns a templated answer so tests can assert without a model.
        return [LocalRollout(text=f"[stub:{self.model_name}] {prompt[:80]}") for _ in range(n)]
