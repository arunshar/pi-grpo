"""Versioned prompt registry for the reasoning policy."""

from __future__ import annotations

from dataclasses import dataclass
from string import Template
from typing import Any

from app.prompts import templates as T


@dataclass(frozen=True)
class _Prompt:
    name: str
    template: str

    def render(self, **kwargs: Any) -> str:
        return Template(self.template).safe_substitute(**{k: str(v) for k, v in kwargs.items()})


_REGISTRY = {
    "reasoning.v1": _Prompt("reasoning.v1", T.REASONING_V1),
    "reasoning.v2": _Prompt("reasoning.v2", T.REASONING_V2),
}


def get_prompt(name: str) -> _Prompt:
    return _REGISTRY[name]
