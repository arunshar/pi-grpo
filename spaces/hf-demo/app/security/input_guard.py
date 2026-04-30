"""Reject inference prompts that try to override the system prompt."""

from __future__ import annotations

import re

from app.errors import PiGrpoError

_BANNED = re.compile(r"ignore (the )?(above|previous) instructions", re.I)


def check(prompt: str) -> None:
    if _BANNED.search(prompt):
        raise PiGrpoError("input_guard.banned_phrase")
    if len(prompt) > 4_000:
        raise PiGrpoError("input_guard.too_long")
