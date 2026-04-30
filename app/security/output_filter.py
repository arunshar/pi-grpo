"""Output filter for the inference path."""

from __future__ import annotations

import re


_EMAIL_RE = re.compile(r"\b[\w.+-]+@\w+(?:\.\w+)+\b")


def scrub(text: str) -> str:
    return _EMAIL_RE.sub("[email]", text)
