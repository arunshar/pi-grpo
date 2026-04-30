"""Strip PII fields from any object that crosses the security boundary."""

from __future__ import annotations

from typing import Any

_REDACT = {"email", "phone", "owner", "operator"}


def scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if k in _REDACT else scrub(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value
