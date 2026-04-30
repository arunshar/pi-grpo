"""HITL feedback ingestion. Mirrors the GeoTrace-Agent shape so the
two systems can share preference data via a shared schema.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeedbackRow:
    trace_id: str
    label: str
    notes: str | None


def write(rows: list[FeedbackRow]) -> int:
    # writes to runs/<run_id>/feedback.jsonl in production
    return len(rows)
