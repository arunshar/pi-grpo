"""DataCuratorAgent. Builds versioned preference datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog

from app.components.preference_builder import (
    PreferencePair,
    from_hitl_jsonl,
    write_jsonl,
)

log = structlog.get_logger(__name__)


@dataclass
class CurationResult:
    n_pairs: int
    out_path: str


class DataCuratorAgent:
    def build(self, *, hitl_jsonl: str | None, out_path: str) -> CurationResult:
        pairs: list[PreferencePair] = []
        if hitl_jsonl and Path(hitl_jsonl).exists():
            pairs.extend(from_hitl_jsonl(hitl_jsonl))
        n = write_jsonl(out_path, pairs)
        return CurationResult(n_pairs=n, out_path=out_path)
