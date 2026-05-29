"""Per-run cost ledger (GPU-hours, tokens out via vLLM)."""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)


def record(run_id: str, *, gpu_hours: float, tokens: int, cost_usd: float) -> None:
    log.info("cost", run_id=run_id, gpu_hours=gpu_hours, tokens=tokens, cost_usd=cost_usd)
