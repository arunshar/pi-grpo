"""W&B adapter. Streams trainer metrics."""

from __future__ import annotations

import os
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class WandbAdapter:
    def __init__(self, project: str, run_name: str | None = None) -> None:
        self.project = project
        self.run_name = run_name
        self._wb: Any | None = None
        if os.environ.get("WANDB_API_KEY"):
            try:
                import wandb  # type: ignore[import-untyped]

                self._wb = wandb
                wandb.init(project=project, name=run_name)
            except Exception as exc:  # pragma: no cover
                log.warning("wandb_init_failed", err=str(exc))

    def log(self, payload: dict[str, float], *, step: int) -> None:
        if self._wb is None:
            log.info("metric", step=step, **payload)
            return
        self._wb.log(payload, step=step)
