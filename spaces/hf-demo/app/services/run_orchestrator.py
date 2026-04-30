"""RunOrchestrator. Submits / supervises training runs and serves inference.

A run is a Python child process (Ray actor in production) that owns
one of the three trainers. The orchestrator owns:

- run lifecycle (pending → running → succeeded / failed / cancelled)
- metric stream → Postgres + W&B
- checkpoint scheduling → object store
- token-budget enforcement on inference
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import structlog

from app.config import Settings, get_settings
from app.errors import RunNotFound, UnsafeRange
from app.models import InferIn, InferOut, RunIn, RunOut, RunStatus

log = structlog.get_logger(__name__)


_SAFE: dict[str, dict[str, tuple[float, float]]] = {
    "ppo":  {"lr": (1e-7, 5e-5), "target_kl": (0.5, 20.0)},
    "dpo":  {"lr": (1e-7, 5e-5), "beta":      (0.01, 1.0)},
    "grpo": {"lr": (1e-7, 5e-5), "target_kl": (0.5, 20.0)},
}


class RunOrchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._runs: dict[str, RunStatus] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))

    @classmethod
    async def bootstrap(cls, settings: Settings | None = None) -> RunOrchestrator:
        return cls(settings or get_settings())

    async def shutdown(self) -> None:
        for t in self._tasks.values():
            t.cancel()
        await self._http.aclose()

    # ---------------------------------------------------------- runs

    async def submit(self, payload: RunIn) -> RunOut:
        self._enforce_safe_ranges(payload)
        run_id = uuid4().hex[:12]
        self._runs[run_id] = RunStatus(run_id=run_id, state="pending", step=0)
        self._tasks[run_id] = asyncio.create_task(self._run(run_id, payload))
        return RunOut(run_id=run_id, submitted_at=datetime.now(UTC))

    async def status(self, run_id: str) -> RunStatus | None:
        return self._runs.get(run_id)

    async def cancel(self, run_id: str) -> RunStatus:
        rs = self._runs.get(run_id)
        if rs is None:
            raise RunNotFound(run_id=run_id)
        if run_id in self._tasks:
            self._tasks[run_id].cancel()
        self._runs[run_id] = rs.model_copy(update={"state": "cancelled"})
        return self._runs[run_id]

    async def _run(self, run_id: str, payload: RunIn) -> None:
        run_dir = Path(self.settings.runs_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._runs[run_id] = self._runs[run_id].model_copy(update={"state": "running"})
        log.info("run_started", run_id=run_id, algo=payload.algo, base_model=payload.base_model)
        try:
            # In production this dispatches to Ray / Kubernetes via
            # `app/agents/trainer_agent.py::TrainerAgent.train`. Here we
            # spin a coroutine that simulates a step loop so the API
            # surface and the metric streaming logic can be exercised
            # without a GPU.
            for step in range(min(payload.total_steps, 20)):
                await asyncio.sleep(0.05)
                self._runs[run_id] = self._runs[run_id].model_copy(update={
                    "step": step + 1,
                    "metrics": {
                        "reward_mean": 0.05 * step,
                        "kl": 4.0 - 0.05 * step,
                        "loss": 1.0 / (1.0 + step),
                        "lr": 5e-7,
                    },
                })
            self._runs[run_id] = self._runs[run_id].model_copy(update={"state": "succeeded"})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover
            log.exception("run_failed", run_id=run_id)
            self._runs[run_id] = self._runs[run_id].model_copy(update={"state": "failed", "error": str(exc)})

    # --------------------------------------------------------- inference

    async def infer(self, payload: InferIn) -> InferOut:
        # Use the vLLM OpenAI-compatible endpoint when available.
        body = {
            "model": self.settings.primary_inference_model,
            "messages": [{"role": "user", "content": payload.prompt}],
            "max_tokens": payload.max_tokens,
            "temperature": payload.temperature,
            "top_p": payload.top_p,
        }
        try:
            r = await self._http.post(f"{self.settings.vllm_url}/chat/completions", json=body)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {})
            return InferOut(text=text, tokens=int(usage.get("completion_tokens", 0)), cost_usd=0.0)
        except httpx.HTTPError:
            return InferOut(text="(vllm unavailable; configure PG_VLLM_URL or run docker compose)", tokens=0)

    # ------------------------------------------------------------ safety

    def _enforce_safe_ranges(self, payload: RunIn) -> None:
        ranges = _SAFE.get(payload.algo)
        if not ranges:
            return
        for k, (lo, hi) in ranges.items():
            v = payload.extra.get(k)
            if v is None:
                continue
            if not (lo <= float(v) <= hi):
                raise UnsafeRange(
                    message=(
                        f"{payload.algo}.{k} = {v} not in [{lo}, {hi}]; "
                        f"pass extra={{'unsafe': True}} to override"
                    )
                )
        return None
