"""FastAPI surface for Pi-GRPO. Submits runs, exposes status, serves inference."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse

from app.config import get_settings
from app.errors import PiGrpoError
from app.models import (
    HealthOut,
    InferIn,
    InferOut,
    RunIn,
    RunOut,
    RunStatus,
)
from app.services.run_orchestrator import RunOrchestrator
from observability.tracer import configure_tracing

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    s = get_settings()
    configure_tracing(s)
    app.state.runs = await RunOrchestrator.bootstrap(s)
    log.info("startup", env=s.env, version=s.version)
    try:
        yield
    finally:
        await app.state.runs.shutdown()


app = FastAPI(
    title="Pi-GRPO",
    version="0.1.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)


@app.exception_handler(PiGrpoError)
async def err_handler(_, exc: PiGrpoError) -> ORJSONResponse:
    return ORJSONResponse(status_code=exc.http_status, content={"code": exc.code, "message": exc.message})


@app.get("/healthz", response_model=HealthOut)
async def healthz() -> HealthOut:
    return HealthOut(status="ok", version=app.version)


@app.post("/v1/runs", response_model=RunOut)
async def submit_run(payload: RunIn) -> RunOut:
    return await app.state.runs.submit(payload)


@app.get("/v1/runs/{run_id}", response_model=RunStatus)
async def run_status(run_id: str) -> RunStatus:
    rs = await app.state.runs.status(run_id)
    if rs is None:
        raise HTTPException(status_code=404, detail="run not found")
    return rs


@app.delete("/v1/runs/{run_id}", response_model=RunStatus)
async def cancel_run(run_id: str) -> RunStatus:
    return await app.state.runs.cancel(run_id)


@app.post("/v1/infer", response_model=InferOut)
async def infer(payload: InferIn) -> InferOut:
    return await app.state.runs.infer(payload)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=1)
