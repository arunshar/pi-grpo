"""End-to-end integration tests against the Pi-GRPO FastAPI surface."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_submit_run_grpo(client: TestClient) -> None:
    r = client.post(
        "/v1/runs",
        json={"algo": "grpo", "base_model": "Qwen/Qwen2-7B-Instruct", "total_steps": 20},
    )
    assert r.status_code == 200
    body = r.json()
    assert "run_id" in body
    assert "submitted_at" in body


def test_submit_run_dpo_then_status(client: TestClient) -> None:
    r1 = client.post(
        "/v1/runs",
        json={"algo": "dpo", "base_model": "Qwen/Qwen2-7B-Instruct", "total_steps": 20},
    )
    assert r1.status_code == 200
    rid = r1.json()["run_id"]
    # the in-process simulator finishes in <2 s; poll briefly
    for _ in range(20):
        r2 = client.get(f"/v1/runs/{rid}")
        if r2.status_code == 200 and r2.json()["state"] in {"succeeded", "failed", "cancelled"}:
            break
        asyncio.run(asyncio.sleep(0.1))
    body = r2.json()
    assert body["state"] in {"running", "succeeded"}
    assert body["step"] >= 0
    assert "metrics" in body


def test_unsafe_range_guard_rejects_out_of_band_lr(client: TestClient) -> None:
    r = client.post(
        "/v1/runs",
        json={"algo": "ppo", "total_steps": 20, "extra": {"lr": 1.0}},
    )
    assert r.status_code == 400
    body = r.json()
    assert body["code"] == "pigrpo.unsafe_range"
    assert "lr" in body["message"]


def test_run_status_404(client: TestClient) -> None:
    r = client.get("/v1/runs/does-not-exist")
    assert r.status_code == 404


def test_unknown_algo_rejected(client: TestClient) -> None:
    r = client.post("/v1/runs", json={"algo": "ppo2", "total_steps": 20})
    assert r.status_code == 422


def test_total_steps_bounds_enforced(client: TestClient) -> None:
    r = client.post("/v1/runs", json={"algo": "ppo", "total_steps": 5})
    assert r.status_code == 422  # below ge=10
    r2 = client.post("/v1/runs", json={"algo": "ppo", "total_steps": 99_999})
    assert r2.status_code == 422  # above le=50_000
