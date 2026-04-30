"""Seed Postgres with a runs table."""

from __future__ import annotations

import asyncio

import asyncpg

from app.config import get_settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id        TEXT PRIMARY KEY,
  algo          TEXT NOT NULL,
  base_model    TEXT NOT NULL,
  state         TEXT NOT NULL,
  step          INT NOT NULL DEFAULT 0,
  metrics       JSONB,
  dataset_sha   TEXT,
  started_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ended_at      TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS metrics_history (
  id        BIGSERIAL PRIMARY KEY,
  run_id    TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  step      INT NOT NULL,
  payload   JSONB NOT NULL,
  ts        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS metrics_history_run_step ON metrics_history (run_id, step);
"""


async def main() -> None:
    s = get_settings()
    conn = await asyncpg.connect(s.pg_dsn.replace("+asyncpg", ""))
    try:
        await conn.execute(SCHEMA)
    finally:
        await conn.close()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
