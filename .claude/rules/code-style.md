# Code Style

- Python 3.11+. `from __future__ import annotations` everywhere.
- Pydantic v2 for the API surface. Strict mypy.
- Async-first I/O around the trainers. Trainers themselves are sync.
- Trainer modules export only `Config` dataclasses and `Trainer` classes; no module-level state.
- Reward modules are pure functions of (trajectory, prompt, optional logits). No I/O.
- Logging: `structlog.get_logger(__name__)` with `event=` and `kv=` fields.
- Metrics flow only through `WandbAdapter.log()` and OTEL spans, never via prints.
