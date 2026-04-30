"""Content-addressed checkpoint manager.

Layout:

    runs/<run_id>/step_<n>/<sha256[:16]>.bin

Manifest at `runs/<run_id>/MANIFEST.jsonl` lists every step + sha so
arbitrary checkpoints are reproducible and auditable.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def save(run_dir: Path, step: int, blob: bytes, *, meta: dict[str, Any]) -> Path:
    sha = hashlib.sha256(blob).hexdigest()[:16]
    out = run_dir / f"step_{step:07d}" / f"{sha}.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    manifest = run_dir / "MANIFEST.jsonl"
    with manifest.open("a") as fh:
        fh.write(json.dumps({
            "step": step,
            "sha": sha,
            "path": str(out),
            "meta": meta,
            "ts": datetime.now(UTC).isoformat(),
        }) + "\n")
    return out


def latest(run_dir: Path) -> Path | None:
    manifest = run_dir / "MANIFEST.jsonl"
    if not manifest.exists():
        return None
    last = manifest.read_text().splitlines()[-1]
    return Path(json.loads(last)["path"])
