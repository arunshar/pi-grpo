"""Streaming dataset over preference jsonl with online batching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from app.components.preference_builder import PreferencePair


def stream_pairs(path: str | Path) -> Iterator[PreferencePair]:
    with Path(path).open() as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            yield PreferencePair(**d)


def batch(it: Iterator[PreferencePair], n: int) -> Iterator[list[PreferencePair]]:
    buf: list[PreferencePair] = []
    for p in it:
        buf.append(p)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf
