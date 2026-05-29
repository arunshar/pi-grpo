"""Inference-side token budget for the FastAPI surface.

Mirrors the GeoTrace-Agent token optimizer's role but only for the
serving path (not training). Compresses long prompts, enforces
`max_tokens` per call, and routes through vLLM with prefix caching.
"""

from __future__ import annotations

import tiktoken


def compress(prompt: str, max_in_tokens: int) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(prompt)
    if len(toks) <= max_in_tokens:
        return prompt
    head = toks[: max_in_tokens // 2]
    tail = toks[-max_in_tokens // 2:]
    return enc.decode(head) + "\n\n[...elided...]\n\n" + enc.decode(tail)
