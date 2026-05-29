"""Tool registry for the trainer / data curator agents."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from . import code_search, vector_search, web_search

REGISTRY: dict[str, dict[str, Any]] = {
    "vector.search": {"description": "Search prior runs and reports", "fn": vector_search.run},
    "web.search":    {"description": "OSINT search",                  "fn": web_search.run},
    "code.search":   {"description": "Grep over training playbooks",  "fn": code_search.run},
}


async def call(name: str, args: dict[str, Any]) -> Any:
    fn: Callable[[dict[str, Any]], Awaitable[Any]] = REGISTRY[name]["fn"]
    return await fn(args)
