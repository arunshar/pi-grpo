"""Smoke test the running service."""

from __future__ import annotations

import sys

import httpx


def main(url: str = "http://localhost:8002") -> int:
    r = httpx.get(f"{url}/healthz", timeout=5.0)
    if r.status_code != 200:
        print(f"healthz failed: {r.status_code}")
        return 1
    print(r.json())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8002"))
