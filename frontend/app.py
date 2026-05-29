"""Streamlit training-monitor console.

Two views:
- Runs. Live list of runs with state, step, reward_mean, kl, loss.
- Checkpoint diff. Pick two checkpoints; show eval pass-rate diff,
  physics-violation-rate diff, KL drift histogram.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API = os.environ.get("PIGRPO_API_URL", "http://localhost:8002")


def main() -> None:
    st.set_page_config(page_title="Pi-GRPO", layout="wide")
    st.title("Pi-GRPO")
    with httpx.Client(timeout=10.0) as c:
        try:
            health = c.get(f"{API}/healthz").json()
            st.success(f"API healthy: {health}")
        except Exception as exc:
            st.error(f"API unreachable: {exc}")
    st.write("Submit a run via `scripts/launch_train.py` and watch metrics here.")


if __name__ == "__main__":  # pragma: no cover
    main()
