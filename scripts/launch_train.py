"""Submit a training run to the running Pi-GRPO API."""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default="http://localhost:8002")
    ap.add_argument("--algo", choices=["ppo", "dpo", "grpo"], required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2-7B-Instruct")
    ap.add_argument("--preferences", default=None)
    ap.add_argument("--reward-config", default="configs/physics_reward.yaml")
    ap.add_argument("--total-steps", type=int, default=2000)
    ap.add_argument("--watch", action="store_true")
    args = ap.parse_args()

    body = {
        "algo": args.algo,
        "base_model": args.base_model,
        "preferences_path": args.preferences,
        "reward_config": args.reward_config,
        "total_steps": args.total_steps,
    }
    with httpx.Client(timeout=30.0) as c:
        r = c.post(f"{args.api_url}/v1/runs", json=body)
        r.raise_for_status()
        run = r.json()
        print(json.dumps(run, indent=2))
        if args.watch:
            while True:
                s = c.get(f"{args.api_url}/v1/runs/{run['run_id']}").json()
                print(s)
                if s["state"] in {"succeeded", "failed", "cancelled"}:
                    return 0 if s["state"] == "succeeded" else 1
                time.sleep(2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
