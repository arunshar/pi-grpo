"""Build a versioned preference dataset from HITL exports."""

from __future__ import annotations

import argparse

from app.agents.data_curator import DataCuratorAgent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hitl-source", required=False, help="path to HITL jsonl")
    ap.add_argument("--out", default="data/preferences/dpo/v1.jsonl")
    args = ap.parse_args()

    res = DataCuratorAgent().build(hitl_jsonl=args.hitl_source, out_path=args.out)
    print(f"wrote {res.n_pairs} pairs to {res.out_path}")


if __name__ == "__main__":
    main()
