# Pull request

## Summary

What changed and why?

## Checklist

- [ ] `ruff check .` is clean.
- [ ] `mypy app` is clean.
- [ ] `pytest -q` is clean.
- [ ] If the reward changed, `tests/test_reward_sanity.py` and `tests/test_physics_reward.py` still pass.
- [ ] If a new trainer flag was added, a row was added to `configs/safe_ranges.yaml` and a sanity test was added.
- [ ] If checkpoint format changed, `MANIFEST.jsonl` is forward-compatible (additive fields only).
- [ ] No new bf16 + fp16 mixing.

## Run examples (optional)

If the change affects training behavior, paste a `run_id` from a successful run and link the W&B chart.
