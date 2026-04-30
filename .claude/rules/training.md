# Training rules

- Reference model is frozen. Verify with a SHA check at run start and run end.
- KL is computed token-wise against the reference, not against the previous policy.
- Adaptive KL controller is bounded `[clip_min, clip_max]`. Never disable bounds.
- Hard physics violations dominate any preference signal. Verified by `tests/test_physics_reward.py`.
- bf16 is the default on H100 / A100. fp16 elsewhere. Never both.
- Checkpoints are content-addressed. `step_<n>/<sha>.bin`. Manifest at `MANIFEST.jsonl`.
- New trainer flag must come with: a row in `configs/safe_ranges.yaml`, a sanity test, and a docs/training.md entry under "failure modes".
