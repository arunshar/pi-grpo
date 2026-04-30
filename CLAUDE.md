# Pi-GRPO: AI Coding Agent Context

## Domain glossary

- S-KBM. Single-axle Kinematic Bicycle Model. State `(x, y, theta, v)`; control `(a, delta)` (acceleration, steering angle). Used by Pi-DPM as a diffusion-decoder prior.
- Pi-DPM. Physics-informed Diffusion Probabilistic Model (Sharma et al., GeoAnomalies '25). Encoder-decoder diffusion with an S-KBM-aware decoder and a physics regularizer on (v, a, heading, curvature, turning rate).
- AGM. Abnormal Gap Measure. Convex combination of `1 - p_phys` and `p_data` (Pi-DPM reconstruction-error tail probability).
- PPO. Proximal Policy Optimization. Clipped surrogate loss + value head + GAE.
- DPO. Direct Preference Optimization. No reward model; the implicit reward is `beta · log(pi(y|x) / pi_ref(y|x))`.
- GRPO. Group Relative Policy Optimization. Sample K trajectories per prompt; advantage = (reward - mean)/std within the group; KL to the reference; no value head. DeepSeek-R1's training algorithm.
- Reference model. Frozen copy of the SFT base used for KL regularization in PPO and GRPO and for the implicit reward in DPO.

## Hard invariants

1. Hard physical violations (max speed envelope, S-KBM curvature limit) MUST yield an unbounded penalty that dominates any pref / data signal. See `app/components/physics_reward.py::PhysicsReward.score`.
2. The reference model is frozen for the entirety of a run. No parameter ever updates the reference. Save it as `ref_model.bin` at run start; verify via SHA at end.
3. KL divergence in PPO / GRPO is computed token-wise against the reference, not the previous policy. Adaptive KL controller (`app/trainers/base.py::AdaptiveKLController`) keeps the per-batch KL near the target.
4. Every gradient step writes a row to `runs.metrics` with `step, reward_mean, kl, loss, lr` and a TensorBoard / W&B point. Skipped steps (NaN loss) write `loss=NaN` so we can audit them.
5. Checkpoints are content-addressed; we never overwrite. `app/services/checkpoint.py::save` writes to `<run_id>/step_<n>/<sha256>.bin`.

## Conventions

- Strict typing. mypy strict + pydantic v2 in the API surface.
- Async-first I/O. The training loop is sync (PyTorch); the I/O around it (data loaders, checkpoint upload, eval submission) is async.
- vLLM is the default rollout backend. The `local_rollout` fallback exists for tests.
- Datasets are versioned. The build artifact is `data/preferences/<algo>/v<n>.jsonl` and the file's SHA is recorded in `runs.dataset_sha`.

## What NOT to do

- Do not use `--bf16` and `--fp16` at the same time. Pick one (default bf16 on H100 / A100, fp16 elsewhere).
- Do not start a PPO run without a sanity-checked reward model. The reward sanity test (`tests/test_reward_sanity.py`) must pass before the trainer accepts the reward config.
- Do not fine-tune above the LR / KL ranges in `configs/safe_ranges.yaml` without an explicit `--unsafe` flag. Bad ranges silently destabilize.
- Do not call vLLM without prefix caching enabled (`--enable-prefix-caching`); the 4x throughput uplift is what makes online RL feasible.
