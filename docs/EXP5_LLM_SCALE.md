# EXP-5: the planned full-scale LLM GRPO run (staging)

This is the runbook and honest status for the Qwen2-7B-Instruct + vLLM full-scale
run that the paper describes as the target configuration. It is **planned, not yet
implemented**. The measured contributions in the paper (the reward-actor scaling
crossover, the reward-hacking probe, the microbenchmarks, the unit suite) stand
without it; EXP-5 fills the trend table (`tab:expected-trends`), the ablation
matrix, and the KL-drift claim.

## What exists today

- The tiny in-process policy GRPO loop (`app/policy/driver.py`, `model.py`) that the
  CPU scaling experiments use. This is what `RunOrchestrator._run` actually runs: it
  does `replace(SMOKE, ...)` and trains the synthetic policy regardless of
  `base_model`.
- A vLLM HTTP inference client for serving (`app/rollouts/vllm_rollout.py`,
  `RunOrchestrator.infer`), and config pointing at `Qwen/Qwen2-7B-Instruct`.
- Ray rollout/staleness scaffolding with explicit `TODO(scale)` markers where the
  vLLM generator would replace `policy.generate`.

## The two blockers

1. **No 7B training code.** There is no entrypoint that loads a Qwen2-7B policy,
   generates rollouts through vLLM, decodes the LLM output into trajectories the
   physics reward can score, and runs a GRPO (or PPO/DPO) update with KL to a frozen
   reference. The base model is currently only logged and used for inference.
2. **No capable container.** `mirror_pnemo.sif` has torch 2.12 + apex + flash_attn
   but no vllm/transformers/accelerate/peft/ray (probed 2026-06-17). It cannot run a
   7B + vLLM job.

## Staging artifacts in this repo

- `docker/pigrpo_llm.def` -- apptainer recipe for the LLM container (vllm base +
  transformers/peft/accelerate/ray/trl). Build on a compute node (internet); the
  first build will likely need one version-pin adjustment.
- `scripts/grpo_llm_train.sbatch` -- 2x H100 launcher, preflight-gated. It checks
  the container, the LLM stack, the HF cache, the trainer entrypoint, the preference
  file, and the H100s, and exits with a precise checklist of what is missing. It
  launches only when all prerequisites are satisfied, so it is a readiness tool, not
  a fake.

## Open code work (`scripts/grpo_llm_train.py`, to implement)

1. Load `Qwen2-7B-Instruct` as the policy with a LoRA adapter (memory: 7B + LoRA fits
   one H100 80GB for training; the frozen reference is the base model served by vLLM).
2. Rollout generation through the vLLM server (`--enable-prefix-caching`) rather than
   `policy.generate`, sharding prompts (replace the `TODO(scale)` in
   `app/policy/ray_rollout.py`).
3. Decode the LLM output into a trajectory the physics reward can score. This is the
   real design question: the reward operates on motion states via `MotionCodebook`;
   the 7B path needs a defined output grammar (motion tokens, or a parseable
   trajectory payload) so `PhysicsReward.score` + `PiDpmScorer` apply unchanged.
4. GRPO update on the LoRA policy reusing the existing `GrpoTrainer` and the hybrid
   reward; KL to the frozen reference; the `AdaptiveKLController` for PPO.
5. Stream metrics for `tab:expected-trends` (hard-violation, soft-envelope,
   pref-win-rate, KL) and the KL-drift logging.

## Build and run sequence (when the trainer exists)

```
# 1. Build the LLM container on a compute node (internet + fakeroot):
srun --account=shekhars --partition=msigpu --gres=gpu:h100:1 --cpus-per-task=16 --mem=64g --time=2:00:00 --pty bash
export APPTAINER_CACHEDIR=/scratch.global/$USER/apptainer_cache
apptainer build --fakeroot /scratch.global/$USER/pigrpo_llm.sif docker/pigrpo_llm.def

# 2. Pre-download the model into the HF cache (compute node has internet):
mkdir -p /scratch.global/$USER/hf_cache
HF_HOME=/scratch.global/$USER/hf_cache apptainer exec /scratch.global/$USER/pigrpo_llm.sif \
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-7B-Instruct')"

# 3. Preflight (prints the checklist; exits 2 until everything is ready):
sbatch scripts/grpo_llm_train.sbatch
```

## Honest status for the paper

The paper now says this run is planned and the code stages but does not yet
implement it, and points here. Do not report `tab:expected-trends`, the ablation
deltas, or the KL-drift numbers as measured until this run is built and completed.
