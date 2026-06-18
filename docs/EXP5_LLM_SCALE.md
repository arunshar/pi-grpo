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

## The trainer (`scripts/grpo_llm_train.py`, implemented, pending GPU validation)

Implemented (structurally complete, byte-compiles; not yet run end to end because the
container and model download are the gating steps):

1. Loads `Qwen2-7B-Instruct` (bf16) with a LoRA adapter (peft); the frozen reference
   is the base model recovered by `model.disable_adapter()`, so there is one model in
   memory, not two.
2. The reward bridge reuses the EXISTING physics path unchanged: the policy is
   prompted to emit a trajectory as `horizon` motion-primitive ids (the
   `MotionCodebook` accel x steer vocabulary); `parse_motion_ids` parses the
   completion, `tokens_to_states` rolls it through the S-KBM, and
   `PhysicsReward.score` + `PiDpmScorer` score it. The trained-policy numbers are
   therefore on the same reward as the CPU scaling / reward-hacking results.
3. GRPO update: K rollouts per prompt, group-baseline advantages, clipped surrogate
   over summed completion logprobs, KL to the frozen reference with an adaptive KL
   coefficient. Optimizes only the LoRA parameters.
4. Streams `metrics.jsonl` (reward mean, hard-violation rate, KL, loss) for
   `tab:expected-trends` and the KL-drift claim.

Rollouts default to Hugging Face `generate` (on-policy and correct). The vLLM path is
the throughput optimization and is left as a hook: vLLM serves the frozen base
weights, so on-policy LoRA rollouts need a per-step adapter sync (vLLM LoRA hot-swap),
which is the next optimization once the correctness run is green.

Still to do: validate end to end on an H100 (the parse grammar, LoRA target modules,
and KL schedule will need tuning on real generations); add PPO/DPO (they reuse the
same reward bridge and policy); wire the vLLM rollout backend.

## Build and run sequence

```
# 1. Build the LLM container (compute node; internet + fakeroot verified by the probe):
sbatch scripts/build_llm_container.sbatch        # -> /scratch.global/$USER/pigrpo_llm.sif + pigrpo_build/DONE

# 2. Pre-download the model (auto-runs after the build via the afterok dependency):
sbatch --dependency=afterok:<build_jobid> scripts/download_qwen.sbatch   # -> hf_cache/

# 3. Preflight + launch (prints a checklist; exits 2 until every prerequisite is met):
sbatch scripts/grpo_llm_train.sbatch
```

## Honest status for the paper

The paper says this run is planned and the code stages it but has not completed it.
Do not report `tab:expected-trends`, the ablation deltas, or the KL-drift numbers as
measured until the GPU run is validated and completed.
