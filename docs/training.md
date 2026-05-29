# Training Guide

## Build a preference dataset

```bash
# from HITL feedback exported by geotrace-agent
python scripts/build_preferences.py \
    --hitl-source /path/to/geotrace-agent/exports/hitl_2026-04.jsonl \
    --out data/preferences/dpo/v1.jsonl
```

## Run DPO

```bash
python scripts/launch_train.py \
    --algo dpo \
    --base-model Qwen/Qwen2-7B-Instruct \
    --preferences data/preferences/dpo/v1.jsonl \
    --total-steps 2000 \
    --watch
```

## Run GRPO

```bash
python scripts/launch_train.py \
    --algo grpo \
    --base-model Qwen/Qwen2-7B-Instruct \
    --reward-config configs/physics_reward.yaml \
    --total-steps 3000 \
    --watch
```

## Run PPO

```bash
python scripts/launch_train.py \
    --algo ppo \
    --base-model Qwen/Qwen2-7B-Instruct \
    --reward-config configs/physics_reward.yaml \
    --total-steps 5000 \
    --watch
```

## Failure modes and what they look like

| Symptom | Likely cause | Fix |
|---|---|---|
| KL doubles every 10 steps | LR too high relative to beta / target_kl | halve LR or raise target_kl |
| reward_mean rises while physics_violation_rate also rises | reward hacking (model exploits soft reward to violate hard reward) | raise w_hard; cap w_pref; verify reward sanity test passes |
| loss is NaN | bf16 underflow on tiny grads | switch to fp16 or upcast logsigmoid in DPO |
| eval pass_rate drops while reward_mean keeps rising | distribution shift between reward and eval | shadow-evaluate with the previous checkpoint to confirm |
