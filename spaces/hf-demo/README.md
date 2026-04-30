---
title: Pi-GRPO
emoji: 🔬
colorFrom: purple
colorTo: pink
sdk: streamlit
sdk_version: "1.36.0"
app_file: streamlit_app.py
pinned: false
license: mit
short_description: Physics-aware reward + reasoner demo for Pi-GRPO
---

# Pi-GRPO — Hugging Face Space

A standalone Streamlit demo of [Pi-GRPO](https://github.com/arunshar/pi-grpo), the physics-informed RL stack (PPO, DPO, GRPO with a hybrid physics-aware reward) described in our [NeurIPS-style preprint](https://github.com/arunshar/pi-grpo/blob/main/paper/pi_grpo_neurips.tex).

This Space runs on the free CPU tier. No GPU, no vLLM, no W\&B required.

## Two tabs

1. **Reward & violations.** Pick a sample trajectory (or upload your own CSV with `lat,lon,t,v`), see the hybrid reward decomposition (`R_hard`, `R_soft`, `R_data`, `R_pref`), the per-segment S-KBM violations, and the trajectory on a Folium map with red markers at hard-violating segments.
2. **Reasoner.** Calls the free Hugging Face Inference API (Qwen2-7B-Instruct or Mistral-7B-Instruct by default) with the `reasoning.v2` prompt; if the API is rate-limited, falls back to a deterministic explainer.

## Optional Space secrets

| Secret | Effect |
|---|---|
| `HF_TOKEN` | Authenticates the Inference API call so the reasoner tab is not rate-limited |
| `PG_PIDPM_CHECKPOINT` | Path to a TorchScripted Pi-DPM checkpoint inside the Space (fall-back is a deterministic surrogate) |

## Cite

```bibtex
@article{sharma2026pigrpo,
  title  = {{Pi-GRPO}: Physics-Informed Reinforcement Learning with Group Relative Policy Optimization for Trajectory Generation and Reasoning},
  author = {Sharma, Arun},
  year   = {2026}
}
```
