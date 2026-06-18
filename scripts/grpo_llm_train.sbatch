#!/bin/bash
#SBATCH --job-name=pigrpo-llm-grpo
#SBATCH --account=shekhars
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# EXP-5: the planned full-scale LLM GRPO run (Qwen2-7B-Instruct + vLLM) on 2x H100.
# GPU 1 hosts a vLLM server (prefix caching) for rollouts; GPU 0 trains the policy
# (LoRA) on the physics reward with KL to a frozen reference.
#
# STATUS: this is a PREFLIGHT-GATED SCAFFOLD. It does NOT yet train, because the
# 7B trainer entrypoint and the LLM container are not built. It checks every
# prerequisite and exits with a precise checklist of what is missing, so it is a
# real readiness tool rather than a fake launcher. When the prerequisites are in
# place it execs the trainer. See docs/EXP5_LLM_SCALE.md for the full plan.
#
# msigpu wall cap is 24h. If the DPO/PPO/GRPO sweep exceeds that, chain DONE-gated
# dependent jobs (the MSI long-job eval-chain pattern), do not rely on one job.
#
# Submit from the repo root:
#   sbatch scripts/grpo_llm_train.sbatch

set -uo pipefail

# ----------------------------------------------------------------- repo root
# Under sbatch the script is copied to a spool dir, so BASH_SOURCE is unreliable;
# prefer an explicit override, then the submit dir, then the script location.
if [ -n "${PIGRPO_REPO:-}" ]; then
  REPO_ROOT="${PIGRPO_REPO}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/grpo_llm_train.sbatch" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

# ----------------------------------------------------------------- knobs
SIF="${PIGRPO_LLM_SIF:-/scratch.global/${USER}/pigrpo_llm.sif}"
MODEL="${PIGRPO_BASE_MODEL:-Qwen/Qwen2-7B-Instruct}"
HF_HOME="${HF_HOME:-/scratch.global/${USER}/hf_cache}"
TRAINER="${REPO_ROOT}/scripts/grpo_llm_train.py"
ALGO="${PIGRPO_ALGO:-grpo}"
PREFS="${PIGRPO_PREFERENCES:-${REPO_ROOT}/data/preferences.jsonl}"

# ----------------------------------------------------------------- preflight
echo "[exp5] preflight on $(hostname) for repo=${REPO_ROOT}"
missing=0
note() { echo "  [MISSING] $*"; missing=$((missing+1)); }
ok()   { echo "  [ok]      $*"; }

[ -f "${SIF}" ] && ok "LLM container ${SIF}" || note "LLM container ${SIF} (build docker/pigrpo_llm.def on a compute node)"
if [ -f "${SIF}" ]; then
  if apptainer exec "${SIF}" python3 -c "import vllm, transformers, peft" >/dev/null 2>&1; then
    ok "container has vllm + transformers + peft"
  else
    note "container is missing vllm/transformers/peft (rebuild docker/pigrpo_llm.def)"
  fi
fi
[ -d "${HF_HOME}" ] && ok "HF cache ${HF_HOME}" || note "HF cache ${HF_HOME} (mkdir -p and pre-download ${MODEL})"
[ -f "${TRAINER}" ] && ok "trainer entrypoint ${TRAINER}" || note "trainer entrypoint scripts/grpo_llm_train.py not found (see docs/EXP5_LLM_SCALE.md)"
[ -f "${PREFS}" ] && ok "preference triples ${PREFS}" || note "preference triples ${PREFS} (export from GeoTrace-Agent HITL)"
nvidia-smi -L 2>/dev/null | grep -qi h100 && ok "H100 visible" || note "no H100 visible on this allocation"

if [ "${missing}" -gt 0 ]; then
  echo "[exp5] ${missing} prerequisite(s) missing; not launching. This is expected until EXP-5 is built." >&2
  exit 2
fi

# ----------------------------------------------------------------- launch
# Reached only when every prerequisite is satisfied.
export HF_HOME PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
echo "[exp5] launching ${ALGO} run: model=${MODEL} on $(nvidia-smi -L | wc -l) GPUs"
# GPU 1 -> vLLM rollout server (prefix caching); GPU 0 -> LoRA trainer.
apptainer exec --nv --env HF_HOME="${HF_HOME}" --bind /scratch.global:/scratch.global --bind "${REPO_ROOT}:${REPO_ROOT}" "${SIF}" \
  bash -c "CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server --model '${MODEL}' --enable-prefix-caching --port 8000 & \
           sleep 60; \
           CUDA_VISIBLE_DEVICES=0 python3 '${TRAINER}' --algo '${ALGO}' --base-model '${MODEL}' --vllm-url http://localhost:8000/v1 --preferences '${PREFS}'"
