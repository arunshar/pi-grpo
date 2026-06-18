#!/bin/bash
#SBATCH --job-name=pigrpo-llm-grpo
#SBATCH --account=shekhars
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# EXP-5: full-scale LLM GRPO run (Qwen2-7B-Instruct + LoRA) on the physics reward
# with KL to a frozen reference (the base model via disable_adapter()).
#
# DEFAULT (correctness) path: on-policy rollouts via Hugging Face generate on the
# single allocated H100. This is what the trainer actually implements and is the
# run that fills tab:expected-trends. Sized via env: PIGRPO_STEPS / PIGRPO_BATCH_PROMPTS
# / PIGRPO_GROUP_SIZE (use small values for a smoke).
#
# THROUGHPUT path (PIGRPO_USE_VLLM=1): hosts a vLLM rollout server on a second GPU.
# This needs --gres=gpu:h100:2 AND an implemented vLLM rollout backend in the trainer
# (the --vllm-url hook is not wired yet), so it is for future use.
#
# This is a PREFLIGHT-GATED launcher: it verifies the container, LLM stack, HF cache,
# trainer, GPU (and preferences only for --algo dpo), exits 2 with a checklist if any
# are missing, and execs the trainer only when satisfied. See docs/EXP5_LLM_SCALE.md.
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
STEPS="${PIGRPO_STEPS:-2000}"
BATCH_PROMPTS="${PIGRPO_BATCH_PROMPTS:-8}"
GROUP_SIZE="${PIGRPO_GROUP_SIZE:-8}"
SEED="${PIGRPO_SEED:-42}"
OUT_DIR="${PIGRPO_OUT_DIR:-${REPO_ROOT}/runs/llm_grpo}"
LOG_COMPLETIONS="${PIGRPO_LOG_COMPLETIONS:-0}"

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
if [ "${ALGO}" = "dpo" ]; then
  [ -f "${PREFS}" ] && ok "preference triples ${PREFS}" || note "preference triples ${PREFS} (export from GeoTrace-Agent HITL; required only for --algo dpo)"
else
  ok "preferences not required for --algo ${ALGO}"
fi
nvidia-smi -L 2>/dev/null | grep -qiE 'h100|a100|nvidia' && ok "GPU visible: $(nvidia-smi -L 2>/dev/null | head -1)" || note "no GPU visible on this allocation"

if [ "${missing}" -gt 0 ]; then
  echo "[exp5] ${missing} prerequisite(s) missing; not launching. This is expected until EXP-5 is built." >&2
  exit 2
fi

# ----------------------------------------------------------------- launch
# Reached only when every prerequisite is satisfied.
export HF_HOME PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
COMMON_ARGS=( --algo "${ALGO}" --base-model "${MODEL}" --steps "${STEPS}" \
              --batch-prompts "${BATCH_PROMPTS}" --group-size "${GROUP_SIZE}" \
              --seed "${SEED}" --out-dir "${OUT_DIR}" --log-completions "${LOG_COMPLETIONS}" )
[ "${ALGO}" = "dpo" ] && COMMON_ARGS+=( --preferences "${PREFS}" )

if [ "${PIGRPO_USE_VLLM:-0}" = "1" ]; then
  # THROUGHPUT path: vLLM rollout server on GPU 1, trainer on GPU 0 (needs gpu:h100:2).
  # NOTE: the trainer's --vllm-url backend is not wired yet, so this path is for future use.
  echo "[exp5] launching ${ALGO} with a vLLM rollout server on $(nvidia-smi -L | wc -l) GPUs"
  apptainer exec --nv --env HF_HOME="${HF_HOME}" --bind /scratch.global:/scratch.global --bind "${REPO_ROOT}:${REPO_ROOT}" "${SIF}" \
    bash -c "CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server --model '${MODEL}' --enable-prefix-caching --port 8000 & \
             sleep 60; \
             CUDA_VISIBLE_DEVICES=0 python3 '${TRAINER}' ${COMMON_ARGS[*]} --vllm-url http://localhost:8000/v1"
else
  # DEFAULT correctness path: on-policy HF-generate rollouts on the single allocated H100.
  echo "[exp5] launching ${ALGO} (HF-generate rollouts) on $(nvidia-smi -L | wc -l) GPU(s): steps=${STEPS} B=${BATCH_PROMPTS} K=${GROUP_SIZE}"
  # expandable_segments avoids allocator fragmentation OOMs over a long (1200-step) run;
  # the K=4 forward fits the 80GB H100 with thin headroom (calibration job 11234165), so
  # this guards the tail. NOTE: this run REQUIRES an 80GB H100 (gpu:h100:1); a 40GB A100
  # cannot hold Qwen2-7B bf16 + KV + the full-vocab logprob forward (job 11234588 OOMed).
  apptainer exec --nv --env HF_HOME="${HF_HOME}" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --bind /scratch.global:/scratch.global --bind "${REPO_ROOT}:${REPO_ROOT}" "${SIF}" \
    python3 "${TRAINER}" "${COMMON_ARGS[@]}"
fi
