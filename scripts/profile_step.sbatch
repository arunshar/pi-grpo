#!/bin/bash
#SBATCH --job-name=pigrpo-profile-step
#SBATCH --account=shekhars
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Profiler pass for the paper's systems table: per-op time + MFU for ONE GRPO LLM
# training step (Qwen2-7B-Instruct + LoRA, on-policy HF-generate rollouts, physics
# reward bridge, clipped-surrogate + KL update), wrapped in torch.profiler, plus a
# (non-blocking) fused-RMSNorm Triton kernel correctness + microbenchmark.
#
# Requires an 80GB H100 (gpu:h100:1): the Qwen2-7B bf16 forward + KV + full-vocab
# logprob forward does NOT fit a 40GB A100 (matches grpo_llm_train.sbatch's note).
#
# Emits (in <repo>/results):
#   profile_step.json   -- measured step time, top-10 ops by CUDA time, MFU
#   profile_step.md     -- markdown table for the paper
#   triton_rmsnorm.json -- Triton kernel correctness + before/after timing
#
# Submit from the repo root:
#   sbatch scripts/profile_step.sbatch
# Useful overrides (env):
#   PIGRPO_PROFILE_TARGET=toy        # profile the tiny in-repo step instead (smoke)
#   PIGRPO_BATCH_PROMPTS=2 PIGRPO_GROUP_SIZE=4 PIGRPO_MAX_NEW_TOKENS=160
#   PIGRPO_LLM_SIF=/path/to/pigrpo_llm.sif

set -uo pipefail

# ----------------------------------------------------------------- repo root
if [ -n "${PIGRPO_REPO:-}" ]; then
  REPO_ROOT="${PIGRPO_REPO}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/profile_step.sbatch" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

# ----------------------------------------------------------------- knobs
SIF="${PIGRPO_LLM_SIF:-/scratch.global/${USER}/pigrpo_llm.sif}"
MODEL="${PIGRPO_BASE_MODEL:-Qwen/Qwen2-7B-Instruct}"
HF_HOME="${HF_HOME:-/scratch.global/${USER}/hf_cache}"
TARGET="${PIGRPO_PROFILE_TARGET:-llm-grpo}"
BATCH_PROMPTS="${PIGRPO_BATCH_PROMPTS:-2}"
GROUP_SIZE="${PIGRPO_GROUP_SIZE:-4}"
MAX_NEW_TOKENS="${PIGRPO_MAX_NEW_TOKENS:-160}"
WARMUP="${PIGRPO_WARMUP:-1}"
ACTIVE="${PIGRPO_ACTIVE:-3}"
TOP_N="${PIGRPO_TOP_N:-10}"

# ----------------------------------------------------------------- env
module purge 2>/dev/null || true
unset PYTHONPATH PYTHONHOME

# ----------------------------------------------------------------- preflight
echo "[profile] on $(hostname) repo=${REPO_ROOT} target=${TARGET}"
missing=0
note() { echo "  [MISSING] $*"; missing=$((missing+1)); }
ok()   { echo "  [ok]      $*"; }

[ -f "${SIF}" ] && ok "LLM container ${SIF}" || note "LLM container ${SIF}"
[ -f "${REPO_ROOT}/scripts/profile_step.py" ] && ok "profiler entrypoint" || note "scripts/profile_step.py"
if [ "${TARGET}" = "llm-grpo" ]; then
  [ -d "${HF_HOME}" ] && ok "HF cache ${HF_HOME}" || note "HF cache ${HF_HOME}"
fi
nvidia-smi -L 2>/dev/null | grep -qiE 'h100|a100|nvidia' \
  && ok "GPU: $(nvidia-smi -L 2>/dev/null | head -1)" || note "no GPU visible"

if [ "${missing}" -gt 0 ]; then
  echo "[profile] ${missing} prerequisite(s) missing; not launching." >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/results"
export HF_HOME PYTHONPATH="${REPO_ROOT}"

BIND="--bind /scratch.global:/scratch.global --bind ${REPO_ROOT}:${REPO_ROOT}"
ALLOC="--env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

# ----------------------------------------------------------------- profiler (the real deliverable)
echo "[profile] running torch.profiler on a ${TARGET} step (B=${BATCH_PROMPTS} K=${GROUP_SIZE})"
apptainer exec --nv --env HF_HOME="${HF_HOME}" --env PYTHONPATH="${REPO_ROOT}" ${ALLOC} ${BIND} "${SIF}" \
  python3 "${REPO_ROOT}/scripts/profile_step.py" \
    --target "${TARGET}" --base-model "${MODEL}" \
    --batch-prompts "${BATCH_PROMPTS}" --group-size "${GROUP_SIZE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --warmup "${WARMUP}" --active "${ACTIVE}" --top-n "${TOP_N}" \
    --out-dir "${REPO_ROOT}/results"
prof_rc=$?
echo "[profile] profiler exit=${prof_rc}"

# ----------------------------------------------------------------- Triton stretch (NON-BLOCKING)
# A failure / not-validated here must NOT fail the job: the profiler is the real result.
echo "[profile] Triton fused-RMSNorm stretch (non-blocking)"
apptainer exec --nv ${ALLOC} ${BIND} "${SIF}" \
  python3 "${REPO_ROOT}/scripts/triton_kernel_demo.py" \
    --rows 4096 --cols 3584 --dtype float32 \
    --out "${REPO_ROOT}/results/triton_rmsnorm.json" || \
  echo "[profile] Triton stretch did not validate (non-blocking; profiler stands)"

echo "[profile] done. results in ${REPO_ROOT}/results/"
exit ${prof_rc}
