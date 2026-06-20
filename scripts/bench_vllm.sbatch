#!/bin/bash
#SBATCH --job-name=pigrpo-bench-vllm
#SBATCH --account=shekhars
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Ray paper (PC-RF) systems experiment: vLLM offline-engine serving throughput /
# latency benchmark for Qwen/Qwen2-7B-Instruct vs a Hugging Face generate baseline,
# on ONE H100 (80GB). Identical model / GPU / prompt set / decoding; reports
# aggregate output tokens/sec, per-request p50/p95 latency, and the vLLM/HF speedup.
#
# Qwen2-7B-Instruct is already cached under HF_HOME (4 safetensors shards). The
# LLM container (vllm + transformers + peft + accelerate) is pigrpo_llm.sif.
#
# NOTE: needs an 80GB H100 (gpu:h100:1). A 40GB A100 cannot hold Qwen2-7B bf16
# for BOTH the vLLM engine and the subsequent HF generate load comfortably; to
# run on a100 (a100-4 partition), split into two jobs (--skip-hf / --skip-vllm)
# or lower --gpu-mem-util and --max-model-len.
#
# Submit from the repo root:
#   sbatch scripts/bench_vllm.sbatch
# Smoke (tiny, real GPU) instead:
#   PIGRPO_BENCH_SMOKE=1 sbatch scripts/bench_vllm.sbatch

set -uo pipefail

# ----------------------------------------------------------------- repo root
if [ -n "${PIGRPO_REPO:-}" ]; then
  REPO_ROOT="${PIGRPO_REPO}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/bench_vllm.sbatch" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

# ----------------------------------------------------------------- knobs
SIF="${PIGRPO_LLM_SIF:-/scratch.global/${USER}/pigrpo_llm.sif}"
MODEL="${PIGRPO_BASE_MODEL:-Qwen/Qwen2-7B-Instruct}"
HF_HOME="${HF_HOME:-/scratch.global/${USER}/hf_cache}"
BENCH="${REPO_ROOT}/scripts/bench_vllm.py"
NUM_PROMPTS="${PIGRPO_BENCH_NUM_PROMPTS:-128}"
MAX_NEW="${PIGRPO_BENCH_MAX_NEW:-128}"
VLLM_BS="${PIGRPO_BENCH_VLLM_BS:-1,8,32,128}"
HF_BS="${PIGRPO_BENCH_HF_BS:-8}"
OUT="${PIGRPO_BENCH_OUT:-${REPO_ROOT}/results/bench_vllm.json}"

SMOKE_ARG=""
if [ "${PIGRPO_BENCH_SMOKE:-0}" = "1" ]; then
  SMOKE_ARG="--smoke"
  OUT="${PIGRPO_BENCH_OUT:-${REPO_ROOT}/results/bench_vllm_smoke.json}"
fi

# ----------------------------------------------------------------- env hygiene
module purge 2>/dev/null || true
unset PYTHONPATH PYTHONHOME 2>/dev/null || true

# ----------------------------------------------------------------- preflight
echo "[bench] node=$(hostname) repo=${REPO_ROOT}"
missing=0
note() { echo "  [MISSING] $*"; missing=$((missing+1)); }
ok()   { echo "  [ok]      $*"; }

[ -f "${SIF}" ] && ok "LLM container ${SIF}" || note "LLM container ${SIF}"
if [ -f "${SIF}" ]; then
  if apptainer exec "${SIF}" python3 -c "import vllm, transformers" >/dev/null 2>&1; then
    ok "container has vllm + transformers"
  else
    note "container missing vllm/transformers"
  fi
fi
[ -d "${HF_HOME}" ] && ok "HF cache ${HF_HOME}" || note "HF cache ${HF_HOME}"
[ -f "${BENCH}" ] && ok "bench script ${BENCH}" || note "bench script ${BENCH}"
nvidia-smi -L 2>/dev/null | grep -qiE 'h100|a100|nvidia' \
  && ok "GPU visible: $(nvidia-smi -L 2>/dev/null | head -1)" \
  || note "no GPU visible on this allocation"

if [ "${missing}" -gt 0 ]; then
  echo "[bench] ${missing} prerequisite(s) missing; not launching." >&2
  exit 2
fi

mkdir -p "$(dirname "${OUT}")"

# ----------------------------------------------------------------- launch
# PYTHONPATH set to the repo root so the script can import from app/* if needed.
# expandable_segments guards against allocator fragmentation when the HF baseline
# loads the same 7B weights right after the vLLM engine is torn down.
echo "[bench] launching: model=${MODEL} num_prompts=${NUM_PROMPTS} max_new=${MAX_NEW} vllm_bs=${VLLM_BS} hf_bs=${HF_BS} smoke=${PIGRPO_BENCH_SMOKE:-0}"
apptainer exec --nv \
  --env HF_HOME="${HF_HOME}" \
  --env PYTHONPATH="${REPO_ROOT}" \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --bind /scratch.global:/scratch.global \
  --bind "${REPO_ROOT}:${REPO_ROOT}" \
  "${SIF}" \
  python3 "${BENCH}" \
    --model "${MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-new-tokens "${MAX_NEW}" \
    --vllm-batch-sizes "${VLLM_BS}" \
    --hf-batch-size "${HF_BS}" \
    --dtype bfloat16 \
    --out "${OUT}" \
    ${SMOKE_ARG}
RC=$?
echo "[bench] exit=${RC} out=${OUT}"
exit "${RC}"
