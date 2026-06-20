#!/bin/bash
#SBATCH --job-name=fsdp-1node
#SBATCH --account=shekhars
#SBATCH --partition=a100-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# ============================================================================
#  FSDP / ZeRO-3 SINGLE-NODE BASELINE for the inter-node strong-scaling curve.
#  This is the 1-node point (world = GPUS_PER_NODE). The 2-node point is
#  fsdp_train_multinode.sbatch. The scaling number is ONLY valid if BOTH runs
#  use the IDENTICAL --model, --batch (per-GPU), and --seq-len. The env defaults
#  below are shared between the two wrappers; do not change one without the other.
#
#  Submit (A100-4, 4 GPUs, easiest to schedule):
#    sbatch scripts/fsdp_train_1node.sbatch
#  H100 alternative (80GB, faster, contended: only ~4 e-nodes):
#    sbatch --partition=msigpu --gres=gpu:h100:4 scripts/fsdp_train_1node.sbatch
#
#  Real 7B run: this loads Qwen2-7B-Instruct from the scratch HF cache. To measure
#  pure systems scaling without the 15G checkpoint load, set RANDOM_INIT=1 (same
#  architecture/shapes/FSDP cost, weights random). On 40GB A100s with FULL_SHARD +
#  grad-checkpointing the 7B fits across 4 shards; if a single GPU OOMs, raise the
#  GPU count or enable RANDOM_INIT and keep GRAD_CKPT=1.
# ============================================================================
set -euo pipefail

REPO_ROOT="${PIGRPO_REPO:-${SLURM_SUBMIT_DIR:-$HOME/paper-work/pi-grpo}}"
SIF="${PIGRPO_LLM_SIF:-/scratch.global/${USER}/pigrpo_llm.sif}"
HF_HOME="${HF_HOME:-/scratch.global/${USER}/hf_cache}"

# ---- Scaling knobs: MUST match fsdp_train_multinode.sbatch exactly. ----
MODEL="${FSDP_MODEL:-Qwen/Qwen2-7B-Instruct}"
PER_GPU_BATCH="${FSDP_BATCH:-1}"
SEQ_LEN="${FSDP_SEQ_LEN:-1024}"
STEPS="${FSDP_STEPS:-20}"
WARMUP="${FSDP_WARMUP:-3}"
GRAD_CKPT="${FSDP_GRAD_CKPT:-1}"        # 1 = activation checkpointing (fits 7B on 40GB)
RANDOM_INIT="${FSDP_RANDOM_INIT:-0}"    # 1 = build from config, skip 15G ckpt load
GPUS_PER_NODE="${FSDP_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"

echo "[fsdp-1node] node=$(hostname) gpus_per_node=${GPUS_PER_NODE} model=${MODEL} batch=${PER_GPU_BATCH} seq=${SEQ_LEN}"
nvidia-smi -L || true

EXTRA=()
[ "${GRAD_CKPT}" = "1" ] && EXTRA+=( --grad-checkpointing )
[ "${RANDOM_INIT}" = "1" ] && EXTRA+=( --random-init )

apptainer exec --nv \
  --env HF_HOME="${HF_HOME}" \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --bind /scratch.global:/scratch.global \
  --bind "${REPO_ROOT}:${REPO_ROOT}" \
  "${SIF}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO_ROOT}'
    export PYTHONPATH='${REPO_ROOT}:\${PYTHONPATH:-}'
    export NCCL_DEBUG=WARN OMP_NUM_THREADS=8
    export TOKENIZERS_PARALLELISM=false
    python3 -c 'import torch; print(\"[fsdp-1node] torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"gpus\", torch.cuda.device_count())'
    torchrun --standalone --nnodes=1 --nproc_per_node='${GPUS_PER_NODE}' \
      scripts/fsdp_train.py \
      --model '${MODEL}' --batch '${PER_GPU_BATCH}' --seq-len '${SEQ_LEN}' \
      --steps '${STEPS}' --warmup '${WARMUP}' ${EXTRA[*]:-}
  "

echo "[fsdp-1node] done; result at ${REPO_ROOT}/results/fsdp_scaling_${GPUS_PER_NODE}.json"
