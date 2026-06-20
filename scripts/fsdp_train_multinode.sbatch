#!/bin/bash
#SBATCH --job-name=fsdp-2node
#SBATCH --account=shekhars
#SBATCH --partition=a100-4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# ============================================================================
#  FSDP / ZeRO-3 TWO-NODE inter-node strong-scaling run (the paper centerpiece).
#  World = nodes x GPUS_PER_NODE (default 2 x 4 = 8). This is the 2-node point;
#  the 1-node baseline is fsdp_train_1node.sbatch. The strong-scaling number is
#  ONLY valid if BOTH runs use the IDENTICAL --model, --batch (per-GPU), --seq-len.
#  These knobs default to the SAME values as the 1-node wrapper; keep them in sync.
#
#  Inter-node rendezvous: ONE task per node (--ntasks-per-node=1); srun launches a
#  torchrun on each node; torchrun forks nproc_per_node ranks per node and they
#  rendezvous via c10d on MASTER_ADDR:PORT (the batch master node, first hostname).
#  Cross-node gradient reduce-scatter / param all-gather rides the InfiniBand fabric.
#
#  Submit (A100-4, 2 nodes x 4 GPU = world 8; abundant, easy to get 2 nodes):
#    sbatch scripts/fsdp_train_multinode.sbatch
#  H100 alternative (80GB, faster, but only ~4 e-nodes so 2 concurrent is contended):
#    sbatch --partition=msigpu --gres=gpu:h100:4 scripts/fsdp_train_multinode.sbatch
# ============================================================================
set -euo pipefail

REPO_ROOT="${PIGRPO_REPO:-${SLURM_SUBMIT_DIR:-$HOME/paper-work/pi-grpo}}"
SIF="${PIGRPO_LLM_SIF:-/scratch.global/${USER}/pigrpo_llm.sif}"
HF_HOME="${HF_HOME:-/scratch.global/${USER}/hf_cache}"

# ---- Scaling knobs: MUST match fsdp_train_1node.sbatch exactly. ----
MODEL="${FSDP_MODEL:-Qwen/Qwen2-7B-Instruct}"
PER_GPU_BATCH="${FSDP_BATCH:-1}"
SEQ_LEN="${FSDP_SEQ_LEN:-1024}"
STEPS="${FSDP_STEPS:-20}"
WARMUP="${FSDP_WARMUP:-3}"
GRAD_CKPT="${FSDP_GRAD_CKPT:-1}"
RANDOM_INIT="${FSDP_RANDOM_INIT:-0}"
GPUS_PER_NODE="${FSDP_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"

# ---- c10d rendezvous endpoint: first node in the allocation is the master. ----
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
# Pick a port deterministic-ish from the job id to avoid collisions on shared nodes.
MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 2000) ))
WORLD=$(( SLURM_NNODES * GPUS_PER_NODE ))

echo "[fsdp-2node] nodes=${SLURM_NNODES} gpus_per_node=${GPUS_PER_NODE} world=${WORLD}"
echo "[fsdp-2node] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} nodelist=${SLURM_JOB_NODELIST}"
echo "[fsdp-2node] model=${MODEL} per_gpu_batch=${PER_GPU_BATCH} seq=${SEQ_LEN} grad_ckpt=${GRAD_CKPT} random_init=${RANDOM_INIT}"

EXTRA=()
[ "${GRAD_CKPT}" = "1" ] && EXTRA+=( --grad-checkpointing )
[ "${RANDOM_INIT}" = "1" ] && EXTRA+=( --random-init )

# One srun task per node. Each task runs torchrun, which forks GPUS_PER_NODE local
# ranks; --node-rank is set per-task from SLURM_PROCID (0..nnodes-1). All ranks
# rendezvous via c10d on MASTER_ADDR:MASTER_PORT. NCCL uses IB across nodes.
srun --kill-on-bad-exit=1 bash -lc "
  apptainer exec --nv \
    --env HF_HOME='${HF_HOME}' \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --bind /scratch.global:/scratch.global \
    --bind '${REPO_ROOT}:${REPO_ROOT}' \
    '${SIF}' \
    bash -lc '
      set -euo pipefail
      cd \"${REPO_ROOT}\"
      export PYTHONPATH=\"${REPO_ROOT}:\${PYTHONPATH:-}\"
      export NCCL_DEBUG=WARN OMP_NUM_THREADS=8
      export TOKENIZERS_PARALLELISM=false
      # PIN to the single IB interface. The Agate nodes are dual-homed (aes0
      # 10.31.96.x + ib0 10.31.104.x); a bare ^lo,docker0 exclude leaves BOTH, so
      # ranks pick mismatched interfaces and the first cross-node collective
      # deadlocks (BUG_JOURNAL #76). Pinning ib0 lets native IB (mlx5_0) come up.
      export NCCL_SOCKET_IFNAME=ib0
      echo \"[fsdp-2node] task host=\$(hostname) SLURM_PROCID=\${SLURM_PROCID} node_rank=\${SLURM_NODEID}\"
      torchrun \
        --nnodes=${SLURM_NNODES} \
        --nproc-per-node=${GPUS_PER_NODE} \
        --node-rank=\${SLURM_NODEID} \
        --rdzv-backend=c10d \
        --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        --rdzv-id=${SLURM_JOB_ID} \
        scripts/fsdp_train.py \
        --model \"${MODEL}\" --batch ${PER_GPU_BATCH} --seq-len ${SEQ_LEN} \
        --steps ${STEPS} --warmup ${WARMUP} ${EXTRA[*]:-}
    '
"

echo "[fsdp-2node] done; result at ${REPO_ROOT}/results/fsdp_scaling_${WORLD}.json"
