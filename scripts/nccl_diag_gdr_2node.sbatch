#!/bin/bash
#SBATCH --job-name=nccl-diag-gdr
#SBATCH --account=shekhars
#SBATCH --partition=a100-4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# ============================================================================
#  GDR bind-mount test (BUG_JOURNAL #77). The container's libmlx5 is too old for
#  NCCL 2.28 (missing mlx5dv_reg_dmabuf_mr), so GPUDirect RDMA is DISABLED and
#  cross-node FSDP is host-staged and slow. The HOST (Rocky 8.10, OFED 25.10)
#  libmlx5.so.1.25.58.0 HAS the symbol, so this run binds the host IB userspace
#  over the container's stale libs (keeping NCCL_SOCKET_IFNAME=ib0) and checks the
#  NCCL INFO log for "GPU Direct RDMA Enabled" instead of the previous "Disabled".
# ============================================================================
set -euo pipefail

REPO_ROOT="${PIGRPO_REPO:-${SLURM_SUBMIT_DIR:-$HOME/paper-work/pi-grpo}}"
SIF="${PIGRPO_LLM_SIF:-/scratch.global/${USER}/pigrpo_llm.sif}"
GPUS_PER_NODE="${FSDP_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
LOGDIR="/scratch.global/${USER}/nccl_logs/${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 2000) ))
WORLD=$(( SLURM_NNODES * GPUS_PER_NODE ))

echo "[pin] nodes=${SLURM_NNODES} world=${WORLD} master=${MASTER_ADDR}:${MASTER_PORT} nodelist=${SLURM_JOB_NODELIST} logdir=${LOGDIR}"

srun --kill-on-bad-exit=1 bash -lc "
  apptainer exec --nv \
    --bind /scratch.global:/scratch.global \
    --bind '${REPO_ROOT}:${REPO_ROOT}' \
    --bind /usr/lib64/libmlx5.so.1.25.58.0:/usr/lib/x86_64-linux-gnu/libmlx5.so.1 \
    --bind /usr/lib64/libibverbs.so.1.14.58.0:/usr/lib/x86_64-linux-gnu/libibverbs.so.1 \
    --bind /etc/libibverbs.d:/etc/libibverbs.d \
    '${SIF}' \
    bash -lc '
      set -euo pipefail
      cd \"${REPO_ROOT}\"
      export PYTHONPATH=\"${REPO_ROOT}:\${PYTHONPATH:-}\"
      export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,ENV
      export NCCL_DEBUG_FILE=\"${LOGDIR}/nccl.%h.%p.log\"
      export NCCL_SOCKET_IFNAME=ib0          # PIN to a single interface (the fix)
      export NCCL_NET_GDR_LEVEL=SYS          # force GDR even across host-bridge/SYS topology (test)
      export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
      export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
      echo \"[gdr] task host=\$(hostname) SLURM_NODEID=\${SLURM_NODEID} NCCL_SOCKET_IFNAME=\${NCCL_SOCKET_IFNAME}\"
      echo \"[gdr] bound libmlx5 (expect host size 502632):\"; ls -lL /usr/lib/x86_64-linux-gnu/libmlx5.so.1 2>/dev/null || true
      torchrun \
        --nnodes=${SLURM_NNODES} \
        --nproc-per-node=${GPUS_PER_NODE} \
        --node-rank=\${SLURM_NODEID} \
        --rdzv-backend=c10d \
        --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        --rdzv-id=${SLURM_JOB_ID} \
        scripts/nccl_diag.py
    '
"

echo "[pin] srun returned rc=$?  (NCCL INFO logs in ${LOGDIR})"
