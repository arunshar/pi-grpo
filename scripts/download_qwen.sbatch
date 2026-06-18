#!/bin/bash
#SBATCH --job-name=pigrpo-dl-qwen
#SBATCH --account=shekhars
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# EXP-5: pre-download Qwen2-7B-Instruct into the scratch HF cache, using the LLM
# container. Submit AFTER the build, gated on its success:
#   sbatch --dependency=afterok:<build_jobid> scripts/download_qwen.sbatch
# (a compute node has outbound internet; the login node does not).

set -uo pipefail

SIF="${PIGRPO_LLM_SIF:-/scratch.global/${USER}/pigrpo_llm.sif}"
MODEL="${PIGRPO_BASE_MODEL:-Qwen/Qwen2-7B-Instruct}"
export HF_HOME="/scratch.global/${USER}/hf_cache"
mkdir -p "${HF_HOME}"

if [ ! -f "${SIF}" ]; then
  echo "[dl] container ${SIF} missing; build it first (scripts/build_llm_container.sbatch)" >&2
  exit 2
fi

echo "[dl] node=$(hostname) model=${MODEL} -> ${HF_HOME}"
# Pass HF_HOME via --env (it overrides the image's %environment, which otherwise
# bakes an empty-$USER path and lands the model in /scratch.global/hf_cache).
apptainer exec --env HF_HOME="${HF_HOME}" --bind /scratch.global:/scratch.global "${SIF}" \
  python3 -c "from huggingface_hub import snapshot_download; p=snapshot_download('${MODEL}'); print('downloaded to', p)"
RC=$?
echo "[dl] exit=${RC}"
exit "${RC}"
