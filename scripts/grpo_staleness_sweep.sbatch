#!/bin/bash
#SBATCH --job-name=pigrpo-exp4-staleness
#SBATCH --account=shekhars
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# EXP-4: async/off-policy staleness sweep (s=0,1,2,4,8). CPU-only, in-process
# harness; runs detached on agsmall so it survives a login-node / session kill.
# Writes staleness_sweep.csv + summary.txt + a DONE marker to scratch.
#
# Submit from the repo root:  sbatch scripts/grpo_staleness_sweep.sbatch

set -uo pipefail

if [ -n "${PIGRPO_REPO:-}" ]; then
  REPO_ROOT="${PIGRPO_REPO}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/grpo_staleness_sweep.py" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
module purge 2>/dev/null || true
unset PYTHONPATH PYTHONHOME
# shellcheck disable=SC1091
source ~/miniforge3/etc/profile.d/conda.sh
conda activate pcrf
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

OUT="${PIGRPO_OUT_DIR:-/scratch.global/${USER}/pigrpo/exp4_staleness}"
mkdir -p "${OUT}"

echo "[exp4] $(date) host=$(hostname) python=$(which python)"
python scripts/grpo_staleness_sweep.py \
  --staleness "${PIGRPO_STALENESS:-0,1,2,4,8}" \
  --steps "${PIGRPO_STEPS:-60}" \
  --batch-prompts "${PIGRPO_BATCH_PROMPTS:-8}" \
  --group-size "${PIGRPO_GROUP_SIZE:-8}" \
  --seed "${PIGRPO_SEED:-42}" \
  --out-dir "${OUT}" 2>&1 | tee "${OUT}/summary.txt"
rc=${PIPESTATUS[0]}

if [ "${rc}" -eq 0 ]; then
  touch "${OUT}/DONE"
  echo "[exp4] DONE -> ${OUT}/staleness_sweep.csv"
else
  echo "[exp4] FAILED rc=${rc}" >&2
fi
exit "${rc}"
