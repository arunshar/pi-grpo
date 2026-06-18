#!/bin/bash
#SBATCH --job-name=pigrpo-llm-build
#SBATCH --account=shekhars
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# EXP-5: build the LLM-training apptainer image from docker/pigrpo_llm.def.
# Runs on a compute node (the login node has no outbound internet; a compute node
# does, verified by the apptainer probe). Writes /scratch.global/$USER/pigrpo_llm.sif
# plus a DONE / FAILED marker so a dependent model-download job can gate on it.
#
# Submit from the repo root:
#   sbatch scripts/build_llm_container.sbatch

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEF="${REPO_ROOT}/docker/pigrpo_llm.def"
OUT="/scratch.global/${USER}/pigrpo_llm.sif"
MARK_DIR="/scratch.global/${USER}/pigrpo_build"
mkdir -p "${MARK_DIR}"
rm -f "${MARK_DIR}/DONE" "${MARK_DIR}/FAILED"

export APPTAINER_CACHEDIR="/scratch.global/${USER}/apptainer_cache"
export APPTAINER_TMPDIR="/scratch.global/${USER}/apptainer_tmp"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

echo "[build] node=$(hostname) def=${DEF} out=${OUT}"
apptainer --version
df -h /scratch.global | tail -1

# Build to a temp name then atomically move, so a half-built sif is never seen as done.
TMP_SIF="${OUT}.partial.${SLURM_JOB_ID:-local}"
set +e
apptainer build --fakeroot "${TMP_SIF}" "${DEF}"
RC=$?
set -e

if [ "${RC}" -eq 0 ] && [ -f "${TMP_SIF}" ]; then
  mv -f "${TMP_SIF}" "${OUT}"
  echo "[build] verifying LLM stack imports..."
  if apptainer exec "${OUT}" python3 -c "import vllm, transformers, peft, accelerate, ray; print('llm stack OK')"; then
    echo "OK ${OUT} $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${MARK_DIR}/DONE"
    echo "[build] DONE -> ${OUT}"
  else
    echo "FAILED import-check $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${MARK_DIR}/FAILED"
    echo "[build] built but import-check failed" >&2; RC=3
  fi
else
  rm -f "${TMP_SIF}"
  echo "FAILED build rc=${RC} $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${MARK_DIR}/FAILED"
  echo "[build] build failed rc=${RC}" >&2
fi
exit "${RC}"
