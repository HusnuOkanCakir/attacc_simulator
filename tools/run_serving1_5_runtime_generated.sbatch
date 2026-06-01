#!/bin/bash
#SBATCH -J serving15_rt
#SBATCH -p high_latency
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH -t 24:00:00
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/%x_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/%x_%j.err

set -euo pipefail

ROOT_DIR="/mnt/galactica/ocakir/attacc_simulator"
LOG_DIR="/mnt/galactica/ocakir/logs"
TMP_ROOT="/mnt/galactica/ocakir/tmp"

mkdir -p "${LOG_DIR}"
mkdir -p "${TMP_ROOT}"
mkdir -p "${ROOT_DIR}/cluster_outputs/online_serving_runs"

cd "${ROOT_DIR}"

source /mnt/galactica/ocakir/miniforge3/etc/profile.d/conda.sh
conda activate attacc

export TMPDIR="${TMP_ROOT}"
export TMP="${TMP_ROOT}"
export TEMP="${TMP_ROOT}"

RUN_LABEL="${RUN_LABEL:-azure_conv_first50_serving15_bs8}"
REQUESTS_CSV="${REQUESTS_CSV:-cluster_outputs/azure/AzureLLMInferenceTrace_conv_first50.csv}"
BATCH="${BATCH:-8}"
MAX_PREFILL_BATCH_SIZE="${MAX_PREFILL_BATCH_SIZE:-8}"
MAX_DECODE_BATCH_SIZE="${MAX_DECODE_BATCH_SIZE:-8}"
HYBRID_COMMAND_SOURCE="${HYBRID_COMMAND_SOURCE:-runtime_generated}"
ONLINE_FRONTEND="${ONLINE_FRONTEND:-serving1_5}"
REBUILD=0

echo "Active conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo "hostname: $(hostname)"
echo "python: $(python -V 2>&1)"
echo "tmpdir: ${TMPDIR}"
echo "run_label: ${RUN_LABEL}"
echo "requests_csv: ${REQUESTS_CSV}"
echo "batch: ${BATCH}"
echo "frontend: ${ONLINE_FRONTEND}"
echo "hybrid_command_source: ${HYBRID_COMMAND_SOURCE}"

REBUILD="${REBUILD}" \
ONLINE_FRONTEND="${ONLINE_FRONTEND}" \
HYBRID_COMMAND_SOURCE="${HYBRID_COMMAND_SOURCE}" \
RUN_LABEL="${RUN_LABEL}" \
REQUESTS_CSV="${REQUESTS_CSV}" \
BATCH="${BATCH}" \
MAX_PREFILL_BATCH_SIZE="${MAX_PREFILL_BATCH_SIZE}" \
MAX_DECODE_BATCH_SIZE="${MAX_DECODE_BATCH_SIZE}" \
bash tools/run_realtime_online_serving_case.sh
