#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR_ROOT="${OUT_DIR:-cluster_outputs/refactor_regression}"
PHASE="${PHASE:-before}"
OUT_DIR="${OUT_DIR_ROOT}/${PHASE}"
TRACE_CACHE="${TRACE_CACHE:-cluster_outputs/trace_cache/AzureLLMInferenceTrace_code.csv}"
GPU_MODEL="${GPU_MODEL:-cluster_outputs/cost_models_full_energy/gpu_only.pkl}"
LPDDR_MODEL="${LPDDR_MODEL:-cluster_outputs/cost_models_full_energy/lpddr5_pim_bank.pkl}"
GPU_TABLE="${GPU_TABLE:-cluster_outputs/cost_tables_full_energy/gpu_only.csv}"
LPDDR_TABLE="${LPDDR_TABLE:-cluster_outputs/cost_tables_full_energy/lpddr5_pim_bank.csv}"
CONDA_ENV="${CONDA_ENV:-lerobot}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${OUT_DIR}"

if [[ -n "${CONDA_ENV}" ]]; then
  RUNNER=(conda run -n "${CONDA_ENV}" python)
else
  RUNNER=("${PYTHON_BIN}")
fi

run_case() {
  local label="$1"
  shift
  echo "Running regression case: ${label}"
  "${RUNNER[@]}" tools/run_trace_replay.py "$@" \
    --requests-csv "${OUT_DIR}/${label}_requests.csv" \
    --summary-json "${OUT_DIR}/${label}_summary.json" \
    --debug-events-csv "${OUT_DIR}/${label}_debug.csv"
}

run_case det_bypass8_table_guarded \
  --azure-trace cluster_outputs/debug_trace_deterministic_bypass8.csv \
  --cost-model table \
  --profile "gpu_only=${GPU_TABLE}" \
  --profile "lpddr5_pim_bank=${LPDDR_TABLE}" \
  --route-policy latency_guarded_energy \
  --energy-latency-guard-ms 5 \
  --slo-e2e-ms 350 \
  --slo-tbt-ms 3 \
  --limit 8 \
  --arrival-time-scale 1.0 \
  --routes gpu_only lpddr5_pim_bank \
  --unsupported-policy clip \
  --local-scheduling-policy prefill_priority_fcfs_decode \
  --fcfs-decode-prediction-mode incremental \
  --enable-decode-batching \
  --max-decode-batch-size 4 \
  --enable-slo-guarded-admission \
  --admission-max-harmed-requests 0 \
  --admission-max-total-harm-ms 0 \
  --admission-max-single-harm-ms 0 \
  --admission-max-own-miss-ms 0 \
  --admission-max-bypass-count 5 \
  --admission-max-wait-ms 150 \
  --admission-aging-harm-ms-per-ms 0.01 \
  --admission-aging-harmed-requests-per-ms 0.001 \
  --admission-retry-interval-ms 10

run_case det_heavy_sparse16_ml_guarded_bs16 \
  --azure-trace cluster_outputs/debug_trace_deterministic_heavy_sparse16.csv \
  --cost-model ml \
  --route-policy latency_guarded_energy \
  --energy-latency-guard-ms 5 \
  --slo-e2e-ms 5000 \
  --slo-tbt-ms 200 \
  --limit 16 \
  --arrival-time-scale 1.0 \
  --ml-profile "gpu_only=${GPU_MODEL}" \
  --ml-profile "lpddr5_pim_bank=${LPDDR_MODEL}" \
  --routes gpu_only lpddr5_pim_bank \
  --unsupported-policy clip \
  --local-scheduling-policy prefill_priority_fcfs_decode \
  --fcfs-decode-prediction-mode incremental \
  --enable-decode-batching \
  --max-decode-batch-size 16 \
  --enable-slo-guarded-admission \
  --admission-max-harmed-requests 1 \
  --admission-max-total-harm-ms 25 \
  --admission-max-single-harm-ms 10 \
  --admission-max-own-miss-ms 25 \
  --admission-max-bypass-count 2 \
  --admission-max-wait-ms 100 \
  --admission-aging-harm-ms-per-ms 0.02 \
  --admission-aging-harmed-requests-per-ms 0.001 \
  --admission-retry-interval-ms 25

run_case azure_l10_min_finish_ml \
  --azure-trace "${TRACE_CACHE}" \
  --cost-model ml \
  --route-policy min_finish \
  --limit 10 \
  --arrival-time-scale 1.0 \
  --ml-profile "gpu_only=${GPU_MODEL}" \
  --ml-profile "lpddr5_pim_bank=${LPDDR_MODEL}" \
  --routes gpu_only lpddr5_pim_bank \
  --unsupported-policy clip \
  --local-scheduling-policy prefill_priority_fcfs_decode \
  --fcfs-decode-prediction-mode incremental \
  --enable-decode-batching \
  --max-decode-batch-size 4

run_case azure_l10_energy_ml \
  --azure-trace "${TRACE_CACHE}" \
  --cost-model ml \
  --route-policy latency_guarded_energy \
  --energy-latency-guard-ms 5 \
  --limit 10 \
  --arrival-time-scale 1.0 \
  --ml-profile "gpu_only=${GPU_MODEL}" \
  --ml-profile "lpddr5_pim_bank=${LPDDR_MODEL}" \
  --routes gpu_only lpddr5_pim_bank \
  --unsupported-policy clip \
  --local-scheduling-policy prefill_priority_fcfs_decode \
  --fcfs-decode-prediction-mode incremental \
  --enable-decode-batching \
  --max-decode-batch-size 4

run_case azure_l100_energy_guarded_ml \
  --azure-trace "${TRACE_CACHE}" \
  --cost-model ml \
  --route-policy latency_guarded_energy \
  --energy-latency-guard-ms 5 \
  --slo-e2e-ms 5000 \
  --slo-tbt-ms 200 \
  --limit 100 \
  --arrival-time-scale 1.0 \
  --ml-profile "gpu_only=${GPU_MODEL}" \
  --ml-profile "lpddr5_pim_bank=${LPDDR_MODEL}" \
  --routes gpu_only lpddr5_pim_bank \
  --unsupported-policy clip \
  --local-scheduling-policy prefill_priority_fcfs_decode \
  --fcfs-decode-prediction-mode incremental \
  --enable-decode-batching \
  --max-decode-batch-size 4 \
  --enable-slo-guarded-admission \
  --admission-max-harmed-requests 1 \
  --admission-max-total-harm-ms 25 \
  --admission-max-single-harm-ms 10 \
  --admission-max-own-miss-ms 25 \
  --admission-max-bypass-count 2 \
  --admission-max-wait-ms 100 \
  --admission-aging-harm-ms-per-ms 0.02 \
  --admission-aging-harmed-requests-per-ms 0.001 \
  --admission-retry-interval-ms 25

echo "Wrote regression outputs to ${OUT_DIR}"
