#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-cluster_outputs/azure_energy_guard_sweep_noguard_v4}"
PLOTS_DIR="${PLOTS_DIR:-cluster_outputs/azure_energy_guard_sweep_noguard_plots_v4}"
TRACE_PATH="${TRACE_PATH:-cluster_outputs/trace_cache/AzureLLMInferenceTrace_code.csv}"
COST_MODEL="${COST_MODEL:-ml}"
GPU_MODEL="${GPU_MODEL:-cluster_outputs/cost_models_full_energy/gpu_only.pkl}"
LPDDR_MODEL="${LPDDR_MODEL:-cluster_outputs/cost_models_full_energy/lpddr5_pim_bank.pkl}"
GPU_TABLE="${GPU_TABLE:-cluster_outputs/cost_tables_full_energy/gpu_only.csv}"
LPDDR_TABLE="${LPDDR_TABLE:-cluster_outputs/cost_tables_full_energy/lpddr5_pim_bank.csv}"
ENABLE_SLO_GUARDED_ADMISSION="${ENABLE_SLO_GUARDED_ADMISSION:-1}"

SLO_E2E_MS="${SLO_E2E_MS:-900}"
SLO_TBT_MS="${SLO_TBT_MS:-3}"
SET_REQUEST_SLOS="${SET_REQUEST_SLOS:-1}"
LIMIT="${LIMIT:-50}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-0.25}"
MAX_DECODE_BATCH_SIZE="${MAX_DECODE_BATCH_SIZE:-4}"
PLOT_MIN_REQUEST_ID="${PLOT_MIN_REQUEST_ID:-0}"
PLOT_MAX_REQUEST_ID="${PLOT_MAX_REQUEST_ID:-$((LIMIT - 1))}"

ADMISSION_MAX_HARMED_REQUESTS="${ADMISSION_MAX_HARMED_REQUESTS:-999}"
ADMISSION_MAX_TOTAL_HARM_MS="${ADMISSION_MAX_TOTAL_HARM_MS:-50}"
ADMISSION_MAX_SINGLE_HARM_MS="${ADMISSION_MAX_SINGLE_HARM_MS:-99999}"
ADMISSION_CHECK_OWN_SLO="${ADMISSION_CHECK_OWN_SLO:-0}"
ADMISSION_MAX_OWN_MISS_MS="${ADMISSION_MAX_OWN_MISS_MS:-0}"
ADMISSION_MAX_BYPASS_COUNT="${ADMISSION_MAX_BYPASS_COUNT:-5}"
ADMISSION_MAX_WAIT_MS="${ADMISSION_MAX_WAIT_MS:-150}"
ADMISSION_AGING_HARM_MS_PER_MS="${ADMISSION_AGING_HARM_MS_PER_MS:-0.01}"
ADMISSION_AGING_HARMED_REQUESTS_PER_MS="${ADMISSION_AGING_HARMED_REQUESTS_PER_MS:-0.001}"
ADMISSION_RETRY_INTERVAL_MS="${ADMISSION_RETRY_INTERVAL_MS:-10}"

mkdir -p "${OUT_DIR}" "${PLOTS_DIR}"

if (($# > 0)); then
  GUARD_VALUES=("$@")
else
  GUARD_VALUES=(5 10 20 30 40 50 75 100 150 200)
fi

for guard_ms in "${GUARD_VALUES[@]}"; do
  guard_tag="${guard_ms//./p}"
  summary_path="${OUT_DIR}/replay_summary_guard${guard_tag}.json"
  requests_path="${OUT_DIR}/replay_requests_guard${guard_tag}.csv"
  debug_path="${OUT_DIR}/debug_events_guard${guard_tag}.csv"
  prefix="guard${guard_tag}"

  echo "Running Azure energy-guard debug case: energy_latency_guard_ms=${guard_ms} cost_model=${COST_MODEL} guarded_admission=${ENABLE_SLO_GUARDED_ADMISSION} request_slos=${SET_REQUEST_SLOS} admission_check_own_slo=${ADMISSION_CHECK_OWN_SLO}"

  cmd=(
    "${PYTHON_BIN}" tools/run_trace_replay.py
    --azure-trace "${TRACE_PATH}"
    --cost-model "${COST_MODEL}"
    --route-policy latency_guarded_energy
    --energy-latency-guard-ms "${guard_ms}"
    --limit "${LIMIT}"
    --arrival-time-scale "${ARRIVAL_SCALE}"
    --routes gpu_only lpddr5_pim_bank
    --unsupported-policy clip
    --local-scheduling-policy prefill_priority_fcfs_decode
    --fcfs-decode-prediction-mode incremental
    --enable-decode-batching
    --max-decode-batch-size "${MAX_DECODE_BATCH_SIZE}"
    --requests-csv "${requests_path}"
    --summary-json "${summary_path}"
    --debug-events-csv "${debug_path}"
  )

  if [[ "${SET_REQUEST_SLOS}" == "1" ]]; then
    cmd+=(
      --slo-e2e-ms "${SLO_E2E_MS}"
      --slo-tbt-ms "${SLO_TBT_MS}"
    )
  fi

  if [[ "${ENABLE_SLO_GUARDED_ADMISSION}" == "1" ]]; then
    cmd+=(
      --enable-slo-guarded-admission
      --admission-max-harmed-requests "${ADMISSION_MAX_HARMED_REQUESTS}"
      --admission-max-total-harm-ms "${ADMISSION_MAX_TOTAL_HARM_MS}"
      --admission-max-single-harm-ms "${ADMISSION_MAX_SINGLE_HARM_MS}"
      --admission-max-bypass-count "${ADMISSION_MAX_BYPASS_COUNT}"
      --admission-max-wait-ms "${ADMISSION_MAX_WAIT_MS}"
      --admission-aging-harm-ms-per-ms "${ADMISSION_AGING_HARM_MS_PER_MS}"
      --admission-aging-harmed-requests-per-ms "${ADMISSION_AGING_HARMED_REQUESTS_PER_MS}"
      --admission-retry-interval-ms "${ADMISSION_RETRY_INTERVAL_MS}"
    )
  fi

  if [[ "${ADMISSION_CHECK_OWN_SLO}" == "1" ]]; then
    cmd+=(
      --admission-check-own-slo
      --admission-max-own-miss-ms "${ADMISSION_MAX_OWN_MISS_MS}"
    )
  fi

  if [[ "${COST_MODEL}" == "ml" ]]; then
    cmd+=(
      --ml-profile "gpu_only=${GPU_MODEL}"
      --ml-profile "lpddr5_pim_bank=${LPDDR_MODEL}"
    )
  else
    cmd+=(
      --profile "gpu_only=${GPU_TABLE}"
      --profile "lpddr5_pim_bank=${LPDDR_TABLE}"
    )
  fi

  "${cmd[@]}"

  "${PYTHON_BIN}" tools/plot_replay_results.py \
    --requests-csv "${requests_path}" \
    --summary-json "${summary_path}" \
    --debug-events-csv "${debug_path}" \
    --prediction-history-min-request-id "${PLOT_MIN_REQUEST_ID}" \
    --prediction-history-max-request-id "${PLOT_MAX_REQUEST_ID}" \
    --out-dir "${PLOTS_DIR}" \
    --prefix "${prefix}"

  "${PYTHON_BIN}" tools/plot_debug_admission.py \
    --events-csv "${debug_path}" \
    --requests-csv "${requests_path}" \
    --out-dir "${PLOTS_DIR}" \
    --prefix "${prefix}"
done

"${PYTHON_BIN}" tools/plot_energy_guard_sweep.py \
  --summary-dir "${OUT_DIR}" \
  --out-dir "${PLOTS_DIR}" \
  --prefix azure_energy_guard_sweep_noguard

OUT_DIR_FOR_PY="${OUT_DIR}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

print("guard_ms,gpu_only,lpddr5_pim_bank,total_energy_nj,decode_energy_per_token_nj,bypassed_count,held_count,best_effort_count")
for path in sorted(Path(os.environ["OUT_DIR_FOR_PY"]).glob("replay_summary_guard*.json")):
    tag = path.stem.split("guard", 1)[1]
    guard_ms = float(tag.replace("p", "."))
    with path.open() as f:
        d = json.load(f)
    route_counts = d.get("route_counts", {})
    admission = d.get("admission", {})
    print(
        f"{guard_ms},"
        f"{route_counts.get('gpu_only', 0)},"
        f"{route_counts.get('lpddr5_pim_bank', 0)},"
        f"{d.get('energy_nj', {}).get('total')},"
        f"{d.get('decode_energy_nj_per_decode_token')},"
        f"{admission.get('bypassed_count', 0)},"
        f"{admission.get('held_count', 0)},"
        f"{admission.get('best_effort_count', 0)}"
    )
PY

echo "Wrote sweep summaries to ${OUT_DIR}"
echo "Wrote sweep plots to ${PLOTS_DIR}"
