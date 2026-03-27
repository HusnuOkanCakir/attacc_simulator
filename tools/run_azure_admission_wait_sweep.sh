#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-cluster_outputs/azure_admission_wait_sweep}"
PLOTS_DIR="${PLOTS_DIR:-cluster_outputs/azure_admission_wait_sweep_plots}"
TRACE_PATH="${TRACE_PATH:-cluster_outputs/trace_cache/AzureLLMInferenceTrace_code.csv}"
COST_MODEL="${COST_MODEL:-ml}"
GPU_MODEL="${GPU_MODEL:-cluster_outputs/cost_models_full_energy/gpu_only.pkl}"
LPDDR_MODEL="${LPDDR_MODEL:-cluster_outputs/cost_models_full_energy/lpddr5_pim_bank.pkl}"
GPU_TABLE="${GPU_TABLE:-cluster_outputs/cost_tables_full_energy/gpu_only.csv}"
LPDDR_TABLE="${LPDDR_TABLE:-cluster_outputs/cost_tables_full_energy/lpddr5_pim_bank.csv}"

# Default to the same route-policy family, but with zero energy guard so the
# sweep isolates admission_max_wait_ms rather than energy-aware route changes.
ROUTE_POLICY="${ROUTE_POLICY:-latency_guarded_energy}"
ENERGY_LATENCY_GUARD_MS="${ENERGY_LATENCY_GUARD_MS:-0}"

SLO_E2E_MS="${SLO_E2E_MS:-900}"
SLO_TBT_MS="${SLO_TBT_MS:-3}"
LIMIT="${LIMIT:-50}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-0.25}"
MAX_DECODE_BATCH_SIZE="${MAX_DECODE_BATCH_SIZE:-4}"
PLOT_MIN_REQUEST_ID="${PLOT_MIN_REQUEST_ID:-0}"
PLOT_MAX_REQUEST_ID="${PLOT_MAX_REQUEST_ID:-$((LIMIT - 1))}"

ADMISSION_MAX_HARMED_REQUESTS="${ADMISSION_MAX_HARMED_REQUESTS:-0}"
ADMISSION_MAX_TOTAL_HARM_MS="${ADMISSION_MAX_TOTAL_HARM_MS:-0}"
ADMISSION_MAX_SINGLE_HARM_MS="${ADMISSION_MAX_SINGLE_HARM_MS:-0}"
# Default to others-only guarded admission. Candidate self-SLO checks stay off
# unless explicitly requested.
ADMISSION_CHECK_OWN_SLO="${ADMISSION_CHECK_OWN_SLO:-0}"
ADMISSION_MAX_OWN_MISS_MS="${ADMISSION_MAX_OWN_MISS_MS:-0}"
ADMISSION_MAX_BYPASS_COUNT="${ADMISSION_MAX_BYPASS_COUNT:-5}"
ADMISSION_AGING_HARM_MS_PER_MS="${ADMISSION_AGING_HARM_MS_PER_MS:-0.01}"
ADMISSION_AGING_HARMED_REQUESTS_PER_MS="${ADMISSION_AGING_HARMED_REQUESTS_PER_MS:-0.001}"
ADMISSION_RETRY_INTERVAL_MS="${ADMISSION_RETRY_INTERVAL_MS:-10}"

mkdir -p "${OUT_DIR}" "${PLOTS_DIR}"

if (($# > 0)); then
  WAIT_VALUES=("$@")
else
  WAIT_VALUES=(25 50 75 100 150 200 300)
fi

for wait_ms in "${WAIT_VALUES[@]}"; do
  wait_tag="${wait_ms//./p}"
  summary_path="${OUT_DIR}/replay_summary_wait${wait_tag}.json"
  requests_path="${OUT_DIR}/replay_requests_wait${wait_tag}.csv"
  debug_path="${OUT_DIR}/debug_events_wait${wait_tag}.csv"
  prefix="wait${wait_tag}"

  echo "Running Azure admission-wait case: admission_max_wait_ms=${wait_ms} route_policy=${ROUTE_POLICY} admission_check_own_slo=${ADMISSION_CHECK_OWN_SLO}"

  cmd=(
    "${PYTHON_BIN}" tools/run_trace_replay.py
    --azure-trace "${TRACE_PATH}"
    --cost-model "${COST_MODEL}"
    --route-policy "${ROUTE_POLICY}"
    --slo-e2e-ms "${SLO_E2E_MS}"
    --slo-tbt-ms "${SLO_TBT_MS}"
    --limit "${LIMIT}"
    --arrival-time-scale "${ARRIVAL_SCALE}"
    --routes gpu_only lpddr5_pim_bank
    --unsupported-policy clip
    --local-scheduling-policy prefill_priority_fcfs_decode
    --fcfs-decode-prediction-mode incremental
    --enable-decode-batching
    --max-decode-batch-size "${MAX_DECODE_BATCH_SIZE}"
    --enable-slo-guarded-admission
    --admission-max-harmed-requests "${ADMISSION_MAX_HARMED_REQUESTS}"
    --admission-max-total-harm-ms "${ADMISSION_MAX_TOTAL_HARM_MS}"
    --admission-max-single-harm-ms "${ADMISSION_MAX_SINGLE_HARM_MS}"
    --admission-max-bypass-count "${ADMISSION_MAX_BYPASS_COUNT}"
    --admission-max-wait-ms "${wait_ms}"
    --admission-aging-harm-ms-per-ms "${ADMISSION_AGING_HARM_MS_PER_MS}"
    --admission-aging-harmed-requests-per-ms "${ADMISSION_AGING_HARMED_REQUESTS_PER_MS}"
    --admission-retry-interval-ms "${ADMISSION_RETRY_INTERVAL_MS}"
    --requests-csv "${requests_path}"
    --summary-json "${summary_path}"
    --debug-events-csv "${debug_path}"
  )

  if [[ "${ROUTE_POLICY}" == "latency_guarded_energy" ]]; then
    cmd+=(--energy-latency-guard-ms "${ENERGY_LATENCY_GUARD_MS}")
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

"${PYTHON_BIN}" tools/plot_admission_wait_sweep.py \
  --summary-dir "${OUT_DIR}" \
  --out-dir "${PLOTS_DIR}" \
  --prefix azure_admission_wait_sweep

OUT_DIR_FOR_PY="${OUT_DIR}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

print("wait_ms,held_count,bypassed_count,best_effort_count,blocked_count,queue_max_len,queue_mean_wait_ms,ttft_p95_ms,e2e_p95_ms,tbt_p95_ms,throughput_tokps,route_counts")
for path in sorted(Path(os.environ["OUT_DIR_FOR_PY"]).glob("replay_summary_wait*.json")):
    tag = path.stem.split("wait", 1)[1]
    wait_ms = float(tag.replace("p", "."))
    with path.open() as f:
        d = json.load(f)
    admission = d.get("admission", {})
    queue = d.get("admission_queue", {})
    print(
        f"{wait_ms},"
        f"{admission.get('held_count', 0)},"
        f"{admission.get('bypassed_count', 0)},"
        f"{admission.get('best_effort_count', 0)},"
        f"{admission.get('blocked_count', 0)},"
        f"{queue.get('max_len', 0)},"
        f"{queue.get('mean_wait_ms', 0)},"
        f"{d.get('ttft_ms', {}).get('p95')},"
        f"{d.get('e2e_ms', {}).get('p95')},"
        f"{d.get('tbt_ms', {}).get('p95')},"
        f"{d.get('throughput_tokps')},"
        f"{d.get('route_counts')}"
    )
PY

echo "Wrote sweep summaries to ${OUT_DIR}"
echo "Wrote sweep plots to ${PLOTS_DIR}"
