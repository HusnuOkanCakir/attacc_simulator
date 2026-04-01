#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

TRACE_PATH="${TRACE_PATH:-cluster_outputs/trace_cache/AzureLLMInferenceTrace_code.csv}"
GPU_MODEL="${GPU_MODEL:-cluster_outputs/cost_models_full_energy/gpu_only.pkl}"
LPDDR_MODEL="${LPDDR_MODEL:-cluster_outputs/cost_models_full_energy/lpddr5_pim_bank.pkl}"

LIMIT="${LIMIT:-50}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-0.25}"
MAX_DECODE_BATCH_SIZE="${MAX_DECODE_BATCH_SIZE:-4}"
ROUTE_LOAD_BALANCE_MS_PER_ACTIVE_REQUEST="${ROUTE_LOAD_BALANCE_MS_PER_ACTIVE_REQUEST:-0}"

ENERGY_LATENCY_GUARD_MS="${ENERGY_LATENCY_GUARD_MS:-50}"
REQUESTS_CSV="${REQUESTS_CSV:-cluster_outputs/replay_requests_azure_route_deltas_guarded_energy.csv}"
SUMMARY_JSON="${SUMMARY_JSON:-cluster_outputs/replay_summary_azure_route_deltas_guarded_energy.json}"
DEBUG_EVENTS_CSV="${DEBUG_EVENTS_CSV:-cluster_outputs/debug_events_azure_route_deltas_guarded_energy.csv}"
PLOTS_DIR="${PLOTS_DIR:-cluster_outputs/azure_route_delta_plots_guarded_energy}"
PREFIX="${PREFIX:-azure_route_delta_guarded_energy}"

PLOT_MIN_REQUEST_ID="${PLOT_MIN_REQUEST_ID:-0}"
PLOT_MAX_REQUEST_ID="${PLOT_MAX_REQUEST_ID:-$((LIMIT - 1))}"

TIMELINE_WIDTH_IN="${TIMELINE_WIDTH_IN:-18}"
TIMELINE_IN_PER_REQUEST="${TIMELINE_IN_PER_REQUEST:-0.08}"
TIMELINE_DPI="${TIMELINE_DPI:-220}"
TIMELINE_MAX_HEIGHT_IN="${TIMELINE_MAX_HEIGHT_IN:-60}"

mkdir -p "$(dirname "${REQUESTS_CSV}")" "$(dirname "${SUMMARY_JSON}")" "$(dirname "${DEBUG_EVENTS_CSV}")" "${PLOTS_DIR}"

echo "Running Azure route-delta guarded-energy case"
echo "  trace=${TRACE_PATH}"
echo "  limit=${LIMIT} arrival_scale=${ARRIVAL_SCALE}"
echo "  energy_latency_guard_ms=${ENERGY_LATENCY_GUARD_MS}"
echo "  route_load_balance_ms_per_active_request=${ROUTE_LOAD_BALANCE_MS_PER_ACTIVE_REQUEST}"
echo "  requests_csv=${REQUESTS_CSV}"
echo "  summary_json=${SUMMARY_JSON}"
echo "  debug_events_csv=${DEBUG_EVENTS_CSV}"
echo "  plots_dir=${PLOTS_DIR}"

"${PYTHON_BIN}" tools/run_trace_replay.py \
  --azure-trace "${TRACE_PATH}" \
  --cost-model ml \
  --route-policy latency_guarded_energy \
  --energy-latency-guard-ms "${ENERGY_LATENCY_GUARD_MS}" \
  --limit "${LIMIT}" \
  --arrival-time-scale "${ARRIVAL_SCALE}" \
  --ml-profile "gpu_only=${GPU_MODEL}" \
  --ml-profile "lpddr5_pim_bank=${LPDDR_MODEL}" \
  --routes gpu_only lpddr5_pim_bank \
  --unsupported-policy clip \
  --local-scheduling-policy prefill_priority_fcfs_decode \
  --fcfs-decode-prediction-mode incremental \
  --enable-decode-batching \
  --max-decode-batch-size "${MAX_DECODE_BATCH_SIZE}" \
  --route-load-balance-ms-per-active-request "${ROUTE_LOAD_BALANCE_MS_PER_ACTIVE_REQUEST}" \
  --requests-csv "${REQUESTS_CSV}" \
  --summary-json "${SUMMARY_JSON}" \
  --debug-events-csv "${DEBUG_EVENTS_CSV}"

"${PYTHON_BIN}" tools/plot_replay_results.py \
  --requests-csv "${REQUESTS_CSV}" \
  --summary-json "${SUMMARY_JSON}" \
  --debug-events-csv "${DEBUG_EVENTS_CSV}" \
  --prediction-history-min-request-id "${PLOT_MIN_REQUEST_ID}" \
  --prediction-history-max-request-id "${PLOT_MAX_REQUEST_ID}" \
  --out-dir "${PLOTS_DIR}" \
  --prefix "${PREFIX}"

"${PYTHON_BIN}" tools/plot_debug_admission.py \
  --events-csv "${DEBUG_EVENTS_CSV}" \
  --requests-csv "${REQUESTS_CSV}" \
  --out-dir "${PLOTS_DIR}" \
  --prefix "${PREFIX}" \
  --timeline-width-in "${TIMELINE_WIDTH_IN}" \
  --timeline-in-per-request "${TIMELINE_IN_PER_REQUEST}" \
  --timeline-dpi "${TIMELINE_DPI}" \
  --timeline-max-height-in "${TIMELINE_MAX_HEIGHT_IN}"

echo "Wrote replay outputs:"
echo "  ${REQUESTS_CSV}"
echo "  ${SUMMARY_JSON}"
echo "  ${DEBUG_EVENTS_CSV}"
echo "Wrote plots to:"
echo "  ${PLOTS_DIR}"
