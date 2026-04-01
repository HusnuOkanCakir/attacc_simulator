#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

TRACE_PATH="${TRACE_PATH:-cluster_outputs/debug_trace_deterministic_fixed_prefill_varied_long_decode16.csv}"
GPU_MODEL="${GPU_MODEL:-cluster_outputs/cost_models_full_energy/gpu_only.pkl}"
LPDDR_MODEL="${LPDDR_MODEL:-cluster_outputs/cost_models_full_energy/lpddr5_pim_bank.pkl}"

OUT_DIR="${OUT_DIR:-cluster_outputs/det_fixed_prefill_varied_long_decode16_energy_guard_sweep_lb25}"
PLOTS_DIR="${PLOTS_DIR:-cluster_outputs/det_fixed_prefill_varied_long_decode16_energy_guard_sweep_lb25_plots}"
PREFIX="${PREFIX:-det_fixed_prefill_varied_long_decode16_energy_guard_sweep_lb25}"

LIMIT="${LIMIT:-16}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-1.0}"
MAX_DECODE_BATCH_SIZE="${MAX_DECODE_BATCH_SIZE:-4}"
SHARE_GPU_PREFILL_ACROSS_ROUTES="${SHARE_GPU_PREFILL_ACROSS_ROUTES:-1}"
ROUTE_LOAD_BALANCE_MS_PER_ACTIVE_REQUEST="${ROUTE_LOAD_BALANCE_MS_PER_ACTIVE_REQUEST:-25}"

PLOT_MIN_REQUEST_ID="${PLOT_MIN_REQUEST_ID:-0}"
PLOT_MAX_REQUEST_ID="${PLOT_MAX_REQUEST_ID:-$((LIMIT - 1))}"

TIMELINE_WIDTH_IN="${TIMELINE_WIDTH_IN:-18}"
TIMELINE_IN_PER_REQUEST="${TIMELINE_IN_PER_REQUEST:-0.08}"
TIMELINE_DPI="${TIMELINE_DPI:-220}"
TIMELINE_MAX_HEIGHT_IN="${TIMELINE_MAX_HEIGHT_IN:-60}"

mkdir -p "${OUT_DIR}" "${PLOTS_DIR}"

if (($# > 0)); then
  GUARD_VALUES=("$@")
else
  GUARD_VALUES=(0 20 50 100 200 300)
fi

for guard_ms in "${GUARD_VALUES[@]}"; do
  guard_tag="${guard_ms//./p}"
  summary_path="${OUT_DIR}/replay_summary_guard${guard_tag}.json"
  requests_path="${OUT_DIR}/replay_requests_guard${guard_tag}.csv"
  debug_path="${OUT_DIR}/debug_events_guard${guard_tag}.csv"
  case_prefix="guard${guard_tag}"

  echo "Running fixed-band energy-aware case: guard_ms=${guard_ms} share_gpu_prefill=${SHARE_GPU_PREFILL_ACROSS_ROUTES} route_load_balance_ms_per_active_request=${ROUTE_LOAD_BALANCE_MS_PER_ACTIVE_REQUEST}"

  cmd=(
    "${PYTHON_BIN}" tools/run_trace_replay.py
    --azure-trace "${TRACE_PATH}"
    --cost-model ml
    --route-policy latency_guarded_energy
    --energy-latency-guard-ms "${guard_ms}"
    --limit "${LIMIT}"
    --arrival-time-scale "${ARRIVAL_SCALE}"
    --ml-profile "gpu_only=${GPU_MODEL}"
    --ml-profile "lpddr5_pim_bank=${LPDDR_MODEL}"
    --routes gpu_only lpddr5_pim_bank
    --unsupported-policy clip
    --local-scheduling-policy prefill_priority_fcfs_decode
    --fcfs-decode-prediction-mode incremental
    --enable-decode-batching
    --max-decode-batch-size "${MAX_DECODE_BATCH_SIZE}"
    --route-load-balance-ms-per-active-request "${ROUTE_LOAD_BALANCE_MS_PER_ACTIVE_REQUEST}"
    --requests-csv "${requests_path}"
    --summary-json "${summary_path}"
    --debug-events-csv "${debug_path}"
  )

  if [[ "${SHARE_GPU_PREFILL_ACROSS_ROUTES}" == "1" ]]; then
    cmd+=(--share-gpu-prefill-across-routes)
  fi

  "${cmd[@]}"

  "${PYTHON_BIN}" tools/plot_replay_results.py \
    --requests-csv "${requests_path}" \
    --summary-json "${summary_path}" \
    --debug-events-csv "${debug_path}" \
    --prediction-history-min-request-id "${PLOT_MIN_REQUEST_ID}" \
    --prediction-history-max-request-id "${PLOT_MAX_REQUEST_ID}" \
    --out-dir "${PLOTS_DIR}" \
    --prefix "${case_prefix}"

  "${PYTHON_BIN}" tools/plot_debug_admission.py \
    --events-csv "${debug_path}" \
    --requests-csv "${requests_path}" \
    --out-dir "${PLOTS_DIR}" \
    --prefix "${case_prefix}" \
    --timeline-width-in "${TIMELINE_WIDTH_IN}" \
    --timeline-in-per-request "${TIMELINE_IN_PER_REQUEST}" \
    --timeline-dpi "${TIMELINE_DPI}" \
    --timeline-max-height-in "${TIMELINE_MAX_HEIGHT_IN}"
done

"${PYTHON_BIN}" tools/plot_energy_guard_sweep.py \
  --summary-dir "${OUT_DIR}" \
  --out-dir "${PLOTS_DIR}" \
  --prefix "${PREFIX}"

OUT_DIR_FOR_PY="${OUT_DIR}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

print("guard_ms,gpu_only,lpddr5_pim_bank,total_energy_nj,decode_energy_per_token_nj,throughput_tokps")
for path in sorted(Path(os.environ["OUT_DIR_FOR_PY"]).glob("replay_summary_guard*.json")):
    tag = path.stem.split("guard", 1)[1]
    guard_ms = float(tag.replace("p", "."))
    with path.open() as f:
        data = json.load(f)
    route_counts = data.get("route_counts", {})
    print(
        f"{guard_ms},"
        f"{route_counts.get('gpu_only', 0)},"
        f"{route_counts.get('lpddr5_pim_bank', 0)},"
        f"{data.get('energy_nj', {}).get('total')},"
        f"{data.get('decode_energy_nj_per_decode_token')},"
        f"{data.get('throughput_tokps')}"
    )
PY

echo "Wrote sweep summaries to ${OUT_DIR}"
echo "Wrote sweep plots to ${PLOTS_DIR}"
