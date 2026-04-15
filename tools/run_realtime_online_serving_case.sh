#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-cluster_outputs/online_serving_runs}"
RUN_LABEL="${RUN_LABEL:-azure_conv_first50_runtime_generated_va_guarded_bs8}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_ROOT}/${TIMESTAMP}_${RUN_LABEL}"
PLOTS_DIR="${RUN_DIR}/plots"
ADMISSION_PLOTS_DIR="${PLOTS_DIR}/admission_debug"

SYSTEM="${SYSTEM:-dgx-attacc}"
GPU="${GPU:-A100a}"
NGPU="${NGPU:-8}"
MODEL="${MODEL:-PI0}"
PIM="${PIM:-bank}"
YAML_TARGET="${YAML_TARGET:-lpddr5-pim}"

REQUESTS_CSV="${REQUESTS_CSV:-cluster_outputs/azure/AzureLLMInferenceTrace_conv_first50.csv}"
TEMPLATE_DIR="${TEMPLATE_DIR:-ramulator2/trace_gen/templates_lpddr5_bank}"
COST_GPU_CSV="${COST_GPU_CSV:-cluster_outputs/cost_tables_full_energy/gpu_only.csv}"
COST_HYBRID_CSV="${COST_HYBRID_CSV:-cluster_outputs/cost_tables_full_energy/lpddr5_pim_bank.csv}"

ONLINE_FRONTEND="${ONLINE_FRONTEND:-realtime}"
HYBRID_COMMAND_SOURCE="${HYBRID_COMMAND_SOURCE:-runtime_generated}"
ROUTE_POLICY="${ROUTE_POLICY:-min_finish}"
UNSUPPORTED_POLICY="${UNSUPPORTED_POLICY:-clip}"
ARRIVAL_TIME_SCALE="${ARRIVAL_TIME_SCALE:-0.1}"
LIN_BUCKET="${LIN_BUCKET:-1}"
LOUT_BUCKET="${LOUT_BUCKET:-1}"
LOCAL_SCHEDULING_POLICY="${LOCAL_SCHEDULING_POLICY:-prefill_priority_fcfs_decode}"
FCFS_DECODE_PREDICTION_MODE="${FCFS_DECODE_PREDICTION_MODE:-incremental}"
ENABLE_SLO_GUARDED_ADMISSION="${ENABLE_SLO_GUARDED_ADMISSION:-1}"
SLO_E2E_MS="${SLO_E2E_MS:-900}"
SLO_TBT_MS="${SLO_TBT_MS:-200}"
ADMISSION_RETRY_INTERVAL_MS="${ADMISSION_RETRY_INTERVAL_MS:-10}"
ADMISSION_MAX_HARMED_REQUESTS="${ADMISSION_MAX_HARMED_REQUESTS:-2}"
ADMISSION_MAX_TOTAL_HARM_MS="${ADMISSION_MAX_TOTAL_HARM_MS:-50}"
ADMISSION_MAX_SINGLE_HARM_MS="${ADMISSION_MAX_SINGLE_HARM_MS:-20}"
ADMISSION_MAX_BYPASS_COUNT="${ADMISSION_MAX_BYPASS_COUNT:-5}"
ADMISSION_MAX_WAIT_MS="${ADMISSION_MAX_WAIT_MS:-150}"
ADMISSION_AGING_HARM_MS_PER_MS="${ADMISSION_AGING_HARM_MS_PER_MS:-0.01}"
ADMISSION_AGING_HARMED_REQUESTS_PER_MS="${ADMISSION_AGING_HARMED_REQUESTS_PER_MS:-0.001}"
ENERGY_LATENCY_GUARD_MS="${ENERGY_LATENCY_GUARD_MS:-20}"

BATCH="${BATCH:-8}"
ENABLE_PREFILL_BATCHING="${ENABLE_PREFILL_BATCHING:-1}"
MAX_PREFILL_BATCH_SIZE="${MAX_PREFILL_BATCH_SIZE:-${BATCH}}"
ENABLE_DECODE_BATCHING="${ENABLE_DECODE_BATCHING:-1}"
MAX_DECODE_BATCH_SIZE="${MAX_DECODE_BATCH_SIZE:-${BATCH}}"

GENERATOR_BACKEND="${GENERATOR_BACKEND:-lpddr5_bank}"
GENERATOR_DHEAD="${GENERATOR_DHEAD:-256}"
GENERATOR_HEADS_PER_HBM="${GENERATOR_HEADS_PER_HBM:-2}"
GENERATOR_DTYPE_BYTES="${GENERATOR_DTYPE_BYTES:-2}"
GENERATOR_NUM_LAYERS="${GENERATOR_NUM_LAYERS:-40}"
GENERATOR_CHANNEL_COUNT="${GENERATOR_CHANNEL_COUNT:-16}"

POWERLIMIT="${POWERLIMIT:-1}"
FFOPT="${FFOPT:-1}"
PIPEOPT="${PIPEOPT:-1}"
REBUILD="${REBUILD:-0}"
PLOT_REALTIME="${PLOT_REALTIME:-1}"
PLOT_REPLAY_STYLE="${PLOT_REPLAY_STYLE:-1}"
CAPTURE_DEBUG_EVENTS="${CAPTURE_DEBUG_EVENTS:-1}"
PLOT_DEBUG_ADMISSION="${PLOT_DEBUG_ADMISSION:-1}"
DEBUG_LOG_PATH="${DEBUG_LOG_PATH:-${RUN_DIR}/frontend_debug.log}"
DEBUG_EVENTS_CSV="${DEBUG_EVENTS_CSV:-${RUN_DIR}/debug_events.csv}"
KV_PLACEMENT_CSV="${KV_PLACEMENT_CSV:-${RUN_DIR}/kv_placement.csv}"
ALLOCATOR_PRESSURE_CSV="${ALLOCATOR_PRESSURE_CSV:-${RUN_DIR}/allocator_pressure.csv}"

mkdir -p "${RUN_DIR}" "${PLOTS_DIR}" "${ADMISSION_PLOTS_DIR}"

if [[ ! -f "${REQUESTS_CSV}" ]]; then
  echo "Missing requests CSV: ${REQUESTS_CSV}" >&2
  exit 1
fi

if [[ ! -f "${COST_GPU_CSV}" ]]; then
  echo "Missing GPU cost table: ${COST_GPU_CSV}" >&2
  exit 1
fi

if [[ ! -f "${COST_HYBRID_CSV}" ]]; then
  echo "Missing hybrid cost table: ${COST_HYBRID_CSV}" >&2
  exit 1
fi

SUMMARY_JSON="${RUN_DIR}/summary.json"
REQUESTS_OUT_CSV="${RUN_DIR}/requests_out.csv"
YAML_FILE="${RUN_DIR}/config.yaml"
RUN_LOG="${RUN_DIR}/run.log"
PREFIX="${RUN_LABEL}"

cmd=(
  "${PYTHON_BIN}" main.py
  --system "${SYSTEM}"
  --gpu "${GPU}"
  --ngpu "${NGPU}"
  --model "${MODEL}"
  --pim "${PIM}"
  --yaml-target "${YAML_TARGET}"
  --batch "${BATCH}"
  --online-serving
  --online-frontend "${ONLINE_FRONTEND}"
  --requests-csv "${REQUESTS_CSV}"
  --template-dir "${TEMPLATE_DIR}"
  --cost-gpu-csv "${COST_GPU_CSV}"
  --cost-hybrid-csv "${COST_HYBRID_CSV}"
  --route-policy "${ROUTE_POLICY}"
  --energy-latency-guard-ms "${ENERGY_LATENCY_GUARD_MS}"
  --unsupported-policy "${UNSUPPORTED_POLICY}"
  --arrival-time-scale "${ARRIVAL_TIME_SCALE}"
  --lin-bucket "${LIN_BUCKET}"
  --lout-bucket "${LOUT_BUCKET}"
  --local-scheduling-policy "${LOCAL_SCHEDULING_POLICY}"
  --fcfs-decode-prediction-mode "${FCFS_DECODE_PREDICTION_MODE}"
  --hybrid-command-source "${HYBRID_COMMAND_SOURCE}"
  --generator-backend "${GENERATOR_BACKEND}"
  --generator-dhead "${GENERATOR_DHEAD}"
  --generator-heads-per-hbm "${GENERATOR_HEADS_PER_HBM}"
  --generator-dtype-bytes "${GENERATOR_DTYPE_BYTES}"
  --generator-num-layers "${GENERATOR_NUM_LAYERS}"
  --generator-channel-count "${GENERATOR_CHANNEL_COUNT}"
)

if [[ "${POWERLIMIT}" == "1" ]]; then
  cmd+=(--powerlimit)
fi
if [[ "${FFOPT}" == "1" ]]; then
  cmd+=(--ffopt)
fi
if [[ "${PIPEOPT}" == "1" ]]; then
  cmd+=(--pipeopt)
fi
if [[ "${ENABLE_PREFILL_BATCHING}" == "1" ]]; then
  cmd+=(--enable-prefill-batching --max-prefill-batch-size "${MAX_PREFILL_BATCH_SIZE}")
fi
if [[ "${ENABLE_DECODE_BATCHING}" == "1" ]]; then
  cmd+=(--enable-decode-batching --max-decode-batch-size "${MAX_DECODE_BATCH_SIZE}")
fi
if [[ "${ENABLE_SLO_GUARDED_ADMISSION}" == "1" ]]; then
  cmd+=(
    --enable-slo-guarded-admission
    --slo-e2e-ms "${SLO_E2E_MS}"
    --slo-tbt-ms "${SLO_TBT_MS}"
    --admission-retry-interval-ms "${ADMISSION_RETRY_INTERVAL_MS}"
    --admission-max-harmed-requests "${ADMISSION_MAX_HARMED_REQUESTS}"
    --admission-max-total-harm-ms "${ADMISSION_MAX_TOTAL_HARM_MS}"
    --admission-max-single-harm-ms "${ADMISSION_MAX_SINGLE_HARM_MS}"
    --admission-max-bypass-count "${ADMISSION_MAX_BYPASS_COUNT}"
    --admission-max-wait-ms "${ADMISSION_MAX_WAIT_MS}"
    --admission-aging-harm-ms-per-ms "${ADMISSION_AGING_HARM_MS_PER_MS}"
    --admission-aging-harmed-requests-per-ms "${ADMISSION_AGING_HARMED_REQUESTS_PER_MS}"
  )
fi
if [[ "${CAPTURE_DEBUG_EVENTS}" == "1" ]]; then
  cmd+=(--capture-debug-events --debug-log-path "${DEBUG_LOG_PATH}")
fi

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

cmd+=(
  --online-yaml "${YAML_FILE}"
  --online-summary-json "${SUMMARY_JSON}"
  --online-requests-out "${REQUESTS_OUT_CSV}"
  --online-kv-placement-out "${KV_PLACEMENT_CSV}"
  --online-allocator-pressure-out "${ALLOCATOR_PRESSURE_CSV}"
)

printf '%q ' "${cmd[@]}" > "${RUN_DIR}/run_command.sh"
printf '\n' >> "${RUN_DIR}/run_command.sh"

{
  echo "run_dir=${RUN_DIR}"
  echo "plots_dir=${PLOTS_DIR}"
  echo "summary_json=${SUMMARY_JSON}"
  echo "requests_out_csv=${REQUESTS_OUT_CSV}"
  echo "yaml_file=${YAML_FILE}"
  echo "request_source=${REQUESTS_CSV}"
  echo "batch=${BATCH}"
  echo "hybrid_command_source=${HYBRID_COMMAND_SOURCE}"
  echo "debug_log_path=${DEBUG_LOG_PATH}"
  echo "debug_events_csv=${DEBUG_EVENTS_CSV}"
  echo "kv_placement_csv=${KV_PLACEMENT_CSV}"
  echo "allocator_pressure_csv=${ALLOCATOR_PRESSURE_CSV}"
  echo "enable_slo_guarded_admission=${ENABLE_SLO_GUARDED_ADMISSION}"
  echo "admission_max_harmed_requests=${ADMISSION_MAX_HARMED_REQUESTS}"
  echo "admission_max_total_harm_ms=${ADMISSION_MAX_TOTAL_HARM_MS}"
  echo "admission_max_single_harm_ms=${ADMISSION_MAX_SINGLE_HARM_MS}"
} > "${RUN_DIR}/run_info.txt"

if [[ "${REBUILD}" == "1" ]]; then
  echo "Rebuilding ramulator2..."
  cmake --build ramulator2/build -j4
  cp ramulator2/build/ramulator2 ramulator2/ramulator2
fi

echo "Running realtime serving case:"
echo "  run_dir=${RUN_DIR}"
echo "  requests_csv=${REQUESTS_CSV}"
echo "  batch=${BATCH}"
echo "  hybrid_command_source=${HYBRID_COMMAND_SOURCE}"

(
  set -x
  "${cmd[@]}"
) |& tee "${RUN_LOG}"

if [[ "${CAPTURE_DEBUG_EVENTS}" == "1" && -f "${DEBUG_LOG_PATH}" ]]; then
  "${PYTHON_BIN}" tools/parse_realtime_debug_log.py \
    --debug-log "${DEBUG_LOG_PATH}" \
    --out-csv "${DEBUG_EVENTS_CSV}"
fi

if [[ "${PLOT_REALTIME}" == "1" ]]; then
  "${PYTHON_BIN}" tools/plot_realtime_serving_results.py \
    --requests-csv "${REQUESTS_OUT_CSV}" \
    --summary-json "${SUMMARY_JSON}" \
    --out-dir "${PLOTS_DIR}" \
    --prefix "${PREFIX}"
fi

if [[ "${PLOT_REPLAY_STYLE}" == "1" ]]; then
  replay_plot_cmd=(
    "${PYTHON_BIN}" tools/plot_replay_results.py
    --requests-csv "${REQUESTS_OUT_CSV}"
    --summary-json "${SUMMARY_JSON}"
    --out-dir "${PLOTS_DIR}"
    --prefix "${PREFIX}"
  )
  if [[ -f "${DEBUG_EVENTS_CSV}" ]]; then
    replay_plot_cmd+=(--debug-events-csv "${DEBUG_EVENTS_CSV}")
  fi
  "${replay_plot_cmd[@]}"
fi

if [[ "${PLOT_DEBUG_ADMISSION}" == "1" && -f "${DEBUG_EVENTS_CSV}" ]]; then
  "${PYTHON_BIN}" tools/plot_debug_admission.py \
    --events-csv "${DEBUG_EVENTS_CSV}" \
    --requests-csv "${REQUESTS_OUT_CSV}" \
    --out-dir "${ADMISSION_PLOTS_DIR}" \
    --prefix "${PREFIX}"
fi

echo "Wrote run outputs to:"
echo "  ${RUN_DIR}"
echo "Wrote plots to:"
echo "  ${PLOTS_DIR}"
if [[ -f "${KV_PLACEMENT_CSV}" ]]; then
  echo "Wrote KV placement CSV to:"
  echo "  ${KV_PLACEMENT_CSV}"
fi
if [[ -f "${ALLOCATOR_PRESSURE_CSV}" ]]; then
  echo "Wrote allocator pressure CSV to:"
  echo "  ${ALLOCATOR_PRESSURE_CSV}"
fi
if [[ "${PLOT_DEBUG_ADMISSION}" == "1" && -f "${DEBUG_EVENTS_CSV}" ]]; then
  echo "Wrote admission-debug plots to:"
  echo "  ${ADMISSION_PLOTS_DIR}"
fi
