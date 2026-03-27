#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-cluster_outputs/azure_guarded_bypass_explore}"
PLOTS_DIR="${PLOTS_DIR:-cluster_outputs/azure_guarded_bypass_explore_plots}"
TRACE_CACHE_DIR="${TRACE_CACHE_DIR:-cluster_outputs/trace_cache}"
AZURE_TRACE="${AZURE_TRACE:-https://raw.githubusercontent.com/Azure/AzurePublicDataset/refs/heads/master/data/AzureLLMInferenceTrace_code.csv}"

GPU_MODEL="${GPU_MODEL:-cluster_outputs/cost_models_full_energy/gpu_only.pkl}"
LPDDR_MODEL="${LPDDR_MODEL:-cluster_outputs/cost_models_full_energy/lpddr5_pim_bank.pkl}"

LIMIT="${LIMIT:-50}"
MAX_DECODE_BATCH_SIZE="${MAX_DECODE_BATCH_SIZE:-4}"
ENERGY_LATENCY_GUARD_MS="${ENERGY_LATENCY_GUARD_MS:-5}"
PLOT_MIN_REQUEST_ID="${PLOT_MIN_REQUEST_ID:-0}"
PLOT_MAX_REQUEST_ID="${PLOT_MAX_REQUEST_ID:-$((LIMIT - 1))}"

# Format per case:
#   label:arrival_scale:slo_e2e_ms:slo_tbt_ms:max_wait_ms:retry_interval_ms
# Defaults are chosen to make guarded-admission behavior visible on Azure.
DEFAULT_CASE_SPECS=(
  "azure_scale025_e1200_t5_w150_r10:0.25:1200:5:150:10"
  "azure_scale025_e900_t3_w150_r10:0.25:900:3:150:10"
  "azure_scale01_e900_t3_w150_r10:0.1:900:3:150:10"
)

if [[ -n "${CASE_SPECS:-}" ]]; then
  IFS=';' read -r -a CASES <<< "${CASE_SPECS}"
else
  CASES=("${DEFAULT_CASE_SPECS[@]}")
fi

mkdir -p "${OUT_DIR}" "${PLOTS_DIR}" "${TRACE_CACHE_DIR}"

if [[ "${AZURE_TRACE}" =~ ^https?:// ]]; then
  TRACE_CACHE_PATH="${TRACE_CACHE_DIR}/AzureLLMInferenceTrace_code.csv"
  if [[ ! -f "${TRACE_CACHE_PATH}" ]]; then
    echo "Caching Azure trace to ${TRACE_CACHE_PATH}"
    "${PYTHON_BIN}" - <<'PY' "${AZURE_TRACE}" "${TRACE_CACHE_PATH}"
import sys
import urllib.request

src, dst = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(src) as resp, open(dst, "wb") as out:
    out.write(resp.read())
PY
  fi
  AZURE_TRACE="${TRACE_CACHE_PATH}"
fi

for spec in "${CASES[@]}"; do
  IFS=':' read -r label arrival_scale slo_e2e_ms slo_tbt_ms max_wait_ms retry_interval_ms <<< "${spec}"

  summary_path="${OUT_DIR}/${label}_summary.json"
  requests_path="${OUT_DIR}/${label}_requests.csv"
  debug_path="${OUT_DIR}/${label}_debug.csv"

  echo "Running Azure guarded exploration case: ${label}"
  echo "  scale=${arrival_scale} slo_e2e=${slo_e2e_ms} slo_tbt=${slo_tbt_ms} max_wait=${max_wait_ms} retry=${retry_interval_ms}"

  "${PYTHON_BIN}" tools/run_trace_replay.py \
    --azure-trace "${AZURE_TRACE}" \
    --cost-model ml \
    --route-policy latency_guarded_energy \
    --energy-latency-guard-ms "${ENERGY_LATENCY_GUARD_MS}" \
    --slo-e2e-ms "${slo_e2e_ms}" \
    --slo-tbt-ms "${slo_tbt_ms}" \
    --limit "${LIMIT}" \
    --arrival-time-scale "${arrival_scale}" \
    --ml-profile "gpu_only=${GPU_MODEL}" \
    --ml-profile "lpddr5_pim_bank=${LPDDR_MODEL}" \
    --routes gpu_only lpddr5_pim_bank \
    --unsupported-policy clip \
    --local-scheduling-policy prefill_priority_fcfs_decode \
    --fcfs-decode-prediction-mode incremental \
    --enable-decode-batching \
    --max-decode-batch-size "${MAX_DECODE_BATCH_SIZE}" \
    --enable-slo-guarded-admission \
    --admission-max-harmed-requests 0 \
    --admission-max-total-harm-ms 0 \
    --admission-max-single-harm-ms 0 \
    --admission-max-own-miss-ms 0 \
    --admission-max-bypass-count 5 \
    --admission-max-wait-ms "${max_wait_ms}" \
    --admission-aging-harm-ms-per-ms 0.01 \
    --admission-aging-harmed-requests-per-ms 0.001 \
    --admission-retry-interval-ms "${retry_interval_ms}" \
    --requests-csv "${requests_path}" \
    --summary-json "${summary_path}" \
    --debug-events-csv "${debug_path}"

  "${PYTHON_BIN}" tools/plot_replay_results.py \
    --requests-csv "${requests_path}" \
    --summary-json "${summary_path}" \
    --debug-events-csv "${debug_path}" \
    --prediction-history-min-request-id "${PLOT_MIN_REQUEST_ID}" \
    --prediction-history-max-request-id "${PLOT_MAX_REQUEST_ID}" \
    --out-dir "${PLOTS_DIR}" \
    --prefix "${label}"

  "${PYTHON_BIN}" tools/plot_debug_admission.py \
    --events-csv "${debug_path}" \
    --requests-csv "${requests_path}" \
    --out-dir "${PLOTS_DIR}" \
    --prefix "${label}"
done

OUT_DIR_FOR_PY="${OUT_DIR}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR_FOR_PY"])
rows = []
for path in sorted(out_dir.glob("*_summary.json")):
    label = path.stem[:-8]
    with path.open() as f:
        d = json.load(f)
    admission = d.get("admission", {})
    queue = d.get("admission_queue", {})
    rows.append({
        "label": label,
        "bypassed_count": admission.get("bypassed_count"),
        "held_count": admission.get("held_count"),
        "best_effort_count": admission.get("best_effort_count"),
        "blocked_count": admission.get("blocked_count"),
        "queue_max_len": queue.get("max_len"),
        "queue_mean_wait_ms": queue.get("mean_wait_ms"),
        "ttft_p95_ms": d.get("ttft_ms", {}).get("p95"),
        "e2e_p95_ms": d.get("e2e_ms", {}).get("p95"),
        "tbt_p95_ms": d.get("tbt_ms", {}).get("p95"),
        "throughput_tokps": d.get("throughput_tokps"),
        "route_counts": d.get("route_counts"),
    })

csv_path = out_dir / "summary_table.csv"
with csv_path.open("w") as f:
    f.write(
        "label,bypassed_count,held_count,best_effort_count,blocked_count,queue_max_len,queue_mean_wait_ms,ttft_p95_ms,e2e_p95_ms,tbt_p95_ms,throughput_tokps,route_counts\n"
    )
    for row in rows:
        f.write(
            f"{row['label']},{row['bypassed_count']},{row['held_count']},{row['best_effort_count']},{row['blocked_count']},{row['queue_max_len']},{row['queue_mean_wait_ms']},{row['ttft_p95_ms']},{row['e2e_p95_ms']},{row['tbt_p95_ms']},{row['throughput_tokps']},{row['route_counts']}\n"
        )

print("label,bypassed_count,held_count,best_effort_count,blocked_count,queue_max_len,queue_mean_wait_ms,ttft_p95_ms,e2e_p95_ms,tbt_p95_ms,throughput_tokps,route_counts")
for row in rows:
    print(
        f"{row['label']},{row['bypassed_count']},{row['held_count']},{row['best_effort_count']},{row['blocked_count']},{row['queue_max_len']},{row['queue_mean_wait_ms']},{row['ttft_p95_ms']},{row['e2e_p95_ms']},{row['tbt_p95_ms']},{row['throughput_tokps']},{row['route_counts']}"
    )
print(f"Wrote summary table: {csv_path}")
PY

echo "Wrote run outputs to ${OUT_DIR}"
echo "Wrote plots to ${PLOTS_DIR}"
