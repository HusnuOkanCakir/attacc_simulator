#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="cluster_outputs/v31_sweep"
mkdir -p "${OUT_DIR}"
TRACE_CACHE_DIR="cluster_outputs/trace_cache"
mkdir -p "${TRACE_CACHE_DIR}"

AZURE_TRACE="${AZURE_TRACE:-https://raw.githubusercontent.com/Azure/AzurePublicDataset/refs/heads/master/data/AzureLLMInferenceTrace_code.csv}"
LIMIT="${LIMIT:-2000}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-0.05}"
UNSUPPORTED_POLICY="${UNSUPPORTED_POLICY:-clip}"
PIM_WAIT_THRESHOLD_MS="${PIM_WAIT_THRESHOLD_MS:-1}"
ROUTE_POLICY="${ROUTE_POLICY:-min_finish}"
GPU_MODEL="${GPU_MODEL:-cluster_outputs/cost_models/gpu_only.pkl}"
LPDDR_MODEL="${LPDDR_MODEL:-cluster_outputs/cost_models/lpddr5_pim_bank.pkl}"

if [[ "${AZURE_TRACE}" =~ ^https?:// ]]; then
  TRACE_CACHE_PATH="${TRACE_CACHE_DIR}/AzureLLMInferenceTrace_code.csv"
  if [[ ! -f "${TRACE_CACHE_PATH}" ]]; then
    echo "Caching Azure trace to ${TRACE_CACHE_PATH}"
    python - <<'PY' "${AZURE_TRACE}" "${TRACE_CACHE_PATH}"
import sys
import urllib.request
src, dst = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(src) as resp, open(dst, "wb") as out:
    out.write(resp.read())
PY
  fi
  AZURE_TRACE="${TRACE_CACHE_PATH}"
fi

run_case() {
  local label="$1"
  shift
  local summary_path="${OUT_DIR}/replay_summary_${label}.json"
  local requests_path="${OUT_DIR}/replay_requests_${label}.csv"
  if [[ -f "${summary_path}" ]]; then
    echo "Skipping existing v3.1 sweep case: ${label}"
    return
  fi
  echo "Running v3.1 sweep case: ${label}"

  python tools/run_trace_replay.py \
    --azure-trace "${AZURE_TRACE}" \
    --cost-model ml \
    --route-policy "${ROUTE_POLICY}" \
    --limit "${LIMIT}" \
    --arrival-time-scale "${ARRIVAL_SCALE}" \
    --ml-profile "gpu_only=${GPU_MODEL}" \
    --ml-profile "lpddr5_pim_bank=${LPDDR_MODEL}" \
    --routes gpu_only lpddr5_pim_bank \
    --unsupported-policy "${UNSUPPORTED_POLICY}" \
    --pim-wait-threshold-ms "${PIM_WAIT_THRESHOLD_MS}" \
    --requests-csv "${requests_path}" \
    --summary-json "${summary_path}" \
    "$@"
}

run_case nobatch
run_case batch2 --enable-decode-batching --max-decode-batch-size 2
run_case batch4 --enable-decode-batching --max-decode-batch-size 4
run_case batch4_guard \
  --enable-decode-batching --max-decode-batch-size 4 \
  --prefill-guard-ms 100
run_case batch4_guard_cap \
  --enable-decode-batching --max-decode-batch-size 4 \
  --prefill-guard-ms 100 \
  --max-consecutive-decode-batches 2 \
  --decode-batch-cap-with-prefill 2

python - <<'PY'
import json
from pathlib import Path

labels = ("nobatch", "batch2", "batch4", "batch4_guard", "batch4_guard_cap")
print("case,throughput_tokps,ttft_p95,prefill_wait_p95,e2e_p95,tbt_p95,gpu_util,pim_util,gpu_route_frac,mean_batch_size,max_batch_size,num_decode_steps,prefill_guard_triggers,decode_limit_triggers")
for label in labels:
    path = Path("cluster_outputs/v31_sweep") / f"replay_summary_{label}.json"
    with path.open() as f:
        d = json.load(f)
    batching = d.get("decode_batching", {})
    local = d.get("local_scheduling", {})
    route_counts = d.get("route_counts", {})
    total_routed = sum(route_counts.values()) or 1
    gpu_route_frac = route_counts.get("gpu_only", 0) / total_routed
    print(
        f"{label},"
        f"{d['throughput_tokps']},"
        f"{d['ttft_ms']['p95']},"
        f"{d['prefill_wait_ms']['p95']},"
        f"{d['e2e_ms']['p95']},"
        f"{d['tbt_ms']['p95']},"
        f"{d['gpu_util']},"
        f"{d['pim_util']},"
        f"{gpu_route_frac},"
        f"{batching.get('mean_batch_size')},"
        f"{batching.get('max_batch_size')},"
        f"{batching.get('num_decode_steps')},"
        f"{local.get('prefill_guard_trigger_count')},"
        f"{local.get('decode_limit_trigger_count')}"
    )
PY
