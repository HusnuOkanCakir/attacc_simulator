#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="cluster_outputs/v31_tuning_sweep"
mkdir -p "${OUT_DIR}"
TRACE_CACHE_DIR="cluster_outputs/trace_cache"
mkdir -p "${TRACE_CACHE_DIR}"

AZURE_TRACE="${AZURE_TRACE:-https://raw.githubusercontent.com/Azure/AzurePublicDataset/refs/heads/master/data/AzureLLMInferenceTrace_code.csv}"
LIMIT="${LIMIT:-2000}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-0.05}"
UNSUPPORTED_POLICY="${UNSUPPORTED_POLICY:-clip}"
PIM_WAIT_THRESHOLD_MS="${PIM_WAIT_THRESHOLD_MS:-1}"
ROUTE_POLICY="${ROUTE_POLICY:-min_finish}"
MAX_DECODE_BATCH_SIZE="${MAX_DECODE_BATCH_SIZE:-4}"
GPU_MODEL="${GPU_MODEL:-cluster_outputs/cost_models/gpu_only.pkl}"
LPDDR_MODEL="${LPDDR_MODEL:-cluster_outputs/cost_models/lpddr5_pim_bank.pkl}"

DECODE_CAPS=(${DECODE_CAPS:-1 2 3})
PREFILL_GUARDS=(${PREFILL_GUARDS:-25 50 100 200})
MAX_CONSEC_DECODE=(${MAX_CONSEC_DECODE:-1 2 4})

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
  local cap="$1"
  local guard="$2"
  local consec="$3"
  local label="cap${cap}_guard${guard}_cons${consec}"
  local summary_path="${OUT_DIR}/replay_summary_${label}.json"
  local requests_path="${OUT_DIR}/replay_requests_${label}.csv"

  if [[ -f "${summary_path}" ]]; then
    echo "Skipping existing v3.1 tuning case: ${label}"
    return
  fi

  echo "Running v3.1 tuning case: ${label}"
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
    --enable-decode-batching \
    --max-decode-batch-size "${MAX_DECODE_BATCH_SIZE}" \
    --prefill-guard-ms "${guard}" \
    --max-consecutive-decode-batches "${consec}" \
    --decode-batch-cap-with-prefill "${cap}" \
    --requests-csv "${requests_path}" \
    --summary-json "${summary_path}"
}

for cap in "${DECODE_CAPS[@]}"; do
  for guard in "${PREFILL_GUARDS[@]}"; do
    for consec in "${MAX_CONSEC_DECODE[@]}"; do
      run_case "${cap}" "${guard}" "${consec}"
    done
  done
done

python - <<'PY'
import json
import re
from pathlib import Path

pat = re.compile(r"replay_summary_cap(\d+)_guard(\d+)_cons(\d+)\.json")
print("decode_cap,prefill_guard_ms,max_consecutive_decode_batches,throughput_tokps,ttft_p95,prefill_wait_p95,e2e_p95,tbt_p95,gpu_util,pim_util,gpu_route_frac,mean_batch_size,max_batch_size,num_decode_steps,prefill_guard_triggers,decode_limit_triggers")
for path in sorted(Path("cluster_outputs/v31_tuning_sweep").glob("replay_summary_*.json")):
    m = pat.match(path.name)
    if not m:
        continue
    with path.open() as f:
        d = json.load(f)
    batching = d.get("decode_batching", {})
    local = d.get("local_scheduling", {})
    route_counts = d.get("route_counts", {})
    total_routed = sum(route_counts.values()) or 1
    gpu_route_frac = route_counts.get("gpu_only", 0) / total_routed
    print(
        f"{m.group(1)},"
        f"{m.group(2)},"
        f"{m.group(3)},"
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
