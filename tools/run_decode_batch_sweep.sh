#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="cluster_outputs/decode_batch_sweep"
mkdir -p "${OUT_DIR}"

AZURE_TRACE="${AZURE_TRACE:-https://raw.githubusercontent.com/Azure/AzurePublicDataset/refs/heads/master/data/AzureLLMInferenceTrace_code.csv}"
LIMIT="${LIMIT:-2000}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-0.05}"
UNSUPPORTED_POLICY="${UNSUPPORTED_POLICY:-clip}"
PIM_WAIT_THRESHOLD_MS="${PIM_WAIT_THRESHOLD_MS:-1}"
ROUTE_POLICY="${ROUTE_POLICY:-min_finish}"
GPU_MODEL="${GPU_MODEL:-cluster_outputs/cost_models/gpu_only.pkl}"
LPDDR_MODEL="${LPDDR_MODEL:-cluster_outputs/cost_models/lpddr5_pim_bank.pkl}"

run_case() {
  local label="$1"
  shift
  echo "Running decode-batch sweep case: ${label}"

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
    --requests-csv "${OUT_DIR}/replay_requests_${label}.csv" \
    --summary-json "${OUT_DIR}/replay_summary_${label}.json" \
    "$@"
}

run_case nobatch
run_case batch2 --enable-decode-batching --max-decode-batch-size 2
run_case batch4 --enable-decode-batching --max-decode-batch-size 4

python - <<'PY'
import json
from pathlib import Path

print("case,throughput_tokps,ttft_p95,e2e_p95,tbt_p95,gpu_util,pim_util,route_counts,mean_batch_size,max_batch_size,num_decode_steps")
for label in ("nobatch", "batch2", "batch4"):
    path = Path("cluster_outputs/decode_batch_sweep") / f"replay_summary_{label}.json"
    with path.open() as f:
        d = json.load(f)
    batching = d.get("decode_batching", {})
    print(
        f"{label},"
        f"{d['throughput_tokps']},"
        f"{d['ttft_ms']['p95']},"
        f"{d['e2e_ms']['p95']},"
        f"{d['tbt_ms']['p95']},"
        f"{d['gpu_util']},"
        f"{d['pim_util']},"
        f"{d['route_counts']},"
        f"{batching.get('mean_batch_size')},"
        f"{batching.get('max_batch_size')},"
        f"{batching.get('num_decode_steps')}"
    )
PY
