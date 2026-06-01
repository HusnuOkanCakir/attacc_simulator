#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ROOT"

REQUESTS_CSV="${REQUESTS_CSV:-cluster_outputs/test_realtime_requests_tiny.csv}"
REQUESTS_OUT_CSV="${REQUESTS_OUT_CSV:-cluster_outputs/online_serving/test_realtime_tiny_requests_out.csv}"
YAML_FILE="${YAML_FILE:-ramulator2/online_serving.yaml}"
TEMPLATE_DIR="${TEMPLATE_DIR:-ramulator2/trace_gen/templates_lpddr5_bank}"
COST_GPU_CSV="${COST_GPU_CSV:-cluster_outputs/cost_tables_full_energy/gpu_only.csv}"
COST_HYBRID_CSV="${COST_HYBRID_CSV:-cluster_outputs/cost_tables_full_energy/lpddr5_pim_bank.csv}"
DEBUG_LOG_PATH="${DEBUG_LOG_PATH:-cluster_outputs/online_serving/frontend_debug.log}"
DEBUG_INTERVAL_CYCLES="${DEBUG_INTERVAL_CYCLES:-100000}"
ONLINE_FRONTEND="${ONLINE_FRONTEND:-realtime}"

export REQUESTS_CSV
export REQUESTS_OUT_CSV
export YAML_FILE
export TEMPLATE_DIR
export COST_GPU_CSV
export COST_HYBRID_CSV
export DEBUG_LOG_PATH
export DEBUG_INTERVAL_CYCLES
export ONLINE_FRONTEND

mkdir -p cluster_outputs/online_serving
mkdir -p log/online_serving

python - <<'PY'
import os
from src.config import make_model_config, make_pim_config
from src.ramulator_wrapper import Ramulator
from src.type import DataType, PIMType, InterfaceType

modelinfos = make_model_config("GPT-175B", DataType.W16A16)
pim_config = make_pim_config(
    PIMType.BA,
    InterfaceType.NVLINK3,
    power_constraint=True,
    yaml_target="lpddr5-pim",
)

ram = Ramulator(modelinfos, "ramulator2", "ramulator.out", pim_config=pim_config)
ram.make_online_yaml_file(
    yaml_file=os.environ["YAML_FILE"],
    requests_csv=os.environ["REQUESTS_CSV"],
    requests_out_csv=os.environ["REQUESTS_OUT_CSV"],
    template_dir=os.environ["TEMPLATE_DIR"],
    cost_gpu_csv=os.environ["COST_GPU_CSV"],
    cost_hybrid_csv=os.environ["COST_HYBRID_CSV"],
    route_policy="min_finish",
    pim_wait_threshold_ms=1.0,
    unsupported_policy="clip",
    arrival_time_scale=1.0,
    lin_bucket=1,
    lout_bucket=1,
    batch_size=1,
    gpu_route_name="gpu_only",
    hybrid_route_name="lpddr5_pim_bank",
    online_frontend=os.environ["ONLINE_FRONTEND"],
)
print(f"wrote {os.environ['YAML_FILE']}")
PY

python - <<'PY'
import os
from pathlib import Path

p = Path(os.environ["YAML_FILE"])
text = p.read_text()
needle = "  batch_size: 1\n  clock_ratio: 1\n"
repl = (
    "  batch_size: 1\n"
    "  debug: true\n"
    f"  debug_interval_cycles: {os.environ['DEBUG_INTERVAL_CYCLES']}\n"
    f"  debug_log_path: {os.environ['DEBUG_LOG_PATH']}\n"
    "  clock_ratio: 1\n"
)

if "debug_log_path:" not in text:
    text = text.replace(needle, repl, 1)
    p.write_text(text)

print(f"patched {p} with debug logging")
PY
