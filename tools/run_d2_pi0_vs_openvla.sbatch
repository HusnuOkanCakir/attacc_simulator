#!/bin/bash
#SBATCH -J d2_pi0_vs_openvla
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/d2_pi0_vs_openvla_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/d2_pi0_vs_openvla_%j.err
#
# D2: Pi0 vs OpenVLA paired comparison with corrected Pi0 architecture
# (18L, 8Q heads, 1 KV head MQA, d_head=256). The old D run was unreliable
# because Pi0 was configured as 40L/8H/MHA — see project_pi0_openvla_audit.
#
# Both models run on the same Azure trace slice, same arrival_scale, same
# kv_pool, same policy, same GPU (A6000). The only difference is the model
# preset. This isolates the model-architecture effect.
#
# Usage:
#   sbatch tools/run_d2_pi0_vs_openvla.sbatch
#
#   # Override n / scale / pool:
#   sbatch --export=ALL,N_REQUESTS=500,ARRIVAL_SCALE=3.0 \
#       tools/run_d2_pi0_vs_openvla.sbatch
#
# Env-var overrides:
#   N_REQUESTS         (default 2000)
#   ARRIVAL_SCALE      (default 1.0)
#   KV_POOL_GB         (default 8)        — sized for Pi0 max_active=32
#   MAX_ACTIVE         (default 32)
#   POLICY             (default max_util_full)
#   MAX_DECODE_BS      (default 8)
#   MAX_CONSEC_DEC     (default 4)
#   PROMPT_PRIORITY    (default false)    — H3 confirmed this wins
#   SLO_E2E_MS         (default 500)
#   SLO_TTFT_MS        (default 200)
#   KV_OOM_POLICY      (default hold_then_fallback)
#   MODELS             (default "pi0 openvla")
#   GPU_TYPE           (default a6000)
#   TRACE_KIND         (default azure)    — "azure" or "synthetic". For synthetic,
#                                           per-model VLA traces are generated
#                                           on-the-fly (Lout=7 OpenVLA, Lout=50 Pi0,
#                                           Poisson @ 6 Hz). Run separately for
#                                           each trace kind to get both data sets.
#
# Output: cluster_outputs/online_serving_runs/d2_sweep_${TS}/
#           01_pi0_a6000/<policy>/...
#           02_openvla_a6000/<policy>/...
#           plots/  (cross-model sweep_compare.png)

set -euo pipefail

ROOT_DIR="/mnt/galactica/ocakir/attacc_simulator"
LOG_DIR="/mnt/galactica/ocakir/logs"
mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

export TMPDIR=/mnt/galactica/ocakir/tmp
mkdir -p "$TMPDIR"

source /mnt/galactica/ocakir/miniforge3/etc/profile.d/conda.sh
conda activate lerobot

echo "hostname:  $(hostname)"
echo "python:    $(python -V 2>&1)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unknown}"

N_REQUESTS="${N_REQUESTS:-2000}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-1.0}"
KV_POOL_GB="${KV_POOL_GB:-8}"
MAX_ACTIVE="${MAX_ACTIVE:-32}"
POLICY="${POLICY:-max_util_full}"
MAX_DECODE_BS="${MAX_DECODE_BS:-8}"
MAX_CONSEC_DEC="${MAX_CONSEC_DEC:-4}"
PROMPT_PRIORITY="${PROMPT_PRIORITY:-false}"
SLO_E2E_MS="${SLO_E2E_MS:-500}"
SLO_TTFT_MS="${SLO_TTFT_MS:-200}"
KV_OOM_POLICY="${KV_OOM_POLICY:-hold_then_fallback}"
KV_OOM_HOLD_LIMIT_MS="${KV_OOM_HOLD_LIMIT_MS:-50}"
MODELS="${MODELS:-pi0 openvla}"
GPU_TYPE="${GPU_TYPE:-a6000}"
TRACE_KIND="${TRACE_KIND:-azure}"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"
TRACE_GEN="${ROOT_DIR}/tools/gen_vla_synthetic_trace.py"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
SYNTH_TRACE_DIR="${ROOT_DIR}/cluster_outputs/synthetic_vla"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/d2_sweep_${TRACE_KIND}_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}" "${SYNTH_TRACE_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

n_models=$(echo "$MODELS" | wc -w)
log "trace_kind=$TRACE_KIND"
log "models=($MODELS)  gpu=$GPU_TYPE"
log "n=$N_REQUESTS  scale=$ARRIVAL_SCALE  kv_pool=${KV_POOL_GB}GiB  max_active=$MAX_ACTIVE"
log "policy=$POLICY  max_decode_bs=$MAX_DECODE_BS  mc=$MAX_CONSEC_DEC  prompt_priority=$PROMPT_PRIORITY"
log "slo_e2e=${SLO_E2E_MS}ms  slo_ttft=${SLO_TTFT_MS}ms  kv_oom=$KV_OOM_POLICY"
log "sweep_dir=$SWEEP_DIR"
log ""

# For trace_kind=synthetic, pre-generate per-model VLA traces if missing.
# Trace path naming: synthetic_vla_<model>_n<N>_seed0.csv
gen_synthetic_if_needed() {
  local model=$1
  local out_csv="${SYNTH_TRACE_DIR}/synthetic_vla_${model}_n${N_REQUESTS}_seed0.csv"
  if [ -s "$out_csv" ]; then
    log "[trace] reusing existing $out_csv ($(wc -l < "$out_csv") lines)"
  else
    log "[trace] generating synthetic VLA trace for model=$model n=$N_REQUESTS"
    "$PY" "$TRACE_GEN" --model "$model" --n-requests "$N_REQUESTS" \
        --rate-hz 6.0 --jitter 1.0 --seed 0 --out-csv "$out_csv" 2>&1 | tee -a "$LOG"
  fi
  echo "$out_csv"  # return path via stdout
}

if [ "$TRACE_KIND" = "synthetic" ]; then
  for model in $MODELS; do
    _trace=$(gen_synthetic_if_needed "$model" | tail -1)
  done
fi

# args: <label> <model>
run_one() {
  local label=$1
  local model=$2
  local run_dir="${SWEEP_DIR}/${label}"
  local cache_file="${CACHE_DIR}/${model}_${GPU_TYPE}_d2_${TRACE_KIND}.bin"

  # Pick the trace: azure default (no --csv-path) or per-model synthetic CSV.
  local csv_arg=""
  if [ "$TRACE_KIND" = "synthetic" ]; then
    local csv_path="${SYNTH_TRACE_DIR}/synthetic_vla_${model}_n${N_REQUESTS}_seed0.csv"
    csv_arg="--csv-path $csv_path"
  fi

  log "── start: $label  model=$model  gpu=$GPU_TYPE  trace=$TRACE_KIND ──"
  log "    cache:       $cache_file"
  if [ -n "$csv_arg" ]; then
    log "    csv:         $csv_arg"
  fi
  log "    watch live:  watch -n 2 cat $run_dir/$POLICY/status.txt"

  local pp_flag="--no-prompt-priority"
  if [ "$PROMPT_PRIORITY" = "true" ]; then
    pp_flag="--prompt-priority"
  fi

  local t0=$(date +%s)
  "$PY" "$RUNNER" \
      --model "$model" \
      --gpu "$GPU_TYPE" \
      --n-requests "$N_REQUESTS" \
      --policies "$POLICY" \
      --kv-pool-gb "$KV_POOL_GB" --ml \
      --max-active "$MAX_ACTIVE" \
      --arrival-scale "$ARRIVAL_SCALE" \
      --slo-e2e-ms "$SLO_E2E_MS" --slo-ttft-ms "$SLO_TTFT_MS" \
      --kv-oom-policy "$KV_OOM_POLICY" \
      --kv-oom-hold-limit-ms "$KV_OOM_HOLD_LIMIT_MS" \
      --max-decode-batch-size "$MAX_DECODE_BS" \
      --max-consecutive-decode-batches "$MAX_CONSEC_DEC" \
      "$pp_flag" \
      $csv_arg \
      --pim-cache-file "$cache_file" \
      --debug-log \
      --run-dir "$run_dir" 2>&1 | tee -a "$LOG"
  local dt=$(( $(date +%s) - t0 ))
  log "── done:  $label  wall=${dt}s ──"
}

# ── Run each (model, gpu) cell sequentially ──────────────────────────────────
i=1
for model in $MODELS; do
  label=$(printf "%02d_%s_%s" "$i" "$model" "$GPU_TYPE")
  run_one "$label" "$model"
  i=$((i+1))
done

# ── Summary ──────────────────────────────────────────────────────────────────
log "=========================================================================="
log "ALL DONE"
log "  sweep_dir: $SWEEP_DIR"
log "  log:       $LOG"
log "  models:    $MODELS"
log "  gpu:       $GPU_TYPE"
log "  trace_kind: $TRACE_KIND"
log "  caches:"
for model in $MODELS; do
  cf="${CACHE_DIR}/${model}_${GPU_TYPE}_d2_${TRACE_KIND}.bin"
  entries=$(grep -cv '^#' "$cf" 2>/dev/null || echo 0)
  log "    $cf  ($entries entries)"
done
log "=========================================================================="

# ── Cross-cell comparison plot ───────────────────────────────────────────────
PLOT_COMPARE="${ROOT_DIR}/tools/plot_sweep_compare.py"
if [ -f "$PLOT_COMPARE" ]; then
  log "[plot] generating cross-cell comparison ..."
  "$PY" "$PLOT_COMPARE" --sweep-dir "$SWEEP_DIR" 2>&1 | tee -a "$LOG"
fi

log ""
log "What to look for in the result:"
log "  1. Pi0 throughput vs OpenVLA throughput — expect modest gap (~1.5–3×),"
log "     NOT the old 7000× artifact. Pi0 is smaller (18L vs 32L), has MQA"
log "     (1 KV head vs 32), and shorter d_head."
log "  2. Routes (PIM/GPU) — Pi0 may route MORE traffic to PIM because its"
log "     PIM decode time is closer to GPU decode (less of a no-brainer)."
log "  3. KV-OOM holds — Pi0 has 320 KB/token vs OpenVLA's 256 KB/token,"
log "     so Pi0 may show more KV pressure at the same pool size."
