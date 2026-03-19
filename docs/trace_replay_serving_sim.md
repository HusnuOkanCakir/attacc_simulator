# Trace Replay Serving Simulator (Azure Trace + PI0/AttAcc Cost Model)

## Goal

Move from single-request benchmarking (`main.py`) to a more realistic serving simulation that replays many requests over time and makes routing decisions:

- `gpu_only`
- `hybrid_pim_gpu` (e.g., `lpddr5_pim_bank`, `hbm3_pim_bank`)

This enables queueing-aware evaluation, tail-latency analysis, and later continuous batching / SLO-aware scheduling.

---

## Important Current Baseline (Reverted Before Starting Replay Work)

Before the serving-simulator study, simulator behavior was restored to the previous/default assumptions:

1. `LPDDR5-PIM` PIM commands are back to **single-stage PIM activation**
- Removed `ACTAB-1`, `ACTSB-1`, `ACTPB-1` PIM-specific two-stage variants in `ramulator2/src/dram/impl/LPDDR5-PIM.cpp`
- Restored PIM activation flow to `ACTAB`, `ACTSB`, `ACTPB`

2. PIM offload in Python simulator is back to **generation stage only** (default behavior)
- Summarization/prefill (`sum`) stage runs on GPU again
- Generation (`gen`) stage uses `Acc` path (`PIM`) for attention-related layers in `dgx-attacc`

This is important because the replay simulator currently assumes the same default calibration behavior when it reads `output.csv`-style profiles.

---

## Motivation for a Trace Replay Simulator

`main.py` is useful for a single scenario:

- one `(Lin, Lout, batch)`
- one hardware config
- one routing mode (`dgx` or `dgx-attacc`)

Real serving is different:

- requests arrive over time
- each request has different prompt/decode lengths
- queues build up
- resource contention changes the best routing choice
- tail latency matters more than average latency

Accordingly, a replay layer was added to simulate:

- request arrivals from a real trace
- per-request route choice (`gpu_only` vs hybrid)
- GPU/PIM resource occupancy over time
- TTFT/TBT/E2E latency distributions

---

## Azure Trace Input (Provided and Missing Signals)

Dataset used:

- Azure dataset: `AzureLLMInferenceTrace_code.csv`

Columns:

- `TIMESTAMP`
- `ContextTokens`
- `GeneratedTokens`

Useful for:

- request arrival times
- prompt length distribution
- decode length distribution

Not provided by the trace:

- per-token streaming timestamps
- SLO/deadline
- request priority
- cancellation/abort
- batching decisions
- model IDs
- route decisions (GPU-only vs PIM+GPU)

Therefore, this dataset is treated as a **request-arrival workload**, not a full execution trace.

---

## Splitwise-Inspired Concepts

`reference_docs/Splitwise.pdf` was reviewed for scheduling inspiration.

Key ideas directly applicable to this simulator:

- Phase-aware thinking: prefill (`sum`) vs decode (`gen`)
- Queue-aware scheduling
- Tail-latency metrics (`TTFT`, `TBT`, `E2E`)
- Two-level structure (routing policy + local scheduling behavior)
- Future support for SLO-aware scheduling and continuous batching

Items **not** implemented at this stage:

- Splitwise-style prompt/token disaggregation across different machines/pools
- KV-cache transfer optimization across machines

Current scope:

- same simulator environment
- route choice between `gpu_only` and `hybrid_pim_gpu`

---

## Implemented Components (v0)

### New Files

- `src/azure_trace.py`
- `src/scheduler_policy.py`
- `src/trace_replay_sim.py`
- `tools/run_trace_replay.py`

### 1) Azure trace loader (`src/azure_trace.py`)

Responsibilities:

- load Azure CSV from local path or URL
- parse `TIMESTAMP`
- sort by time
- normalize arrival times to `t=0` (milliseconds)
- support:
  - request limit
  - start offset
  - arrival time scaling (to change offered load)

Output object per row:

- `request_id`
- `arrival_ms`
- `context_tokens`
- `generated_tokens`

### 2) Routing policy (`src/scheduler_policy.py`)

Implemented baseline policy:

- queue-aware predicted-finish-time routing

Current route candidates are scored using:

- predicted finish time
- predicted GPU wait
- predicted PIM wait

Implemented heuristic (important):

- `PIM-threshold heuristic`
- If `predicted_pim_wait > X`, reject PIM route and fall back to GPU-only (if available)

This is configurable in the CLI with:

- `--pim-wait-threshold-ms`

### 3) Event-driven replay simulator (`src/trace_replay_sim.py`)

Core responsibilities:

- manage request states
- model prefill and decode phases separately
- simulate GPU/PIM resource contention
- perform admission-time route selection
- produce latency/utilization/throughput metrics

Request lifecycle (v0):

- `new`
- `waiting_prefill`
- `waiting_decode`
- `done`
- `dropped`

Metrics tracked per request:

- `TTFT` (time to first token)
- `E2E` (end-to-end latency)
- `TBT` intervals (time between tokens)
- mapped / clipped `(Lin, Lout)` used for cost lookup

### 4) CLI runner (`tools/run_trace_replay.py`)

Purpose:

- easy execution of replay experiments from Azure trace
- select route profiles (calibration CSVs)
- configure unsupported-length policy
- set routing heuristic threshold
- export summary and per-request results

---

## Cost Model Source (Reuses Existing Simulator Outputs)

The replay simulator does **not** call Ramulator per request.

Instead, it reuses calibrated outputs from `main.py` (`output.csv` schema), such as:

- `cluster_outputs/output_gpu.csv`
- `cluster_outputs/output_hbm3.csv`
- `cluster_outputs/output_lpddr5.csv`

### How fields are interpreted

From each row:

- `s_time`
  - used as prefill (`sum`) end-to-end latency in ms

- `g_time (ms)`
  - used as per-token decode latency in ms
  - this is already averaged over `lout - 1` in `System.simulate()`

Stage breakdown fields (used to approximate resource occupancy split):

- decode GPU-related occupancy:
  - `g_fc`
  - `g_etc`
  - `g2g_comm`

- decode PIM-related occupancy (hybrid routes):
  - `g_matmul`
  - `g_softmax`
  - `c2g_comm`

This occupancy split is used to estimate queue wait and predicted finish time for route selection.

---

## Current Mapping Policy for Unsupported Lengths

Problem:

- Azure trace contains contexts up to ~7.4k tokens
- current PI0 calibration examples are limited (e.g., one point around `Lin=867`, `Lout=50`)

So the replay simulator supports:

- `nearest`
- `clip`
- `drop`

Configurable via:

- `--unsupported-policy`

Diagnostics are reported:

- `mapping_clipped_lin_count`
- `mapping_clipped_lout_count`

This makes it visible when replay results are heavily extrapolated / clipped.

---

## Current Assumptions (v0)

These are intentionally simple to get a correct baseline first:

1. No continuous batching yet
- Requests are scheduled as individual tasks in the event loop
- Phase-aware, but not batched

2. Route is fixed at admission
- No mid-flight switching (`hybrid -> gpu_only` or vice versa)
- This avoids KV placement / migration complexity for now

3. Prefill (`sum`) remains GPU-only by default
- Matches current restored simulator behavior
- There is an optional hook in cost model loading (`--sum-offload-to-pim`) for future experiments

4. Cost lookup uses `output.csv` calibration rows
- Fast replay
- No per-request Ramulator invocation

---

## Design Concepts Reserved for Future Stages

These items are not all fully implemented yet; however, the current design preserves extension points for them:

### Continuous batching (later)

Why it matters:

- decode throughput and tail latency depend strongly on batching

Planned direction:

- add batch selection hooks for prefill/decode
- start with decode continuous batching first

### Deadline / slack (SLO-aware scheduling)

Why it matters:

- real services are often driven by SLO compliance, not only average throughput

Planned direction:

- optional `deadline`
- `slack = deadline - now - predicted_remaining_time`
- EDF/slack-aware policy variants

### PIM-threshold heuristic (already added)

Rule:

- if `predicted_pim_wait > X`, route request to GPU-only

Purpose:

- prevent hybrid queue congestion from hurting TTFT and tail latency

### Tail latency (already reported)

The replay summary reports:

- `TTFT`: `mean`, `p50`, `p95`, `p99`
- `TBT`: `mean`, `p50`, `p95`, `p99`
- `E2E`: `mean`, `p50`, `p95`, `p99`

This is a core metric set for comparing scheduling policies.

---

## Example Commands

### 1) Replay with GPU-only vs LPDDR5-PIM hybrid

```bash
python tools/run_trace_replay.py \
  --limit 1000 \
  --profile gpu_only=cluster_outputs/output_gpu.csv \
  --profile lpddr5_pim_bank=cluster_outputs/output_lpddr5.csv \
  --routes gpu_only lpddr5_pim_bank \
  --unsupported-policy nearest \
  --pim-wait-threshold-ms 5
```

### 2) Higher offered load (stress queueing / tail latency)

```bash
python tools/run_trace_replay.py \
  --limit 2000 \
  --arrival-time-scale 0.05 \
  --profile gpu_only=cluster_outputs/output_gpu.csv \
  --profile hbm3_pim_bank=cluster_outputs/output_hbm3.csv \
  --profile lpddr5_pim_bank=cluster_outputs/output_lpddr5.csv \
  --routes gpu_only lpddr5_pim_bank \
  --unsupported-policy clip \
  --pim-wait-threshold-ms 1 \
  --slo-ttft-ms 100 \
  --slo-e2e-ms 500 \
  --requests-csv cluster_outputs/replay_requests.csv \
  --summary-json cluster_outputs/replay_summary.json
```

---

## Output Summary (What the Replay Prints)

The simulator prints a JSON summary with fields such as:

- request counts
  - `total_requests`
  - `admitted_requests`
  - `completed_requests`
  - `dropped_requests`

- throughput / utilization
  - `throughput_rps`
  - `throughput_tokps`
  - `gpu_util`
  - `pim_util`

- routing behavior
  - `route_counts`
  - `route_fallback_count`

- clipping/extrapolation diagnostics
  - `mapping_clipped_lin_count`
  - `mapping_clipped_lout_count`

- latency metrics
  - `ttft_ms`
  - `tbt_ms`
  - `e2e_ms`

- optional SLO stats
  - `slo_ttft_miss_rate`
  - `slo_e2e_miss_rate`

---

## Relationship to Existing Files / Flow

This new replay layer does **not** replace the existing PI0 + AttAcc flow.

Use the flows together:

1. `docs/pi0_attacc_flow.md`
- calibrate PI0 attention and generate PIM results

2. `main.py` + `output.csv`
- generate route-specific performance profiles (`gpu_only`, `hbm3_pim_bank`, `lpddr5_pim_bank`)

3. `tools/run_trace_replay.py`
- replay many requests with queueing/routing using those profiles

This separation keeps:

- `main.py` as the single-scenario evaluator
- replay simulator as the serving/policy evaluator

---

## Known Limitations (Current v0)

- Only as good as the calibration table (`output.csv` rows)
- No continuous batching yet
- No route switching after admission
- No explicit KV migration cost model
- Azure trace does not contain SLOs or priorities (must be synthesized)
- Many requests may be clipped/mapped if calibration grid is sparse




---

## Files Added in This Stage

- `src/azure_trace.py`
- `src/scheduler_policy.py`
- `src/trace_replay_sim.py`
- `tools/run_trace_replay.py`

---

## Cost Table and ML Interpolation (v1.5)

A denser cost-calibration and interpolation path was added on top of the replay simulator.

### Files

- `tools/gen_cost_table.py`
- `tools/train_cost_model.py`
- `tools/plot_cost_model_diagnostics.py`
- `src/learned_cost_model.py`

### Purpose

The original replay path used only:

- exact lookup
- nearest lookup
- clipping
- drop

This is workable for sparse calibration tables, but it is weak when many requests fall between calibrated points.

The new flow supports a lightweight ML interpolation model:

- per-route training data from `output.csv`-style cost tables
- one model per route
- one regressor per target
- gradient-boosted trees (`HistGradientBoostingRegressor`)

Targets predicted:

- `prefill_e2e_ms`
- `prefill_gpu_ms`
- `prefill_pim_ms`
- `decode_e2e_ms`
- `decode_gpu_ms`
- `decode_pim_ms`

### Why this model

This follows the same general direction as `throttLLeM`, which uses gradient-boosted trees for bounded tabular performance prediction.

This is a good fit here because:

- inputs are low-dimensional (`Lin`, `Lout`, `bs`)
- cost surfaces are nonlinear
- inference-time prediction must stay lightweight

### Modes in replay

`tools/run_trace_replay.py` now supports:

- `--cost-model table`
  - exact/nearest/clip/drop from CSV tables
- `--cost-model ml`
  - learned model only
- `--route-policy min_finish`
  - baseline route choice: smallest predicted finish time
- `--route-policy slack_then_finish`
  - SLO-aware route choice: prefer routes that meet the E2E deadline, otherwise choose least lateness

Note:

- `hybrid_pim_gpu` is still a valid execution route name in the replay scheduler.
- The old `--cost-model hybrid` prediction mode was removed to avoid confusion.
- Cost prediction is now either pure table lookup or pure learned-model prediction.

Recommended mode:

- `ml`

This keeps cost prediction behavior simple: one learned model is used everywhere, with clipping only when a request is outside the trained bounds.

For SLO-aware routing, use:

- `--route-policy slack_then_finish`
- `--slo-e2e-ms <deadline>`

For `v2.1` queue-pressure-aware prediction, add one or more of:

- `--gpu-queue-alpha`
- `--pim-queue-alpha`
- `--active-request-alpha`
- `--decode-token-alpha`

These terms inflate the admission-time predicted finish based on already queued
work, making the predictor less myopic under bursty load.

---

## Dense PI0 Sweep Commands

### Narrow PI0 sweep

Use this command first for a conservative PI0-focused table around the current operating region:

```bash
python tools/gen_cost_table.py \
  --preset pi0-narrow \
  --route gpu_only \
  --route lpddr5_pim_bank \
  --route hbm3_pim_bank \
  --resume \
  --sort
```

### Denser PI0 sweep

Use this command when a denser set of points is required for ML interpolation:

```bash
python tools/gen_cost_table.py \
  --preset pi0-dense \
  --route gpu_only \
  --route lpddr5_pim_bank \
  --resume \
  --sort
```

### Wide Azure-inspired sweep

Use this command when broad heterogeneous coverage is required for ML-based cost prediction:

```bash
python tools/gen_cost_table.py \
  --preset azure-wide \
  --route gpu_only \
  --route lpddr5_pim_bank \
  --resume \
  --sort
```

Built-in presets:

- `pi0-point`
- `pi0-narrow`
- `pi0-dense`
- `azure-wide`

Explicit `--lin-values`, `--lout-values`, and `--batch-values` always override the preset values.

This produces per-route cost tables in:

- `cluster_outputs/cost_tables/gpu_only.csv`
- `cluster_outputs/cost_tables/lpddr5_pim_bank.csv`
- `cluster_outputs/cost_tables/hbm3_pim_bank.csv`

---

## ML Training + Diagnostics Commands

### Train route models

```bash
python tools/train_cost_model.py \
  --profile gpu_only=cluster_outputs/cost_tables/gpu_only.csv \
  --profile lpddr5_pim_bank=cluster_outputs/cost_tables/lpddr5_pim_bank.csv \
  --profile hbm3_pim_bank=cluster_outputs/cost_tables/hbm3_pim_bank.csv \
  --out-dir cluster_outputs/cost_models \
  --summary-json cluster_outputs/cost_models/train_summary.json
```

### Plot interpolation diagnostics

```bash
python tools/plot_cost_model_diagnostics.py \
  --profile gpu_only=cluster_outputs/cost_tables/gpu_only.csv \
  --profile lpddr5_pim_bank=cluster_outputs/cost_tables/lpddr5_pim_bank.csv \
  --profile hbm3_pim_bank=cluster_outputs/cost_tables/hbm3_pim_bank.csv \
  --ml-profile gpu_only=cluster_outputs/cost_models/gpu_only.pkl \
  --ml-profile lpddr5_pim_bank=cluster_outputs/cost_models/lpddr5_pim_bank.pkl \
  --ml-profile hbm3_pim_bank=cluster_outputs/cost_models/hbm3_pim_bank.pkl \
  --out-dir cluster_outputs/cost_model_plots \
  --summary-json cluster_outputs/cost_model_plots/metrics_summary.json
```

This writes:

- `<route>_pred_vs_actual.png`
- `<route>_error_by_shape.png`
- `<route>_metrics.json`

Relevant diagnostics from the current runs:

- `gpu_only` predicted vs actual  
  <img src="../cluster_outputs/cost_model_plots/gpu_only_pred_vs_actual.png" alt="gpu-only cost model predicted vs actual" width="49%">

- `gpu_only` error by shape  
  <img src="../cluster_outputs/cost_model_plots/gpu_only_error_by_shape.png" alt="gpu-only cost model error by shape" width="49%">

- `lpddr5_pim_bank` predicted vs actual  
  <img src="../cluster_outputs/cost_model_plots/lpddr5_pim_bank_pred_vs_actual.png" alt="lpddr5-pim-bank cost model predicted vs actual" width="49%">

- `lpddr5_pim_bank` error by shape  
  <img src="../cluster_outputs/cost_model_plots/lpddr5_pim_bank_error_by_shape.png" alt="lpddr5-pim-bank cost model error by shape" width="49%">

### Replay with ML mode

```bash
python tools/run_trace_replay.py \
  --cost-model ml \
  --limit 2000 \
  --arrival-time-scale 0.05 \
  --ml-profile gpu_only=cluster_outputs/cost_models/gpu_only.pkl \
  --ml-profile lpddr5_pim_bank=cluster_outputs/cost_models/lpddr5_pim_bank.pkl \
  --routes gpu_only lpddr5_pim_bank \
  --unsupported-policy clip \
  --pim-wait-threshold-ms 1 \
  --requests-csv cluster_outputs/replay_requests_ml.csv \
  --summary-json cluster_outputs/replay_summary_ml.json
```

### Replay with ML mode and slack-aware routing

```bash
python tools/run_trace_replay.py \
  --cost-model ml \
  --route-policy slack_then_finish \
  --limit 2000 \
  --arrival-time-scale 0.05 \
  --ml-profile gpu_only=cluster_outputs/cost_models/gpu_only.pkl \
  --ml-profile lpddr5_pim_bank=cluster_outputs/cost_models/lpddr5_pim_bank.pkl \
  --routes gpu_only lpddr5_pim_bank \
  --unsupported-policy clip \
  --pim-wait-threshold-ms 1 \
  --slo-e2e-ms 500 \
  --requests-csv cluster_outputs/replay_requests_ml_slack.csv \
  --summary-json cluster_outputs/replay_summary_ml_slack.json
```

Plot the corresponding replay outputs:

```bash
python tools/plot_replay_results.py \
  --requests-csv cluster_outputs/replay_requests_ml_slack.csv \
  --summary-json cluster_outputs/replay_summary_ml_slack.json \
  --out-dir cluster_outputs/replay_plots \
  --prefix ml_slack
```

### Replay with ML mode, slack-aware routing, and queue-pressure prediction (`v2.1`)

```bash
python tools/run_trace_replay.py \
  --cost-model ml \
  --route-policy slack_then_finish \
  --limit 2000 \
  --arrival-time-scale 0.05 \
  --ml-profile gpu_only=cluster_outputs/cost_models/gpu_only.pkl \
  --ml-profile lpddr5_pim_bank=cluster_outputs/cost_models/lpddr5_pim_bank.pkl \
  --routes gpu_only lpddr5_pim_bank \
  --unsupported-policy clip \
  --pim-wait-threshold-ms 1 \
  --slo-e2e-ms 500 \
  --gpu-queue-alpha 0.05 \
  --pim-queue-alpha 0.05 \
  --decode-token-alpha 0.001 \
  --requests-csv cluster_outputs/replay_requests_ml_slack_qp.csv \
  --summary-json cluster_outputs/replay_summary_ml_slack_qp.json
```

Plot the corresponding replay outputs:

```bash
python tools/plot_replay_results.py \
  --requests-csv cluster_outputs/replay_requests_ml_slack_qp.csv \
  --summary-json cluster_outputs/replay_summary_ml_slack_qp.json \
  --out-dir cluster_outputs/replay_plots \
  --prefix ml_slack_qp
```

## Current v2 Observations

Using:

- `--cost-model ml`
- `--route-policy slack_then_finish`
- `--arrival-time-scale 0.05`
- `--slo-e2e-ms 500`

the replay plots and summaries show:

- `lpddr5_pim_bank` still wins every admission decision
  - route mix stays entirely on the LPDDR5-PIM route
  - slack-aware routing does not yet create route diversity
- the admission-time predictor is optimistic under load
  - in `predicted_vs_actual`, most points sit well above the ideal `y=x` line
  - predicted finish is much smaller than actual E2E latency
- slack is mostly positive at admission, but actual deadline misses are still very high
  - `slo_e2e_miss_rate = 0.9945`
  - this means the slack policy is working on underestimated completion times
- latency is queue-dominated
  - `TTFT` is already multi-second
  - `E2E` is tens of seconds
  - the replay is dominated by queue buildup, not only per-request compute cost
- arrival scatter plots show burst structure
  - requests arriving in dense regions see much higher `TTFT` and `E2E`
  - clipped requests contribute some distortion, but the main effect is queue growth

Interpretation:

- `v2` is implemented correctly, but the current workload and route pair do not create a strong routing tradeoff
- the next limitation is not ML interpolation quality; it is admission-time latency prediction under future queue growth
- the next useful improvement is a `v2.1` queue-pressure-aware predictor before or alongside continuous batching

Related replay plots:

- latency ECDF  
  <img src="../cluster_outputs/replay_plots/ml_slack_latency_ecdf.png" alt="v2 latency ECDF" width="49%">

- arrival scatter  
  <img src="../cluster_outputs/replay_plots/ml_slack_arrival_scatter.png" alt="v2 arrival scatter" width="49%">

- predicted vs actual  
  <img src="../cluster_outputs/replay_plots/ml_slack_predicted_vs_actual.png" alt="v2 predicted vs actual" width="49%">

- route mix  
  <img src="../cluster_outputs/replay_plots/ml_slack_route_mix.png" alt="v2 route mix" width="49%">

- slack histogram  
  <img src="../cluster_outputs/replay_plots/ml_slack_slack_hist.png" alt="v2 slack histogram" width="49%">

## v2.1 Queue-Pressure-Aware Prediction

To address the optimism seen in `predicted_vs_actual`, the replay simulator now
supports additive queue-pressure terms at admission time:

- GPU backlog term: proportional to pending GPU work already in the system
- PIM backlog term: proportional to pending PIM work already in the system
- active-request term: additive penalty per queued request
- pending-decode-token term: additive penalty based on queued decode tokens

These penalties are route-sensitive:

- GPU backlog is weighted by the route's GPU work share
- PIM backlog is weighted by the route's PIM work share

This is intended as a `v2.1` fix for the main weakness exposed by `v2`:

- the original predictor only saw current free times (`gpu_free_ms`, `pim_free_ms`)
- it did not account for already admitted waiting work
- under bursty load, this caused systematically optimistic slack estimates

Diagnostics added to the per-request CSV:

- `chosen_queue_pressure_ms`
- `chosen_predicted_finish_ms`
- `chosen_slack_ms`

Related replay plots:

- latency ECDF  
  <img src="../cluster_outputs/replay_plots/ml_slack_qp_latency_ecdf.png" alt="v2.1 latency ECDF" width="49%">

- arrival scatter  
  <img src="../cluster_outputs/replay_plots/ml_slack_qp_arrival_scatter.png" alt="v2.1 arrival scatter" width="49%">

- predicted vs actual  
  <img src="../cluster_outputs/replay_plots/ml_slack_qp_predicted_vs_actual.png" alt="v2.1 predicted vs actual" width="49%">

- route mix  
  <img src="../cluster_outputs/replay_plots/ml_slack_qp_route_mix.png" alt="v2.1 route mix" width="49%">

- slack histogram  
  <img src="../cluster_outputs/replay_plots/ml_slack_qp_slack_hist.png" alt="v2.1 slack histogram" width="49%">

- queue-pressure histogram  
  <img src="../cluster_outputs/replay_plots/ml_slack_qp_queue_pressure_hist.png" alt="v2.1 queue-pressure histogram" width="49%">

- queue pressure vs E2E  
  <img src="../cluster_outputs/replay_plots/ml_slack_qp_queue_pressure_vs_e2e.png" alt="v2.1 queue pressure vs E2E" width="49%">

## SLO Sweep Findings

Sweeping:

- `--route-policy slack_then_finish`
- `--arrival-time-scale 0.05`
- `--slo-e2e-ms` in `{500, 1000, 2000, 5000, 10000, 20000, 50000, 100000}`

shows:

- changing the SLO changes only the reported `slo_e2e_miss_rate`
- throughput, p95 latencies, route counts, and utilization remain unchanged
- all requests still route to `lpddr5_pim_bank`
- `route_fallback_count` stays `0`

Observed miss-rate trend:

- `500` -> `0.9945`
- `1000` -> `0.9940`
- `2000` -> `0.9800`
- `5000` -> `0.9685`
- `10000` -> `0.9685`
- `20000` -> `0.9685`
- `50000` -> `0.8230`
- `100000` -> `0.0630`

Interpretation:

- for the current route pair (`gpu_only`, `lpddr5_pim_bank`), slack-aware routing is not behaviorally distinct from `min_finish`
- the SLO sweep is currently evaluation-only, not policy-active
- the deadline threshold changes how many requests count as misses, but it does not change the chosen route or the underlying queue dynamics
- this confirms that the next missing capability is a stronger queue-growth-aware prediction model, not another SLO sweep

## v2.1 Queue-Pressure Sweep Findings

Sweeping queue-pressure coefficients with:

- `gpu_queue_alpha` in `{0.0, 0.02, 0.05, 0.1}`
- `pim_queue_alpha` in `{0.0, 0.02, 0.05}`
- `decode_token_alpha` in `{0.0, 0.0005, 0.001, 0.002}`

shows:

- `slo_e2e_miss_rate` stays essentially flat across the sweep
- route selection still does not change
  - all requests continue to route to `lpddr5_pim_bank`
  - `route_fallback_count` remains `0`
- the only visible behavioral change is throughput degradation in the corner where:
  - `gpu_queue_alpha = 0.0`
  - `pim_queue_alpha > 0`

Examples from the sweep:

- `g=0.0, p=0.02, d=0.0` -> `396.57 tok/s`
- `g=0.0, p=0.05, d=0.0` -> `364.62 tok/s`
- once `gpu_queue_alpha > 0`, throughput returns close to the baseline `402.83 tok/s`

Interpretation:

- the current additive queue-pressure correction is active, but it is not yet improving deadline outcomes
- in this coefficient range, it mostly penalizes the PIM route when GPU pressure is not also modeled
- adding a GPU-pressure term cancels that asymmetric penalty, but does not create a useful routing tradeoff
- `v2.1` improves admission-time pessimism somewhat, but it is still not strong enough to change the system-level outcome

Implication:

- the next useful improvement is not more coefficient sweeping
- the next useful improvement is either:
  - a more structured queue-growth model with separate GPU/PIM phase backlog terms, or
  - moving to `v3` continuous batching where queue dynamics become materially different

## v3 Decode Continuous Batching

`v3.0` adds decode-side continuous batching to the replay simulator.

Scope:

- route choice is still fixed at admission time
- prefill remains single-request
- decode steps can now execute as same-route batches
- batch membership is dynamic from step to step

Current implementation details:

- batching is enabled with `--enable-decode-batching`
- batch size is capped with `--max-decode-batch-size`
- only requests in `waiting_decode` are eligible for batching
- only same-route requests are batched together
- the scheduler forms decode batches from the earliest-ready cohort for each route

Batch cost estimation:

- for a decode batch of size `B`, the simulator queries the cost model with:
  - `Lin = max(current_context_tokens)` across the batched requests
  - `Lout = 2` to represent a single decode token step
  - `bs = B`
- each completed batch decrements `remaining_decode_tokens` by `1` for all
  requests in the batch

New summary diagnostics:

- `decode_batching.enabled`
- `decode_batching.mean_batch_size`
- `decode_batching.max_batch_size`
- `decode_batching.num_decode_steps`

Current status:

- `v3.0` is implemented as a first decode-only batching pass
- it changes execution semantics correctly, but it still needs calibration and
  evaluation on larger replay runs before drawing performance conclusions
- the next follow-up is to compare batched vs non-batched replay under the same
  workload and then decide whether prefill batching (`v3.1`) is worthwhile

## v3.1 Prompt-Aware Decode Batching

`v3.1` adds prompt-protection controls on top of decode batching so batched
decode does not starve prefill work.

New controls:

- `--prefill-guard-ms`
  - if any waiting prefill has waited at least this long, schedule a prefill task next
- `--max-consecutive-decode-batches`
  - if prefills are waiting, force a prefill task after this many consecutive decode batches
- `--decode-batch-cap-with-prefill`
  - if prefills are waiting, cap decode batch size to this value

New diagnostics:

- `prefill_wait_ms`
- `local_scheduling.prefill_guard_trigger_count`
- `local_scheduling.decode_limit_trigger_count`

Example:

```bash
python tools/run_trace_replay.py \
  --cost-model ml \
  --route-policy min_finish \
  --limit 2000 \
  --arrival-time-scale 0.05 \
  --ml-profile gpu_only=cluster_outputs/cost_models/gpu_only.pkl \
  --ml-profile lpddr5_pim_bank=cluster_outputs/cost_models/lpddr5_pim_bank.pkl \
  --routes gpu_only lpddr5_pim_bank \
  --unsupported-policy clip \
  --pim-wait-threshold-ms 1 \
  --enable-decode-batching \
  --max-decode-batch-size 4 \
  --prefill-guard-ms 100 \
  --max-consecutive-decode-batches 2 \
  --decode-batch-cap-with-prefill 2 \
  --requests-csv cluster_outputs/replay_requests_v31.csv \
  --summary-json cluster_outputs/replay_summary_v31.json
```

Sweep and plot commands:

```bash
bash tools/run_v31_sweep.sh
```

```bash
python tools/plot_v31_sweep_results.py \
  --summary-dir cluster_outputs/v31_sweep \
  --out-dir cluster_outputs/v31_sweep_plots \
  --prefix v31
```

## v3 Decode-Batching Findings

Comparing:

- `nobatch`
- `batch2`
- `batch4`

on the same replay workload shows a clear tradeoff.

Observed summary:

- `nobatch`
  - `throughput_tokps = 402.83`
  - `ttft_p95 = 21723 ms`
  - `e2e_p95 = 102239 ms`
  - `tbt_p95 = 3109 ms`
- `batch2`
  - `throughput_tokps = 406.32`
  - `ttft_p95 = 60164 ms`
  - `e2e_p95 = 113477 ms`
  - `tbt_p95 = 1491 ms`
- `batch4`
  - `throughput_tokps = 286.25`
  - `ttft_p95 = 138841 ms`
  - `e2e_p95 = 182471 ms`
  - `tbt_p95 = 790 ms`

Batching statistics:

- `nobatch`
  - `mean_batch_size = 1.00`
  - `num_decode_steps = 57024`
- `batch2`
  - `mean_batch_size = 1.95`
  - `num_decode_steps = 29286`
- `batch4`
  - `mean_batch_size = 3.63`
  - `num_decode_steps = 15709`

Route mix:

- `nobatch`: all `2000` requests route to `lpddr5_pim_bank`
- `batch2`: `114` requests route to `gpu_only`, `1886` to `lpddr5_pim_bank`
- `batch4`: `11` requests route to `gpu_only`, `1989` to `lpddr5_pim_bank`

Interpretation:

- decode batching is now behaviorally active
- `batch2` gives a small throughput improvement
- both `batch2` and `batch4` significantly improve `TBT`
- larger decode batches hurt `TTFT` and `E2E` badly
- `batch4` is too aggressive with the current batching policy and cost model
- the system becomes increasingly decode-heavy, which improves token cadence but starves prompt service

Resource utilization trend:

- GPU utilization stays very high in all cases
- PIM utilization drops as decode batching becomes more aggressive
  - this suggests the current batched execution model is pushing the system toward more GPU-dominated behavior rather than better hybrid overlap

Conclusion:

- `v3.0` decode batching works mechanically and exposes a real throughput-vs-latency tradeoff
- small decode batching (`batch2`) is the only promising operating point so far
- the next useful improvement is not simply larger batch size
- the next useful improvement is better local scheduling so decode batching does not starve prompt service

Related plots:

- Throughput  
  <img src="../cluster_outputs/decode_batch_sweep_plots/decode_batch_throughput.png" alt="v3 decode-batch throughput" width="49%">

- `TTFT p95`  
  <img src="../cluster_outputs/decode_batch_sweep_plots/decode_batch_ttft_p95.png" alt="v3 decode-batch TTFT p95" width="49%">

- `E2E p95`  
  <img src="../cluster_outputs/decode_batch_sweep_plots/decode_batch_e2e_p95.png" alt="v3 decode-batch E2E p95" width="49%">

- `TBT p95`  
  <img src="../cluster_outputs/decode_batch_sweep_plots/decode_batch_tbt_p95.png" alt="v3 decode-batch TBT p95" width="49%">

- Mean decode batch size  
  <img src="../cluster_outputs/decode_batch_sweep_plots/decode_batch_mean_batch_size.png" alt="v3 mean decode batch size" width="49%">

- Decode steps  
  <img src="../cluster_outputs/decode_batch_sweep_plots/decode_batch_num_decode_steps.png" alt="v3 decode steps" width="49%">

- Resource utilization  
  <img src="../cluster_outputs/decode_batch_sweep_plots/decode_batch_utilization.png" alt="v3 decode-batch utilization" width="49%">

## v3.1 Prompt-Aware Batching Findings

Comparing:

- `nobatch`
- `batch2`
- `batch4`
- `batch4_guard`
- `batch4_guard_cap`

shows that the prompt-aware cap is the first `v3` mechanism that materially
improves system-level behavior.

Observed summary:

- `nobatch`
  - `throughput_tokps = 402.83`
  - `ttft_p95 = 21723 ms`
  - `prefill_wait_p95 = 21706 ms`
  - `e2e_p95 = 102239 ms`
  - `tbt_p95 = 3109 ms`
- `batch2`
  - `throughput_tokps = 406.32`
  - `ttft_p95 = 60164 ms`
  - `prefill_wait_p95 = 60132 ms`
  - `e2e_p95 = 113477 ms`
  - `tbt_p95 = 1491 ms`
- `batch4`
  - `throughput_tokps = 286.25`
  - `ttft_p95 = 138841 ms`
  - `prefill_wait_p95 = 138815 ms`
  - `e2e_p95 = 182471 ms`
  - `tbt_p95 = 790 ms`
- `batch4_guard`
  - effectively identical to `batch4`
- `batch4_guard_cap`
  - `throughput_tokps = 477.47`
  - `ttft_p95 = 60164 ms`
  - `prefill_wait_p95 = 60132 ms`
  - `e2e_p95 = 99872 ms`
  - `tbt_p95 = 1018 ms`

Main observations:

- `batch4_guard` alone does not help, even though `prefill_guard_trigger_count` is high
- the real improvement comes from `decode_batch_cap_with_prefill`
- `batch4_guard_cap` is the best overall point so far
  - highest throughput
  - best `E2E p95`
  - much better `TBT p95` than `nobatch`
- `TTFT p95` is still much worse than `nobatch`
- the cap matters more than the guard by itself

Route and utilization behavior:

- `batch4_guard_cap` restores a healthier route mix
  - `gpu_route_frac = 0.057`, matching `batch2`
- `batch4_guard_cap` also recovers PIM utilization relative to unguarded `batch4`
  - `batch4`: `pim_util = 0.0436`
  - `batch4_guard_cap`: `pim_util = 0.0712`

Interpretation:

- prompt-aware decode batching is necessary once decode batch size becomes large
- simply forcing prefill occasionally is not enough
- limiting decode batch size while prefills are waiting is what actually changes queue dynamics
- `v3.1` makes batching useful at system level, but prompt latency is still too high

Conclusion:

- `batch4_guard_cap` is the new best configuration in the current `v3` study
- the next tuning step should focus on:
  - `decode_batch_cap_with_prefill`
  - `prefill_guard_ms`
  - `max_consecutive_decode_batches`

Related plots:

- Throughput  
  <img src="../cluster_outputs/v31_sweep_plots/v31_throughput.png" alt="v3.1 throughput" width="49%">

- `TTFT p95`  
  <img src="../cluster_outputs/v31_sweep_plots/v31_ttft_p95.png" alt="v3.1 TTFT p95" width="49%">

- `Prefill wait p95`  
  <img src="../cluster_outputs/v31_sweep_plots/v31_prefill_wait_p95.png" alt="v3.1 prefill wait p95" width="49%">

- `E2E p95`  
  <img src="../cluster_outputs/v31_sweep_plots/v31_e2e_p95.png" alt="v3.1 E2E p95" width="49%">

- `TBT p95`  
  <img src="../cluster_outputs/v31_sweep_plots/v31_tbt_p95.png" alt="v3.1 TBT p95" width="49%">

- Mean decode batch size  
  <img src="../cluster_outputs/v31_sweep_plots/v31_mean_batch_size.png" alt="v3.1 mean decode batch size" width="49%">

- Decode steps  
  <img src="../cluster_outputs/v31_sweep_plots/v31_num_decode_steps.png" alt="v3.1 decode steps" width="49%">

- GPU-only route fraction  
  <img src="../cluster_outputs/v31_sweep_plots/v31_gpu_route_frac.png" alt="v3.1 GPU-only route fraction" width="49%">

- Prefill guard triggers  
  <img src="../cluster_outputs/v31_sweep_plots/v31_prefill_guard_triggers.png" alt="v3.1 prefill guard triggers" width="49%">

- Decode-limit triggers  
  <img src="../cluster_outputs/v31_sweep_plots/v31_decode_limit_triggers.png" alt="v3.1 decode limit triggers" width="49%">

- Resource utilization  
  <img src="../cluster_outputs/v31_sweep_plots/v31_utilization.png" alt="v3.1 utilization" width="49%">

## v3.1 Tuning Sweep

After the initial `v3.1` sweep, the next step was to tune the three local
scheduling knobs:

- `decode_batch_cap_with_prefill`
- `prefill_guard_ms`
- `max_consecutive_decode_batches`

Sweep command:

```bash
bash tools/run_v31_tuning_sweep.sh
```

Plot command:

```bash
python tools/plot_v31_tuning_sweep_results.py \
  --summary-dir cluster_outputs/v31_tuning_sweep \
  --out-dir cluster_outputs/v31_tuning_sweep_plots \
  --prefix v31_tuning
```

The sweep covered:

- `decode_batch_cap_with_prefill = 1, 2, 3`
- `prefill_guard_ms = 25, 50, 100, 200`
- `max_consecutive_decode_batches = 1, 2, 4`

### Main observations

The tuning sweep shows a very strong and simple pattern:

- only `decode_batch_cap_with_prefill` materially changes the result
- `prefill_guard_ms` has no visible effect in this sweep
- `max_consecutive_decode_batches` also has no visible effect in this sweep

For a fixed `decode_batch_cap_with_prefill`, the rows are effectively
identical across all guard and consecutive-decode settings.

### Best configuration

The best point in the sweep is:

- `decode_batch_cap_with_prefill = 1`

Representative metrics for `cap=1`:

- `throughput_tokps = 717.10`
- `ttft_p95 = 21764.53 ms`
- `prefill_wait_p95 = 21746.63 ms`
- `e2e_p95 = 58565.19 ms`
- `tbt_p95 = 791.22 ms`
- `gpu_route_frac = 0.0`
- `mean_batch_size = 3.639`

Compared with the earlier `nobatch` baseline:

- throughput improves from `402.83` to `717.10 tok/s`
- `TTFT p95` stays almost unchanged
- `E2E p95` improves substantially
- `TBT p95` improves substantially

Compared with the earlier `batch4_guard_cap` point:

- throughput improves further
- `TTFT p95` improves a lot
- `E2E p95` improves a lot
- `TBT p95` also improves

### Interpretation

The tuning sweep indicates that the effective rule is:

- when prefills are waiting, decode should be capped very aggressively
- `cap=1` is the useful operating point
- `cap=2` and `cap=3` remain in the worse prompt-starvation regime

This means:

- the important control is not the guard timing itself
- the important control is whether decode is allowed to keep batching while
  prompt work is queued

In practice, `cap=1` behaves like:

- keep large decode batches when no prompt is waiting
- once prompt work appears, restrict decode to single-request service until
  the prompt backlog clears

That preserves most of the batching benefit without allowing decode to dominate
prompt latency.

### Practical conclusion

The tuning sweep sharpens the earlier `v3.1` conclusion:

- `decode_batch_cap_with_prefill` is the key `v3.1` knob
- the best value in the current workload is `1`
- `prefill_guard_ms` and `max_consecutive_decode_batches` are not currently
  useful tuning axes

So the current best prompt-aware decode batching policy is:

- large decode batches when prompts are absent
- `decode_batch_cap_with_prefill = 1` when prompts are waiting

### Related plots

- Throughput heatmap  
  <img src="../cluster_outputs/v31_tuning_sweep_plots/v31_tuning_throughput_heatmap.png" alt="v3.1 tuning throughput heatmap" width="49%">

- `TTFT p95` heatmap  
  <img src="../cluster_outputs/v31_tuning_sweep_plots/v31_tuning_ttft_p95_heatmap.png" alt="v3.1 tuning TTFT p95 heatmap" width="49%">

- `Prefill wait p95` heatmap  
  <img src="../cluster_outputs/v31_tuning_sweep_plots/v31_tuning_prefill_wait_p95_heatmap.png" alt="v3.1 tuning prefill wait p95 heatmap" width="49%">

- `E2E p95` heatmap  
  <img src="../cluster_outputs/v31_tuning_sweep_plots/v31_tuning_e2e_p95_heatmap.png" alt="v3.1 tuning E2E p95 heatmap" width="49%">

- `TBT p95` heatmap  
  <img src="../cluster_outputs/v31_tuning_sweep_plots/v31_tuning_tbt_p95_heatmap.png" alt="v3.1 tuning TBT p95 heatmap" width="49%">

- GPU-only route fraction heatmap  
  <img src="../cluster_outputs/v31_tuning_sweep_plots/v31_tuning_gpu_route_frac_heatmap.png" alt="v3.1 tuning GPU route fraction heatmap" width="49%">

- Top throughput configs  
  <img src="../cluster_outputs/v31_tuning_sweep_plots/v31_tuning_throughput_top3.png" alt="v3.1 tuning top throughput configs" width="49%">

- Top `E2E p95` configs  
  <img src="../cluster_outputs/v31_tuning_sweep_plots/v31_tuning_e2e_p95_top3.png" alt="v3.1 tuning top E2E p95 configs" width="49%">

## v3.1 Best-Cap Replot

After identifying `decode_batch_cap_with_prefill = 1` as the best setting in
the tuning sweep, the `v3.1` comparison was rerun with one extra case:

- `batch4_guard_cap1`

This makes the direct comparison:

- `nobatch`
- `batch2`
- `batch4`
- `batch4_guard`
- `batch4_guard_cap` (`cap=2`)
- `batch4_guard_cap1` (`cap=1`, best so far)

Plot command:

```bash
python tools/plot_v31_sweep_results.py \
  --summary-dir cluster_outputs/v31_sweep \
  --out-dir cluster_outputs/v31_sweep_plots_cap1 \
  --prefix v31_cap1
```

### Main observations

- `batch4_guard_cap1` is now clearly the best overall operating point
- it improves throughput strongly over all previous `v3.1` cases
- it brings `TTFT p95` and `prefill_wait_p95` back close to `nobatch`
- it also gives the best `E2E p95`
- it keeps `TBT p95` far better than `nobatch`

Observed summary:

- `nobatch`
  - `throughput_tokps = 402.83`
  - `ttft_p95 = 21723 ms`
  - `e2e_p95 = 102239 ms`
  - `tbt_p95 = 3109 ms`
- `batch4_guard_cap1`
  - `throughput_tokps = 717.10`
  - `ttft_p95 = 21765 ms`
  - `e2e_p95 = 58565 ms`
  - `tbt_p95 = 791 ms`

Interpretation:

- `cap=1` keeps the decode-batching benefit while eliminating most of the
  prompt-starvation penalty seen in `batch2`, `batch4`, and `cap=2`
- the best policy so far is:
  - large decode batches when no prefill is waiting
  - `decode_batch_cap_with_prefill = 1` when prefill is waiting

This is the strongest result in the current replay study.

### Related plots

- Throughput  
  <img src="../cluster_outputs/v31_sweep_plots_cap1/v31_cap1_throughput.png" alt="v3.1 cap1 throughput" width="49%">

- `TTFT p95`  
  <img src="../cluster_outputs/v31_sweep_plots_cap1/v31_cap1_ttft_p95.png" alt="v3.1 cap1 TTFT p95" width="49%">

- `Prefill wait p95`  
  <img src="../cluster_outputs/v31_sweep_plots_cap1/v31_cap1_prefill_wait_p95.png" alt="v3.1 cap1 prefill wait p95" width="49%">

- `E2E p95`  
  <img src="../cluster_outputs/v31_sweep_plots_cap1/v31_cap1_e2e_p95.png" alt="v3.1 cap1 E2E p95" width="49%">

- `TBT p95`  
  <img src="../cluster_outputs/v31_sweep_plots_cap1/v31_cap1_tbt_p95.png" alt="v3.1 cap1 TBT p95" width="49%">

- Mean decode batch size  
  <img src="../cluster_outputs/v31_sweep_plots_cap1/v31_cap1_mean_batch_size.png" alt="v3.1 cap1 mean batch size" width="49%">

- Decode steps  
  <img src="../cluster_outputs/v31_sweep_plots_cap1/v31_cap1_num_decode_steps.png" alt="v3.1 cap1 decode steps" width="49%">

- GPU-only route fraction  
  <img src="../cluster_outputs/v31_sweep_plots_cap1/v31_cap1_gpu_route_frac.png" alt="v3.1 cap1 GPU route fraction" width="49%">

- Resource utilization  
  <img src="../cluster_outputs/v31_sweep_plots_cap1/v31_cap1_utilization.png" alt="v3.1 cap1 utilization" width="49%">

## v3.2 Prefill Batching

`v3.2` adds same-route prefill batching on top of the current best `v3.1`
decode policy:

- decode batching enabled
- `max_decode_batch_size = 4`
- `decode_batch_cap_with_prefill = 1`
- route policy unchanged
- no mixed-route batching

Implementation summary:

- new replay config knobs:
  - `enable_prefill_batching`
  - `max_prefill_batch_size`
- prefill batches are built per route from ready `waiting_prefill` requests
- prefill batch cost uses:
  - `Lin = max(context_tokens)`
  - `Lout = max(generated_tokens)`
  - `bs = batch size`
- batched prefills complete all requests in the batch at the same `finish_ms`
- replay summary now includes:
  - `prefill_batching.enabled`
  - `prefill_batching.mean_batch_size`
  - `prefill_batching.max_batch_size`
  - `prefill_batching.num_prefill_steps`

Run command (single run example):

```bash
python tools/run_trace_replay.py \
  --cost-model ml \
  --route-policy min_finish \
  --limit 2000 \
  --arrival-time-scale 0.05 \
  --ml-profile gpu_only=cluster_outputs/cost_models/gpu_only.pkl \
  --ml-profile lpddr5_pim_bank=cluster_outputs/cost_models/lpddr5_pim_bank.pkl \
  --routes gpu_only lpddr5_pim_bank \
  --unsupported-policy clip \
  --pim-wait-threshold-ms 1 \
  --enable-decode-batching \
  --max-decode-batch-size 4 \
  --prefill-guard-ms 100 \
  --max-consecutive-decode-batches 2 \
  --decode-batch-cap-with-prefill 1 \
  --enable-prefill-batching \
  --max-prefill-batch-size 2 \
  --requests-csv cluster_outputs/replay_requests_v32.csv \
  --summary-json cluster_outputs/replay_summary_v32.json
```

Sweep command:

```bash
bash tools/run_v32_sweep.sh
```

Plot command:

```bash
python tools/plot_v32_sweep_results.py \
  --summary-dir cluster_outputs/v32_sweep \
  --out-dir cluster_outputs/v32_sweep_plots \
  --prefix v32
```

Cases in the `v3.2` comparison:

- `v31_best`
- `v32_prefill2`
- `v32_prefill4`

Validation status:

- static checks passed
- a small table-based replay run with prefill batching completed successfully
- full `v3.2` sweep results are the next measurement step

Expected evaluation focus:

- `throughput_tokps`
- `ttft_p95`
- `prefill_wait_p95`
- `e2e_p95`
- `tbt_p95`
- `gpu_route_frac`
- `decode_batching.*`
- `prefill_batching.*`

## v3.2 Findings

From `cluster_outputs/v32_sweep_plots/v32_summary.csv`:

- `v31_best`
  - `throughput_tokps = 717.10`
  - `ttft_p95 = 21764.53 ms`
  - `prefill_wait_p95 = 21746.63 ms`
  - `e2e_p95 = 58565.19 ms`
  - `tbt_p95 = 791.22 ms`
- `v32_prefill2`
  - `throughput_tokps = 603.997`
  - `ttft_p95 = 36493.26 ms`
  - `prefill_wait_p95 = 36422.43 ms`
  - `e2e_p95 = 73999.21 ms`
  - `tbt_p95 = 791.22 ms`
- `v32_prefill4`
  - `throughput_tokps = 531.923`
  - `ttft_p95 = 48837.84 ms`
  - `prefill_wait_p95 = 48654.17 ms`
  - `e2e_p95 = 87217.61 ms`
  - `tbt_p95 = 789.59 ms`

Main observations:

- `v31_best` remains the best overall configuration
- prefill batching is active mechanically:
  - `prefill_mean_batch_size` rises `1.00 -> 1.996 -> 3.960`
  - `prefill_steps` fall `2000 -> 1002 -> 505`
- larger prefill batches hurt prompt latency monotonically:
  - `ttft_p95`: `21764 -> 36493 -> 48838`
  - `prefill_wait_p95`: `21747 -> 36422 -> 48654`
- throughput and `e2e_p95` also degrade monotonically:
  - throughput: `717.10 -> 603.99 -> 531.92`
  - `e2e_p95`: `58565 -> 73999 -> 87218`
- decode-side behavior stays almost unchanged:
  - `decode_mean_batch_size`: `3.639`, `3.632`, `3.627`
  - `decode_steps`: `15670`, `15700`, `15724`
  - `tbt_p95` stays near `790 ms`
- route behavior barely changes:
  - `gpu_route_frac`: `0.000`, `0.000`, `0.001`
- utilization shifts in a bad direction:
  - `gpu_util`: `0.916 -> 0.935 -> 0.946`
  - `pim_util`: `0.108 -> 0.092 -> 0.081`

Interpretation:

- current `v3.2` prefill batching is implemented correctly but is not beneficial for this workload
- batching prefills forces requests to wait for the batch and then complete prefill together, which amplifies prompt delay
- reducing the number of prefill steps does not compensate for the extra prompt-side waiting
- the likely weak point is the coarse prefill batch approximation:
  - `Lin = max(context_tokens)`
  - `Lout = max(generated_tokens)`
  - `bs = batch size`

Bottom line:

- `v31_best` remains the working policy
- `v32_prefill2` and `v32_prefill4` are both regressions
- prefill batching is a negative result in the current form
- if revisited later, the prefill batch cost model or batch completion policy should change first

Related plots:

- Throughput  
  <img src="../cluster_outputs/v32_sweep_plots/v32_throughput.png" alt="v3.2 throughput" width="49%">

- `TTFT p95`  
  <img src="../cluster_outputs/v32_sweep_plots/v32_ttft_p95.png" alt="v3.2 TTFT p95" width="49%">

- `Prefill wait p95`  
  <img src="../cluster_outputs/v32_sweep_plots/v32_prefill_wait_p95.png" alt="v3.2 prefill wait p95" width="49%">

- `E2E p95`  
  <img src="../cluster_outputs/v32_sweep_plots/v32_e2e_p95.png" alt="v3.2 E2E p95" width="49%">

- `TBT p95`  
  <img src="../cluster_outputs/v32_sweep_plots/v32_tbt_p95.png" alt="v3.2 TBT p95" width="49%">

- Mean decode batch size  
  <img src="../cluster_outputs/v32_sweep_plots/v32_decode_mean_batch_size.png" alt="v3.2 mean decode batch size" width="49%">

- Mean prefill batch size  
  <img src="../cluster_outputs/v32_sweep_plots/v32_prefill_mean_batch_size.png" alt="v3.2 mean prefill batch size" width="49%">

- Decode steps  
  <img src="../cluster_outputs/v32_sweep_plots/v32_decode_steps.png" alt="v3.2 decode steps" width="49%">

- Prefill steps  
  <img src="../cluster_outputs/v32_sweep_plots/v32_prefill_steps.png" alt="v3.2 prefill steps" width="49%">

- GPU-only route fraction  
  <img src="../cluster_outputs/v32_sweep_plots/v32_gpu_route_frac.png" alt="v3.2 GPU route fraction" width="49%">

- Resource utilization  
  <img src="../cluster_outputs/v32_sweep_plots/v32_utilization.png" alt="v3.2 utilization" width="49%">

## Current state
* v0: 
  - trace replay
  - admission-time route choice
  - no batching

* v1: 
  -  PIM-threshold heuristic
  - tail-latency reporting

* v1.5: 
  - cost-table generation
  - learned ML cost model
  - diagnostics plots

* v2: 
  - slack / deadline-aware routing

* v2.1: 
  - queue-pressure correction on predicted finish

* v3.0: 
  - decode continuous batching

* v3.1: 
  - prompt-aware decode batching
  - decode_batch_cap_with_prefill
  - guard / consecutive-decode controls
  - tuning sweep

* v3.2:
  - same-route prefill batching
  - prefill batching summary/diagnostics
  - `v32` comparison sweep and plot script

## ML Retraining (Expanded Range) Findings

The cost predictor was retrained after expanding the cost-table range, and replay was rerun with:

- `--cost-model ml`
- `--route-policy min_finish`
- no batching
- same `arrival-time-scale = 0.05`

Training quality improved substantially with the larger dataset (`840` rows per route):

- `gpu_only`
  - `prefill_e2e_ms`: `R^2 = 0.99983`, `MAPE = 0.12575`, `MAE = 1.64711`
  - `decode_e2e_ms`: `R^2 = 0.9999985`, `MAPE = 0.0003845`, `MAE = 0.001133`
- `lpddr5_pim_bank`
  - `prefill_e2e_ms`: `R^2 = 0.99977`, `MAPE = 0.07506`, `MAE = 0.94280`
  - `decode_e2e_ms`: `R^2 = 0.9999970`, `MAPE = 0.0002259`, `MAE = 0.000479`
  - `decode_pim_ms`: `R^2 = 0.9999982`, `MAPE = 0.003166`, `MAE = 0.000379`

### Replay comparison (`ml` before vs after retrain)

| Metric | Previous `ml` | New `ml` (expanded-range retrain) |
|---|---:|---:|
| `throughput_tokps` | 402.832 | 387.745 |
| `ttft_p95 (ms)` | 21,723.475 | 26,981.824 |
| `e2e_p95 (ms)` | 102,238.599 | 107,924.563 |
| `route_counts.gpu_only` | 0 | 321 |
| `route_counts.lpddr5_pim_bank` | 2000 | 1679 |
| `mapping_clipped_lin_count` | 341 | 116 |
| `mapping_clipped_lout_count` | 60 | 36 |

Interpretation:

- The wider training range reduced clipping significantly, so replay relies less on boundary behavior.
- Route behavior is now more realistic: `gpu_only` is selected for a nontrivial subset of requests (`321/2000`), whereas earlier runs were always LPDDR5-PIM.
- The replay became less optimistic overall (lower throughput, higher p95 latency), which is consistent with reduced clipping and broader shape coverage.
- The predictor quality itself is strong, but under heavy load admission-time prediction is still optimistic relative to actual E2E (queue growth remains the core challenge).

## Predicted Finish-Time Formulation (Admission Model)

The admission model in `src/trace_replay_sim.py` computes predicted completion
for each candidate route in `_predict_route_candidate(...)`.

### Notation

For a request:

- arrival time: `a` (ms)
- input tokens: `L_in`
- output tokens: `L_out`
- decode tokens: `n = max(0, L_out - 1)`

Current resource availability:

- GPU next-free time: `G_0`
- PIM next-free time: `P_0`

Cost-model outputs for route `r`:

- prefill times: `(p_g, p_p, p_e)`
  - GPU, PIM, E2E prefill time
- decode per-token times: `(d_g, d_p, d_e)`
  - GPU, PIM, E2E decode time

Decode totals:

- `D_g = n * d_g`
- `D_p = n * d_p`
- `D_e = n * d_e`

### Stage-start function

The helper `_predict_stage_start(...)` is equivalent to:

`start(ready, x_g, x_p, G, P) = max(ready, [x_g > 0] * G, [x_p > 0] * P)`

where `[x > 0]` indicates inclusion of that resource constraint only if the
stage uses that resource.

### Prefill prediction

- prefill start: `S_p = start(a, p_g, p_p, G_0, P_0)`
- prefill end: `E_p = S_p + p_e`

Resource free-times after reserving prefill service:

- `G_1 = G_0` if `p_g = 0`, else `max(G_0, S_p + p_g)`
- `P_1 = P_0` if `p_p = 0`, else `max(P_0, S_p + p_p)`

### Decode prediction

- decode start: `S_d = start(E_p, D_g, D_p, G_1, P_1)`
- base predicted finish: `F_0 = S_d + D_e`

### Queue-pressure correction (v2.1)

If queue-pressure terms are enabled, an additive correction `Δ` is applied:

`F = F_0 + Δ`

Snapshot values from `_queue_pressure_snapshot(...)`:

- `Q_g`: pending GPU work (ms)
- `Q_p`: pending PIM work (ms)
- `A`: active request count
- `T`: pending decode tokens

Route work shares:

- `w_g = (p_g + D_g) / max(p_e + D_e, ε)`
- `w_p = (p_p + D_p) / max(p_e + D_e, ε)`
- `ε = 1e-9`

Queue-pressure term:

- `Δ = α_g * Q_g * w_g + α_p * Q_p * w_p + α_a * A + α_t * T * max(d_e, 0)`

with coefficients:

- `α_g = gpu_queue_alpha`
- `α_p = pim_queue_alpha`
- `α_a = active_request_alpha`
- `α_t = decode_token_alpha`

### Predicted E2E and wait terms

The candidate stores absolute predicted finish time `F`. Predicted E2E is:

- `E2E_pred = F - a`

Additional wait terms used by routing:

- predicted GPU wait: `W_g = max(0, S_p - a)`
- predicted PIM wait:
  - `W_p = max(0, P_1 - E_p)` when route uses PIM and decode PIM service is nonzero
  - `W_p = 0` otherwise

### Batch-size coupling in prediction

The cost-model query uses predicted lookup batch size from
`_predicted_lookup_batch_size(...)`. Therefore, admission estimates are based on
`(L_in, L_out, b_hat)`, where `b_hat` depends on decode-batching settings and
same-route decode backlog at the current simulation state.

## v4.0 Predictive Admission Control (throttLLeM-style)

### Implemented features

- Admission lifecycle was extended with `waiting_admission` before `waiting_prefill`.
- A strict FCFS admission queue was implemented:
  - the queue head is evaluated first,
  - if the head is rejected (and not marked lost), later arrivals cannot overtake.
- Admission feasibility is evaluated with step-level shadow simulation:
  - candidate route is projected with current local scheduling semantics,
  - projected candidate E2E and mean TBT are checked against SLOs,
  - projected impact on existing non-lost requests is checked.
- Lost-but-run behavior was implemented:
  - if a request is infeasible under all routes, it is marked `is_lost=true`,
  - it is still admitted and executed as best effort.
- New outputs were added:
  - summary: `admission_queue.{max_len,mean_wait_ms}`, `admission.{reject_count,lost_count,shadow_timeout_count}`,
  - request CSV: `admission_time_ms`, `admission_queue_wait_ms`, `admission_attempts`, `admission_last_reject_reason`, `is_lost`.

### Implemented CLI controls

- `--enable-predictive-admission`
- `--slo-tbt-ms`
- `--admission-shadow-max-steps`
- `--admission-retry-interval-ms`

### v4.0 deterministic debug findings

From `cluster_outputs/debug_summary_v40_base.json` (baseline) and `cluster_outputs/debug_summary_v40_pred.json` (predictive admission):

- predictive admission became active immediately:
  - `reject_count: 0 -> 298`
  - `lost_count: 0 -> 5`
  - `admission_queue.max_len: 0 -> 3`
  - `admission_queue.mean_wait_ms: 0 -> 56.5`
- tail latency increased under tight SLOs in this debug scenario:
  - `ttft_p95: 104.66 -> 174.80 ms`
  - `e2e_p95: 705.89 -> 874.33 ms`
  - `tbt_p95: 3.26 -> 4.87 ms`

Interpretation:

- v4.0 admission control is functionally correct and observable.
- Under tight SLO settings and heavy queue pressure, queue-head blocking plus repeated rejects/lost transitions dominate.

Related plots:

- v4.0 latency ECDF  
  <img src="../cluster_outputs/replay_plots/v40_pred_latency_ecdf.png" alt="v4.0 latency ECDF" width="49%">

- v4.0 predicted E2E vs actual E2E  
  <img src="../cluster_outputs/replay_plots/v40_pred_predicted_vs_actual.png" alt="v4.0 predicted vs actual" width="49%">

- v4.0 route mix  
  <img src="../cluster_outputs/replay_plots/v40_pred_route_mix.png" alt="v4.0 route mix" width="49%">

## v4.1 Optional `fcfs_strict` Local Scheduling Mode

### Implemented features

- A new local scheduling policy selector was added:
  - `priority` (default, existing prompt-aware policy),
  - `fcfs_strict` (new, run-to-completion by request).
- In `fcfs_strict`:
  - oldest admitted active request is selected by `(admission_time_ms, request_id)`,
  - prefill then decode is served to completion for that request,
  - prefill/decode batching is operationally disabled (effective batch size `1`),
  - route admission logic remains unchanged.
- Shadow projection parity was added:
  - the admission shadow engine runs the same `fcfs_strict` semantics when selected,
  - this avoids projection/execution policy mismatch.
- Summary now includes:
  - `local_scheduling.local_scheduling_policy`.

### Implemented CLI control

- `--local-scheduling-policy {priority,fcfs_strict}`

### v4.1 findings from current long-run outputs

From `cluster_outputs/replay_summary_v41_fcfs_nopred_l1000.json` (FCFS strict, no predictive admission):

- `total_requests = 1000`
- `throughput_tokps = 52.93`
- `ttft_p95 = 5937.34 ms`
- `e2e_p95 = 6808.17 ms`
- `tbt_p95 = 2.85 ms`
- route mix: `gpu_only = 187`, `lpddr5_pim_bank = 813`

From `cluster_outputs/replay_summary_v41_fcfs_pred_l1000.json` (FCFS strict + predictive admission):

- current file contains `total_requests = 300` (run configuration/output should be verified before direct 1:1 throughput comparison against the 1000-request run),
- `throughput_tokps = 263.85`
- `ttft_p95 = 14392.11 ms`
- `e2e_p95 = 14879.21 ms`
- `tbt_p95 = 2.86 ms`
- admission metrics show heavy gate pressure:
  - `reject_count = 2531`
  - `lost_count = 273`
  - `shadow_timeout_count = 5062`
  - `admission_queue.max_len = 171`
  - `admission_queue.mean_wait_ms = 3494.45`

Interpretation:

- `fcfs_strict` provides deterministic execution semantics and improved interpretability.
- Predictive admission under aggressive SLO/load settings can produce large lost fractions and queue buildup.
- High orange density in v4.1 predictive plots corresponds to `is_lost=true` requests.

Related v4.1 plots:

- FCFS no-predictive latency ECDF  
  <img src="../cluster_outputs/replay_plots/v41_fcfs_nopred_l1000_latency_ecdf.png" alt="v4.1 FCFS no predictive latency ECDF" width="49%">

- FCFS no-predictive predicted E2E vs actual E2E  
  <img src="../cluster_outputs/replay_plots/v41_fcfs_nopred_l1000_predicted_vs_actual.png" alt="v4.1 FCFS no predictive predicted vs actual" width="49%">

- FCFS predictive latency ECDF  
  <img src="../cluster_outputs/replay_plots/v41_fcfs_pred_l1000_latency_ecdf.png" alt="v4.1 FCFS predictive latency ECDF" width="49%">

- FCFS predictive predicted E2E vs actual E2E  
  <img src="../cluster_outputs/replay_plots/v41_fcfs_pred_l1000_predicted_vs_actual.png" alt="v4.1 FCFS predictive predicted vs actual" width="49%">

- FCFS predictive predicted finish vs actual finish  
  <img src="../cluster_outputs/replay_plots/v41_fcfs_pred_l1000_predicted_finish_vs_actual_finish.png" alt="v4.1 FCFS predictive finish vs finish" width="49%">

- FCFS predictive route mix  
  <img src="../cluster_outputs/replay_plots/v41_fcfs_pred_l1000_route_mix.png" alt="v4.1 FCFS predictive route mix" width="49%">



## v4.2 Prefill-Priority + FCFS-Decode with Alternative Prediction Modes

### Implemented features

- A new local scheduling policy was added:
  - `prefill_priority_fcfs_decode`
- In this policy:
  - prefills remain highest priority at task boundaries,
  - the decode queue is ordered by stable `decode_enqueue_seq`,
  - decode batching is continuous but restricted to the FCFS head plus the contiguous same-route prefix behind it.
- A prediction-mode selector was added for this policy:
  - `shadow`
  - `legacy`
  - `incremental`
- `shadow` reuses the generic forward scheduler projection.
- `legacy` reuses the earlier analytic finish-time predictor.
- `incremental` introduces a compact deterministic forecaster specialized for the `prefill_priority_fcfs_decode` scheduler:
  - it tracks forecasted prefills, decode queue order, resource free times, and remaining decode tokens,
  - it evaluates both route candidates (`gpu_only`, `lpddr5_pim_bank`) at arrival,
  - it refreshes `latest_predicted_finish_ms` for active requests after each new admission and after decode-route rebind.

### Implemented CLI controls

- `--local-scheduling-policy prefill_priority_fcfs_decode`
- `--fcfs-decode-prediction-mode {shadow,legacy,incremental}`

### v4.2 prediction-mode sweep findings

From `cluster_outputs/v42_prediction_mode_sweep_plots/v42_prediction_modes_summary.csv`:

- `shadow`
  - `throughput_tokps = 364.92`
  - `ttft_p95 = 4733.63 ms`
  - `e2e_p95 = 8460.60 ms`
  - `tbt_p95 = 2.878 ms`
  - `gpu_route_frac = 0.0`
  - `decode_mean_batch_size = 3.799`
- `incremental`
  - numerically identical to `shadow` in the 300-request sweep
- `legacy`
  - `throughput_tokps = 186.52`
  - `ttft_p95 = 21542.78 ms`
  - `e2e_p95 = 27138.71 ms`
  - `tbt_p95 = 3.798 ms`
  - `gpu_route_frac = 0.1267`
  - `decode_mean_batch_size = 2.590`

Interpretation:

- `incremental` reproduces the same scheduler-level prediction quality as `shadow` for this workload, while avoiding the generic shadow projection path.
- `legacy` is not adequate for `prefill_priority_fcfs_decode`; it underestimates finish time and materially degrades throughput and latency.
- For this scheduler/workload pair, both `shadow` and `incremental` route all requests to `lpddr5_pim_bank`.
- The `gpu_only` decisions observed in `legacy` appear to be artifacts of prediction error rather than useful routing opportunities.
- Resource utilization is also consistent with this interpretation:
  - `shadow` / `incremental` keep GPU utilization lower and PIM utilization higher than `legacy`,
  - `legacy` shifts more work toward GPU (`gpu_util = 0.902`, `pim_util = 0.048`), which aligns with its inferior routing and latency behavior.

Related sweep plots:

- Throughput  
  <img src="../cluster_outputs/v42_prediction_mode_sweep_plots/v42_prediction_modes_vs_v31_throughput.png" alt="v4.2 prediction-mode throughput vs v3.1" width="49%">

- `TTFT p95`  
  <img src="../cluster_outputs/v42_prediction_mode_sweep_plots/v42_prediction_modes_vs_v31_ttft_p95.png" alt="v4.2 prediction-mode TTFT p95 vs v3.1" width="49%">

- `E2E p95`  
  <img src="../cluster_outputs/v42_prediction_mode_sweep_plots/v42_prediction_modes_vs_v31_e2e_p95.png" alt="v4.2 prediction-mode E2E p95 vs v3.1" width="49%">

- GPU-only route fraction  
  <img src="../cluster_outputs/v42_prediction_mode_sweep_plots/v42_prediction_modes_vs_v31_gpu_route_frac.png" alt="v4.2 prediction-mode GPU route fraction vs v3.1" width="49%">

- Resource utilization  
  <img src="../cluster_outputs/v42_prediction_mode_sweep_plots/v42_prediction_modes_vs_v31_utilization.png" alt="v4.2 prediction-mode utilization vs v3.1" width="49%">

### Finish-prediction diagnostics

From the absolute finish-time scatter plots:

- `incremental` and `shadow` lie almost exactly on the diagonal:
  - predicted finish time and actual finish time are nearly identical,
  - the FCFS-decode forecaster is therefore consistent with the realized scheduler behavior.
- `legacy` remains far from the diagonal:
  - finish times are substantially underpredicted,
  - this explains the degraded latency and throughput in the sweep.
- `v31_best` remains behaviorally strong, but its finish-time prediction is visibly less aligned than the new `v4.2` forecaster.

Interpretation:

- `v4.2 incremental` is currently the best finish-time predictor in the replay framework.
- `v3.1` still provides stronger route diversity and slightly better prompt/E2E tails in the 300-request comparison, but `v4.2 incremental` provides much better finish-time calibration and much lower `TBT p95`.
- The absence of `gpu_only` routing in `v4.2 incremental` indicates that, under the FCFS-decode policy and the current workload, `lpddr5_pim_bank` remains the consistently preferred route once finish-time prediction is corrected.

Related finish-time plots:

- `v31_best` predicted finish vs actual finish  
  <img src="../cluster_outputs/replay_plots/v31_best_l300_predicted_finish_vs_actual_finish.png" alt="v31 best predicted finish vs actual finish" width="49%">

- `v4.2 shadow` predicted finish vs actual finish  
  <img src="../cluster_outputs/replay_plots/v42_fcfs_decode_l1000_predicted_finish_vs_actual_finish.png" alt="v4.2 shadow predicted finish vs actual finish" width="49%">

- `v4.2 incremental` predicted finish vs actual finish  
  <img src="../cluster_outputs/replay_plots/v42_fcfs_decode_incremental_l1000_predicted_finish_vs_actual_finish.png" alt="v4.2 incremental predicted finish vs actual finish" width="49%">

- `v4.2 legacy` predicted finish vs actual finish  
  <img src="../cluster_outputs/replay_plots/v42_fcfs_decode_legacy_l1000_predicted_finish_vs_actual_finish.png" alt="v4.2 legacy predicted finish vs actual finish" width="49%">

### Modeling note: controller overhead

The replay simulator currently models:

- waiting due to GPU/PIM resource contention,
- waiting in `waiting_admission` when predictive admission is enabled.

It does not model the wall-clock control-plane time spent by the scheduler itself to:

- evaluate route candidates,
- run `shadow` or `incremental` forecasters,
- refresh active-request predictions.

Therefore, both predicted and actual request latencies currently assume negligible controller-compute overhead. The existing results should be interpreted under that assumption.

### `heuristic_refresh` prediction model

The `heuristic_refresh` mode was introduced as a cheaper alternative to `incremental` for the `prefill_priority_fcfs_decode` scheduler. The real execution policy remains unchanged:

- prefills are dispatched as single-request tasks,
- prefills have priority at task boundaries,
- decode requests are ordered by `decode_enqueue_seq`,
- decode batching uses the FCFS head plus the contiguous same-route prefix behind it.

Only the prediction mechanism changes. Instead of replaying future tasks step by step, `heuristic_refresh` estimates completion time analytically from the current queue structure.

#### Inputs used by the heuristic

For a candidate request on route `r`, the predictor uses:

- current `gpu_free_ms`,
- current `pim_free_ms`,
- the set of `waiting_decode` requests,
- each decode request's `decode_enqueue_seq`,
- each request's remaining decode tokens,
- each request's route,
- the current decode context length,
- `max_decode_batch_size`,
- the existing ML/table cost model for prefill and decode-step latency.

#### Step 1: prefill estimate

The candidate prefill start is estimated using the same stage-start rule as the main simulator:

```text
t_prefill_start = max(t_arrival, gpu_free_if_needed, pim_free_if_needed)
```

The candidate prefill end is then:

```text
t_prefill_end = t_prefill_start + prefill_e2e(route, Lin, Lout, bs=1)
```

Prefill is always evaluated with `bs = 1` in this mode.

#### Step 2: FCFS decode-queue snapshot

The heuristic then constructs the current decode queue by sorting all `waiting_decode` requests by:

```text
(decode_enqueue_seq, request_id)
```

The candidate request is appended to the tail of this queue, because it can only enter decode after its own prefill completes.

#### Step 3: contiguous route blocks

The FCFS decode queue is then collapsed into maximal contiguous same-route segments. For example:

```text
[r0:H, r1:H, r2:G, r3:G, r4:H, candidate:H]
```

becomes:

- block 1: `[r0:H, r1:H]`
- block 2: `[r2:G, r3:G]`
- block 3: `[r4:H, candidate:H]`

This block representation is used because the FCFS-prefix batching policy can only batch the head request with the contiguous same-route prefix behind it.

#### Step 4: per-block decode-step estimate

For each block `b`:

- `n_b` is the number of requests in the block,
- `bs_eff_b = min(max_decode_batch_size, n_b)` if decode batching is enabled, otherwise `1`,
- `Lin_b = max(context_tokens + 1 + decoded_tokens_done)` across all requests in the block.

The heuristic then queries the cost model for one representative decode step:

```text
step_cost_b = f(route_b, Lin_b, 2, bs_eff_b)
```

where `f(...)` is the ML/table decode-step latency oracle.

#### Step 5: block drain approximation

The total time to drain block `b` is approximated as:

```text
T_b = max_i(remaining_decode_tokens_i) * step_cost_b
```

This is an intentional approximation. It assumes that a block drains over roughly as many rounds as the longest remaining request in that block, with each round costing `step_cost_b`.

#### Step 6: candidate finish estimate

If the candidate request lands in block `k`, its predicted finish time is approximated by:

```text
F_hat_candidate ≈ t_prefill_end
                 + sum_{b < k} T_b
                 + T_intra_block_candidate
```

The first term accounts for the candidate's own prefill. The summation accounts for all earlier FCFS route blocks that must drain before the candidate's block can become the active head block.

#### Step 7: intra-block completion estimate

Inside the candidate's own block, completion is estimated as:

```text
T_intra_block_candidate ≈ remaining_decode_tokens_candidate * step_cost_k
                          + excess_wait_candidate
```

The `excess_wait` term captures the case where the candidate is deeper than the first effective batch window of its block and therefore cannot participate in the earliest rounds immediately.

#### Step 8: wait breakdowns

The heuristic also produces approximate wait components:

```text
predicted_gpu_wait = max(0, t_prefill_start - t_arrival)
```

For hybrid routes, `predicted_pim_wait_ms` is estimated from the delay between candidate prefill completion and the first expected decode opportunity of the candidate block.

#### Step 9: optional alpha residual

After the FCFS block estimate is produced, an optional residual queue-pressure correction may be added when nonzero alpha parameters are supplied:

```text
Delta_alpha =
    alpha_gpu    * pending_gpu_work
  + alpha_pim    * pending_pim_work
  + alpha_active * active_requests
  + alpha_tok    * pending_decode_tokens * decode_step_scale
```

The final heuristic prediction becomes:

```text
F_hat_candidate <- F_hat_candidate + Delta_alpha
```

These alpha terms are residual corrections only. They do not replace the FCFS queue/block calculation.

#### Interpretation

The distinction between the three `v4.2` predictors is therefore:

- `legacy`: immediate resource-free-time estimate plus optional queue-pressure correction,
- `incremental`: scheduler-aligned forward forecast of the currently known queue,
- `heuristic_refresh`: FCFS queue/block approximation with rolling refresh, but without explicit future task replay.

In practice, `heuristic_refresh` is faster than `incremental`, but its block-drain approximation can still underpredict finish time and distort route choice. That effect was visible in the 300-request comparison, where `heuristic_refresh` selected `gpu_only` more often than `incremental` and produced worse system-level latency and throughput.

## v4.3 Deferred SLO-Guarded Admission on Top of FCFS Decode

The next step was to keep the `prefill_priority_fcfs_decode` executor, but place a deferred SLO-aware admission queue in front of it. The admission queue evaluates queued arrivals in FCFS order, but blocked requests can be bypassed by later arrivals that are currently admissible. Requests that remain blocked eventually enter as best-effort traffic.

The real executor is unchanged:

- prefills are still highest-priority work at task boundaries,
- decode still runs one iteration at a time,
- decode batches are still the FCFS head plus the contiguous same-route prefix.

### Baseline comparison

Against the earlier baselines on the 300-request setting:

- `v31_best` remained the best end-to-end latency point in this comparison (`E2E p95 = 7603.84 ms`),
- `v4.2 incremental` and `v4.3 guarded` had almost identical throughput (`364.92` vs `364.91 tok/s`),
- but `v4.3 guarded` with the tested harm budgets produced many forced best-effort admissions:
  - `best_effort_count = 120`
  - `lost_count = 120`
  - `admission_queue.max_len = 33`
  - `admission_queue.mean_wait_ms = 34.75`
  - `slo_e2e_miss_rate = 0.79`

This means the guarded-admission mechanism worked mechanically, but the tested queue budgets were too strict under the `limit=300`, `scale=0.05` load.

![](../cluster_outputs/v43_sweep_plots/v43_vs_baselines_throughput.png)

![](../cluster_outputs/v43_sweep_plots/v43_vs_baselines_e2e_p95.png)

![](../cluster_outputs/v43_sweep_plots/v43_vs_baselines_tbt_p95.png)

![](../cluster_outputs/v43_sweep_plots/v43_vs_baselines_utilization.png)

### Debug behavior

For small debug runs, the queue plots make the guarded-admission behavior readable directly. The key events are:

- arrival to `waiting_admission`,
- repeated `admission_eval` / `admission_hold`,
- `admission_bypass` when a later request is admitted while an earlier one stays queued,
- eventual `admit_route` or `admission_best_effort`.

The debug timeline below is useful because it shows the actual request state progression instead of only aggregate percentiles.

![](../cluster_outputs/replay_plots/v43_guarded_scale1_l10_debug_request_timeline.png)

![](../cluster_outputs/replay_plots/v43_guarded_scale1_l10_debug_queue_counts.png)

### Clean scale=1 TBT sweep

At `scale=1.0`, the system is mostly arrival-limited, not compute-limited. That makes the TBT sweep a clean way to study admission behavior without conflating it with throughput saturation.

Across the clean `scale=1.0` TBT sweep:

- throughput stayed flat at `69.18 tok/s`,
- GPU utilization stayed around `0.124`,
- PIM utilization stayed around `0.018`,
- tighter TBT SLOs mostly increased admission intervention instead of changing throughput.

Examples:

- `slo_tbt_ms = 50`
  - `best_effort_count = 230`
  - `blocked_count = 234`
  - `bypassed_count = 178`
  - `TTFT p95 = 345.40 ms`
- `slo_tbt_ms = 400`
  - `best_effort_count = 44`
  - `blocked_count = 45`
  - `bypassed_count = 10`
  - `TTFT p95 = 315.55 ms`

So at `scale=1.0`, tighter TBT protection mostly controls the admission tail, while throughput is still determined by the arrival trace.

![](../cluster_outputs/v43_scale1_tbt_sweep_plots/v43_scale1_tbt_sweep_admission_counts.png)

![](../cluster_outputs/v43_scale1_tbt_sweep_plots/v43_scale1_tbt_sweep_latency_p95.png)

![](../cluster_outputs/v43_scale1_tbt_sweep_plots/v43_scale1_tbt_sweep_utilization.png)

## v5 Decode-Energy-Aware Routing

The next addition was energy-aware routing on top of the same replay executor. In `v5`, routing still respects latency first, but when two routes are close enough in predicted finish time, the scheduler can prefer the lower-energy route.

### Implementation model

`v5` uses:

- the existing decode energy column already present in the cost tables: `g_energy (nJ)`,
- retrained ML bundles that predict `decode_energy_nj`,
- the incremental FCFS forecaster to accumulate future decode energy,
- a new route policy: `latency_guarded_energy`.

The policy is:

1. predict finish time and incremental decode energy for each route,
2. keep routes within `fastest_finish + 5 ms`,
3. choose the route with the lowest predicted incremental decode energy inside that latency guard.

Important current limitation:

- `v5` is **decode-energy only**
- prefill energy is still not modeled in route selection

### What changed in decision quality

The useful point is that `v5` does not always choose the fastest route. It chooses the lowest-energy route only when the finish-time penalty is small enough.

The `limit=10`, `scale=1.0` debug run shows this clearly:

- `request 4`
  - `gpu_only`: finish `462.759 ms`, energy `1.0936e9 nJ`
  - `lpddr5_pim_bank`: finish `463.710 ms`, energy `1.0736e9 nJ`
  - chosen: `lpddr5_pim_bank`
  - reason: latency gap is only `0.95 ms`, inside the `5 ms` guard, so lower energy wins

- `request 5`
  - `gpu_only`: finish `575.381 ms`, energy `1.3437e9 nJ`
  - `lpddr5_pim_bank`: finish `571.521 ms`, energy `1.5113e9 nJ`
  - chosen: `gpu_only`
  - reason: `gpu_only` is slightly slower (`3.86 ms`), still inside the guard, and lower energy

- `request 7`
  - `gpu_only`: finish `1051.592 ms`, energy `2.1872e9 nJ`
  - `lpddr5_pim_bank`: finish `1052.485 ms`, energy `2.1473e9 nJ`
  - chosen: `lpddr5_pim_bank`
  - reason: finish gap is only `0.89 ms`, so lower energy wins

- `request 8`
  - `gpu_only`: finish `1339.691 ms`, energy `6.2652e8 nJ`
  - `lpddr5_pim_bank`: finish `1326.283 ms`, energy `7.6919e8 nJ`
  - chosen: `lpddr5_pim_bank`
  - reason: finish gap is `13.4 ms`, outside the guard, so latency dominates and energy is ignored

This is the intended behavior: energy is active, but it is a secondary objective inside a bounded latency window.

### Aggregate v5 result so far

For the current `limit=2000`, `scale=1.0` replay:

- route mix:
  - `gpu_only = 63`
  - `lpddr5_pim_bank = 1937`
- `throughput_tokps = 69.18`
- `E2E p95 = 4257.04 ms`
- `TTFT p95 = 314.08 ms`
- `TBT p95 = 2.878 ms`
- actual decode energy:
  - `decode_energy_nj.total = 7.096e12 nJ`
  - `decode_energy_nj_per_decode_token = 1.244e8 nJ/token`

So far, the aggregate result says:

- the workload still strongly prefers the hybrid route,
- energy did not flip the system into GPU-heavy routing,
- but route choice now changes in near-tie cases.

The standard replay plots below confirm that prediction quality remained stable while adding the energy-aware route selection logic.

![](../cluster_outputs/replay_plots/v5_energy_l2000_predicted_finish_vs_actual_finish.png)

![](../cluster_outputs/replay_plots/v5_energy_l2000_route_mix.png)

For the small debug run:

![](../cluster_outputs/replay_plots/v5_energy_l10_debug_predicted_finish_vs_actual_finish.png)

![](../cluster_outputs/replay_plots/v5_energy_l10_debug_route_mix.png)

### How to interpret the new energy fields

`v5` now records two different energy notions:

- `decode_energy_nj` in the cost model:
  - predicted energy of one decode step at a given `(route, Lin, Lout, bs)`
- `chosen_predicted_incremental_decode_energy_nj` in the replay CSV:
  - predicted **marginal future decode energy added to the system** by choosing that route

The second one is the routing signal that matters. It is not just isolated request energy; it includes batching effects through the incremental forecast. That is why `v5` can prefer a route that creates more efficient future decode batching, even if the isolated single-step energy is not the whole story.

## v5 Full-Request Energy and Guarded Bypass Debugging

After the decode-only `v5` pass, the simulator was extended to model **full request energy** instead of decode energy alone.

### Full-request energy implementation

The following changes were added:

- generated cost tables now include:
  - `s_energy (nJ)` for prefill
  - `g_energy (nJ)` for decode
- learned ML bundles now predict:
  - `prefill_energy_nj`
  - `decode_energy_nj`
- runtime service estimates now carry:
  - `prefill_energy_nj`
  - `decode_energy_nj`
- the incremental FCFS forecaster now accumulates:
  - total prefill energy
  - total decode energy
- `latency_guarded_energy` now routes using **full marginal predicted energy**, not only decode energy

This means the routing signal is now:

```text
Delta_E_r = E_with(r) - E_base
```

where `E` includes both forecasted prefill energy and forecasted decode energy.

### Aggregate full-energy run

For the `limit=2000`, `scale=1.0` replay with full-energy ML models:

- `route_counts`
  - `gpu_only = 98`
  - `lpddr5_pim_bank = 1902`
- `energy_nj.total = 1.3682e13`
- `energy_nj.prefill_total = 7.1029e12`
- `energy_nj.decode_total = 6.5794e12`
- `energy_nj_per_request = 6.8412e9`
- `decode_energy_nj_per_decode_token = 1.1538e8`

This shows that:

- the hybrid route still dominates under this workload,
- prefill energy is now a first-class part of the accounting,
- full-request energy can be compared directly against latency metrics in the replay summaries.

![](../cluster_outputs/replay_plots/full_energy_l2000_predicted_finish_vs_actual_finish.png)

![](../cluster_outputs/replay_plots/full_energy_l2000_route_mix.png)

![](../cluster_outputs/replay_plots/full_energy_l2000_latency_ecdf.png)

### Pure-energy routing vs time-only baseline

To isolate the effect of energy-aware routing itself, two matched `limit=10`, `scale=1.0` replays were compared:

- baseline:
  - `route_policy = min_finish`
- pure-energy:
  - `route_policy = latency_guarded_energy`
  - `energy_latency_guard_ms = 1e9`
  - no E2E deadline and no guarded admission

With this setup, routing is effectively:

```text
choose the lowest predicted marginal full-request energy route
among all eligible routes
```

#### Deterministic 10-request trace

For `cluster_outputs/debug_trace_deterministic_heavy_sparse.csv`:

- baseline route mix:
  - `lpddr5_pim_bank = 8`
- pure-energy route mix:
  - `gpu_only = 8`
- total energy:
  - `5.4583e10 nJ -> 3.7586e10 nJ`
  - `-31.1%`
- energy per request:
  - `6.8229e9 nJ -> 4.6982e9 nJ`
  - `-31.1%`
- decode energy per token:
  - `5.1668e7 nJ -> 3.0806e7 nJ`
  - `-40.4%`
- throughput:
  - `1412.6 tok/s -> 1179.8 tok/s`
  - `-16.5%`
- `E2E p95`:
  - `488.5 ms -> 604.7 ms`
  - `+23.8%`

Interpretation:

- on this deterministic debug slice, pure-energy routing flips all requests from hybrid to `gpu_only`,
- that buys a substantial energy reduction,
- but the energy savings come with a clear latency and throughput penalty.

![](../cluster_outputs/full_energy_det_compare_plots/full_energy_det_compare_energy_total.png)

![](../cluster_outputs/full_energy_det_compare_plots/full_energy_det_compare_decode_energy_per_token.png)

![](../cluster_outputs/full_energy_det_compare_plots/full_energy_det_compare_gpu_route_frac.png)

![](../cluster_outputs/full_energy_det_compare_plots/full_energy_det_compare_e2e_p95.png)

![](../cluster_outputs/replay_plots/full_energy_min_finish_det_l10_debug_request_execution_timeline.png)

![](../cluster_outputs/replay_plots/full_energy_pure_energy_det_l10_debug_request_execution_timeline.png)

#### Azure 10-request slice

For the first `10` requests from `AzureLLMInferenceTrace_code.csv` at `scale=1.0`:

- baseline route mix:
  - `lpddr5_pim_bank = 10`
- pure-energy route mix:
  - `gpu_only = 9`
  - `lpddr5_pim_bank = 1`
- total energy:
  - `6.0201e10 nJ -> 4.9413e10 nJ`
  - `-17.9%`
- energy per request:
  - `6.0201e9 nJ -> 4.9413e9 nJ`
  - `-17.9%`
- decode energy per token:
  - `1.2096e8 nJ -> 6.3259e7 nJ`
  - `-47.7%`
- throughput:
  - `93.59 tok/s -> 88.63 tok/s`
  - `-5.3%`
- `E2E p95`:
  - `1482.0 ms -> 1530.5 ms`
  - `+3.3%`
- `TBT p95`:
  - `286.5 ms -> 140.4 ms`
  - `-51.0%`

Interpretation:

- even on a tiny Azure slice, the pure-energy policy materially changes route choice,
- the route mix shifts strongly toward `gpu_only`,
- total energy drops meaningfully,
- latency impact is milder than in the deterministic slice,
- this small sample should be treated as directional, not definitive.

![](../cluster_outputs/replay_plots/full_energy_min_finish_l10_debug_route_mix.png)

![](../cluster_outputs/replay_plots/full_energy_pure_energy_l10_debug_route_mix.png)

![](../cluster_outputs/replay_plots/full_energy_min_finish_l10_debug_latency_ecdf.png)

![](../cluster_outputs/replay_plots/full_energy_pure_energy_l10_debug_latency_ecdf.png)

### Deterministic guarded-bypass debug trace

To make the deferred guarded-admission behavior inspectable, a heterogeneous 8-request deterministic debug trace was added:

- `cluster_outputs/debug_trace_deterministic_bypass8.csv`

This trace is intentionally shaped so that:

- `r2` and `r4` are expensive and get held,
- later short requests such as `r3`, `r5`, `r6`, and `r7` can be admitted first,
- the admission queue therefore exhibits real bypass behavior.

On the table-model guarded run with this trace:

- `bypassed_count = 7`
- `blocked_count = 2`
- `best_effort_count = 2`
- `held_count = 30`
- `admission_queue.max_len = 3`

The queue snapshots show the intended behavior directly:

- `r2` is held,
- `r3` bypasses `r2`,
- `r5` bypasses both `r2` and `r4`,
- `r6` and `r7` also bypass the blocked long requests,
- the long blocked requests are eventually admitted as best-effort after `max_wait_ms`.

![](../cluster_outputs/replay_plots/full_energy_guarded_bypass8_table_waiting_queue_slots.png)

![](../cluster_outputs/replay_plots/full_energy_guarded_bypass8_table_request_execution_timeline.png)

![](../cluster_outputs/replay_plots/full_energy_guarded_bypass8_table_predicted_finish_vs_actual_finish.png)

### Runtime optimization for guarded-admission ML

Full-energy ML guarded-admission runs became significantly slower than table runs because guarded admission repeatedly triggers:

- route estimates for queued requests,
- incremental baseline forecasts,
- projected candidate forecasts,
- multiple ML target predictions per estimate.

To reduce this cost, the simulator now includes:

- stronger estimate caching for repeated `(route, Lin, Lout, bs)` tuples,
- prefetch of request estimates for arrivals and waiting-admission requests,
- cached incremental baseline forecasts when the active state has not changed,
- projected candidate cache keyed by waiting request, route, and baseline signature,
- batched ML target inference in `src/learned_cost_model.py`

Replay summaries now expose `cache_stats`, including:

- `estimate_cache_hits`
- `estimate_cache_misses`
- `estimate_prefetch_populated`
- `incremental_active_forecast_cache_hits`
- `incremental_active_forecast_cache_misses`
- `projection_cache_hits`
- `projection_cache_misses`
- cost-model batched inference counters when using learned models

This makes it possible to see whether runtime is dominated by estimate generation, baseline forecasting, or projected-candidate replay.

## Commands Used

### Main run: full-energy ML replay

```bash
conda run -n lerobot python tools/run_trace_replay.py \
  --azure-trace cluster_outputs/trace_cache/AzureLLMInferenceTrace_code.csv \
  --cost-model ml \
  --route-policy latency_guarded_energy \
  --energy-latency-guard-ms 5 \
  --slo-e2e-ms 5000 \
  --limit 2000 \
  --arrival-time-scale 1.0 \
  --ml-profile gpu_only=cluster_outputs/cost_models_full_energy/gpu_only.pkl \
  --ml-profile lpddr5_pim_bank=cluster_outputs/cost_models_full_energy/lpddr5_pim_bank.pkl \
  --routes gpu_only lpddr5_pim_bank \
  --unsupported-policy clip \
  --local-scheduling-policy prefill_priority_fcfs_decode \
  --fcfs-decode-prediction-mode incremental \
  --enable-decode-batching \
  --max-decode-batch-size 4 \
  --requests-csv cluster_outputs/replay_requests_full_energy_l2000.csv \
  --summary-json cluster_outputs/replay_summary_full_energy_l2000.json
```

### Debug run: deterministic guarded bypass

This is the stable debug configuration used to demonstrate actual bypass behavior:

```bash
python tools/run_trace_replay.py \
  --azure-trace cluster_outputs/debug_trace_deterministic_bypass8.csv \
  --cost-model table \
  --profile gpu_only=cluster_outputs/cost_tables_full_energy/gpu_only.csv \
  --profile lpddr5_pim_bank=cluster_outputs/cost_tables_full_energy/lpddr5_pim_bank.csv \
  --route-policy latency_guarded_energy \
  --energy-latency-guard-ms 5 \
  --slo-e2e-ms 350 \
  --slo-tbt-ms 3 \
  --limit 8 \
  --arrival-time-scale 1.0 \
  --routes gpu_only lpddr5_pim_bank \
  --unsupported-policy clip \
  --local-scheduling-policy prefill_priority_fcfs_decode \
  --fcfs-decode-prediction-mode incremental \
  --enable-decode-batching \
  --max-decode-batch-size 4 \
  --enable-slo-guarded-admission \
  --admission-max-harmed-requests 0 \
  --admission-max-total-harm-ms 0 \
  --admission-max-single-harm-ms 0 \
  --admission-max-own-miss-ms 0 \
  --admission-max-bypass-count 5 \
  --admission-max-wait-ms 150 \
  --admission-aging-harm-ms-per-ms 0.01 \
  --admission-aging-harmed-requests-per-ms 0.001 \
  --admission-retry-interval-ms 10 \
  --requests-csv cluster_outputs/replay_requests_full_energy_guarded_bypass8_table.csv \
  --summary-json cluster_outputs/replay_summary_full_energy_guarded_bypass8_table.json \
  --debug-events-csv cluster_outputs/debug_events_full_energy_guarded_bypass8_table.csv
```

### Plot replay results

For the main full-energy replay:

```bash
python tools/plot_replay_results.py \
  --requests-csv cluster_outputs/replay_requests_full_energy_l2000.csv \
  --summary-json cluster_outputs/replay_summary_full_energy_l2000.json \
  --out-dir cluster_outputs/replay_plots \
  --prefix full_energy_l2000
```

### Plot debug results

Regular replay/debug plots:

```bash
python tools/plot_replay_results.py \
  --requests-csv cluster_outputs/replay_requests_full_energy_guarded_bypass8_table.csv \
  --summary-json cluster_outputs/replay_summary_full_energy_guarded_bypass8_table.json \
  --debug-events-csv cluster_outputs/debug_events_full_energy_guarded_bypass8_table.csv \
  --prediction-history-min-request-id 0 \
  --prediction-history-max-request-id 7 \
  --out-dir cluster_outputs/replay_plots \
  --prefix full_energy_guarded_bypass8_table
```

Admission-queue / execution-debug plots:

```bash
python tools/plot_debug_admission.py \
  --events-csv cluster_outputs/debug_events_full_energy_guarded_bypass8_table.csv \
  --requests-csv cluster_outputs/replay_requests_full_energy_guarded_bypass8_table.csv \
  --out-dir cluster_outputs/replay_plots \
  --prefix full_energy_guarded_bypass8_table
```
