# Trace Replay Serving Simulator (Azure Trace + PI0/AttAcc Cost Model)

## Goal

Move from single-request benchmarking (`main.py`) to a more realistic serving simulation that replays many requests over time and makes routing decisions:

- `gpu_only`
- `hybrid_pim_gpu` (e.g., `lpddr5_pim_bank`, `hbm3_pim_bank`)

This enables queueing-aware evaluation, tail-latency analysis, and later continuous batching / SLO-aware scheduling.

---

## Important Current Baseline (Reverted Before Starting Replay Work)

Before starting the serving simulator work, we intentionally restored the simulator behavior to the previous/default assumptions:

1. `LPDDR5-PIM` PIM commands are back to **single-stage PIM activation**
- Removed `ACTAB-1`, `ACTSB-1`, `ACTPB-1` PIM-specific two-stage variants in `ramulator2/src/dram/impl/LPDDR5-PIM.cpp`
- Restored PIM activation flow to `ACTAB`, `ACTSB`, `ACTPB`

2. PIM offload in Python simulator is back to **generation stage only** (default behavior)
- Summarization/prefill (`sum`) stage runs on GPU again
- Generation (`gen`) stage uses `Acc` path (`PIM`) for attention-related layers in `dgx-attacc`

This is important because the replay simulator currently assumes the same default calibration behavior when it reads `output.csv`-style profiles.

---

## Why We Need a Trace Replay Simulator

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

So we added a new replay layer to simulate:

- request arrivals from a real trace
- per-request route choice (`gpu_only` vs hybrid)
- GPU/PIM resource occupancy over time
- TTFT/TBT/E2E latency distributions

---

## Azure Trace Input (What It Gives / What It Does Not)

We use:

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

So we treat it as a **request arrival workload**, not a full execution trace.

---

## Splitwise-Inspired Concepts We Adopted

We reviewed `reference_docs/Splitwise.pdf` for scheduling inspiration.

Key ideas that directly apply to our simulator:

- Phase-aware thinking: prefill (`sum`) vs decode (`gen`)
- Queue-aware scheduling
- Tail-latency metrics (`TTFT`, `TBT`, `E2E`)
- Two-level structure (routing policy + local scheduling behavior)
- Future support for SLO-aware scheduling and continuous batching

What we are **not** doing yet:

- Splitwise-style prompt/token disaggregation across different machines/pools
- KV-cache transfer optimization across machines

Our current scope is:

- same simulator environment
- route choice between `gpu_only` and `hybrid_pim_gpu`

---

## What We Implemented (v0)

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

## Important Concepts We Intentionally Designed For (Future Stages)

These are not all fully implemented yet, but the current design keeps space for them:

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

## Next Steps (Recommended)

1. Build a denser calibration table
- sweep `(Lin, Lout)` buckets using `main.py`
- separate profiles for:
  - `gpu_only`
  - `hbm3_pim_bank`
  - `lpddr5_pim_bank`

2. Add continuous batching (decode first)
- this will materially improve realism for token-generation behavior

3. Add SLO/deadline-aware policies
- slack-aware routing
- tail-latency optimization

4. Compare policies under the same trace
- FCFS baseline
- queue-aware predicted finish time
- PIM-threshold heuristic
- later: SLO-aware / fairness-aware variants

---

## Files Added in This Stage

- `src/azure_trace.py`
- `src/scheduler_policy.py`
- `src/trace_replay_sim.py`
- `tools/run_trace_replay.py`

---

## Cost Table + ML Interpolation (v1.5)

We added a denser cost-calibration and interpolation path on top of the replay simulator.

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

Use this first if you want a conservative PI0-focused table around the current operating region:

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

Use this when you want enough points for ML interpolation:

```bash
python tools/gen_cost_table.py \
  --preset pi0-dense \
  --route gpu_only \
  --route lpddr5_pim_bank \
  --resume \
  --sort
```

### Wide Azure-inspired sweep

Use this when you want broad heterogeneous coverage for ML-based cost prediction:

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
