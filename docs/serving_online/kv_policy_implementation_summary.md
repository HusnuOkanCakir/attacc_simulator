# KV Cache Policy Implementation — Design & Results

Implemented across the `serving_online` Ramulator2 frontend on branch
`feat/pi0-lpddr5-pim-flow`. All changes live under
`ramulator2/src/frontend/impl/serving_online/` (mirrored to
`pim_ramulator_src/serving_online/`).

---

## What was implemented

### M1 — Chunked decode allocation

**Problem.** The original `max_utilization` policy grew the KV reservation
one token at a time via `allocate_append`, causing O(generated_tokens) allocator
calls and one preemption decision per decode step.

**Fix.** At admission, reserve `prefill_bytes + kv_decode_chunk_tokens` in one
shot. During decode, only call `grow_kv_or_preempt` when the already-reserved
chunk is exhausted — then grow by another whole `kv_decode_chunk_tokens` worth.
Default chunk = 32 tokens, configurable via `kv_decode_chunk_tokens`.

**Effect.** Preemption decisions amortize across 32 decode steps. With Azure
first-50 and a 1.5 GB pool, preempt events dropped from ~29 (per-token growth)
to 13 (chunked, full-evict mode).

---

### M2 — Tail-trim preemption

**Problem.** `preempt_lru` (full evict) frees a victim's entire K and V pages
and bounces it back to `WaitingAdmission`, forcing a full re-prefill +
all decode steps on re-admission. This wastes all completed decode work.

**Fix.** `preempt_tail_trim`: free only the last N pages of the victim's K and
V regions — exactly enough to cover the growing request's delta, rounded up to
whole tokens. Victim stays in `WaitingDecode`; `remaining_decode` is bumped by
the trim count so it re-runs only those tokens on its next decode step.

**Correctness.** Pages are allocated chronologically (prefill first, decode
appended to the back). Popping from the back is a strict rewind of the most
recent decode tokens — prefill is never touched. Trim is capped at
`completed_decodes` to avoid trimming into unfilled prefill space.

**Effect on Azure first-50, 1.5 GB pool:**

| Mode | Events | Tokens recomputed | TTFT p50 |
|------|--------|-------------------|----------|
| full evict | 13 | 280 | 61 ms |
| tail trim | 139 | 1 358 | 42 ms |

More events but far cheaper per event (decode steps only, no re-prefill).
TTFT p50 drops 19 ms because victims never lose prefill state.

---

### M3 — Oracle-predictive admission

**Problem.** Under memory pressure, `max_utilization` admits any request that
fits at prefill size, even if its final KV footprint would oversubscribe the
pool and trigger many preemptions.

**Fix.** Before admitting candidate R, compute the projected final KV bytes of
all in-flight requests + R's full footprint:

```
projected_in_flight  = Σ kv_bytes_for_ctx(r.ctx + r.gen - 1) - r.kv_bytes_reserved/2   (per side) × 2
candidate_total      = 2 × kv_bytes_for_ctx(R.ctx + R.gen - 1)
if projected_in_flight + candidate_total > kv_pool_bytes × threshold:
    hold R for predictive_hold_ms and retry
```

Oracle predictor uses `generated_tokens` from the Azure CSV. Short requests
skip the gate naturally (their contribution doesn't push the projection over
the threshold) — implicit SJF without queue reordering.

**Effect on Azure first-50, 1.5 GB pool:**

| Policy | Preempts | KV-OOM holds | Pred holds | TTFT p50 | Throughput |
|--------|----------|--------------|------------|----------|------------|
| guaranteed_no_evict | 0 | 813 | 0 | 81 ms | 10.39 rps |
| max_util_full | 13 | 579 | 0 | 61 ms | 10.23 rps |
| max_util_tail | 139 | 860 | 0 | 42 ms | 10.06 rps |
| **max_util_tail_pred** | **0** | **0** | 159 | **12 ms** | **10.41 rps** |

`max_util_tail_pred` eliminates all preemption and KV-OOM holds. TTFT p50
drops to 12 ms (6.8× better than baseline) and throughput exceeds
`guaranteed_no_evict`.

---

### Livelock guard (`kv_tail_trim_max_per_request`)

**Problem discovered.** At `arrival_time_scale=0.01` (10× faster arrivals),
the pool is oversubscribed and plain `max_util_tail` livelocked: the same
request was tail-trimmed 999+ times, aggregate in-flight work never shrank,
and the simulation diverged (~21,000 trims, only 7/50 requests completed
after 31 seconds of simulated time).

**Root cause.** Tail-trim alone has no admission back-pressure. When offered
load exceeds pool capacity, every growing request trims a victim, but the
victim re-queues the trimmed tokens, so total work is conserved — not reduced.

**Fix.** Per-request `tail_trim_events` counter (resets on full-evict). When
a victim's counter reaches `kv_tail_trim_max_per_request` (default 8), it is
excluded from the tail-trim victim pool. The caller falls through to
`preempt_lru` (full evict), which removes the request from in-flight entirely
and restores net progress.

**Effect at 10× load (`arrival_time_scale=0.01`, Azure first-50):**

| | Without guard | With guard (cap=8) |
|---|---|---|
| Done | 7 / 50 (diverging) | 50 / 50 ✓ |
| Tail-trims | 21 546 | 278 |
| Full-evict escalations | 0 | 17 |

---

## New YAML knobs

All under `Frontend:` in the simulator YAML. Ignored by `guaranteed_no_evict`.

| Knob | Default | Description |
|------|---------|-------------|
| `kv_scheduler_policy` | `guaranteed_no_evict` | `guaranteed_no_evict` or `max_utilization` |
| `kv_decode_chunk_tokens` | `32` | M1: decode reservation granularity (tokens) |
| `kv_evict_granularity` | `tail` | M2: `tail` = tail-trim, `full` = whole-request evict |
| `kv_tail_trim_max_per_request` | `8` | Livelock guard: trim cap per request per admission cycle; `0` disables |
| `predictive_admission` | `false` | M3: oracle pressure gate at admission |
| `predictive_pressure_threshold` | `1.0` | M3: hold when `projected + cand > pool × threshold` |
| `predictive_hold_ms` | `50.0` | M3: per-attempt hold budget for held candidates |

---

## Results at scale — Azure first-500, `arrival_time_scale=0.1`

Run: `python tools/run_serving_online_policy_compare.py --n-requests 500 --arrival-scale 0.1`

| Metric | guaranteed_no_evict | max_util_full | max_util_tail | max_util_tail_pred |
|---|---|---|---|---|
| Done | 500 / 500 | 500 / 500 | 500 / 500 | 500 / 500 |
| Admitted | 500 | 567 | 581 | 500 |
| Preempts (tail/full) | 0 | 215 (0/215) | 2759 (2553/206) | 0 |
| Tokens recomputed | 0 | 4 582 | 41 784 | 0 |
| Predictive holds | 0 | 0 | 0 | 228 543 |
| TTFT p50 / p90 (ms) | 13 740 / 61 701 | 11 064 / 43 027 | 16 219 / 54 145 | 15 896 / 64 989 |
| E2E p50 / p90 (ms) | 14 536 / 62 324 | 11 667 / 43 507 | 16 621 / 54 543 | 16 667 / 65 963 |
| **Throughput (req/s)** | 5.59 | **7.12** | 6.14 | 5.43 |

At 500-request scale with `arrival_time_scale=0.1`, `max_util_full` wins
throughput (7.12 vs 5.59 rps for baseline). `max_util_tail_pred` holds 228k
candidates — the predictive gate is conservative at this scale and slightly
hurts throughput. The 206 full-evict escalations in `max_util_tail` (from the
livelock guard) confirm the guard is necessary even at moderate load.

---

## New tools

| Script | Purpose |
|--------|---------|
| `tools/run_serving_online_policy_compare.py` | Run all 4 policies; flags: `--n-requests N`, `--arrival-scale S`, `--skip-run`, `--run-dir` |
| `tools/plot_policy_compare_per_run.py` | Per-run plots (execution timeline, route mix, predicted vs actual) for each policy subdir |
| `tools/plot_kv_pool_timeline.py` | KV pool occupancy stacked timeline by replaying `debug.log`; flags: `--subdir`, `--t-max`, `--combined` |

Timeline plots are auto-generated at the end of `run_serving_online_policy_compare.py`.

---

## Code pointers

| File | Role |
|------|------|
| `scheduler.cpp` | `grow_kv_or_preempt` (M1 chunked growth), `preempt_tail_trim` (M2 + livelock guard), `process_admission` (M3 predictive gate) |
| `pim_allocator.cpp` | `free_tail` — pops pages from the back of a request's allocation |
| `types.h` | `ServingOnlineConfig` knobs; `RuntimeRequest.tail_trim_tokens`, `.tail_trim_events` |
| `frontend.cpp` | YAML reads, validation, summary keys (`tail_trim_count`, `full_evict_count`, `tokens_recomputed`, `predictive_holds`) |

Reference docs: [kv_scheduler_policies.md](kv_scheduler_policies.md) ·
[kv_tail_trim.md](kv_tail_trim.md)
