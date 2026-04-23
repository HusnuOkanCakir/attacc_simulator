# KV Scheduler Policies (`kv_scheduler_policy`)

The `serving_online` frontend supports two KV cache management policies,
selected by the top-level YAML knob:

```yaml
Frontend:
  impl: ServingOnlineFrontend
  kv_scheduler_policy: guaranteed_no_evict   # or: max_utilization
```

Both follow established LLM-serving scheduler literature — TRT-LLM's
`batch_scheduler_policy` names are used directly. The underlying tradeoff is
the same one discussed in the vLLM paper (Kwon et al., SOSP '23, §4.5):
*over-reserve and never evict* vs *reserve lazily and preempt*.

---

## `guaranteed_no_evict` (default)

**Admission.** A request is admitted only when the PIM KV pool can
reserve its full worst-case footprint **up front**, per side:

```
kv_bytes_per_side = num_layers * num_heads * (ctx + gen - 1) * d_head * dtype_bytes
```

Both K and V regions must fit independently. If either side lacks capacity,
the request stays in `WaitingAdmission` and the frontend applies the existing
`kv_oom_policy` knob for recovery:

| `kv_oom_policy`        | Behavior on admission KV-OOM                      |
|------------------------|---------------------------------------------------|
| `hold` (default)       | Retry after `admission_retry_interval_ms`.         |
| `fallback_gpu`         | Immediately re-route to `gpu_only` (no KV needed). |
| `hold_then_fallback`   | Hold up to `kv_oom_hold_limit_ms`, then GPU.       |

**In-flight.** Once admitted, a request's KV pages are pinned until `Done`.
**No preemption happens.** `preempt_count` is always 0.

**When to use.** Reproducible per-request latency; no recomputation-induced
tail spikes. You pay for unused decode slots (ctx+gen-1 is the tail of the
step, not the average), which caps the maximum in-flight batch when the pool
is tight.

---

## `max_utilization`

**Admission.** A request is admitted with **prefill-only** KV reservation:

```
kv_bytes_per_side = num_layers * num_heads * context_tokens * d_head * dtype_bytes
```

This is the exact footprint the prefill pass needs. No per-decode-step
reservation happens at admission time. If even prefill-sized bytes don't fit,
the request holds FCFS — as in vLLM §4.5. The `kv_oom_policy` knob is
**ignored** under this policy (a stderr warning is emitted at init if set to
anything other than `hold`), because admission itself never rejects a request
that would otherwise have fit at prefill.

**In-flight growth (chunked).** After prefill and after every decode step
that leaves `remaining_decode > 0`, the scheduler checks whether the current
reservation still covers the next step's context. If not, it grows by *whole
`kv_decode_chunk_tokens` chunks* via `PimAllocator::allocate_append`:

```
need_per_side = kv_bytes_for_ctx(next_step_ctx)
have_per_side = req.kv_bytes_reserved / 2
if need_per_side > have_per_side:
    chunk_bytes   = kv_decode_chunk_tokens × bytes_per_token_per_side
    chunks_needed = ceil((need - have) / chunk_bytes)
    append chunks_needed × chunk_bytes   # to both Key and Value
```

Admission also reserves *prompt + one chunk* in a single call (instead of
just the prompt), so the first `kv_decode_chunk_tokens` decode tokens never
trigger a growth. Default chunk is 32 tokens; set `kv_decode_chunk_tokens: 1`
for strict per-token growth.

**Preemption on growth OOM.** If the delta can't be satisfied, the scheduler
tries two recovery paths in order, both targeting the **most recently
admitted** in-flight decode request (largest `admitted_ms` — LIFO, per vLLM §4.5):

1. **Tail-trim** (`kv_evict_granularity: tail`, default). Frees only the
   last `N` tokens of the victim's K and V regions — exactly enough to cover
   the growing request's delta, rounded up to whole tokens. Victim **stays
   in `WaitingDecode`**; `remaining_decode` is bumped so it redoes the
   trimmed tokens on its next decode step. Cheaper than a full evict: only
   a handful of tokens are recomputed instead of prefill + every prior
   decode. See [kv_tail_trim.md](kv_tail_trim.md).

2. **Full evict** (`kv_evict_granularity: full`, or as a fallback when no
   tail-trim victim has `completed_decodes > 0`). Frees the victim's entire
   Key and Value pages via `PimAllocator::free`, resets its state to
   `WaitingAdmission` with reason `kv_evicted`, and bumps
   `m_full_evict_count`. The victim re-runs prefill + all decode steps on
   its next admission. This matches vLLM §4.5's default recovery strategy
   (no swap-to-host — the simulator doesn't model host DRAM bandwidth).

Self-preemption: if no other PIM request is in-flight, the growing request
preempts itself.

**Predictive (size-aware) admission.** When
`predictive_admission: true`, the scheduler runs an oracle pressure gate
*before* each PIM admission. Using `generated_tokens` from the Azure CSV as
an oracle predictor, it computes the projected final KV footprint of all
in-flight PIM requests + the candidate. If
`projected + candidate > kv_pool_bytes × predictive_pressure_threshold`, the
candidate is held with reason `predictive_pressure` and re-tried after
`predictive_hold_ms`. Short later-arriving requests skip the gate because
their contribution doesn't push the projection past the threshold — giving
implicit SJF without rearranging the admission order. The predictor is
swappable (a learned or heuristic predictor is a drop-in replacement for
`kv_bytes_for_ctx(max_ctx)` over the Azure-oracle value).

**When to use.** Maximizing pool utilization and throughput under memory
pressure. Mean / p50 latency improves (more requests are in-flight in
parallel), but tail p99 worsens when a request is preempted and re-prefilled.
`preempt_count` reports how many evictions happened.

---

## Comparison on Azure first-50 (pi0, LPDDR5-PIM, `kv_pool_bytes=1.5 GB`)

Run with `tools/run_serving_online_policy_compare.py` (four configs). Full
figure at [`docs/figures/serving_online_policy_compare.png`](../figures/serving_online_policy_compare.png).

| Metric                        | `guaranteed_no_evict` | `max_util_full` | `max_util_tail` | `max_util_tail_pred` |
|-------------------------------|-----------------------|-----------------|-----------------|----------------------|
| Admitted / completed          | 50 / 50               | 57 / 50         | 50 / 50         | 50 / 50              |
| KV-OOM holds                  | 813                   | 579             | 860             | **0**                |
| Preempt count (tail / full)   | 0 (0 / 0)             | 13 (0 / 13)     | 139 (139 / 0)   | **0 (0 / 0)**        |
| Tokens recomputed             | 0                     | 280             | 1 358           | **0**                |
| Predictive holds              | 0                     | 0               | 0               | 159                  |
| TTFT p50 / p90 (ms)           | 81 / 335              | 61 / 390        | 42 / 377        | **12 / 284**         |
| TTFT p99 (ms)                 | 2359                  | **2282**        | 2517            | 2319                 |
| E2E p50 / p90 (ms)            | 434 / 711             | 412 / 1012      | 432 / 983       | **354 / 734**        |
| E2E p99 (ms)                  | 2522                  | **2440**        | 2679            | 2481                 |
| Throughput (req/s)            | 10.39                 | 10.23           | 10.06           | **10.41**            |

Reading the table:

- **`max_util_full`** (chunked growth + full-evict). 13 full preemptions on
  Azure first-50 — fewer than the 29 from the earlier per-token-growth
  configuration because the 32-token chunks let the pool hold more
  unpreempted decodes for longer. Each full evict discards the victim's
  prefill + every completed decode: on this trace, 280 tokens total.
- **`max_util_tail`** (chunked growth + tail-trim). Replaces the 13 full
  evictions with 139 fine-grained trims averaging ~10 tokens each. Total
  recomputation is larger in raw tokens (1 358) but every trimmed token is
  a single decode step, whereas a full-evict token is worth a whole prefill
  step's arithmetic. TTFT p50 drops 39 ms (81 → 42) because victims stay in
  `WaitingDecode` and never lose their prefill.
- **`max_util_tail_pred`** (tail-trim + oracle-predictive admission). The
  predictive gate holds candidates that would oversubscribe the pool; 159
  holds completely eliminate preemption and KV-OOM misses (zero of each).
  TTFT p50 drops to 12 ms and throughput ticks up to 10.41 req/s — above
  the `guaranteed_no_evict` baseline.

`guaranteed_no_evict` remains the **regression anchor**: its numbers stay
byte-for-byte identical to the pre-knob behavior.

---

## Which knob does what

| Knob                           | Default                | Purpose |
|--------------------------------|------------------------|---------|
| `kv_scheduler_policy`          | `guaranteed_no_evict`  | Top-level policy: how much KV is reserved at admission and whether in-flight requests can be preempted. |
| `kv_oom_policy`                | `hold`                 | Recovery when `guaranteed_no_evict` admission fails (`hold` / `fallback_gpu` / `hold_then_fallback`). Ignored under `max_utilization`. |
| `kv_oom_hold_limit_ms`         | `50.0`                 | Per-request grace period for `hold_then_fallback`. |
| `admission_retry_interval_ms`  | `1.0`                  | Retry cadence for KV-OOM held requests. |
| `kv_pool_bytes`                | `0`                    | Total PIM-DRAM bytes reserved for the KV cache. |
| `kv_decode_chunk_tokens`       | `32`                   | **max_utilization only.** Grow-and-reserve granularity in decode tokens. `1` = per-token growth. |
| `kv_evict_granularity`         | `tail`                 | **max_utilization only.** `tail` = free only the victim's recent tokens; `full` = evict the whole request. |
| `kv_tail_trim_max_per_request` | `8`                    | **max_utilization + tail only.** Livelock guard: a victim excluded from tail-trim after this many trims since its current admission (caller escalates to full-evict). Reset on full-evict. `0` disables. |
| `predictive_admission`         | `false`                | **max_utilization only.** Enable the oracle size-aware admission gate. |
| `predictive_pressure_threshold`| `1.0`                  | Hold the candidate when `projected + candidate > pool × threshold`. |
| `predictive_hold_ms`           | `50.0`                 | Per-attempt hold budget when the predictive gate rejects a candidate. |

---

## References

- Kwon et al., *Efficient Memory Management for Large Language Model Serving
  with PagedAttention*, SOSP '23. §4.5 covers the FCFS + LIFO eviction +
  recomputation strategy used by `max_utilization` with `kv_evict_granularity: full`.
- Wu et al., *FastServe: Fast Distributed Inference Serving for Large
  Language Models*, 2024. Preemption at output-token granularity; our
  tail-trim is the ramulator/PIM analogue that recomputes instead of swapping.
- NVIDIA TRT-LLM docs, `batch_scheduler_policy`. Uses the same two top-level
  names (`guaranteed_no_evict` / `max_utilization`).
- Patel, Choukse, Zhang, Goiri, Warrier, Mahajan, Bianchini,
  *Splitwise: Efficient Generative LLM Inference Using Phase Splitting*,
  ISCA '24. Motivates phase-aware memory management.
