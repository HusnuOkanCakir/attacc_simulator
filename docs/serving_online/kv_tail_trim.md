# Tail-trim preemption

`kv_scheduler_policy: max_utilization` + `kv_evict_granularity: tail` replaces
whole-request eviction with a finer-grained recovery: when an in-flight
request needs to grow its KV reservation and the pool is full, the scheduler
frees only the **most recent N tokens** of the LIFO victim's K and V regions
— *exactly* enough to cover the growing request's delta, rounded up to whole
tokens.

## State transitions

Full evict (`kv_evict_granularity: full`, legacy):

```
victim: WaitingDecode ─► WaitingAdmission       (will redo prefill + every decode)
        kv_bytes_reserved ─► 0
        preempt_count++, m_full_evict_count++
```

Tail trim (`kv_evict_granularity: tail`, default):

```
victim: WaitingDecode ─► WaitingDecode          (keeps prefill + earlier decodes)
        kv_bytes_reserved -= 2 × trim_tokens × per_token_per_side
        remaining_decode  += trim_tokens        (redo only the trimmed tail)
        tail_trim_tokens  += trim_tokens
        preempt_count++, m_tail_trim_count++, m_tokens_recomputed += trim_tokens
```

## Correctness

1. **Page ordering is chronological.** `PimAllocator::allocate` and
   `allocate_append` both pop pages from `pool.free_pages.back()` and append
   them to `m_allocs[key]`. Prefill writes happen first (admission reservation
   + the initial decode chunk), so the first N pages of each request's
   allocation hold prefill output. Decode appends land at the back. Freeing
   pages from the back of `m_allocs[key]` is therefore a strict rewind of
   the most recent decode tokens — it never touches prefill or earlier
   completed-decode tokens.

2. **Trim is capped at `completed_decodes`.** The trim token count is
   `min(ceil(delta / per_token), completed_decodes)`. If the victim has
   completed zero decode tokens (only finished prefill), tail-trim declines;
   the caller falls through to `preempt_lru` (full evict). This keeps the
   prefill tail safe and avoids an infinite loop where a fresh decoder
   tail-trims itself down to zero.

3. **Redo semantics match the cost-table decode step.** Incrementing
   `remaining_decode` by `trim_tokens` means the victim's next `pick_decode`
   call charges `trim_tokens × decode_e2e_ms` of work — the same cost-table
   formula as the original decodes, because the simulator uses greedy
   decoding and does not track token *values*. Each recomputed step reads
   the same-or-smaller KV depth as the original and produces a fresh
   reservation via chunked `allocate_append` on its natural growth path.

4. **Livelock guard** (`kv_tail_trim_max_per_request`, default 8). Under
   heavy oversubscription (offered load exceeds pool capacity), pure
   tail-trim thrashes: the LIFO victim is re-trimmed faster than it can
   decode, aggregate in-flight work never shrinks, and simulation diverges.
   The guard excludes victims whose `tail_trim_events` (since their current
   admission) has reached the cap — `grow_kv_or_preempt` then falls through
   to `preempt_lru` (full evict), removing that request from in-flight and
   restoring net progress. The counter resets on full-evict (fresh admission
   cycle). `0` disables the guard (legacy — only safe when paired with
   predictive admission or a workload known to fit). Empirically, at
   `arrival_time_scale=0.01` (10× the Azure first-50 baseline rate), a
   pool-oversubscribed run loops 20 000+ trims without finishing when the
   guard is off; with the default cap of 8 it completes 50/50 after ~15
   escalations to full evict.

## Why "tail" is cheaper than "full" on this simulator

A full evict discards prefill + every completed decode token; the victim
pays `prefill_e2e_ms + N × decode_e2e_ms` to catch back up. A tail-trim
frees one chunk of decode pages and pays
`trim_tokens × decode_e2e_ms` to recompute only those tokens — no prefill
reuse overhead. On Azure first-50 with a 1.5 GB pool and `chunk=32`, the
runner reports:

| Mode  | Events | Recomputed tokens |
|-------|--------|-------------------|
| full  | 13     | 280               |
| tail  | 139    | 1 358             |

More events, but the per-event recompute is a few decode steps rather than
a whole prefill. Mean TTFT drops accordingly; `max_util_tail_pred` (tail +
predictive admission) reaches zero preempts altogether on this workload.

## Relation to FastServe

FastServe (2024) preempts at output-token granularity by **swapping** the
victim's K/V to host DRAM and restoring it on resume. Tail-trim uses the
same granularity but substitutes **recomputation** for swap — the ramulator
simulator does not model host DRAM bandwidth, and for small trim counts
recompute is cheaper than a round-trip swap through PCIe. Both strategies
trade off in the same axis: retain more victim state, pay less to resume.

## Where it lives in code

- `PimAllocator::free_tail(KvRegion, request_id, bytes_to_free)` — pops
  pages from the back of the allocation. Returns bytes actually freed.
  Declared in [pim_allocator.h](../../ramulator2/src/frontend/impl/serving_online/pim_allocator.h);
  implemented in [pim_allocator.cpp](../../ramulator2/src/frontend/impl/serving_online/pim_allocator.cpp).
- `Scheduler::preempt_tail_trim(excluded_index, bytes_needed_per_side, now_ms)`
  — picks the LIFO victim (`state == WaitingDecode`, `completed_decodes > 0`),
  calls `free_tail` on Key and Value, bumps `remaining_decode`. In
  [scheduler.cpp](../../ramulator2/src/frontend/impl/serving_online/scheduler.cpp).
- `Scheduler::grow_kv_or_preempt` — tries tail-trim first when
  `kv_evict_granularity == "tail"`, falls back to `preempt_lru` when no
  tail-trimmable victim exists (or granularity is `full`).

## Debug log events

With `debug_log_path` enabled, every tail-trim emits a line like:

```
[  1152.341ms] TAIL_TRIM req= 15  tokens=7  freed_bytes=2293760  by_req=14
```

Sum of `tokens` across all `TAIL_TRIM` lines equals the summary key
`tokens_recomputed` (plus the imputed contribution from any
`PREEMPT redo_decodes=N` lines when `kv_evict_granularity: full` is in use).
