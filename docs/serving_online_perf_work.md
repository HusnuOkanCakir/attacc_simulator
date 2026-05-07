# serving_online performance work — diagnosis, cache, verification

Session record for the wall-clock perf work on the `serving_online` PIM frontend.
Covers diagnostic instrumentation, the per-shape decode-cycle cache (Lever 1),
verification results, literature grounding, and what's next.

---

## 1. Context

After porting the realistic LPDDR5-PIM command stream (Pi0 ≈ 16K cmds/decode-step,
OpenVLA ≈ 86K cmds/decode-step at typical contexts), the `serving_online`
frontend became unusable:
- n=1 OpenVLA: ~3 minutes wall-clock for ~84 ms simulated
- n=50 cluster runs: estimated days

Cluster jobs (`tools/run_openvla_slo_cluster.sbatch`, ready since prior sessions)
were blocked from being submitted. The cost-table pipeline runs offline and was
unaffected; the bottleneck was inline Ramulator processing the realistic PIM
command stream during the serving timeline.

## 2. Realistic PIM commands (recap, prior session)

Replaced the simplified READ-only generator with the full LPDDR5-PIM sequence,
ported from `serving1_5/pim_commands.cpp`:

```
Per (layer, head):
  K-side score matmul:
    for each block ∈ ceil(ctx/32):
      PIM_WR_GB(K_block) → PIM_MAC_AB(K_block)
    PIM_MV_SB → PIM_SFM
  V-side context matmul:
    for each block ∈ ceil(ctx/32):
      PIM_MV_GB(V_block) → PIM_MAC_AB(V_block)
    PIM_MV_SB → PIM_BARRIER
```

Total commands: `num_layers × num_heads × (4 × blocks + 4)` where
`blocks = ceil(context_tokens / 32)` (FP16, 64-byte cache line). `PIM_BARRIER`
uses an empty callback so it does not decrement the outstanding-counter,
matching `serving1_5`.

Files: [pim_ramulator_src/serving_online/pim_commands.{h,cpp}](pim_ramulator_src/serving_online/)
plus the [ramulator2 mirror](ramulator2/src/frontend/impl/serving_online/).
Status: `docs/status.md` point 3 marked `[FIXED]`.

## 3. Phase 1 + 2 — diagnostic instrumentation

Added periodic and per-task timing visibility to the inline Ramulator path
in [frontend.cpp](pim_ramulator_src/serving_online/frontend.cpp).

### Phase 1 — PROGRESS log every 1 s wall-clock
In `Runtime::tick()`, gated on `--debug-log`. Format:

```
[ 1234.567ms] PROGRESS    cycle=12345  task=decode/lpddr5_pim_bank
              cmds_total=86016  cmds_sent=43200  cmds_outstanding=128
              wall_elapsed=12.345s  sim_per_wall=3.4x
```

Cost: one `steady_clock::now()` per tick + one duration compare.

### Phase 2 — Per-section TIMING breakdown at TASK_DONE
`std::chrono::nanoseconds` accumulators around the five hot sections, summed
per task and printed at `TASK_DONE`:

| Section | Wraps |
|---|---|
| `t_cmd_gen` | `m_command_gen->decode_step()` |
| `t_translate` | `m_translation->translate()` (per command) |
| `t_send` | `m_memory_system->send()` (per command) |
| `t_sched` | `m_scheduler.tick()` |
| `t_complete` | `complete_active_task_if_ready()` |
| `t_unaccounted` | `wall_total - sum(others)` ≈ Ramulator DRAM tick |

### Diagnostic finding
Pi0 n=1 with `--debug-log`, per decode step (16,640 cmds):

| Section | Time | % of step |
|---|---|---|
| `t_unaccounted` (Ramulator DRAM tick) | **2.33 s** | **64%** |
| `t_complete` (`is_finished()` over-calls) | 0.63 s | 17% |
| `t_send` | 0.34 s | 9% |
| `t_translate` | 0.33 s | 9% |
| `t_cmd_gen` | 0.001 s | <1% |
| `t_sched` | 0.000 s | <1% |
| **wall** | **3.6 s** | 100% |

Ramulator's DRAM tick is the dominant cost. Translation/send/cmd_gen are not
the bottleneck.

## 4. Phase 3 — `pim_command_mode` A/B toggle

Added a runtime switch ([pim_commands.h](pim_ramulator_src/serving_online/pim_commands.h),
[types.h](pim_ramulator_src/serving_online/types.h)) and `--pim-command-mode`
CLI flag:

- `realistic` (default) — full per-block sequence (above)
- `simple` — `num_layers × num_heads × 2` total `PIM_MAC_AB` only,
  context-length-independent

A/B result, Pi0 n=1: **170 s realistic vs 16 s simple** (~10× faster, identical
TTFT/E2E because the cost-table fast-forward bounds simulated time).
Confirmed the bottleneck scales with command count → DRAM-tick cost.

`pim_command_mode` is kept as a debug knob; not the recommended speed fix.
The principled fix is Lever 1 (cache).

## 5. Lever 1 — per-shape PIM-decode cycle cache

### Design

A `std::unordered_map<PimShapeKey, PimShapeStats>` on `Runtime`:

```cpp
struct PimShapeKey {
  std::string route;       // "lpddr5_pim_bank"
  int         ctx_bucket;  // = ceil(context_tokens / 32)
  int         batch_size;
};

struct PimShapeStats {
  Clk_t  drain_clk;     // simulated DRAM cycles to drain inline (first sighting)
  size_t cmds_total;    // command count at first sighting
  size_t hits;          // cache-hit counter
};
```

Bucket size = 32 tokens = `kLineBytes / dtype_bytes`. Two requests with
`ctx ∈ (32k, 32(k+1)]` emit the same number of blocks and exercise the bank
schedule near-identically.

### Cache-miss path (first sighting)
Identical to the prior inline path: pre-generate commands, pump into Ramulator,
drain. At the moment `m_issue_done && m_pim_outstanding == 0` becomes true
(in `complete_active_task_if_ready()`), record
`drain_clk = m_clk - m_clk_at_task_start` under the shape key. Emit `CACHE_MISS`
and `CACHE_FILL` events to debug.log.

### Cache-hit path
At the top of `start_task()` for PIM decode, check the cache. On hit:
fast-forward by `max(cost_table_pim_ms, drain_ms_from_cache)` and skip command
generation, translation, and `m_memory_system->send()` entirely. Increment
`hits`, emit `CACHE_HIT`.

### Why it's safe
- **Cost-table fast-forward already overrides drain time when drain < pim_ms** ([frontend.cpp:215-219](pim_ramulator_src/serving_online/frontend.cpp#L215-L219)). Cache hit reproduces the same outcome.
- **`m_pim_free_ms` is advanced by the cost table at task start** ([frontend.cpp:371-372](pim_ramulator_src/serving_online/frontend.cpp#L371-L372)), independent of inline drain. Scheduler accounting is unchanged on the hit path.
- **Bucket-by-max-context-in-batch** is the same approximation the realistic command stream already makes (longest-context request dominates the step).

### CLI / YAML
- YAML: `pim_cache_enabled: true|false` (default `true`)
- CLI: `--no-pim-cache` (opt out for verification)

### Diagnostic events (debug.log)
`CACHE_HIT`, `CACHE_MISS`, `CACHE_FILL` per task, plus `PIM_CACHE_STATS` and
per-shape `SHAPE` breakdown at `finalize()`. YAML stats also include
`pim_cache_hits`, `pim_cache_misses`, `pim_cache_distinct_shapes`.

## 6. Verification

### Correctness — cached vs `--no-pim-cache` produce identical results

| Run | Wall-clock | TTFT p50 | E2E p50 |
|---|---|---|---|
| Pi0 n=1, cache on | **8.8 s** | 10 ms | 84 ms |
| Pi0 n=1, `--no-pim-cache` | 168 s | 10 ms ✓ | 84 ms ✓ |

Identical simulated behavior; **19× wall-clock speedup**.

### Speed — n=1 OpenVLA
| Run | Wall-clock | Cache stats |
|---|---|---|
| Baseline (realistic, no cache) | ~180 s | n/a |
| Cache on | **20 s** | 31 hits / 2 misses / 94% / 2 distinct shapes |

**~9× speedup**. E2E = 236 ms (matches baseline).

### n=10 OpenVLA — continuous-batching-like (multi-bs) regression
| Run | Wall-clock | Cache stats |
|---|---|---|
| Cache on | **11m44s** | 200 hits / 9 misses / **95.7%** / 9 distinct shapes |

Hit-rate ≥90% with realistic batch-size diversity (`bs ∈ {1,2,3,4,5,6,7}`).
Cache key correctly distinguishes contention regimes: at `ctx_bucket=50` with
`bs=2` `drain_ms=4.66`, with `bs=5` `drain_ms=5.82` — same bucket, different
batch size, different cycle counts.

### Pi0 roofline — sanity refresh
`tools/plot_roofline.py` regenerated cleanly. PIM attention vs GPU attention at
decode AI ≈ 1 FLOP/byte: **PIM = 1.614 TFLOPS, GPU = 0.466 TFLOPS → 3.46×**
on the memory-bound attention component (Lin=512). Numbers consistent with
empirical PIM bandwidth assumptions hard-coded in `plot_roofline.py`
(`PIM_ATTN_BW_TBS = 1.62`).

## 7. Files modified

| File | Change |
|---|---|
| [pim_ramulator_src/serving_online/types.h](pim_ramulator_src/serving_online/types.h) | `pim_command_mode`, `pim_cache_enabled` config fields |
| [pim_ramulator_src/serving_online/pim_commands.{h,cpp}](pim_ramulator_src/serving_online/) | `Mode { Realistic, Simple }` + branch |
| [pim_ramulator_src/serving_online/frontend.cpp](pim_ramulator_src/serving_online/frontend.cpp) | Phase 1+2 instrumentation, Phase 3 toggle plumbing, Lever 1 cache (state, hit-path, miss-path drain capture, stats summary), YAML param wiring |
| [ramulator2/src/frontend/impl/serving_online/](ramulator2/src/frontend/impl/serving_online/) | Mirror of the above |
| [tools/run_serving_online_policy_compare.py](tools/run_serving_online_policy_compare.py) | `--pim-command-mode`, `--no-pim-cache` flags + YAML template fields + global `PIM_COMMAND_MODE`, `PIM_CACHE_ENABLED` |
| [docs/status.md](docs/status.md) | Marked point 3 `[FIXED]`, point 9 `[PARTIALLY FIXED]`, added "Recent additions" section |

## 8. Research grounding (literature review)

Sanity-checked the methodology against the field. Key findings:

- **Inline cycle-accurate Ramulator is the dominant methodology** in PIM-LLM
  simulation: AttAcc, NeuPIMs, CD-PIM, HPIM, PIMphony, LP-Spec, SAL-PIM,
  PIM-GPT all do this. Offline trace-and-cache is the unusual choice in this
  niche.

- **Inline simulation models cross-request bank contention** that cost tables
  cannot represent. With continuous batching, multiple requests' PIM commands
  interleave on shared banks; that interleaving is what inline Ramulator
  measures and what the cost table averages over.

- **VLA + LPDDR5-PIM + edge robotics is unpublished whitespace** as of late 2025.
  Closest neighbors:
  - **AttAcc** (ASPLOS '24) — datacenter HBM3-PIM for batched LLMs, our
    simulator's parent. Datacenter, not edge.
  - **CD-PIM** (arxiv 2601.12298) — LPDDR5-PIM for *edge* low-batch LLM decode.
    Closest hardware match. Generic LLM, not VLA.
  - **LP-Spec** (arxiv 2508.07227) — LPDDR-PIM mobile speculative decoding.
  - **ActionFlow** (arxiv 2512.20276) — *VLA-side* analog: cross-request
    pipelining for VLA on Jetson Orin (GPU-only). 2.55× on OpenVLA-7B.
    Establishes prior art that VLA continuous batching matters.
  - **VLA-Perf** (arxiv 2602.18397) — analytical VLA inference performance
    model, useful for citing KV-bandwidth pressure.

  Nothing combines all three. That's the project's contribution.

## 9. What's next

### Immediate — Path A (chosen)
- [ ] scp local changes to the cluster (`ocakir@safari-proxy`)
- [ ] Rebuild ramulator2 on the cluster
- [ ] Submit `tools/run_openvla_slo_cluster.sbatch` (n=50, max_util_full,
      kv-pool=2GB, fallback_gpu, SLO 500/200ms). Expected: ~1 hour with cache;
      would have been ~9 hours without.
- [ ] Pull results, inspect for the three failure modes:
  1. Cache hit rate < 90% → continuous batching shape fragmentation, would
     suggest Lever 3 needed
  2. All-GPU fallback → cost table pessimistic; need denser bs/Lout sweep
     (cluster sweep `output.csv` blocker still open)
  3. n=50 takes >2 hours → distinct_shapes high; investigate
- [ ] Refresh roofline + add serving-throughput companion figure

### Deferred (may not be needed)
- **Lever 2** — hardware-aligned command granularity (per all-bank MAC step
  matching AttAcc paper Figure 6 / `gen_trace_attacc_bank.py`). Phase 3's
  `pim_command_mode` slot ready to host this.
- **Lever 3** — continuous batching in scheduler. Structural change. Cache is
  a prerequisite (already in place); without it, batched runs intractable.
  Needed when the cluster results show fragmentation or when continuous
  batching becomes the headline figure.
- **Cluster cost-table sweep failure** (`output.csv not written`) — separate
  open issue.
- **Cross-run cache persistence** (write/read cache to disk between runs).
  Would amortize first-shape misses across cluster sweeps.
