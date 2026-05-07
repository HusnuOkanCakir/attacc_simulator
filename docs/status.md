What the simulator actually does — two-tier architecture
Tier 1: Cost table / analytical model (everything except DRAM contention)

For every task (prefill or decode), the scheduler looks up (Lin, Lout, bs) in the trained cost model → gets (gpu_ms, pim_ms, e2e_ms, energy). These are plugged into two wall-clock counters m_gpu_free_ms and m_pim_free_ms. The task is "done" when both counters are satisfied. This is the primary source of latency for everything.

Tier 2: Ramulator DRAM simulation (contention only)

Only triggered for PIM decode. Now generates the full realistic LPDDR5-PIM command sequence per (layer, head) — see point 3 below for details. Once all commands drain AND the wall-clock cost-table time is met, the task completes. The DRAM simulation exists to model bank contention between concurrent requests in the same decode batch — not to determine absolute latency.

Crucially, after all commands drain, the code fast-forwards to max(gpu_ms, pim_ms, e2e_ms) from the cost table — so the cost table almost always dominates, not the DRAM simulation.

Pi0 vs OpenVLA DRAM command count (updated — now realistic, not simplified READs):

Pi0:     40L × 8H × (4 × ceil(ctx/32) + 4)  — at ctx=374: ~16,640 cmds per decode step
OpenVLA: 32L × 32H × (4 × ceil(ctx/32) + 4) — at ctx=630: ~86,016 cmds per decode step
(Old simplified: Pi0=640, OpenVLA=2048 — fixed READ-only generator, context-length-independent)

What's modeled realistically
Aspect	How realistic
Request scheduling, admission, preemption	Very detailed — full state machine
KV pool management (paging, pool pressure)	Realistic — byte-accurate allocations
Request inter-arrival timing	Real Azure trace → realistic burstiness
LPDDR5 timing parameters	Cycle-accurate (tCK=1250ps, tRCD, tRAS, bank groups)
Bank contention between concurrent decode batches	Now modeled via realistic PIM command stream (WR_GB/MAC_AB/MV_SB/SFM/MV_GB/BARRIER)
Energy accounting	From profiled cost tables
SLO enforcement, route selection	Fully implemented
OpenVLA-7B support	DONE — 32L/32H/d128, +256 vision prefix tokens, cost tables + ML models trained

What's missing or simplified
1. GPU is 100% analytical — no simulation
GPU timing is a lookup from a table measured on an A100 once. No simulation of SM utilization, HBM bandwidth saturation, tensor core scheduling, or how running multiple requests simultaneously affects GPU throughput (at bs>1, the cost table captures this via the bs axis, but there's no dynamic feedback from actual GPU state).

2. Prefill is fully analytical — PIM is never used during prefill
Even on the lpddr5_pim_bank route, prefill runs on GPU only. The cost table's prefill_pim_ms is always 0.0. In a real PIM system you might pipeline attention prefill across DRAM banks.

3. [FIXED] PIM commands now realistic — ported from serving1_5
Previously: simplified READ-only generator emitting num_layers × num_heads × 2 READ commands per decode step, independent of context length.

Now: full realistic LPDDR5-PIM sequence per (layer, head), at cache-line block granularity:

  K-side:  for each block: PIM_WR_GB(K) → PIM_MAC_AB(K)
           PIM_MV_SB (collect partial scores)
           PIM_SFM   (softmax)
  V-side:  for each block: PIM_MV_GB(V) → PIM_MAC_AB(V)
           PIM_MV_SB (collect partial contexts)
           PIM_BARRIER (fire-and-forget, empty callback)

blocks = ceil(context_tokens / 32) for FP16 (64-byte cache line / 2 bytes per element).
Total = num_layers × num_heads × (4 × blocks + 4) per decode step.
Files: pim_ramulator_src/serving_online/pim_commands.{h,cpp} + ramulator2 mirror.
Verified: Pi0 n=1 (84ms E2E) and OpenVLA n=1 (236ms E2E) both pass.

Note: SETM/SETH (SET_MODEL/SET_HEAD) are not included — amortized as one-time setup.
Note: absolute latency still gated by cost table; command count now scales with context.

4. FFN layers stay on GPU, not in PIM
Only attention KV reads go to PIM. In LLaMA-7B decode, the FFN is ~2/3 of total compute and is weight-bandwidth-bound. If FFN weights were also in PIM DRAM, the PIM advantage would be larger. This is not modeled.

5. Vision encoder not simulated (OpenVLA)
SigLIP processes the camera image into 256 tokens on GPU. This has its own latency (~20–50ms on A100) and runs before every prefill. We skip it and just add 256 tokens to context via vision_prefix_tokens. For a real robot this matters — every inference cycle starts with image encoding.

6. DRAM refresh disabled
The timing preset has nRFCab=-1, nRFCpb=-1, nREFI=-1 — refresh is disabled. Real LPDDR5 refreshes every ~3.9µs, causing brief stalls that reduce sustained bandwidth by ~5-10%.

7. No GPU–PIM pipelining
In principle, while PIM is doing KV reads for one layer's attention, the GPU could be running the next layer's FFN. The simulator models GPU and PIM as co-running (e2e = max(gpu_ms, pim_ms)) but the actual overlap between layers is not scheduled at operator granularity.

8. No multi-GPU, no interconnect
Single A100 assumed. No NVLink, PCIe, or host-device transfer modeling. For real OpenVLA at 7B you'd likely use 1 GPU, so this is fine.

9. [PARTIALLY FIXED] Fixed bs for cost table — interpolation error at non-trained batch sizes
Root cause: HistGBR is piecewise-constant in bs. Old training set: bs ∈ {1, 8, 32}. For batch sizes
in between (e.g. bs=2, 3, 5, 6), the model returns stale values from the nearest trained point.

What was done:
- run_cost_table_sweep.sbatch updated: BATCH_VALUES now 1,2,4,8,16,32 (was 1,8,32)
- LOUT_VALUES extended to 2,16:400:16 (was 2,16:208:16) to cover long-generation requests

What remains:
- Cluster sweep with new bs/Lout values must be re-run (previous cluster run failed with
  "output.csv not written" error — root cause not yet diagnosed; local tests pass)
- After sweep: retrain cost models, export .bin files, push to cluster
- Until then: OpenVLA cost tables still use old bs={1,8,32} and Lout max=208

10. No data layout optimization
KV is stored linearly (K[layer][head][token][d_head]). A real LPDDR5-PIM system would bank-stripe across all banks to maximize all-bank MAC parallelism. The current layout may map multiple heads to the same bank — creating artificial serialization in the DRAM simulation.

Summary of what a "run" actually measures
A run measures scheduling quality and KV pool efficiency under realistic arrival patterns. The latency numbers come almost entirely from the cost table (which was measured on real hardware at specific (Lin, Lout, bs) grid points). Ramulator adds relative contention effects between concurrent requests sharing DRAM banks — and now uses the correct PIM command stream (WR_GB/MAC_AB/MV_SB/SFM/BARRIER) to model the actual broadcast-then-reduce all-bank compute pattern.

The simulator is a high-fidelity scheduler and memory pressure model layered on top of a lower-fidelity single-request performance model (the cost table). That's an honest and useful design for comparing KV policies — but it means absolute latency numbers are only as accurate as the cost table's coverage of (Lin, Lout, bs) space.

Recent additions (since original status)
- OpenVLA-7B: added to MODEL_PRESETS, cost tables swept on cluster (Lin∈[512:6144:512],
  Lout∈[16:208:16], bs={1,8,32}), HistGBR models trained (R²>0.999), .bin exported.
- Runner flags added: --model {pi0,openvla}, --kv-pool-gb, --slo-e2e-ms, --slo-ttft-ms,
  --kv-oom-policy {hold,fallback_gpu,hold_then_fallback}, --kv-oom-hold-limit-ms
- run_openvla_slo_cluster.sbatch: dedicated cluster script for OpenVLA SLO sweep
  (high_latency partition, n=50, max_active=32, fallback_gpu, --ml)
