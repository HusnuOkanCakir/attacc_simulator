/**
 * serving_online/types.h
 *
 * Core data structures for the serving_online frontend: request state machine,
 * cost estimates, active tasks, and configuration.
 *
 * Request lifecycle:
 *
 *   Arrival → WaitingAdmission
 *             │  admission succeeds (route + memory available)
 *             ▼
 *         WaitingPrefill
 *             │  scheduler picks this request for prefill
 *             ▼
 *         WaitingDecode  ←─┐
 *             │            │ one decode token done, more remaining
 *             │ all decode tokens done
 *             ▼
 *           Done
 *
 *   WaitingAdmission can also go to Dropped if admission fails permanently.
 */

#ifndef RAMULATOR_FRONTEND_IMPL_SERVING_ONLINE_TYPES_H
#define RAMULATOR_FRONTEND_IMPL_SERVING_ONLINE_TYPES_H

#include <string>
#include <vector>
#include <optional>

#include "base/base.h"

namespace Ramulator::ServingOnline {

// ─── Request state ─────────────────────────────────────────────────────────

enum class ReqState {
  WaitingAdmission,   // arrived but not yet admitted to a route
  WaitingPrefill,     // admitted; waiting for the prefill task to start
  WaitingDecode,      // prefill done; waiting for next decode task
  Done,               // all decode tokens completed
  Dropped,            // permanently rejected (no eligible route after max wait)
};

// ─── Source request (loaded from Azure CSV) ─────────────────────────────────

struct SourceRequest {
  int    id;
  double arrival_ms;
  int    context_tokens;
  int    generated_tokens;
};

// ─── Cost estimate (from the lookup table) ──────────────────────────────────
//
// Holds predicted latencies for one (route, context_tokens, generated_tokens, bs)
// combination. Both prefill and per-decode-token costs are stored.

struct CostEstimate {
  std::string route;
  bool   uses_pim = false;       // true for pim+gpu routes
  // Prefill phase
  double prefill_gpu_ms    = 0.0;
  double prefill_pim_ms    = 0.0;
  double prefill_e2e_ms    = 0.0;
  double prefill_energy_nj = 0.0;
  // Per-decode-step (one token)
  double decode_gpu_ms    = 0.0;
  double decode_pim_ms    = 0.0;
  double decode_e2e_ms    = 0.0;
  double decode_energy_nj = 0.0;
  int    decode_tokens    = 0;   // generated_tokens - 1
};

// ─── Live runtime request ───────────────────────────────────────────────────

struct RuntimeRequest {
  // Identity (set at arrival, never changes)
  int    id                 = -1;
  double arrival_ms         = 0.0;
  int    context_tokens     = 0;
  int    generated_tokens   = 0;

  // Scheduling state
  ReqState    state         = ReqState::WaitingAdmission;
  std::string route;               // chosen at admission
  CostEstimate svc;                // cost estimate chosen at admission

  int    remaining_decode         = 0;      // decode tokens left to process
  double ready_ms                 = 0.0;    // earliest time this request can be scheduled
  double deadline_ms              = -1.0;   // SLO deadline (-1 = disabled)
  bool   is_lost                  = false;  // true if admitted after SLO miss

  // Admission retry state
  double      admission_next_retry_ms      = 0.0;
  int         admission_attempts           = 0;
  std::string admission_last_reject_reason;  // e.g. "kv_oom", "no_eligible_route"
  double      kv_oom_first_ms              = -1.0;  // time of first KV OOM hit (-1 = never)

  // KV memory reserved in PIM DRAM (freed when request completes)
  uint64_t kv_bytes_reserved = 0;

  // Time this request was admitted (set in process_admission when the
  // transition WaitingAdmission → WaitingPrefill fires). Used by the
  // max_utilization policy to pick the most recently admitted victim (LIFO)
  // when a decode step runs out of KV pages. Reset back to -1 on preemption.
  double admitted_ms = -1.0;

  // Admission-time scheduler prediction for the chosen route. These are
  // absolute/relative times predicted before future arrivals and later decode
  // batching are known, so they are useful for validating routing estimates
  // against eventual completion under load.
  double scheduler_predicted_start_ms        = -1.0;
  double scheduler_predicted_finish_ms       = -1.0;
  double scheduler_predicted_e2e_ms          = -1.0;
  double scheduler_predicted_prefill_ms      = -1.0;
  double scheduler_predicted_decode_total_ms = -1.0;

  // Number of times this request was preempted (evicted from in-flight back
  // to WaitingAdmission) under the max_utilization policy. Always 0 under
  // guaranteed_no_evict.
  int preempt_count = 0;

  // Cumulative count of decode tokens this request recomputed as a tail-trim
  // victim under kv_evict_granularity="tail". Full evictions do not touch
  // this counter (they go into preempt_count + redo_decodes in the log).
  uint64_t tail_trim_tokens = 0;

  // Number of tail-trim events this request has absorbed since its current
  // admission. Reset to 0 when the request is preempted (full-evicted) back
  // to WaitingAdmission. Used by the livelock guard in preempt_tail_trim:
  // once this counter reaches kv_tail_trim_max_per_request, the request is
  // no longer tail-trim-eligible and the caller escalates to full evict.
  int tail_trim_events = 0;

  // Timing (filled in as phases complete, -1 = not yet reached)
  double prefill_start_ms = -1.0;
  double prefill_end_ms   = -1.0;
  double first_token_ms   = -1.0;  // = prefill_end_ms (TTFT)
  double completion_ms    = -1.0;  // set when all decode tokens done

  // Time-between-tokens samples (one entry per decode step)
  std::vector<double> tbt_ms;

  // Drop reason (if state == Dropped)
  std::string dropped_reason;
};

// ─── Active task ────────────────────────────────────────────────────────────
//
// One unit of scheduled work: a prefill or a decode batch.
// The scheduler returns one of these; the frontend executes it.

struct ActiveTask {
  std::string          phase;            // "prefill" or "decode"
  std::string          route;            // "gpu_only" or "pim+gpu"
  std::vector<int>     request_indices;  // indices into RuntimeRequest vector
  int                  context_tokens;   // max context_tokens in batch (= KV depth)
  int                  batch_size;       // number of requests in task
  double               gpu_ms;          // GPU execution time (advance gpu_free_ms by this)
  double               pim_ms;          // PIM execution time (PIM commands drive actual timing)
  double               e2e_ms;          // wall-clock duration (max of gpu+pim in parallel)
  double               start_ms;        // simulation time when task starts
};

// ─── Configuration ──────────────────────────────────────────────────────────
//
// All fields are read from the YAML config in frontend.cpp init().

struct ServingOnlineConfig {
  // Input
  std::string csv_path;           // Azure LLM inference CSV
  std::string cost_gpu_csv;       // cost table for gpu_only route
  std::string cost_pim_csv;       // cost table for pim+gpu route
  // Optional ML tree models (binary .bin from export_cost_model_trees.py).
  // When set, ML inference replaces nearest-neighbor lookup for that route.
  std::string cost_gpu_model;     // e.g. "cluster_outputs/cost_models_full_energy/gpu_only_trees.bin"
  std::string cost_hybrid_model;  // e.g. "cluster_outputs/cost_models_full_energy/lpddr5_pim_bank_trees.bin"

  // Route names (must match what is in the cost CSVs)
  std::string gpu_route_name = "gpu_only";
  std::string pim_route_name = "pim+gpu";

  // Routing policy
  std::string route_policy           = "min_finish";  // or "latency_guarded_energy"
  double      energy_latency_guard_ms = 5.0;           // guard window for latency_guarded_energy

  // SLO thresholds (-1 = disabled)
  double slo_e2e_ms  = -1.0;
  double slo_ttft_ms = -1.0;

  // Scheduling
  int    max_decode_batch_size          = 8;
  bool   prompt_priority                = true;  // always pick prefill first
  int    max_consecutive_decode_batches = 4;     // force prefill after this many decode batches

  // Bounded active set: cap how many requests the scheduler reasons about at
  // once. Arrivals beyond this stay in the FIFO arrival queue at zero per-tick
  // cost until a slot opens (Done/Dropped). 0 = unlimited (legacy behavior, all
  // arrivals enter immediately — O(N²) admission). Recommended: 4× the decode
  // batch size, which keeps the active set wide enough to fill batches with
  // route diversity while bounding per-tick scans to a small constant.
  int    max_active_requests = 0;

  // Admission
  double admission_max_wait_ms      = 0.0;  // 0 = no timeout (hold forever until feasible)
  double admission_retry_interval_ms = 1.0; // ms between admission retries

  // Phase C — admission-control violation budget. When a request D is
  // considered for admission, the scheduler can predict how many
  // currently-running requests would become newly-violated if D joined the
  // decode batch (per-step time grows from bs=N to bs=N+1, slowing every
  // in-flight request). If that count exceeds this budget, D is held in
  // WaitingAdmission and the FCFS loop moves on to the next candidate.
  // Values:
  //   -1 (default) — disabled, preserves prior behavior.
  //    0           — strict, hold D if any in-flight request would be newly violated.
  //    N >= 1      — tolerate up to N newly-violated in-flight requests.
  // Already-violated requests are excluded from the count (we don't penalize
  // D for adding pain to requests that already missed their deadline).
  int admission_violation_budget = -1;

  // Behavior when the chosen PIM route hits KV OOM at admission time.
  //   "hold"         — (default) keep the request in WaitingAdmission and retry
  //                    once KV memory is freed. Preserves route choice at the
  //                    cost of TTFT bloat under pressure.
  //   "fallback_gpu" — immediately re-route the request to the GPU route
  //                    (no KV reservation), so it admits right away.
  //   "hold_then_fallback" — hold for up to kv_oom_hold_limit_ms, then
  //                    fall back to GPU. Balances TTFT with PIM decode speed.
  std::string kv_oom_policy = "hold";
  double kv_oom_hold_limit_ms = 50.0;

  // Top-level KV scheduler policy. Governs how much KV is reserved at
  // admission and whether in-flight requests can be preempted.
  //   "guaranteed_no_evict" — (default) reserve full worst-case KV
  //                    (ctx + gen - 1) per side at admission; never evict.
  //                    kv_oom_policy above chooses the admission-failure
  //                    recovery (hold / fallback_gpu / hold_then_fallback).
  //   "max_utilization"  — reserve only prefill-sized KV at admission;
  //                    grow per decode step via allocate_append. If growth
  //                    hits OOM, preempt the last-admitted request (LIFO)
  //                    and re-admit it later (recomputation per vLLM §4.5).
  //                    kv_oom_policy is ignored (logged) under this policy.
  std::string kv_scheduler_policy = "guaranteed_no_evict";

  // ── max_utilization tuning (ignored under guaranteed_no_evict) ──────────
  //
  // kv_decode_chunk_tokens: number of decode tokens' worth of KV pages to
  //   reserve in a single allocate_append call. 1 = grow per decode step;
  //   higher values amortize allocator work and preemption decisions across
  //   many steps. At admission, prefill_bytes + chunk_tokens_worth is
  //   reserved in one shot.
  int kv_decode_chunk_tokens = 32;

  // kv_evict_granularity: how much of a victim's KV to free on OOM.
  //   "tail" — (default) free only the tail (most recent N tokens) matching
  //            the growing request's need. Victim stays in WaitingDecode and
  //            redoes just the trimmed decode tail on its next step.
  //   "full" — free the victim's entire KV; victim goes back to
  //            WaitingAdmission and re-runs prefill + all decode steps.
  // Only consulted under kv_scheduler_policy == "max_utilization".
  std::string kv_evict_granularity = "tail";

  // kv_tail_trim_max_per_request: livelock guard for tail-trim mode. A
  // request that has already absorbed this many tail-trim events (since its
  // current admission) is excluded from the victim pool; the caller falls
  // through to preempt_lru (full evict). Counter resets to 0 when the
  // request is full-evicted back to WaitingAdmission. 0 disables the guard
  // (legacy behavior — unsafe under oversubscription). Ignored unless
  // kv_scheduler_policy == "max_utilization" and kv_evict_granularity ==
  // "tail".
  int kv_tail_trim_max_per_request = 8;

  // kv_tail_trim_multiplier: how many times the immediate need to free per
  //   tail-trim event. 1 = free exactly what the growing request needs (default,
  //   current behavior). Higher values free more memory per event (e.g., 4 frees
  //   4× the chunk size) so subsequent chunks don't immediately trigger another
  //   OOM, trading more recomputation per event for far fewer total events.
  //   Effective only under kv_scheduler_policy == "max_utilization" and
  //   kv_evict_granularity == "tail".
  int kv_tail_trim_multiplier = 1;

  // predictive_admission: oracle-based size-aware admission gate. When true,
  //   at admission time the scheduler projects the sum of remaining KV bytes
  //   across all in-flight requests + the candidate's final footprint. If the
  //   projection exceeds kv_pool_bytes * predictive_pressure_threshold, the
  //   candidate is held for predictive_hold_ms and re-checked later. Oracle
  //   predictor: uses generated_tokens from the Azure CSV directly.
  bool   predictive_admission           = false;
  double predictive_pressure_threshold  = 1.0;
  double predictive_hold_ms             = 50.0;

  // Arrival scaling (< 1.0 increases offered load)
  double arrival_time_scale = 1.0;
  int    arrival_limit      = -1;   // -1 = load all requests

  // PIM DRAM memory budget for KV cache
  uint64_t kv_pool_bytes   = 0;
  uint64_t page_size_bytes = 4096;
  int      num_channels    = 8;     // LPDDR5 channel count for KV interleaving

  // Model architecture parameters (used for KV size formula and command
  // generation). Defaults are Pi0 (Gemma-2B backbone): 18 layers, 8 query
  // heads, 1 KV head (MQA). Override via YAML for other models
  // (e.g. OpenVLA-7B / Llama-2-7B MHA: 32 / 32 / 32 / 128).
  int num_layers   = 18;
  int num_heads    = 8;     // Q projection heads
  int num_kv_heads = 1;     // K/V projection heads. MHA when == num_heads; MQA when 1.
  int d_head       = 256;
  int dtype_bytes  = 2;     // 2 = FP16

  // Vision-encoder prefix: number of pre-encoded image patch tokens that
  // every request implicitly carries before its text context. Added to
  // context_tokens at request load time so all downstream cost lookups,
  // KV reservations, and command generation see the inflated context.
  // 0 disables (Pi0). 256 is typical for OpenVLA's SigLIP/DINO encoder.
  int vision_prefix_tokens = 0;

  // PIM command stream complexity. Determines how many DRAM commands the
  // PIM command generator emits per (batched) decode step.
  //
  // Per-request modes (cmds scale linearly with batch_size B):
  //   "realistic" (default) — full per-block sequence emitted per request.
  //                           Per (layer, kv_head): (PIM_WR_GB+PIM_MAC_AB)×
  //                           blocks K-side, PIM_MV_SB+PIM_SFM, (PIM_MV_GB+
  //                           PIM_MAC_AB)×blocks V-side, PIM_MV_SB+PIM_BARRIER.
  //                           Total: B × N_L × N_KVH × (4·blocks + 4).
  //   "simple"              — two PIM_MAC_AB per (layer, kv_head, request).
  //                           Context-independent diagnostic.
  //
  // Shared-command modes (Q/V broadcasts + control are amortized across the
  // batch; K/V data MAC behavior differs):
  //   "attacc_qbroadcast"   — AttAcc-faithful. Per (layer, kv_head): 1 PIM_WR_GB
  //                           + B per-request K MAC_AB + 1 PIM_MV_SB + 1 PIM_SFM
  //                           + 1 PIM_MV_GB + B per-request V MAC_AB + 1 PIM_MV_SB
  //                           + 1 PIM_BARRIER. Total: N_L × N_KVH × (2·B·blocks + 6).
  //                           Linear-in-B but with smaller coefficient than
  //                           realistic (Q/V broadcasts and control hoisted).
  //   "bank_stripe"         — Experimental. Per (layer, kv_head): 1 PIM_WR_GB
  //                           + 1 shared MAC_AB per K block + softmax controls
  //                           + 1 PIM_MV_GB + 1 shared MAC_AB per V block +
  //                           final controls. Total: N_L × N_KVH × (2·blocks + 6)
  //                           — independent of B. Models the ideal case where
  //                           bank-level parallelism amortizes per-request
  //                           K/V reads into a single MAC_AB window.
  std::string pim_command_mode = "realistic";

  // Per-shape PIM-decode cycle cache. When true (default), the runtime
  // measures the inline DRAM-drain cycle count the first time it sees a
  // (route, ctx_bucket=ceil(ctx/32), batch_size) shape and reuses it for
  // every subsequent decode task with the same shape — no Ramulator
  // re-run on the hit path. Big speedup on long runs with repeated
  // shapes (continuous batching, multi-step decode loops). False = always
  // run inline (slower but useful for verification).
  bool pim_cache_enabled = true;

  // Cross-run persistence of the per-shape PIM-decode cycle cache. When
  // non-empty, the runtime tries to load this file at init time
  // (populating m_pim_shape_cache before any task runs) and rewrites it
  // atomically at finalize(). Header records model+DRAM fingerprint; a
  // mismatching file is rejected and the cache starts fresh.
  std::string pim_cache_file;     // path or empty = disabled

  // Output files
  std::string requests_out_csv;   // per-request timing CSV (empty = skip)
  std::string debug_log_path;     // structured event log (empty = disabled)
};

}  // namespace Ramulator::ServingOnline

#endif  // RAMULATOR_FRONTEND_IMPL_SERVING_ONLINE_TYPES_H
