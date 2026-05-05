/**
 * serving_online/scheduler.h
 *
 * Admission queue and task selection for the serving_online frontend.
 *
 * The Scheduler is the CPU-side runtime logic. It:
 *   1. Enqueues arriving requests (read from SourceRequest list by arrival_ms).
 *   2. Admits requests: selects a route using the cost table + routing policy,
 *      checks KV memory availability, and transitions to WaitingPrefill.
 *      If KV memory is full, the request is held and retried later.
 *   3. Picks the next task (prefill or decode batch) once the frontend is free.
 *
 * KV OOM hold/retry:
 *   When a PIM route is chosen but PIM DRAM is full, the request stays in
 *   WaitingAdmission with admission_last_reject_reason = "kv_oom". The retry
 *   clock (admission_next_retry_ms) is reset to now + retry_interval.
 *   When any request completes and frees KV memory, wake_kv_held() resets
 *   all kv_oom-blocked retries to now, so they are reconsidered immediately.
 *
 * Routing policies:
 *   "min_finish"              — pick the route with the earliest predicted finish time.
 *   "latency_guarded_energy"  — pick the lowest-energy route within
 *                               energy_latency_guard_ms of the fastest route.
 */

#ifndef RAMULATOR_FRONTEND_IMPL_SERVING_ONLINE_SCHEDULER_H
#define RAMULATOR_FRONTEND_IMPL_SERVING_ONLINE_SCHEDULER_H

#include <functional>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "frontend/impl/serving_online/cost_table.h"
#include "frontend/impl/serving_online/pim_allocator.h"
#include "frontend/impl/serving_online/types.h"

namespace Ramulator::ServingOnline {

class Scheduler {
 public:
  /**
   * Initialize the scheduler.
   *
   * @param cfg       Runtime configuration (routes, scheduling params, etc.).
   * @param arrivals  All source requests, sorted by arrival_ms.
   * @param table     Cost table (must outlive this object).
   * @param alloc     KV allocator (must outlive this object).
   */
  void initialize(const ServingOnlineConfig& cfg,
                  std::vector<SourceRequest> arrivals,
                  const CostTable& table,
                  PimAllocator& alloc);

  /**
   * Advance the scheduler to simulation time `now_ms`.
   *
   * Enqueues new arrivals, processes the admission queue, and selects the
   * next task. Returns nullopt if there is no work ready to start yet
   * (e.g., no admitted request is ready, or all requests are done).
   *
   * Call this once per frontend tick (or whenever the active task finishes
   * and a new one is needed).
   */
  std::optional<ActiveTask> tick(double now_ms, double gpu_free_ms, double pim_free_ms);

  /**
   * Mark a task as completed at `finish_ms`.
   *
   * Updates all request states in the task (WaitingDecode → Done or
   * remaining_decode--), frees KV memory for completed requests,
   * and wakes any kv_oom-held admission candidates.
   */
  void complete_task(const ActiveTask& task, double finish_ms);

  // Returns true once all source requests have arrived, been admitted or
  // dropped, and completed (state == Done or Dropped).
  bool all_done() const;

  // Optional debug log callback. When set, the scheduler emits one line per
  // notable event (arrival, admission, KV alloc/free/OOM) as a formatted
  // string. The caller writes the string wherever it likes (file, stdout …).
  void set_log_fn(std::function<void(const std::string&)> fn) {
    m_log_fn = std::move(fn);
  }

  // Read-only access to request list (for stats / output CSV writing).
  const std::vector<RuntimeRequest>& requests() const { return m_requests; }

  // Mutable access — used by the frontend to stamp prefill_start_ms when a
  // prefill task begins (the scheduler only sets prefill_end_ms on completion).
  std::vector<RuntimeRequest>& mutable_requests() { return m_requests; }

  // Summary counters
  int admitted_count()     const { return m_admitted_count; }
  int dropped_count()      const { return m_dropped_count; }
  int kv_hold_count()      const { return m_kv_hold_count; }
  int preempt_count()      const { return m_preempt_count; }
  int tail_trim_count()    const { return m_tail_trim_count; }
  int full_evict_count()   const { return m_full_evict_count; }
  uint64_t tokens_recomputed() const { return m_tokens_recomputed; }
  int predictive_holds()   const { return m_predictive_holds; }

  // Earliest future time at which scheduler state may change without any
  // active task running. Returns nullopt if no future event is known.
  std::optional<double> next_wakeup_ms() const;

 private:
  // ── Arrival ────────────────────────────────────────────────────────────
  void enqueue_arrivals(double now_ms);

  // ── Admission ──────────────────────────────────────────────────────────
  void process_admission(double now_ms, double gpu_free_ms, double pim_free_ms);

  // Evaluate a route and return its predicted finish time.
  // Returns -1.0 if no cost estimate exists for this route.
  double predict_finish(const std::string& route,
                        int context_tokens, int generated_tokens,
                        double now_ms, double gpu_free_ms, double pim_free_ms) const;

  // Choose the best route among candidates using the configured policy.
  // Returns the route name, or "" if no eligible route exists.
  std::string choose_route(int context_tokens, int generated_tokens,
                           double now_ms, double gpu_free_ms, double pim_free_ms,
                           double deadline_ms) const;

  // KV bytes needed to cache all K and V tensors for one request at its
  // maximum decode context (ctx + gen - 1). Used by guaranteed_no_evict.
  uint64_t kv_bytes_per_side(const RuntimeRequest& req) const;

  // KV bytes needed for an arbitrary context-length. Used by max_utilization
  // to compute the smaller "prefill-only" reservation at admission, and the
  // per-decode-step delta during growth.
  uint64_t kv_bytes_for_ctx(int ctx_tokens) const;

  // KV bytes produced per decode token per side (= num_layers * num_heads
  // * d_head * dtype_bytes). Used by tail-trim to convert a byte delta into
  // a token count.
  uint64_t kv_bytes_per_token_per_side() const;

  // Sum of predicted final KV footprint (per-side × 2) across all in-flight
  // requests (WaitingPrefill + WaitingDecode). Used by predictive_admission
  // to gate new admissions.
  uint64_t projected_in_flight_kv_bytes() const;

  // Admission-time reservation size per side. Dispatches on
  // cfg.kv_scheduler_policy: full worst-case for guaranteed_no_evict,
  // prefill-only for max_utilization.
  uint64_t kv_admission_bytes_per_side(const RuntimeRequest& req) const;

  // Preempt the most recently admitted in-flight request (largest admitted_ms)
  // whose index != excluded_index. Frees its KV pages and returns it to
  // WaitingAdmission for later re-prefill. Returns true on success, false if
  // no suitable victim exists.
  bool preempt_lru(int excluded_index, double now_ms);

  // Tail-trim preemption: free only the most recent pages of the LIFO victim,
  // enough to cover bytes_needed_per_side (rounded up to whole tokens).
  // Victim stays in WaitingDecode; its remaining_decode is bumped so the
  // trimmed tail is recomputed on its next decode step. Returns false if
  // no victim has any completed decode tokens to trim — caller should fall
  // back to preempt_lru.
  bool preempt_tail_trim(int excluded_index, uint64_t bytes_needed_per_side,
                          double now_ms);

  // Grow the KV reservation for request at req_index to cover the next
  // scheduled step (prefill output token or next decode read). Only active
  // under max_utilization. If growth can't be satisfied, preempts the
  // last-admitted other request and retries. If no other victim is
  // available, self-preempts req_index. Returns true if the request kept
  // running; false if it was self-preempted.
  bool grow_kv_or_preempt(int req_index, double now_ms);

  // ── KV OOM wakeup ──────────────────────────────────────────────────────
  // After freeing KV memory, reset retry clocks for all kv_oom-held requests.
  void wake_kv_held(double now_ms);

  // ── Task selection ─────────────────────────────────────────────────────
  std::optional<ActiveTask> pick_task(double now_ms, double gpu_free_ms,
                                      double pim_free_ms) const;
  std::optional<ActiveTask> pick_prefill(double now_ms) const;
  std::optional<ActiveTask> pick_decode_batch(double now_ms) const;

  // ── Helpers ────────────────────────────────────────────────────────────
  RuntimeRequest* find_req(int id);

  // ── Debug log ──────────────────────────────────────────────────────────
  std::function<void(const std::string&)> m_log_fn;
  void dlog(const std::string& msg) const { if (m_log_fn) m_log_fn(msg); }

  // ── State ──────────────────────────────────────────────────────────────
  ServingOnlineConfig             m_cfg;
  const CostTable*           m_table   = nullptr;
  PimAllocator*              m_alloc   = nullptr;

  std::vector<SourceRequest>  m_arrivals;
  size_t                      m_arrival_idx = 0;
  std::vector<RuntimeRequest> m_requests;

  // Effective cap on concurrent active (non-terminal) requests in m_requests.
  // 0 = unlimited. Set in initialize() from cfg.max_active_requests, with the
  // auto-default applied when cfg.max_active_requests <= 0.
  int m_active_cap   = 0;
  int m_active_count = 0;   // requests currently in non-terminal state

  // Per-state indices into m_requests. Kept in sync via set_state(); all
  // scheduler hot paths iterate only the relevant index instead of scanning
  // all of m_requests (which grows monotonically as requests complete and
  // accumulate in Done/Dropped state).
  std::unordered_set<int> m_waiting_admission;
  std::unordered_set<int> m_waiting_prefill;
  std::unordered_set<int> m_waiting_decode;

  // Centralized state transition. Updates m_requests[idx].state AND keeps the
  // per-state indices in sync. ALWAYS use this helper for state transitions.
  void set_state(int idx, ReqState new_state);

  int m_consecutive_decode = 0;

  // Summary counters
  int m_admitted_count   = 0;
  int m_dropped_count    = 0;
  int m_kv_hold_count    = 0;
  int m_preempt_count    = 0;           // total preemption events (tail + full)
  int m_tail_trim_count  = 0;           // tail-trim events (M2)
  int m_full_evict_count = 0;           // full-request preemptions (preempt_lru)
  uint64_t m_tokens_recomputed = 0;     // decode tokens trimmed across all events
  int m_predictive_holds = 0;           // admission holds under M3
};

}  // namespace Ramulator::ServingOnline

#endif  // RAMULATOR_FRONTEND_IMPL_SERVING_ONLINE_SCHEDULER_H
