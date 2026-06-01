/**
 * serving_online/scheduler.cpp
 *
 * Admission queue and task selection. See scheduler.h for design overview.
 */

#include "frontend/impl/serving_online/scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace {
// Small snprintf wrapper that returns a std::string (avoids fmt dependency).
template<typename... Args>
static std::string sfmt(const char* fmt_str, Args... args) {
  char buf[512];
  std::snprintf(buf, sizeof(buf), fmt_str, args...);
  return {buf};
}
} // namespace

namespace Ramulator::ServingOnline {

// ─── set_state() ─────────────────────────────────────────────────────────────
// Centralized state transition. Removes the request from its current state's
// index (if any) and inserts into the new state's index. Done/Dropped have no
// index — terminal states never get scanned.

void Scheduler::set_state(int idx, ReqState new_state) {
  RuntimeRequest& r = m_requests[idx];
  switch (r.state) {
    case ReqState::WaitingAdmission: m_waiting_admission.erase(idx); break;
    case ReqState::WaitingPrefill:   m_waiting_prefill.erase(idx);   break;
    case ReqState::WaitingDecode:    m_waiting_decode.erase(idx);    break;
    case ReqState::Done:
    case ReqState::Dropped:                                          break;
  }
  r.state = new_state;
  switch (new_state) {
    case ReqState::WaitingAdmission: m_waiting_admission.insert(idx); break;
    case ReqState::WaitingPrefill:   m_waiting_prefill.insert(idx);   break;
    case ReqState::WaitingDecode:    m_waiting_decode.insert(idx);    break;
    case ReqState::Done:
    case ReqState::Dropped:                                           break;
  }
}

// ─── initialize() ────────────────────────────────────────────────────────────

void Scheduler::initialize(const ServingOnlineConfig& cfg,
                            std::vector<SourceRequest> arrivals,
                            const CostTable& table,
                            PimAllocator& alloc) {
  m_cfg      = cfg;
  m_arrivals = std::move(arrivals);
  m_table    = &table;
  m_alloc    = &alloc;
  m_arrival_idx = 0;
  m_requests.clear();
  m_consecutive_decode = 0;
  m_admitted_count = m_dropped_count = m_kv_hold_count = 0;
  m_preempt_count = m_tail_trim_count = m_full_evict_count = 0;
  m_tokens_recomputed = 0;
  m_predictive_holds = 0;
  m_active_count = 0;
  m_waiting_admission.clear();
  m_waiting_prefill.clear();
  m_waiting_decode.clear();
  // Resolve effective active-set cap.
  //   < 0  → unlimited (legacy behavior, no cap, all arrivals enter immediately)
  //   = 0  → unlimited (default keeps legacy behavior; explicit positive value
  //          activates the cap)
  //   > 0  → use that as the cap
  m_active_cap = (cfg.max_active_requests > 0) ? cfg.max_active_requests : 0;
}

// ─── all_done() ──────────────────────────────────────────────────────────────

bool Scheduler::all_done() const {
  // All arrivals must have been enqueued.
  if (m_arrival_idx < m_arrivals.size()) return false;
  // No request may be in a non-terminal state.
  return m_waiting_admission.empty()
      && m_waiting_prefill.empty()
      && m_waiting_decode.empty();
}

// ─── tick() ──────────────────────────────────────────────────────────────────

std::optional<ActiveTask> Scheduler::tick(double now_ms,
                                          double gpu_free_ms,
                                          double pim_free_ms) {
  enqueue_arrivals(now_ms);
  process_admission(now_ms, gpu_free_ms, pim_free_ms);
  auto task = pick_task(now_ms, gpu_free_ms, pim_free_ms);
  // Phase E (telemetry): record the actual batch size of any decode task
  // launched this tick. pick_task is const, so we record here in the
  // (non-const) tick path. Prefill tasks are recorded separately if needed
  // — for now we only need decode-batch stats.
  if (task.has_value() && task->phase == "decode") {
    record_decode_batch_size(task->batch_size);
  }
  return task;
}

// ─── enqueue_arrivals() ──────────────────────────────────────────────────────

void Scheduler::enqueue_arrivals(double now_ms) {
  while (m_arrival_idx < m_arrivals.size() &&
         m_arrivals[m_arrival_idx].arrival_ms <= now_ms) {
    // Bounded active set: stop promoting once we've hit the cap. Remaining
    // arrivals stay in m_arrivals[] at zero per-tick cost; they get pulled in
    // on the next slot-opening (Done/Dropped) → enqueue_arrivals call. Keying
    // on now_ms (not the trace arrival time) preserves arrival_ms semantics:
    // queue wait shows up naturally as bigger TTFT/E2E.
    if (m_active_cap > 0 && m_active_count >= m_active_cap) break;

    const auto& src = m_arrivals[m_arrival_idx++];
    RuntimeRequest rr;
    rr.id               = src.id;
    rr.arrival_ms       = src.arrival_ms;
    // Vision-encoder prefix is added once, here, so every downstream
    // path (cost-table lookup, KV reservation, command generation) sees
    // the inflated context. 0 for text-only models (Pi0).
    rr.context_tokens   = src.context_tokens + m_cfg.vision_prefix_tokens;
    rr.generated_tokens = src.generated_tokens;
    rr.ready_ms         = src.arrival_ms;
    rr.admission_next_retry_ms = src.arrival_ms;  // eligible immediately on arrival
    if (m_cfg.slo_e2e_ms > 0.0) {
      rr.deadline_ms = src.arrival_ms + m_cfg.slo_e2e_ms;
    }
    dlog(sfmt("[%10.3fms] ARRIVE    req=%3d  ctx=%5d  gen=%4d  deadline=%.1fms",
              src.arrival_ms, src.id, src.context_tokens, src.generated_tokens,
              rr.deadline_ms < 0 ? -1.0 : rr.deadline_ms));
    const int new_idx = static_cast<int>(m_requests.size());
    const int req_id = m_requests.back().id;  // before move; but m_requests was already pushed
    (void)req_id;  // unused — using src.id below for the snapshot tag
    m_requests.push_back(std::move(rr));
    m_active_count++;
    m_waiting_admission.insert(new_idx);   // request enters in WaitingAdmission
    log_queue_event(now_ms, "arrival_enqueued", src.id);
  }
}

// ─── log_queue_event() — Phase E telemetry ──────────────────────────────────
// Append one QueueEvent capturing the current admission-queue order. No I/O
// here — events are drained at end-of-run by frontend.cpp::write_queue_snapshots().
void Scheduler::log_queue_event(double now_ms, const std::string& event,
                                 int req_id, const std::string& reason,
                                 double wait_ms, const std::string& note) {
  if (m_queue_events.size() >= kMaxQueueEvents) {
    if (!m_queue_events_truncated_logged) {
      QueueEvent ev;
      ev.event_index = m_queue_event_counter++;
      ev.sim_now_ms  = now_ms;
      ev.event       = "truncated";
      ev.request_id  = -1;
      ev.note        = "queue_events log capped at kMaxQueueEvents";
      m_queue_events.push_back(std::move(ev));
      m_queue_events_truncated_logged = true;
    }
    return;
  }
  QueueEvent ev;
  ev.event_index = m_queue_event_counter++;
  ev.sim_now_ms  = now_ms;
  ev.event       = event;
  ev.request_id  = req_id;
  ev.reason      = reason;
  ev.wait_ms     = wait_ms;
  // Capture the first 8 admission-queue ids (in arbitrary order — unordered_set).
  ev.queue_order.reserve(std::min<size_t>(8, m_waiting_admission.size()));
  for (int idx : m_waiting_admission) {
    if (ev.queue_order.size() >= 8) break;
    ev.queue_order.push_back(m_requests[idx].id);
  }
  ev.note = note;
  m_queue_events.push_back(std::move(ev));
}

// ─── kv_bytes_per_side() ─────────────────────────────────────────────────────

uint64_t Scheduler::kv_bytes_for_ctx(int ctx_tokens) const {
  // Per-side (K or V) bytes. K and V each scale with num_kv_heads, NOT
  // num_heads (the Q-projection heads). Callers multiply by 2 externally
  // when they need total K+V bytes (e.g. scheduler.cpp:528, 848 / 184, 433).
  if (ctx_tokens < 0) ctx_tokens = 0;
  return static_cast<uint64_t>(m_cfg.num_layers)
       * static_cast<uint64_t>(m_cfg.num_kv_heads)
       * static_cast<uint64_t>(ctx_tokens)
       * static_cast<uint64_t>(m_cfg.d_head)
       * static_cast<uint64_t>(m_cfg.dtype_bytes);
}

uint64_t Scheduler::kv_bytes_per_side(const RuntimeRequest& req) const {
  // Reserve KV for the largest decode-time context the request will reach.
  // Prefill produces the first token, and the decode loop generates the
  // remaining generated_tokens - 1 tokens. The final decode step therefore
  // reads a KV cache of context_tokens + generated_tokens - 1.
  const int max_ctx = req.context_tokens + std::max(0, req.generated_tokens - 1);
  return kv_bytes_for_ctx(max_ctx);
}

uint64_t Scheduler::kv_bytes_per_token_per_side() const {
  // Per-token, per-side (K or V). Scales with num_kv_heads, not num_heads.
  return static_cast<uint64_t>(m_cfg.num_layers)
       * static_cast<uint64_t>(m_cfg.num_kv_heads)
       * static_cast<uint64_t>(m_cfg.d_head)
       * static_cast<uint64_t>(m_cfg.dtype_bytes);
}

uint64_t Scheduler::kv_admission_bytes_per_side(const RuntimeRequest& req) const {
  if (m_cfg.kv_scheduler_policy == "max_utilization") {
    // Reserve the prompt + one decode chunk in a single shot; growth per
    // chunk after that. Capped at the worst-case final context.
    const int chunk  = std::max(1, m_cfg.kv_decode_chunk_tokens);
    const int max_ctx = req.context_tokens + std::max(0, req.generated_tokens - 1);
    const int init_ctx = std::min(req.context_tokens + chunk, max_ctx);
    return kv_bytes_for_ctx(init_ctx);
  }
  // guaranteed_no_evict (default): reserve the full worst case.
  return kv_bytes_per_side(req);
}

uint64_t Scheduler::projected_in_flight_kv_bytes() const {
  // Iterate the WaitingPrefill ∪ WaitingDecode indices instead of all of
  // m_requests. With bounded active set this is O(K) regardless of total N.
  uint64_t sum = 0;
  auto add = [&](int idx) {
    const auto& r = m_requests[idx];
    if (!r.svc.uses_pim) return;          // GPU-only in-flights don't use KV
    const int max_ctx = r.context_tokens + std::max(0, r.generated_tokens - 1);
    sum += 2ull * kv_bytes_for_ctx(max_ctx);
  };
  for (int i : m_waiting_prefill) add(i);
  for (int i : m_waiting_decode)  add(i);
  return sum;
}

// ─── predict_finish() ────────────────────────────────────────────────────────

double Scheduler::predict_finish(const std::string& route,
                                  int context_tokens, int generated_tokens,
                                  double now_ms, double gpu_free_ms,
                                  double pim_free_ms) const {
  auto pred = predict_finish_detail(route, context_tokens, generated_tokens,
                                    now_ms, gpu_free_ms, pim_free_ms);
  return pred ? pred->finish_ms : -1.0;
}

std::optional<Scheduler::PredictionEstimate> Scheduler::predict_finish_detail(
    const std::string& route,
    int context_tokens, int generated_tokens,
    double now_ms, double gpu_free_ms, double pim_free_ms) const {
  auto est = m_table->estimate(route, context_tokens, generated_tokens, 1);
  if (!est) return std::nullopt;

  struct DecodeBacklog {
    double token_work_ms = 0.0;
    int    request_count = 0;
  };

  DecodeBacklog pim_decode;
  DecodeBacklog gpu_decode;
  double queued_prefill_ms = 0.0;

  auto add_decode = [&](const RuntimeRequest& r, int tokens) {
    if (tokens <= 0) return;
    DecodeBacklog& b = (r.route == m_cfg.pim_route_name) ? pim_decode : gpu_decode;
    b.token_work_ms += static_cast<double>(tokens) * r.svc.decode_e2e_ms;
    b.request_count++;
  };

  // Requests already admitted but not prefetched are ahead of this candidate's
  // prefill under prompt-priority scheduling. Do not count their full future
  // decode here: once they reach decode, the candidate can batch/interleave with
  // them instead of waiting for all of their tokens to finish first.
  for (int i : m_waiting_prefill) {
    const auto& r = m_requests[i];
    queued_prefill_ms += r.svc.prefill_e2e_ms;
  }
  for (int i : m_waiting_decode) {
    const auto& r = m_requests[i];
    add_decode(r, r.remaining_decode);
  }

  auto batched_decode_ms = [&](const DecodeBacklog& b) {
    if (b.request_count <= 0) return 0.0;
    const int effective_bs = std::max(1, std::min(m_cfg.max_decode_batch_size,
                                                  b.request_count));
    return b.token_work_ms / static_cast<double>(effective_bs);
  };

  double queued_decode_ms = 0.0;
  if (route == m_cfg.pim_route_name) {
    // PIM requests wait behind other PIM decode work only.
    queued_decode_ms = batched_decode_ms(pim_decode);
  } else if (route == m_cfg.gpu_route_name) {
    // GPU-only requests interleave with PIM via pick_decode_batch() (each
    // tick picks whichever cohort has the earliest ready_ms). A GPU-only
    // request does NOT have to drain the entire PIM backlog first — they
    // take turns. Use only the GPU-specific backlog so the prediction
    // reflects the actual competition for the GPU resource, not the PIM
    // queue depth. This allows choose_route() to prefer GPU when PIM is
    // heavily backlogged.
    queued_decode_ms = batched_decode_ms(gpu_decode);
  } else {
    queued_decode_ms = batched_decode_ms(pim_decode) + batched_decode_ms(gpu_decode);
  }
  // Phase D — cross-resource contention accounting:
  //   PIM-route tasks consume BOTH gpu_free_ms (for FFN+projection) AND
  //   pim_free_ms (for attention). If a candidate D is admitted on PIM
  //   while gpu_free_ms is busy from prior PIM tasks' FFN portions, D
  //   has to wait for the GPU as well. Without this max, the predictor
  //   systematically underestimates PIM's true cost → too many requests
  //   route to PIM → GPU queue silently grows from PIM tasks' FFN work
  //   → overflow GPU-routed requests pay the bill in tail latency
  //   (E5, 2026-05-27: hybrid p99 7.7 s for 10% GPU traffic vs 524 ms
  //    pure GPU-only).
  //   GPU-route's start uses gpu_free_ms only (unchanged) since pure
  //   GPU tasks don't touch pim_free_ms.
  const double hw_free_ms = (route == m_cfg.pim_route_name)
      ? std::max(pim_free_ms, gpu_free_ms)
      : gpu_free_ms;
  const double base_ms = std::max(now_ms, hw_free_ms);
  const double start = base_ms + queued_prefill_ms;
  const double decode_total =
      static_cast<double>(est->decode_tokens) * est->decode_e2e_ms;

  PredictionEstimate pred;
  pred.start_ms        = start;
  pred.prefill_ms      = est->prefill_e2e_ms;
  pred.decode_total_ms = decode_total;
  pred.finish_ms       = start + est->prefill_e2e_ms
                       + queued_decode_ms + decode_total;
  return pred;
}

// ─── choose_route() ──────────────────────────────────────────────────────────

std::string Scheduler::choose_route(int context_tokens, int generated_tokens,
                                     double now_ms, double gpu_free_ms,
                                     double pim_free_ms, double deadline_ms) const {
  // Collect candidates for all configured routes.
  struct Candidate {
    std::string route;
    bool        uses_pim;
    double      finish_ms;
    double      energy_nj;   // total prefill+decode energy (for energy policy)
    bool        eligible;    // true if finish_ms <= deadline_ms (or no deadline)
  };

  std::vector<Candidate> cands;
  for (const std::string& route : {m_cfg.pim_route_name, m_cfg.gpu_route_name}) {
    double finish = predict_finish(route, context_tokens, generated_tokens,
                                   now_ms, gpu_free_ms, pim_free_ms);
    if (finish < 0.0) continue;  // no cost data for this route

    auto est = m_table->estimate(route, context_tokens, generated_tokens, 1);
    double energy = 0.0;
    if (est) {
      energy = est->prefill_energy_nj
             + static_cast<double>(est->decode_tokens) * est->decode_energy_nj;
    }

    bool eligible = (deadline_ms < 0.0) || (finish <= deadline_ms);
    cands.push_back({route, est ? est->uses_pim : false,
                     finish, energy, eligible});
  }

  if (cands.empty()) return "";

  // Prefer eligible routes; fall back to best available if none are eligible.
  std::vector<Candidate*> pool;
  for (auto& c : cands) {
    if (c.eligible) pool.push_back(&c);
  }
  if (pool.empty()) {
    // No deadline-feasible route — use all candidates (best-effort).
    for (auto& c : cands) pool.push_back(&c);
  }
  if (pool.empty()) return "";

  if (m_cfg.route_policy == "latency_guarded_energy") {
    // Find the fastest finish time among candidates in the pool.
    double fastest = std::numeric_limits<double>::infinity();
    for (const auto* c : pool) fastest = std::min(fastest, c->finish_ms);

    // Keep only routes within guard_ms of the fastest.
    double guard = m_cfg.energy_latency_guard_ms;
    std::vector<Candidate*> guarded;
    for (auto* c : pool) {
      if (c->finish_ms <= fastest + guard + 1e-9) guarded.push_back(c);
    }
    if (guarded.empty()) guarded = pool;

    // Among the guarded set, pick the lowest energy.
    auto* best = *std::min_element(guarded.begin(), guarded.end(),
        [](const Candidate* a, const Candidate* b) {
          if (a->energy_nj != b->energy_nj) return a->energy_nj < b->energy_nj;
          return a->finish_ms < b->finish_ms;
        });
    return best->route;
  }

  // Default: "min_finish" — pick earliest predicted finish.
  auto* best = *std::min_element(pool.begin(), pool.end(),
      [](const Candidate* a, const Candidate* b) {
        if (a->finish_ms != b->finish_ms) return a->finish_ms < b->finish_ms;
        // Tie: prefer GPU route (more predictable tail latency).
        return (!a->uses_pim) > (!b->uses_pim);
      });
  return best->route;
}

// ─── newly_violated_for_route() — Phase C admission-control helper ──────────

int Scheduler::newly_violated_for_route(const RuntimeRequest& D,
                                         const std::string& route,
                                         double now_ms,
                                         double gpu_free_ms,
                                         double pim_free_ms) const {
  // Phase C′: predict_finish-based interference model.
  //
  // The original per-step delta model (est(N+1) - est(N)) returned 0 in
  // practice because the cost-table predicts batching is beneficial —
  // adding a request to a sub-saturated batch typically REDUCES per-step
  // decode time. Real interference comes from D consuming hardware time:
  // its prefill/decode advance gpu_free_ms (and pim_free_ms for PIM-routed),
  // delaying in-flight requests gated by those clocks.
  //
  // We project the hardware-free clocks forward as if D were admitted on
  // `route` right now, then call predict_finish() for each in-flight r
  // under both the OLD and NEW clocks. The queue term and batching gain
  // are already inside predict_finish — we don't double-count them.

  // 1. Estimate D's hardware footprint on `route` at single-request bs (the
  //    same bs used everywhere predict_finish is called).
  auto D_est = m_table->estimate(route, D.context_tokens, D.generated_tokens, 1);
  if (!D_est) return 0;  // no cost data → can't reason → don't hold

  // 2. Collect in-flight requests on `route` (WaitingPrefill ∪ WaitingDecode).
  std::vector<int> inflight;
  inflight.reserve(m_waiting_prefill.size() + m_waiting_decode.size());
  for (int i : m_waiting_prefill) {
    if (m_requests[i].route == route) inflight.push_back(i);
  }
  for (int i : m_waiting_decode) {
    if (m_requests[i].route == route) inflight.push_back(i);
  }
  if (inflight.empty()) return 0;  // empty batch — D affects nobody

  // 3. Project gpu_free_ms / pim_free_ms forward by D's admission.
  //    - Prefill is GPU-only on both routes (cost-table has prefill_pim_ms = 0).
  //    - Decode: D.decode_gpu_ms × (Lout-1) on GPU; D.decode_pim_ms × (Lout-1) on PIM.
  //    - For a GPU-route admission, only gpu_free_ms advances.
  //    - For a PIM-route admission, BOTH advance (PIM-routed decode also runs
  //      FFN/projections on GPU in parallel).
  const double base_gpu = std::max(now_ms, gpu_free_ms);
  const double base_pim = std::max(now_ms, pim_free_ms);
  const double decode_tokens = static_cast<double>(D_est->decode_tokens);
  double new_gpu_free = base_gpu + D_est->prefill_e2e_ms
                      + decode_tokens * D_est->decode_gpu_ms;
  double new_pim_free = pim_free_ms;  // GPU-route doesn't touch PIM clock
  if (route == m_cfg.pim_route_name) {
    new_pim_free = base_pim + decode_tokens * D_est->decode_pim_ms;
  }

  // 4. For each in-flight r, recompute predict_finish under both clock sets
  //    and count crossings of r.deadline_ms.
  int newly_violated = 0;
  for (int i : inflight) {
    const auto& r = m_requests[i];
    if (r.deadline_ms < 0.0) continue;  // SLO disabled for r

    const double r_old = predict_finish(r.route, r.context_tokens,
                                        r.generated_tokens, now_ms,
                                        gpu_free_ms, pim_free_ms);
    if (r_old < 0.0) continue;                       // no cost data for r
    if (r_old > r.deadline_ms) continue;             // already violated → exclude
    const double r_new = predict_finish(r.route, r.context_tokens,
                                        r.generated_tokens, now_ms,
                                        new_gpu_free, new_pim_free);
    if (r_new > r.deadline_ms) ++newly_violated;
  }
  return newly_violated;
}

// ─── process_admission() ─────────────────────────────────────────────────────

void Scheduler::process_admission(double now_ms, double gpu_free_ms,
                                   double pim_free_ms) {
  if (m_waiting_admission.empty()) return;   // fast no-op when nothing to admit

  // Sort eligible candidates by (arrival_ms, id) — FCFS. Carry the index so
  // we can erase from m_waiting_admission when a state transition fires.
  std::vector<int> waiting;
  waiting.reserve(m_waiting_admission.size());
  for (int idx : m_waiting_admission) {
    auto& r = m_requests[idx];
    if (r.admission_next_retry_ms <= now_ms) waiting.push_back(idx);
  }
  std::sort(waiting.begin(), waiting.end(),
            [&](int a, int b) {
              const auto& ra = m_requests[a];
              const auto& rb = m_requests[b];
              if (ra.arrival_ms != rb.arrival_ms) return ra.arrival_ms < rb.arrival_ms;
              return ra.id < rb.id;
            });

  for (int wait_idx : waiting) {
    RuntimeRequest* req = &m_requests[wait_idx];
    req->admission_attempts++;

    // Choose the best route.
    std::string route = choose_route(req->context_tokens, req->generated_tokens,
                                     now_ms, gpu_free_ms, pim_free_ms, req->deadline_ms);

    // Phase C‴ — admission-control violation budget, HOLD-ONLY (no reroute).
    //
    // Spec (user-confirmed 2026-05-24, after Phase C″ wrong-direction):
    //   Check the min_finish-chosen route's newly_violated count. If it
    //   exceeds the budget, HOLD D in WaitingAdmission. Do NOT try the
    //   alternative route — empirical evidence (E4 v2 + v3 OpenVLA x=0)
    //   showed that rerouting D to the slower route triggers a cascade:
    //   D piles onto PIM, PIM queue grows, subsequent admissions also
    //   reroute to PIM, violators climb.
    //
    //   Hold-only matches the original user spec: when admitting D NOW
    //   would push X+1 in-flight past deadline, wait. The natural FCFS
    //   loop in process_admission() picks the next waiting request E;
    //   if E doesn't violate, E gets admitted first (queue-jumping).
    //   Held D is re-evaluated on the next tick when admission_next_retry_ms
    //   <= now_ms — usually after a completion has freed up the route.
    if (!route.empty() && m_cfg.admission_violation_budget >= 0) {
      const int budget = m_cfg.admission_violation_budget;
      const int nv_chosen = newly_violated_for_route(*req, route, now_ms,
                                                     gpu_free_ms, pim_free_ms);
      if (nv_chosen > budget) {
        // Hold — no reroute attempt. Wait for in-flight to drain.
        req->admission_next_retry_ms = now_ms + m_cfg.admission_retry_interval_ms;
        req->admission_last_reject_reason = "violation_budget";
        ++m_admission_violation_holds;
        dlog(sfmt("[%10.3fms] HOLD_VIOL req=%3d  route=%s  nv_chosen=%d  "
                  "budget=%d  retry_at=%.1fms",
                  now_ms, req->id, route.c_str(), nv_chosen, budget,
                  req->admission_next_retry_ms));
        log_queue_event(now_ms, "hold", req->id, "violation_budget",
                        now_ms - req->arrival_ms);
        continue;
      }
      // nv_chosen <= budget → admit on min_finish's chosen route (no change).
    }

    if (route.empty()) {
      // No route available yet.
      double waited = now_ms - req->arrival_ms;
      if (m_cfg.admission_max_wait_ms > 0.0 &&
          waited >= m_cfg.admission_max_wait_ms) {
        // Waited too long: force best-effort admission on any available route.
        // Try PIM first, then GPU.
        for (const std::string& r : {m_cfg.pim_route_name, m_cfg.gpu_route_name}) {
          if (m_table->estimate(r, req->context_tokens, req->generated_tokens, 1)) {
            route = r;
            break;
          }
        }
        if (route.empty()) {
          // No cost data for any route — drop the request.
          set_state(wait_idx, ReqState::Dropped);
          req->dropped_reason = "no_route_for_best_effort";
          m_dropped_count++;
          m_active_count--;
          continue;
        }
        req->is_lost = true;
      } else {
        // Hold: try again later.
        req->admission_next_retry_ms = now_ms + m_cfg.admission_retry_interval_ms;
        continue;
      }
    }

    // Get cost estimate for chosen route.
    auto est_opt = m_table->estimate(route, req->context_tokens, req->generated_tokens, 1);
    if (!est_opt) {
      req->admission_next_retry_ms = now_ms + m_cfg.admission_retry_interval_ms;
      continue;
    }
    const CostEstimate& est = *est_opt;

    // For PIM routes: check KV memory availability.
    if (est.uses_pim) {
      // M3: predictive-admission pressure gate. Under max_utilization with
      // oracle prediction, hold the candidate if admitting it would push the
      // total *final* KV footprint of in-flight + candidate past the pool
      // pressure threshold. Short later-arriving requests pass through.
      if (m_cfg.predictive_admission &&
          m_cfg.kv_scheduler_policy == "max_utilization") {
        const uint64_t projected = projected_in_flight_kv_bytes();
        const int      cand_ctx  = req->context_tokens
                                 + std::max(0, req->generated_tokens - 1);
        const uint64_t cand_bytes = 2ull * kv_bytes_for_ctx(cand_ctx);
        const uint64_t limit = static_cast<uint64_t>(
            static_cast<double>(m_cfg.kv_pool_bytes)
            * m_cfg.predictive_pressure_threshold);
        if (projected + cand_bytes > limit) {
          dlog(sfmt("[%10.3fms] PRED_HOLD req=%3d  projected=%llu  cand=%llu  limit=%llu",
                    now_ms, req->id,
                    (unsigned long long)projected,
                    (unsigned long long)cand_bytes,
                    (unsigned long long)limit));
          req->admission_last_reject_reason = "predictive_pressure";
          req->admission_next_retry_ms      = now_ms + m_cfg.predictive_hold_ms;
          ++m_predictive_holds;
          log_queue_event(now_ms, "hold", req->id, "predictive_pressure",
                          now_ms - req->arrival_ms);
          continue;
        }
      }

      uint64_t kv_side = kv_admission_bytes_per_side(*req);
      if (!m_alloc->can_fit(KvRegion::Key, kv_side) ||
          !m_alloc->can_fit(KvRegion::Value, kv_side)) {
        m_kv_hold_count++;

        // Record when this request first hit KV OOM.
        if (req->kv_oom_first_ms < 0.0) req->kv_oom_first_ms = now_ms;

        // If this request's initial KV footprint exceeds the per-region pool
        // capacity it can never fit in PIM even with an empty pool — force GPU
        // fallback immediately, regardless of scheduler policy. This prevents
        // infinite KV_OOM_HOLD loops under max_utilization (which otherwise
        // ignores kv_oom_policy).
        const bool max_util = (m_cfg.kv_scheduler_policy == "max_utilization");
        const uint64_t pool_half = m_cfg.kv_pool_bytes / 2;
        const bool inherently_oversized = (pool_half > 0 && kv_side >= pool_half);
        bool do_fallback = inherently_oversized;
        if (inherently_oversized) {
          dlog(sfmt("[%10.3fms] KV_OOM_OVERSIZED  req=%3d  kv_side_bytes=%llu  pool_half_bytes=%llu  -> fallback_gpu",
                    now_ms, req->id,
                    (unsigned long long)kv_side,
                    (unsigned long long)pool_half));
        } else if (!max_util) {
          if (m_cfg.kv_oom_policy == "fallback_gpu") {
            do_fallback = true;
          } else if (m_cfg.kv_oom_policy == "hold_then_fallback") {
            double waited = now_ms - req->kv_oom_first_ms;
            do_fallback = (waited >= m_cfg.kv_oom_hold_limit_ms);
          }
        }

        if (do_fallback) {
          auto gpu_est = m_table->estimate(
              m_cfg.gpu_route_name,
              req->context_tokens, req->generated_tokens, 1);
          if (gpu_est && !gpu_est->uses_pim) {
            const CostEstimate& gpu_ref = *gpu_est;
            dlog(sfmt("[%10.3fms] KV_OOM_FALLBACK  req=%3d  waited_ms=%.1f  threshold_ms=%.1f  -> route=%s",
                      now_ms, req->id,
                      now_ms - req->kv_oom_first_ms,
                      m_cfg.kv_oom_hold_limit_ms,
                      m_cfg.gpu_route_name.c_str()));
            req->admission_last_reject_reason = "kv_oom_fallback_gpu";
            auto pred = predict_finish_detail(
                m_cfg.gpu_route_name,
                req->context_tokens, req->generated_tokens,
                now_ms, gpu_free_ms, pim_free_ms);
            if (pred) {
              req->scheduler_predicted_start_ms        = pred->start_ms;
              req->scheduler_predicted_finish_ms       = pred->finish_ms;
              req->scheduler_predicted_e2e_ms          = pred->finish_ms - req->arrival_ms;
              req->scheduler_predicted_prefill_ms      = pred->prefill_ms;
              req->scheduler_predicted_decode_total_ms = pred->decode_total_ms;
            }
            set_state(wait_idx, ReqState::WaitingPrefill);
            req->route              = m_cfg.gpu_route_name;
            req->svc                = gpu_ref;
            req->remaining_decode   = gpu_ref.decode_tokens;
            req->ready_ms           = now_ms;
            req->admitted_ms        = now_ms;
            m_admitted_count++;
            log_queue_event(now_ms, "admit_route", req->id,
                            "kv_oom_fallback_gpu", now_ms - req->arrival_ms);
            continue;
          }
        }

        // Hold and retry when memory is freed.
        dlog(sfmt("[%10.3fms] KV_OOM_HOLD  req=%3d  policy=%s  kv_side_bytes=%llu  free_key_bytes=%llu  waited_ms=%.1f",
                  now_ms, req->id, m_cfg.kv_oom_policy.c_str(),
                  (unsigned long long)kv_side,
                  (unsigned long long)m_alloc->free_bytes(KvRegion::Key),
                  req->kv_oom_first_ms >= 0 ? now_ms - req->kv_oom_first_ms : 0.0));
        req->admission_next_retry_ms    = now_ms + m_cfg.admission_retry_interval_ms;
        req->admission_last_reject_reason = "kv_oom";
        log_queue_event(now_ms, "hold", req->id, "kv_oom",
                        req->kv_oom_first_ms >= 0 ? now_ms - req->kv_oom_first_ms : 0.0);
        continue;
      }
      // Reserve KV memory.
      m_alloc->allocate(KvRegion::Key,   req->id, kv_side);
      m_alloc->allocate(KvRegion::Value, req->id, kv_side);
      req->kv_bytes_reserved = kv_side * 2;
      // Under max_utilization, `kv_side` only covers the prompt — prefill and
      // each decode step will grow the allocation via allocate_append.
      dlog(sfmt("[%10.3fms] KV_ALLOC  req=%3d  region=Key    bytes=%llu  free_after=%llu",
                now_ms, req->id, (unsigned long long)kv_side,
                (unsigned long long)m_alloc->free_bytes(KvRegion::Key)));
      dlog(sfmt("[%10.3fms] KV_ALLOC  req=%3d  region=Value  bytes=%llu  free_after=%llu",
                now_ms, req->id, (unsigned long long)kv_side,
                (unsigned long long)m_alloc->free_bytes(KvRegion::Value)));
    }

    // Admit the request.
    dlog(sfmt("[%10.3fms] ADMIT     req=%3d  route=%-20s  ctx=%5d  gen=%4d  attempts=%d%s",
              now_ms, req->id, route.c_str(),
              req->context_tokens, req->generated_tokens,
              req->admission_attempts,
              req->is_lost ? "  [SLO_MISS]" : ""));
    auto pred = predict_finish_detail(route, req->context_tokens,
                                      req->generated_tokens,
                                      now_ms, gpu_free_ms, pim_free_ms);
    if (pred) {
      req->scheduler_predicted_start_ms        = pred->start_ms;
      req->scheduler_predicted_finish_ms       = pred->finish_ms;
      req->scheduler_predicted_e2e_ms          = pred->finish_ms - req->arrival_ms;
      req->scheduler_predicted_prefill_ms      = pred->prefill_ms;
      req->scheduler_predicted_decode_total_ms = pred->decode_total_ms;
    }
    set_state(wait_idx, ReqState::WaitingPrefill);
    req->route              = route;
    req->svc                = est;
    req->remaining_decode   = est.decode_tokens;
    req->ready_ms           = now_ms;
    req->admitted_ms        = now_ms;
    req->admission_last_reject_reason.clear();
    m_admitted_count++;
    log_queue_event(now_ms, "admit_route", req->id, route,
                    now_ms - req->arrival_ms);
  }
}

// ─── complete_task() ─────────────────────────────────────────────────────────

void Scheduler::complete_task(const ActiveTask& task, double finish_ms) {
  bool any_done = false;

  if (task.phase == "prefill") {
    m_consecutive_decode = 0;
    for (int idx : task.request_indices) {
      auto& req = m_requests[idx];
      req.prefill_end_ms  = finish_ms;
      req.first_token_ms  = finish_ms;
      req.ready_ms        = finish_ms;
      if (req.remaining_decode > 0) {
        set_state(idx, ReqState::WaitingDecode);
        // max_utilization: grow KV to cover the first decode step's read.
        // No-op under guaranteed_no_evict (full reservation already in place).
        grow_kv_or_preempt(idx, finish_ms);
      } else {
        set_state(idx, ReqState::Done);
        req.completion_ms = finish_ms;
        m_active_count--;
        any_done = true;
      }
    }
  } else {
    // decode
    m_consecutive_decode++;
    for (int idx : task.request_indices) {
      auto& req = m_requests[idx];
      double last_tok = req.completion_ms > 0.0 ? req.completion_ms
                      : (req.first_token_ms > 0.0 ? req.first_token_ms
                                                   : req.prefill_end_ms);
      if (last_tok > 0.0) req.tbt_ms.push_back(finish_ms - last_tok);

      // A batch-mate processed earlier in this loop may have evicted this
      // request via grow_kv_or_preempt → preempt_lru. If so, skip completion:
      // the request is already back in WaitingAdmission with its state clean.
      if (req.state == ReqState::WaitingAdmission) continue;

      req.remaining_decode--;
      req.ready_ms = finish_ms;

      if (req.remaining_decode <= 0) {
        set_state(idx, ReqState::Done);
        req.completion_ms = finish_ms;
        m_active_count--;
        dlog(sfmt("[%10.3fms] DONE      req=%3d  e2e_ms=%.3f  ttft_ms=%.3f",
                  finish_ms, req.id,
                  finish_ms - req.arrival_ms,
                  req.first_token_ms >= 0 ? req.first_token_ms - req.arrival_ms : -1.0));
        // Free KV memory and wake any kv_oom-held requests.
        if (req.kv_bytes_reserved > 0) {
          m_alloc->free(KvRegion::Key,   req.id);
          m_alloc->free(KvRegion::Value, req.id);
          dlog(sfmt("[%10.3fms] KV_FREE   req=%3d  region=Key    free_after=%llu",
                    finish_ms, req.id,
                    (unsigned long long)m_alloc->free_bytes(KvRegion::Key)));
          dlog(sfmt("[%10.3fms] KV_FREE   req=%3d  region=Value  free_after=%llu",
                    finish_ms, req.id,
                    (unsigned long long)m_alloc->free_bytes(KvRegion::Value)));
          req.kv_bytes_reserved = 0;
          any_done = true;
        }
      } else {
        set_state(idx, ReqState::WaitingDecode);
        // max_utilization: grow KV for the next decode step's read.
        // If growth forces preemption of THIS request, state is reset to
        // WaitingAdmission and kv_bytes_reserved is zeroed by preempt_lru.
        grow_kv_or_preempt(idx, finish_ms);
      }
    }
  }

  if (any_done) {
    wake_kv_held(finish_ms);
    // Slot(s) just freed — pull more arrivals into the active set if any are
    // ready. Cheap no-op when arrival queue is empty or cap is unlimited.
    enqueue_arrivals(finish_ms);
  }
}

// ─── wake_kv_held() ──────────────────────────────────────────────────────────

void Scheduler::wake_kv_held(double now_ms) {
  int woke = 0;
  // Iterate WaitingAdmission index instead of all of m_requests.
  for (int idx : m_waiting_admission) {
    auto& r = m_requests[idx];
    if (r.admission_last_reject_reason == "kv_oom" ||
        r.admission_last_reject_reason == "kv_evicted") {
      r.admission_next_retry_ms = now_ms;
      woke++;
    }
  }
  if (woke > 0) {
    dlog(sfmt("[%10.3fms] KV_WAKE   woke=%d held request(s) reset for retry",
              now_ms, woke));
  }
}

// ─── preempt_lru() ───────────────────────────────────────────────────────────
//
// Used by the max_utilization policy when a running request's KV growth
// can't be satisfied. Evicts the most recently admitted PIM request (largest
// admitted_ms) other than excluded_index, freeing its pages and returning it
// to WaitingAdmission. The victim will re-run prefill on its next admission —
// equivalent to vLLM §4.5's "recomputation" recovery path.

bool Scheduler::preempt_lru(int excluded_index, double now_ms) {
  int best = -1;
  double best_admit = -1.0;
  // In-flight victims live in WaitingPrefill ∪ WaitingDecode. Iterate those
  // indices instead of scanning all of m_requests.
  auto consider = [&](int i) {
    if (i == excluded_index) return;
    const auto& r = m_requests[i];
    if (r.kv_bytes_reserved == 0) return;
    if (r.admitted_ms > best_admit) {
      best_admit = r.admitted_ms;
      best       = i;
    }
  };
  for (int i : m_waiting_prefill) consider(i);
  for (int i : m_waiting_decode)  consider(i);
  if (best < 0) return false;

  auto& v = m_requests[best];
  const uint64_t freed = v.kv_bytes_reserved;
  // Capture completed-decodes before we clear remaining_decode — on re-admit
  // the victim redoes prefill + this many decode tokens.
  const int redone = std::max(0, v.svc.decode_tokens - v.remaining_decode);
  m_alloc->free(KvRegion::Key,   v.id);
  m_alloc->free(KvRegion::Value, v.id);
  dlog(sfmt("[%10.3fms] PREEMPT   req=%3d  freed_bytes=%llu  admitted_ms=%.3f  by_req=%d  redo_decodes=%d",
            now_ms, v.id, (unsigned long long)freed,
            v.admitted_ms,
            excluded_index >= 0 ? m_requests[excluded_index].id : -1,
            redone));
  v.kv_bytes_reserved            = 0;
  set_state(best, ReqState::WaitingAdmission);   // returns to admission queue
  v.route.clear();
  v.remaining_decode             = 0;
  v.admission_last_reject_reason = "kv_evicted";
  v.admission_next_retry_ms      = now_ms;
  v.admitted_ms                  = -1.0;
  v.preempt_count++;
  v.tail_trim_events             = 0;    // fresh admission cycle
  m_preempt_count++;
  m_full_evict_count++;
  m_tokens_recomputed += static_cast<uint64_t>(redone);
  return true;
}

// ─── preempt_tail_trim() ─────────────────────────────────────────────────────
//
// Free just the most-recent pages of the LIFO decode victim — enough to cover
// the growing request's needed delta, rounded up to whole decode tokens.
// Victim stays in WaitingDecode; remaining_decode is bumped so it re-runs
// the trimmed tail on its next decode step (partial recomputation).
// Returns false if no in-flight decode victim with completed_decodes > 0
// exists — caller should fall back to preempt_lru (full evict).

bool Scheduler::preempt_tail_trim(int excluded_index,
                                   uint64_t bytes_needed_per_side,
                                   double now_ms) {
  const int trim_cap = m_cfg.kv_tail_trim_max_per_request;
  int best = -1;
  double best_admit = -1.0;
  // Tail-trim victims must be in WaitingDecode (they've completed at least one
  // decode step). Iterate the index instead of scanning all of m_requests.
  for (int i : m_waiting_decode) {
    if (i == excluded_index) continue;
    const auto& r = m_requests[i];
    if (r.kv_bytes_reserved == 0) continue;
    const int completed = std::max(0, r.svc.decode_tokens - r.remaining_decode);
    if (completed <= 0) continue;
    // Livelock guard: once a request has absorbed too many tail-trims since
    // its current admission, stop re-trimming it and let the caller escalate
    // to full evict. trim_cap == 0 disables the guard (legacy behavior).
    if (trim_cap > 0 && r.tail_trim_events >= trim_cap) continue;
    if (r.admitted_ms > best_admit) {
      best_admit = r.admitted_ms;
      best       = i;
    }
  }
  if (best < 0) return false;

  auto& v = m_requests[best];
  const uint64_t per_tok    = kv_bytes_per_token_per_side();
  if (per_tok == 0) return false;
  const int      completed  = std::max(0, v.svc.decode_tokens - v.remaining_decode);
  const uint64_t need_tokens = (bytes_needed_per_side + per_tok - 1) / per_tok;
  const uint64_t mult = static_cast<uint64_t>(std::max(1, m_cfg.kv_tail_trim_multiplier));
  int trim = static_cast<int>(std::min<uint64_t>(
      need_tokens * mult, static_cast<uint64_t>(completed)));
  if (trim <= 0) return false;

  const uint64_t bytes_per_side = static_cast<uint64_t>(trim) * per_tok;
  const uint64_t freed_k = m_alloc->free_tail(KvRegion::Key,   v.id, bytes_per_side);
  const uint64_t freed_v = m_alloc->free_tail(KvRegion::Value, v.id, bytes_per_side);
  const uint64_t freed   = freed_k + freed_v;

  v.kv_bytes_reserved               -= std::min<uint64_t>(v.kv_bytes_reserved, freed);
  v.remaining_decode                += trim;
  v.preempt_count                   += 1;
  v.tail_trim_tokens                += static_cast<uint64_t>(trim);
  v.tail_trim_events                += 1;
  v.admission_last_reject_reason    = "kv_tail_trimmed";
  m_tokens_recomputed               += static_cast<uint64_t>(trim);
  m_preempt_count                   += 1;
  m_tail_trim_count                 += 1;
  dlog(sfmt("[%10.3fms] TAIL_TRIM req=%3d  tokens=%d  freed_bytes=%llu  by_req=%d",
            now_ms, v.id, trim, (unsigned long long)freed,
            excluded_index >= 0 ? m_requests[excluded_index].id : -1));
  return true;
}

// ─── grow_kv_or_preempt() ────────────────────────────────────────────────────
//
// Called after every prefill/decode step that leaves more work to do for the
// request. Under max_utilization, grows the KV allocation to cover the next
// step's read. Growth is done in chunks of `kv_decode_chunk_tokens` tokens
// so admission, growth, and preemption decisions amortize across many decode
// steps. If the pool lacks free pages, first tries tail-trimming a LIFO
// decode victim (M2); failing that, falls back to a full evict (preempt_lru).
// Self-preempts if no victim exists.

bool Scheduler::grow_kv_or_preempt(int req_index, double now_ms) {
  if (m_cfg.kv_scheduler_policy != "max_utilization") return true;

  auto& req = m_requests[req_index];
  if (!req.svc.uses_pim) return true;         // GPU route: no KV reservation
  if (req.remaining_decode <= 0) return true; // no further step needs growth

  const int      prefill_token = req.generated_tokens > 0 ? 1 : 0;
  const int      completed     = std::max(0, req.svc.decode_tokens - req.remaining_decode);
  const int      next_ctx      = req.context_tokens + prefill_token + completed;
  const uint64_t strict_need   = kv_bytes_for_ctx(next_ctx);
  const uint64_t have_per_side = req.kv_bytes_reserved / 2;
  if (strict_need <= have_per_side) return true;   // current chunk still covers us

  // Grow by whole chunks of `chunk` tokens. Cap at the worst-case final ctx.
  const int      max_ctx   = req.context_tokens + std::max(0, req.generated_tokens - 1);
  const uint64_t max_bytes = kv_bytes_for_ctx(max_ctx);
  const int      chunk     = std::max(1, m_cfg.kv_decode_chunk_tokens);
  const uint64_t per_tok   = kv_bytes_per_token_per_side();
  const uint64_t chunk_bytes = static_cast<uint64_t>(chunk) * per_tok;

  const uint64_t shortfall  = strict_need - have_per_side;
  const uint64_t chunks_needed = (shortfall + chunk_bytes - 1) / chunk_bytes;
  uint64_t delta = chunks_needed * chunk_bytes;
  if (have_per_side + delta > max_bytes) delta = max_bytes - have_per_side;
  if (delta == 0) return true;

  // Free pages (preempt) until both regions can fit `delta`. Prefer tail-trim
  // when the policy requests it; fall back to full evict. If no victim at
  // all, self-evict.
  const bool tail_mode = (m_cfg.kv_evict_granularity == "tail");
  while (!m_alloc->can_fit(KvRegion::Key, delta) ||
         !m_alloc->can_fit(KvRegion::Value, delta)) {
    bool freed = false;
    if (tail_mode) {
      freed = preempt_tail_trim(req_index, delta, now_ms);
    }
    if (!freed) {
      freed = preempt_lru(req_index, now_ms);
    }
    if (!freed) {
      // No eligible victim other than this request — self-evict.
      preempt_lru(-1, now_ms);
      return false;
    }
  }

  const bool ok_k = m_alloc->allocate_append(KvRegion::Key,   req.id, delta);
  const bool ok_v = m_alloc->allocate_append(KvRegion::Value, req.id, delta);
  if (!ok_k || !ok_v) {
    dlog(sfmt("[%10.3fms] KV_GROW_FAIL  req=%3d  delta_bytes=%llu",
              now_ms, req.id, (unsigned long long)delta));
    preempt_lru(-1, now_ms);
    return false;
  }
  req.kv_bytes_reserved += 2 * delta;
  dlog(sfmt("[%10.3fms] KV_GROW   req=%3d  delta_bytes=%llu  new_reserved=%llu  next_ctx=%d  chunk=%d",
            now_ms, req.id,
            (unsigned long long)delta,
            (unsigned long long)req.kv_bytes_reserved,
            next_ctx, chunk));
  return true;
}

// ─── pick_task() ─────────────────────────────────────────────────────────────

std::optional<ActiveTask> Scheduler::pick_task(double now_ms,
                                                double gpu_free_ms,
                                                double pim_free_ms) const {
  bool has_prefill = false;
  for (const auto& r : m_requests) {
    if (r.state == ReqState::WaitingPrefill && r.ready_ms <= now_ms) {
      has_prefill = true;
      break;
    }
  }

  // --- Prefill priority ---
  // Always pick prefill first if prompt_priority is set, OR if we have run
  // too many consecutive decode batches.
  if (has_prefill &&
      (m_cfg.prompt_priority ||
       m_consecutive_decode >= m_cfg.max_consecutive_decode_batches)) {
    return pick_prefill(now_ms);
  }

  // --- Try decode batch ---
  auto decode = pick_decode_batch(now_ms);
  if (decode) return decode;

  // --- Fall back to prefill if decode had nothing ready ---
  if (has_prefill) return pick_prefill(now_ms);

  return std::nullopt;
}

// ─── pick_prefill() ──────────────────────────────────────────────────────────

std::optional<ActiveTask> Scheduler::pick_prefill(double now_ms) const {
  int best_idx = -1;
  // Iterate WaitingPrefill index instead of all of m_requests.
  for (int i : m_waiting_prefill) {
    const auto& r = m_requests[i];
    if (r.ready_ms > now_ms) continue;
    if (best_idx < 0) { best_idx = i; continue; }
    const auto& cur = m_requests[best_idx];
    if (r.ready_ms < cur.ready_ms ||
        (r.ready_ms == cur.ready_ms && r.id < cur.id)) {
      best_idx = i;
    }
  }
  if (best_idx < 0) return std::nullopt;
  const RuntimeRequest* best = &m_requests[best_idx];
  int idx = best_idx;
  ActiveTask t;
  t.phase            = "prefill";
  t.route            = best->route;
  t.request_indices  = {idx};
  t.context_tokens   = best->context_tokens;
  t.batch_size       = 1;
  t.gpu_ms           = best->svc.prefill_gpu_ms;
  t.pim_ms           = best->svc.prefill_pim_ms;
  t.e2e_ms           = best->svc.prefill_e2e_ms;
  t.start_ms         = now_ms;
  return t;
}

// ─── pick_decode_batch() ─────────────────────────────────────────────────────

std::optional<ActiveTask> Scheduler::pick_decode_batch(double now_ms) const {
  // Group WaitingDecode requests by route.
  // For each route, form a cohort of requests whose ready_ms <= cutoff,
  // capped at max_decode_batch_size.

  struct Cohort {
    std::string          route;
    std::vector<int>     indices;
    double               cutoff_ms;  // all requests in cohort are ready by this time
    double               earliest_start_ms;
  };

  std::vector<Cohort> cohorts;
  for (const std::string& route : {m_cfg.pim_route_name, m_cfg.gpu_route_name}) {
    // Collect and sort decode-ready requests for this route.
    // Iterate WaitingDecode index — bounded by active set, not m_requests.size().
    std::vector<std::pair<double, int>> ready_reqs;  // (ready_ms, index)
    for (int i : m_waiting_decode) {
      const auto& r = m_requests[i];
      if (r.route != route) continue;
      if (r.ready_ms > now_ms) continue;
      ready_reqs.push_back({r.ready_ms, i});
    }
    if (ready_reqs.empty()) continue;

    std::sort(ready_reqs.begin(), ready_reqs.end());

    // Continuous batching: include every request whose ready_ms <= now_ms
    // (all entries in ready_reqs already satisfy this). FCFS order by ready_ms,
    // capped at max_decode_batch_size.
    std::vector<int> cohort_indices;
    for (auto& [rms, idx] : ready_reqs) {
      if (static_cast<int>(cohort_indices.size()) >= m_cfg.max_decode_batch_size) break;
      cohort_indices.push_back(idx);
    }
    double cutoff = ready_reqs[0].first;

    // Estimate batch decode times (use max context_tokens as the representative).
    int max_ctx = 0;
    for (int idx : cohort_indices) max_ctx = std::max(max_ctx, m_requests[idx].context_tokens);

    auto est = m_table->estimate(route, max_ctx, 2, static_cast<int>(cohort_indices.size()));
    if (!est) continue;

    double start = std::max({now_ms, cutoff});  // earliest start for this cohort
    cohorts.push_back({route, cohort_indices, cutoff, start});
  }

  if (cohorts.empty()) return std::nullopt;

  // Pick the cohort whose earliest_start_ms is smallest (least waiting).
  auto& best = *std::min_element(cohorts.begin(), cohorts.end(),
      [](const Cohort& a, const Cohort& b) {
        return a.earliest_start_ms < b.earliest_start_ms;
      });

  int max_ctx = 0;
  for (int idx : best.indices) max_ctx = std::max(max_ctx, m_requests[idx].context_tokens);
  int bs = static_cast<int>(best.indices.size());
  auto est = m_table->estimate(best.route, max_ctx, 2, bs);
  if (!est) return std::nullopt;

  ActiveTask t;
  t.phase           = "decode";
  t.route           = best.route;
  t.request_indices = best.indices;
  t.context_tokens  = max_ctx;
  t.batch_size      = bs;
  t.gpu_ms          = est->decode_gpu_ms;
  t.pim_ms          = est->decode_pim_ms;
  t.e2e_ms          = est->decode_e2e_ms;
  t.start_ms        = now_ms;
  return t;
}

// ─── find_req() ──────────────────────────────────────────────────────────────

RuntimeRequest* Scheduler::find_req(int id) {
  for (auto& r : m_requests) {
    if (r.id == id) return &r;
  }
  return nullptr;
}

std::optional<double> Scheduler::next_wakeup_ms() const {
  std::optional<double> next_ms;

  auto consider = [&](double value) {
    if (!next_ms.has_value() || value < *next_ms) {
      next_ms = value;
    }
  };

  if (m_arrival_idx < m_arrivals.size()) {
    consider(m_arrivals[m_arrival_idx].arrival_ms);
  }

  // Iterate per-state indices instead of all of m_requests.
  for (int i : m_waiting_admission) consider(m_requests[i].admission_next_retry_ms);
  for (int i : m_waiting_prefill)   consider(m_requests[i].ready_ms);
  for (int i : m_waiting_decode)    consider(m_requests[i].ready_ms);

  return next_ms;
}

}  // namespace Ramulator::ServingOnline
