/**
 * serving_online/frontend.cpp
 *
 * ServingOnlineFrontend: clean IFrontEnd for pi0 PIM+GPU LLM serving.
 *
 * Architecture:
 *   - Scheduler admits arriving requests to routes (pim+gpu or gpu_only).
 *   - GPU work advances a wall-clock free-time estimate directly.
 *   - PIM work issues Request objects to the ramulator memory system, which
 *     drives the actual LPDDR5-PIM timing model.
 *   - The frontend ticks until all requests are Done or Dropped.
 *
 * File organisation:
 *   Runtime     (namespace Ramulator::ServingOnline) -- all simulation logic.
 *   ServingOnlineFrontend (namespace Ramulator) -- IFrontEnd wrapper, reads YAML params.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "base/exception.h"
#include "frontend/frontend.h"
#include "frontend/impl/serving_online/cost_table.h"
#include "frontend/impl/serving_online/csv_loader.h"
#include "frontend/impl/serving_online/pim_allocator.h"
#include "frontend/impl/serving_online/pim_commands.h"
#include "frontend/impl/serving_online/scheduler.h"
#include "translation/translation.h"
#include "translation/impl/serving_online_translation.h"

namespace {

template<typename... Args>
static std::string sfmt(const char* fmt_str, Args... args) {
  char buf[512];
  std::snprintf(buf, sizeof(buf), fmt_str, args...);
  return {buf};
}

// Decode a PIM VA into its constituent bit fields (mirrors pim_allocator.cpp constants).
struct VaFields {
  bool     is_pim_va;
  uint64_t region;      // 0=Key, 1=Value
  uint64_t request_id;
  uint64_t page_index;
  uint64_t page_offset;
};
static VaFields decode_va(uint64_t va) {
  VaFields f;
  f.is_pim_va   = (va >> 63) & 1;
  f.region      = (va >> 60) & 0x7;
  f.request_id  = (va >> 40) & 0xFFFFF;
  f.page_index  = (va >> 12) & 0xFFFFFFF;
  f.page_offset =  va        & 0xFFF;
  return f;
}

} // namespace

namespace Ramulator::ServingOnline {

// ─── Helpers ─────────────────────────────────────────────────────────────────

static std::string state_name(ReqState state) {
  switch (state) {
    case ReqState::WaitingAdmission: return "waiting_admission";
    case ReqState::WaitingPrefill:   return "waiting_prefill";
    case ReqState::WaitingDecode:    return "waiting_decode";
    case ReqState::Done:             return "done";
    case ReqState::Dropped:          return "dropped";
  }
  return "unknown";
}

static double mean_or_neg1(const std::vector<double>& v) {
  if (v.empty()) return -1.0;
  return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

// ─── Runtime ─────────────────────────────────────────────────────────────────
//
// Contains all simulation state. The outer IFrontEnd class delegates to this.

class Runtime {
 public:
  explicit Runtime(Logger_t logger) : m_logger(std::move(logger)) {}

  // ── Initialization ─────────────────────────────────────────────────────────

  void initialize(const ServingOnlineConfig& cfg) {
    m_cfg = cfg;

    // Load Azure inference CSV.
    m_arrivals = load_azure_csv(cfg.csv_path, cfg.arrival_limit, cfg.arrival_time_scale);
    m_total_requests = m_arrivals.size();

    // Load cost table(s). Both routes may share a file or have separate files.
    m_cost_table.load(cfg.cost_gpu_csv);
    if (!cfg.cost_pim_csv.empty() && cfg.cost_pim_csv != cfg.cost_gpu_csv) {
      m_cost_table.load(cfg.cost_pim_csv);
    }

    // Build KV allocator.
    PimAllocator::Config alloc_cfg;
    alloc_cfg.kv_pool_bytes   = cfg.kv_pool_bytes;
    alloc_cfg.page_size_bytes = cfg.page_size_bytes;
    alloc_cfg.num_channels    = cfg.num_channels;
    m_allocator = std::make_shared<PimAllocator>(alloc_cfg);

    // Build PIM command generator.
    PimCommandGen::Params cmd_params;
    cmd_params.num_layers  = cfg.num_layers;
    cmd_params.num_heads   = cfg.num_heads;
    cmd_params.d_head      = cfg.d_head;
    cmd_params.dtype_bytes = cfg.dtype_bytes;
    m_command_gen = std::make_unique<PimCommandGen>(cmd_params, m_allocator.get());

    // Initialize scheduler.
    m_scheduler.initialize(cfg, m_arrivals, m_cost_table, *m_allocator);

    // Open debug log if requested.
    if (!cfg.debug_log_path.empty()) {
      m_debug_log.open(cfg.debug_log_path);
      if (!m_debug_log.is_open()) {
        throw std::runtime_error("ServingOnline runtime: cannot open debug_log_path: "
                                 + cfg.debug_log_path);
      }
      m_debug_log << "# serving_online debug log\n"
                  << "# format: [wall_time_ms] EVENT  fields...\n"
                  << "# events: ARRIVE ADMIT KV_ALLOC KV_FREE KV_OOM_HOLD KV_OOM_FALLBACK "
                     "KV_WAKE TASK_START TRANSLATE TASK_DONE DONE\n"
                  << "#\n";
      // Forward scheduler events to the same log file.
      m_scheduler.set_log_fn([this](const std::string& line) {
        m_debug_log << line << '\n';
      });
    }
  }

  void attach_translation(ITranslation* translation) {
    m_translation = translation;
  }

  std::shared_ptr<PimAllocator> allocator_shared() const { return m_allocator; }

  // ── Memory system attachment ───────────────────────────────────────────────

  void connect_memory_system(IMemorySystem* memory_system) {
    m_memory_system = memory_system;
    if (m_memory_system == nullptr) {
      throw ConfigurationError("ServingOnline runtime: memory system is null");
    }
    if (m_memory_system->get_tCK() <= 0.0f) {
      throw ConfigurationError("ServingOnline runtime: invalid tCK from memory system");
    }
    m_cycles_per_ms = 1.0e6 / static_cast<double>(m_memory_system->get_tCK());
  }

  // ── Simulation tick ────────────────────────────────────────────────────────

  void tick() {
    if (is_finished()) return;

    process_callbacks();
    complete_active_task_if_ready();

    if (!m_active_task.has_value()) {
      auto next = m_scheduler.tick(now_ms(), m_gpu_free_ms, m_pim_free_ms);
      if (next) {
        start_task(std::move(*next));
      } else if (auto wake = m_scheduler.next_wakeup_ms(); wake.has_value()) {
        fast_forward_to_ms(*wake);
        return;
      }
    }

    if (m_active_task.has_value()) {
      if (m_active_task->phase == "prefill") {
        fast_forward_to_ms(m_active_task->start_ms + m_active_task->e2e_ms);
        complete_active_task_if_ready();
        return;
      }
      if (m_active_task->phase == "decode"
          && m_active_task->route != m_cfg.pim_route_name) {
        fast_forward_to_ms(m_active_task->start_ms + m_active_task->gpu_ms);
        complete_active_task_if_ready();
        return;
      }
    }

    pump_active_pim();
    process_callbacks();
    complete_active_task_if_ready();

    // Once all PIM reads for a PIM-decode task have been issued and drained,
    // the task is only gated on wall-clock gpu/pim time. Fast-forward instead
    // of burning millions of empty cycles.
    if (m_active_task.has_value()
        && m_active_task->phase == "decode"
        && m_active_task->route == m_cfg.pim_route_name
        && m_issue_done
        && m_pim_outstanding == 0) {
      const double target = m_active_task->start_ms
                          + std::max({m_active_task->gpu_ms,
                                      m_active_task->pim_ms,
                                      m_active_task->e2e_ms});
      fast_forward_to_ms(target);
      complete_active_task_if_ready();
      return;
    }

    if (!is_finished()) {
      ++m_clk;
    }
  }

  // ── Status ────────────────────────────────────────────────────────────────

  bool is_finished() const {
    return m_scheduler.all_done()
        && !m_active_task.has_value()
        && m_completed_callback_events == 0
        && m_pim_outstanding == 0;
  }

  // ── Output ────────────────────────────────────────────────────────────────

  void finalize() const {
    if (m_cfg.requests_out_csv.empty()) return;

    std::ofstream out(m_cfg.requests_out_csv);
    if (!out.is_open()) {
      throw std::runtime_error(
          "ServingOnline runtime: cannot open requests_out_csv: " + m_cfg.requests_out_csv);
    }

    out << "request_id,arrival_ms,context_tokens,generated_tokens,state,route,is_lost,"
           "admission_attempts,admission_last_reject_reason,dropped_reason,"
           "prefill_start_ms,prefill_end_ms,first_token_ms,completion_ms,"
           "ttft_ms,e2e_ms,tbt_mean_ms,preempt_count,tail_trim_tokens\n";

    for (const auto& req : m_scheduler.requests()) {
      const double ttft_ms =
          req.first_token_ms >= 0.0 ? (req.first_token_ms - req.arrival_ms) : -1.0;
      const double e2e_ms =
          req.completion_ms >= 0.0 ? (req.completion_ms - req.arrival_ms) : -1.0;

      out << req.id << ','
          << req.arrival_ms << ','
          << req.context_tokens << ','
          << req.generated_tokens << ','
          << state_name(req.state) << ','
          << req.route << ','
          << (req.is_lost ? 1 : 0) << ','
          << req.admission_attempts << ','
          << req.admission_last_reject_reason << ','
          << req.dropped_reason << ','
          << req.prefill_start_ms << ','
          << req.prefill_end_ms << ','
          << req.first_token_ms << ','
          << req.completion_ms << ','
          << ttft_ms << ','
          << e2e_ms << ','
          << mean_or_neg1(req.tbt_ms) << ','
          << req.preempt_count << ','
          << req.tail_trim_tokens << '\n';
    }
  }

  void print_stats(YAML::Emitter& emitter,
                   const std::string& ifce_name,
                   const std::string& impl_name) const {
    size_t completed = 0;
    size_t dropped   = 0;
    size_t pim_routes = 0;
    size_t gpu_routes = 0;
    for (const auto& req : m_scheduler.requests()) {
      if (req.state == ReqState::Done)    completed++;
      if (req.state == ReqState::Dropped) dropped++;
      if (req.route == m_cfg.pim_route_name) pim_routes++;
      if (req.route == m_cfg.gpu_route_name) gpu_routes++;
    }

    emitter << YAML::Key << ifce_name << YAML::Value << YAML::BeginMap;
    emitter << YAML::Key << "impl"              << YAML::Value << impl_name;
    emitter << YAML::Key << "total_requests"    << YAML::Value << m_total_requests;
    emitter << YAML::Key << "admitted_requests" << YAML::Value << m_scheduler.admitted_count();
    emitter << YAML::Key << "completed_requests"<< YAML::Value << completed;
    emitter << YAML::Key << "dropped_requests"  << YAML::Value << dropped;
    emitter << YAML::Key << "kv_oom_holds"      << YAML::Value << m_scheduler.kv_hold_count();
    emitter << YAML::Key << "preempt_count"     << YAML::Value << m_scheduler.preempt_count();
    emitter << YAML::Key << "tail_trim_count"   << YAML::Value << m_scheduler.tail_trim_count();
    emitter << YAML::Key << "full_evict_count"  << YAML::Value << m_scheduler.full_evict_count();
    emitter << YAML::Key << "tokens_recomputed" << YAML::Value << m_scheduler.tokens_recomputed();
    emitter << YAML::Key << "predictive_holds"  << YAML::Value << m_scheduler.predictive_holds();
    emitter << YAML::Key << "kv_scheduler_policy" << YAML::Value << m_cfg.kv_scheduler_policy;
    emitter << YAML::Key << "route_counts"      << YAML::Value << YAML::BeginMap;
    emitter << YAML::Key << m_cfg.gpu_route_name << YAML::Value << gpu_routes;
    emitter << YAML::Key << m_cfg.pim_route_name << YAML::Value << pim_routes;
    emitter << YAML::EndMap;
    emitter << YAML::Key << "frontend_cycles"   << YAML::Value << m_clk;
    emitter << YAML::Key << "gpu_free_ms"       << YAML::Value << m_gpu_free_ms;
    emitter << YAML::Key << "pim_free_ms"       << YAML::Value << m_pim_free_ms;
    emitter << YAML::Key << "requests_out_csv"  << YAML::Value << m_cfg.requests_out_csv;
    emitter << YAML::EndMap << YAML::Newline;
  }

  // ── Accessors ─────────────────────────────────────────────────────────────

  size_t total_requests()   const { return m_total_requests; }
  size_t cost_point_count() const { return m_cost_table.total_points(); }
  Clk_t  clk()              const { return m_clk; }

 private:
  // ── Time conversion ────────────────────────────────────────────────────────

  double now_ms() const {
    if (m_cycles_per_ms <= 0.0) return 0.0;
    return static_cast<double>(m_clk) / m_cycles_per_ms;
  }

  Clk_t clk_for_ms(double time_ms) const {
    if (m_cycles_per_ms <= 0.0 || time_ms <= 0.0) return 0;
    return static_cast<Clk_t>(std::ceil(time_ms * m_cycles_per_ms));
  }

  void fast_forward_to_ms(double target_ms) {
    const Clk_t target_clk = clk_for_ms(target_ms);
    if (target_clk > m_clk) {
      m_clk = target_clk;
    }
  }

  // Effective decode-time context depth: prompt + the first token produced by
  // prefill + completed decode tokens.
  int current_context_tokens(const RuntimeRequest& req) const {
    const int completed = std::max(0, req.svc.decode_tokens - req.remaining_decode);
    const int prefill_token = req.generated_tokens > 0 ? 1 : 0;
    return req.context_tokens + prefill_token + completed;
  }

  // ── Task lifecycle ─────────────────────────────────────────────────────────

  void start_task(ActiveTask task) {
    const double start_ms = now_ms();
    task.start_ms = start_ms;

    // Stamp prefill_start_ms for every request in this task.
    if (task.phase == "prefill") {
      for (const int idx : task.request_indices) {
        auto& req = m_scheduler.mutable_requests()[idx];
        if (req.prefill_start_ms < 0.0) {
          req.prefill_start_ms = start_ms;
        }
      }
    }

    // Advance the GPU and PIM free-time clocks.
    m_gpu_free_ms = std::max(m_gpu_free_ms, start_ms) + task.gpu_ms;
    m_pim_free_ms = std::max(m_pim_free_ms, start_ms) + task.pim_ms;

    // Reset PIM command state.
    m_pending_pim_requests.clear();
    m_next_pim_request          = 0;
    m_pim_outstanding           = 0;
    m_completed_callback_events = 0;
    m_issue_done                = true;

    // For PIM decode tasks: pre-generate all the LPDDR5 memory requests.
    if (task.phase == "decode" && task.route == m_cfg.pim_route_name) {
      m_issue_done = false;
      for (const int idx : task.request_indices) {
        const auto& req = m_scheduler.requests()[idx];
        auto on_complete = [this](Request&) { m_completed_callback_events++; };
        auto cmds = m_command_gen->decode_step(
            req.id, current_context_tokens(req), on_complete);
        m_pending_pim_requests.insert(m_pending_pim_requests.end(),
                                      std::make_move_iterator(cmds.begin()),
                                      std::make_move_iterator(cmds.end()));
      }
    }

    // Log task start (before move — use local `task`, not m_active_task).
    m_translate_sample_count = 0;
    if (m_debug_log.is_open()) {
      std::string reqs_str;
      for (int idx : task.request_indices) {
        if (!reqs_str.empty()) reqs_str += ',';
        reqs_str += std::to_string(m_scheduler.requests()[idx].id);
      }
      if (task.phase == "decode" && task.route == m_cfg.pim_route_name) {
        dlog(sfmt("[%10.3fms] TASK_START  phase=decode   route=%-20s  reqs=[%s]"
                  "  ctx=%5d  bs=%d  gpu_ms=%6.2f  pim_ms=%6.2f  e2e_ms=%6.2f  cmds=%zu",
                  start_ms, task.route.c_str(), reqs_str.c_str(),
                  task.context_tokens, task.batch_size,
                  task.gpu_ms, task.pim_ms, task.e2e_ms,
                  m_pending_pim_requests.size()));
      } else {
        dlog(sfmt("[%10.3fms] TASK_START  phase=%-7s  route=%-20s  reqs=[%s]"
                  "  ctx=%5d  bs=%d  gpu_ms=%6.2f  pim_ms=%6.2f  e2e_ms=%6.2f",
                  start_ms, task.phase.c_str(), task.route.c_str(), reqs_str.c_str(),
                  task.context_tokens, task.batch_size,
                  task.gpu_ms, task.pim_ms, task.e2e_ms));
      }
    }

    m_active_task = std::move(task);
  }

  // Send pending PIM requests to the memory system one per tick (back-pressure).
  void pump_active_pim() {
    if (!m_active_task.has_value()
        || m_active_task->phase != "decode"
        || m_active_task->route != m_cfg.pim_route_name
        || m_memory_system == nullptr) {
      return;
    }

    while (m_next_pim_request < m_pending_pim_requests.size()) {
      Request req = m_pending_pim_requests[m_next_pim_request];

      // Log first 4 VA→PA translations per task so the address path is visible.
      const uint64_t orig_va = static_cast<uint64_t>(req.addr);

      // VA-to-PA translation (bit 63 indicates a KV virtual address).
      if (m_translation != nullptr && !m_translation->translate(req)) {
        throw std::runtime_error("ServingOnline runtime: request translation failed");
      }

      if (m_debug_log.is_open() && m_translate_sample_count < 4) {
        const uint64_t pa = static_cast<uint64_t>(req.addr);
        const VaFields f  = decode_va(orig_va);
        dlog(sfmt("[%10.3fms] TRANSLATE  req=%3d  cmd=%4zu  "
                  "va=0x%016llx  [%s  req_id=%llu  page=%llu  off=%llu]  "
                  "pa=0x%016llx",
                  now_ms(), req.source_id,
                  m_next_pim_request,
                  (unsigned long long)orig_va,
                  f.region == 0 ? "Key" : "Val",
                  (unsigned long long)f.request_id,
                  (unsigned long long)f.page_index,
                  (unsigned long long)f.page_offset,
                  (unsigned long long)pa));
        m_translate_sample_count++;
        if (m_translate_sample_count == 4) {
          const size_t remaining = m_pending_pim_requests.size() - m_next_pim_request - 1;
          if (remaining > 0) {
            dlog(sfmt("[%10.3fms]   ... (%zu more cmds not logged)",
                      now_ms(), remaining));
          }
        }
      }

      if (!m_memory_system->send(req)) {
        break;  // memory system full; retry next tick
      }
      if (req.callback) {
        m_pim_outstanding++;
      }
      m_next_pim_request++;
    }
    m_issue_done = (m_next_pim_request == m_pending_pim_requests.size());
  }

  // Drain completed PIM callbacks.
  void process_callbacks() {
    while (m_completed_callback_events > 0 && m_pim_outstanding > 0) {
      m_completed_callback_events--;
      m_pim_outstanding--;
    }
  }

  // Check whether the active task is finished; if so, complete it.
  void complete_active_task_if_ready() {
    if (!m_active_task.has_value()) return;

    const double now = now_ms();

    if (m_active_task->phase == "prefill") {
      // Prefill: wall-clock latency gates completion.
      if (now < m_active_task->start_ms + m_active_task->e2e_ms) return;
    } else {
      // Decode: both GPU time and PIM commands must be done.
      const bool gpu_done = (now >= m_active_task->start_ms + m_active_task->gpu_ms);
      const bool pim_done = (m_active_task->route != m_cfg.pim_route_name)
                         || (m_issue_done && m_pim_outstanding == 0);
      if (!gpu_done || !pim_done) return;
    }

    if (m_debug_log.is_open()) {
      std::string reqs_str;
      for (int idx : m_active_task->request_indices) {
        if (!reqs_str.empty()) reqs_str += ',';
        reqs_str += std::to_string(m_scheduler.requests()[idx].id);
      }
      dlog(sfmt("[%10.3fms] TASK_DONE   phase=%-7s  route=%-20s  reqs=[%s]  duration_ms=%.3f",
                now, m_active_task->phase.c_str(), m_active_task->route.c_str(),
                reqs_str.c_str(), now - m_active_task->start_ms));
    }

    m_scheduler.complete_task(*m_active_task, now);
    m_active_task.reset();
    m_pending_pim_requests.clear();
    m_next_pim_request          = 0;
    m_issue_done                = true;
    m_pim_outstanding           = 0;
    m_completed_callback_events = 0;
  }

  // ── State ──────────────────────────────────────────────────────────────────

  Logger_t       m_logger;
  ServingOnlineConfig m_cfg;
  IMemorySystem* m_memory_system = nullptr;
  ITranslation*  m_translation   = nullptr;

  CostTable                       m_cost_table;
  std::vector<SourceRequest>      m_arrivals;
  std::shared_ptr<PimAllocator>   m_allocator;
  std::unique_ptr<PimCommandGen>  m_command_gen;
  Scheduler                       m_scheduler;

  size_t m_total_requests  = 0;
  double m_cycles_per_ms   = 0.0;
  Clk_t  m_clk             = 0;
  double m_gpu_free_ms     = 0.0;
  double m_pim_free_ms     = 0.0;

  std::optional<ActiveTask>  m_active_task;
  std::vector<Request>       m_pending_pim_requests;
  size_t m_next_pim_request          = 0;
  size_t m_pim_outstanding           = 0;
  size_t m_completed_callback_events = 0;
  bool   m_issue_done                = true;

  // Debug log (null stream when disabled).
  std::ofstream m_debug_log;
  int m_translate_sample_count = 0;   // resets per task; cap TRANSLATE lines at 4

  void dlog(const std::string& line) {
    if (m_debug_log.is_open()) m_debug_log << line << '\n';
  }
};

}  // namespace Ramulator::ServingOnline

// ─── IFrontEnd wrapper ────────────────────────────────────────────────────────

namespace Ramulator {

class ServingOnlineFrontend : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(
      IFrontEnd,
      ServingOnlineFrontend,
      "ServingOnlineFrontend",
      "Clean pi0 PIM+GPU LLM serving with runtime LPDDR5 command generation.");

 public:
  void init() override {
    ServingOnline::ServingOnlineConfig cfg;

    // Input files
    cfg.csv_path        = param<std::string>("requests_csv").required();
    cfg.requests_out_csv = param<std::string>("requests_out_csv")
                              .default_val("cluster_outputs/serving_online_requests.csv");
    cfg.cost_gpu_csv    = param<std::string>("cost_gpu_csv").required();
    cfg.cost_pim_csv    = param<std::string>("cost_hybrid_csv").required();

    // Route names
    cfg.gpu_route_name  = param<std::string>("gpu_route_name").default_val("gpu_only");
    cfg.pim_route_name  = param<std::string>("hybrid_route_name").default_val("pim+gpu");

    // Routing policy
    cfg.route_policy           = param<std::string>("route_policy").default_val("min_finish");
    cfg.energy_latency_guard_ms = param<float>("energy_latency_guard_ms").default_val(5.0f);

    // SLO thresholds
    cfg.slo_e2e_ms  = param<float>("slo_e2e_ms").default_val(-1.0f);
    cfg.slo_ttft_ms = param<float>("slo_ttft_ms").default_val(-1.0f);

    // Scheduling
    cfg.max_decode_batch_size          = param<int>("max_decode_batch_size").default_val(8);
    cfg.prompt_priority                = param<bool>("prompt_priority").default_val(true);
    cfg.max_consecutive_decode_batches = param<int>("max_consecutive_decode_batches").default_val(4);

    // Admission
    cfg.admission_max_wait_ms       = param<float>("admission_max_wait_ms").default_val(0.0f);
    cfg.admission_retry_interval_ms = param<float>("admission_retry_interval_ms").default_val(1.0f);
    cfg.kv_oom_policy               = param<std::string>("kv_oom_policy").default_val("hold");
    cfg.kv_oom_hold_limit_ms        = param<float>("kv_oom_hold_limit_ms").default_val(50.0f);
    cfg.kv_scheduler_policy         = param<std::string>("kv_scheduler_policy")
                                          .default_val("guaranteed_no_evict");

    // Fine-grained KV knobs (read only under kv_scheduler_policy=max_utilization).
    cfg.kv_decode_chunk_tokens         = param<int>("kv_decode_chunk_tokens").default_val(32);
    cfg.kv_evict_granularity           = param<std::string>("kv_evict_granularity")
                                            .default_val("tail");
    cfg.predictive_admission           = param<bool>("predictive_admission").default_val(false);
    cfg.predictive_pressure_threshold  = param<float>("predictive_pressure_threshold")
                                            .default_val(1.0f);
    cfg.predictive_hold_ms             = param<float>("predictive_hold_ms").default_val(50.0f);
    cfg.kv_tail_trim_max_per_request   = param<int>("kv_tail_trim_max_per_request").default_val(8);

    // Arrival
    cfg.arrival_time_scale = param<float>("arrival_time_scale").default_val(1.0f);
    cfg.arrival_limit      = param<int>("arrival_limit").default_val(-1);

    // Memory
    cfg.kv_pool_bytes   = param<uint64_t>("kv_pool_bytes").default_val(0);
    cfg.page_size_bytes = (param<Addr_t>("translation_pagesize_KB").default_val(4) << 10);
    cfg.num_channels    = param<int>("generator_channel_count").default_val(8);

    // Pi0 model
    cfg.num_layers  = param<int>("generator_num_layers").default_val(18);
    cfg.num_heads   = param<int>("generator_num_heads").default_val(8);
    cfg.d_head      = param<int>("generator_dhead").default_val(256);
    cfg.dtype_bytes = param<int>("generator_dtype_bytes").default_val(2);

    // Debug log (empty string = disabled)
    cfg.debug_log_path = param<std::string>("debug_log_path").default_val("");

    // Validate route policy.
    if (cfg.route_policy != "min_finish" &&
        cfg.route_policy != "latency_guarded_energy") {
      throw ConfigurationError(
          "ServingOnlineFrontend: route_policy must be 'min_finish' or 'latency_guarded_energy'");
    }
    // Validate kv_oom_policy.
    if (cfg.kv_oom_policy != "hold" && cfg.kv_oom_policy != "fallback_gpu" &&
        cfg.kv_oom_policy != "hold_then_fallback") {
      throw ConfigurationError(
          "ServingOnlineFrontend: kv_oom_policy must be 'hold', 'fallback_gpu', or 'hold_then_fallback'");
    }
    // Validate kv_scheduler_policy.
    if (cfg.kv_scheduler_policy != "guaranteed_no_evict" &&
        cfg.kv_scheduler_policy != "max_utilization") {
      throw ConfigurationError(
          "ServingOnlineFrontend: kv_scheduler_policy must be 'guaranteed_no_evict' or 'max_utilization'");
    }
    // Validate fine-grained KV knobs.
    if (cfg.kv_decode_chunk_tokens < 1) {
      throw ConfigurationError(
          "ServingOnlineFrontend: kv_decode_chunk_tokens must be >= 1");
    }
    if (cfg.kv_evict_granularity != "tail" && cfg.kv_evict_granularity != "full") {
      throw ConfigurationError(
          "ServingOnlineFrontend: kv_evict_granularity must be 'tail' or 'full'");
    }
    if (!(cfg.predictive_pressure_threshold > 0.0 &&
          cfg.predictive_pressure_threshold <= 4.0)) {
      throw ConfigurationError(
          "ServingOnlineFrontend: predictive_pressure_threshold must be in (0, 4]");
    }
    if (cfg.kv_tail_trim_max_per_request < 0) {
      throw ConfigurationError(
          "ServingOnlineFrontend: kv_tail_trim_max_per_request must be >= 0 "
          "(0 disables the livelock guard)");
    }
    if (cfg.kv_scheduler_policy == "max_utilization" &&
        cfg.kv_oom_policy != "hold") {
      // Not an error — surface as a warning so the user knows the kv_oom knob
      // is inert under max_utilization. The scheduler always holds FCFS when
      // even prefill bytes don't fit (per vLLM §4.5).
      std::fprintf(
          stderr,
          "[ServingOnlineFrontend] kv_scheduler_policy=max_utilization "
          "ignores kv_oom_policy='%s'. Admission holds FCFS; in-flight KV "
          "growth triggers LIFO preemption.\n",
          cfg.kv_oom_policy.c_str());
    }

    m_clock_ratio = param<uint>("clock_ratio").required();

    m_logger  = Logging::create_logger("ServingOnlineFrontend");
    m_runtime = std::make_unique<ServingOnline::Runtime>(m_logger);
    m_runtime->initialize(cfg);

    // Create and wire the VA-to-PA translation child.
    m_translation = create_child_ifce<ITranslation>();
    auto* control = dynamic_cast<IServingOnlineTranslationControl*>(m_translation);
    if (control == nullptr) {
      throw ConfigurationError(
          "ServingOnlineFrontend: Translation child must implement IServingOnlineTranslationControl "
          "(use Translation: impl: ServingOnlineTranslation)");
    }
    control->attach_pim_allocator(m_runtime->allocator_shared());
    m_runtime->attach_translation(m_translation);
  }

  void connect_memory_system(IMemorySystem* memory_system) override {
    IFrontEnd::connect_memory_system(memory_system);
    if (!m_runtime) {
      throw ConfigurationError("ServingOnlineFrontend: runtime not initialized");
    }
    m_runtime->connect_memory_system(memory_system);
    m_clk = m_runtime->clk();
  }

  void tick() override {
    if (!m_runtime) return;
    m_runtime->tick();
    m_clk = m_runtime->clk();
  }

  bool is_finished() override {
    return !m_runtime || m_runtime->is_finished();
  }

  void finalize() override {
    if (m_runtime) {
      m_runtime->finalize();
      m_clk = m_runtime->clk();
    }
    IFrontEnd::finalize();
  }

  void print_stats(YAML::Emitter& emitter) override {
    if (!m_runtime) return;
    m_runtime->print_stats(emitter, get_ifce_name(), get_name());
  }

 private:
  Logger_t m_logger;
  ITranslation* m_translation = nullptr;
  std::unique_ptr<ServingOnline::Runtime> m_runtime;
};

}  // namespace Ramulator
