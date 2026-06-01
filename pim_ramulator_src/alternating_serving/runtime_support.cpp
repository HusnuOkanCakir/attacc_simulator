#include <algorithm>
#include <cmath>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include "base/exception.h"
#include "frontend/impl/alternating_serving/inputs.h"
#include "frontend/impl/alternating_serving/runtime.h"

namespace Ramulator::AlternatingServing {

namespace fs = std::filesystem;

AlternatingServingRuntime::AlternatingServingRuntime(Logger_t logger) : m_logger(std::move(logger)) {}

void AlternatingServingRuntime::initialize(const FrontendConfig& config) {
  m_config = config;
  validate_static_config(m_config);
  m_source_requests = load_requests_csv(m_config);
  m_cost_points = load_cost_tables(m_config);
  m_templates = load_templates(m_config);
  m_stats.total_requests = static_cast<int>(m_source_requests.size());
  open_debug_log_if_needed();
}

void AlternatingServingRuntime::connect_memory_system(IMemorySystem* memory_system) {
  m_memory_system = memory_system;
  finalize_post_connect_setup();
}

void AlternatingServingRuntime::finalize_post_connect_setup() {
  if (m_memory_system == nullptr) {
    throw ConfigurationError("AlternatingServingFrontend: memory system is not connected");
  }
  if (m_memory_system->get_clock_ratio() != 1) {
    throw ConfigurationError("AlternatingServingFrontend: MemorySystem.clock_ratio must be 1");
  }

  m_tck_ns = m_memory_system->get_tCK();
  if (!(m_tck_ns > 0.0f)) {
    throw ConfigurationError("AlternatingServingFrontend: memory system returned invalid tCK {}", m_tck_ns);
  }

  m_cycles_per_ms = 1.0e6 / static_cast<double>(m_tck_ns);
  if (!(m_cycles_per_ms > 0.0)) {
    throw ConfigurationError("AlternatingServingFrontend: failed to derive cycles_per_ms from tCK {}", m_tck_ns);
  }

  for (auto& req : m_source_requests) {
    req.arrival_clk = ms_to_cycles(req.arrival_ms);
  }
  m_post_connected = true;
}

size_t AlternatingServingRuntime::gpu_cost_point_count() const {
  auto it = m_cost_points.find(m_config.gpu_route_name);
  return (it == m_cost_points.end()) ? 0 : it->second.size();
}

size_t AlternatingServingRuntime::hybrid_cost_point_count() const {
  auto it = m_cost_points.find(m_config.hybrid_route_name);
  return (it == m_cost_points.end()) ? 0 : it->second.size();
}

Clk_t AlternatingServingRuntime::ms_to_cycles(double ms) const {
  if (!(ms > 0.0)) {
    return 0;
  }
  const double raw = ms * m_cycles_per_ms;
  const auto cyc = static_cast<Clk_t>(std::ceil(raw - 1e-12));
  return std::max<Clk_t>(1, cyc);
}

double AlternatingServingRuntime::cycles_to_ms(Clk_t clk) const {
  if (clk <= 0) {
    return 0.0;
  }
  return static_cast<double>(clk) * static_cast<double>(m_tck_ns) / 1.0e6;
}

std::optional<ServiceEstimate> AlternatingServingRuntime::estimate_request_for_route(
    const std::string& route,
    int context_tokens,
    int generated_tokens,
    int bs) const {
  return estimate_request(m_cost_points, m_config, m_cycles_per_ms, route, context_tokens, generated_tokens, bs);
}

bool AlternatingServingRuntime::has_template_for_lin(const std::string& route, int lin) const {
  auto route_it = m_templates.find(route);
  if (route_it == m_templates.end()) {
    return false;
  }
  return route_it->second.find(lin) != route_it->second.end();
}

const TemplateTrace* AlternatingServingRuntime::get_template_for_lin(const std::string& route, int lin) const {
  auto route_it = m_templates.find(route);
  if (route_it == m_templates.end()) {
    return nullptr;
  }
  auto trace_it = route_it->second.find(lin);
  return (trace_it == route_it->second.end()) ? nullptr : &trace_it->second;
}

void AlternatingServingRuntime::open_debug_log_if_needed() {
  if (!m_config.debug_enabled) {
    return;
  }

  fs::path debug_path(m_config.debug_log_path);
  if (!debug_path.parent_path().empty()) {
    fs::create_directories(debug_path.parent_path());
  }
  m_debug_log.open(debug_path, std::ios::out | std::ios::trunc);
  if (!m_debug_log.is_open()) {
    throw ConfigurationError("AlternatingServingFrontend: cannot open debug_log_path {}", m_config.debug_log_path);
  }

  std::ostringstream oss;
  oss << "[dbg:init] requests=" << m_stats.total_requests
      << " gpu_cost_points=" << gpu_cost_point_count()
      << " hybrid_cost_points=" << hybrid_cost_point_count()
      << " debug_interval_clk=" << m_config.debug_interval_clk;
  debug_write_line(oss.str());
}

void AlternatingServingRuntime::debug_write_line(const std::string& line) {
  if (!m_config.debug_enabled || !m_debug_log.is_open()) {
    return;
  }
  m_debug_log << line << '\n';
  m_debug_log.flush();
}

void AlternatingServingRuntime::maybe_log_progress(const char* reason) {
  if (!m_config.debug_enabled || m_clk < m_next_debug_clk) {
    return;
  }

  const bool mem_pending = (m_memory_system != nullptr) && m_memory_system->is_pending();
  int active_req_id = -1;
  const char* active_phase = "-";
  Clk_t active_start = 0;
  Clk_t active_finish = 0;
  size_t pim_next = 0;
  size_t pim_total = 0;
  size_t pim_outstanding = 0;
  bool pim_issue_done = false;
  bool pim_completed = false;

  if (m_active_task.has_value()) {
    const auto& task = *m_active_task;
    active_req_id = m_requests[task.request_idx].request_id;
    active_phase = phase_name(task.phase);
    active_start = task.start_clk;
    active_finish = task.predicted_finish_clk;
    if (task.pim_job.has_value()) {
      const auto& job = *task.pim_job;
      pim_next = job.next_cmd_idx;
      pim_total = (job.trace == nullptr) ? 0 : job.trace->commands.size();
      pim_outstanding = job.outstanding_cmds;
      pim_issue_done = job.issue_done;
      pim_completed = job.completed;
    }
  }

  std::ostringstream oss;
  oss << "[dbg:" << reason << "] clk=" << m_clk
      << " arrival_idx=" << m_arrival_idx << "/" << m_source_requests.size()
      << " next_route_slot=" << m_next_route_slot
      << " active_req=" << active_req_id
      << " phase=" << active_phase
      << " start=" << active_start
      << " finish=" << active_finish
      << " mem_pending=" << mem_pending
      << " pim_next=" << pim_next << "/" << pim_total
      << " pim_outstanding=" << pim_outstanding
      << " pim_issue_done=" << pim_issue_done
      << " pim_completed=" << pim_completed
      << " completed_pim_events=" << m_completed_pim_events.size();
  debug_write_line(oss.str());

  m_next_debug_clk = m_clk + m_config.debug_interval_clk;
}

std::optional<Clk_t> AlternatingServingRuntime::next_frontend_event_clk() const {
  Clk_t next_clk = kInfClk;

  if (m_arrival_idx < m_source_requests.size()) {
    const Clk_t arrival_clk = m_source_requests[m_arrival_idx].arrival_clk;
    if (arrival_clk > m_clk) {
      next_clk = std::min(next_clk, arrival_clk);
    }
  }

  if (m_active_task.has_value()) {
    const auto& task = *m_active_task;
    if (task.start_clk > m_clk) {
      next_clk = std::min(next_clk, task.start_clk);
    }
    if (task.gpu_done_clk > m_clk) {
      next_clk = std::min(next_clk, task.gpu_done_clk);
    }
    if (task.link_done_clk > m_clk) {
      next_clk = std::min(next_clk, task.link_done_clk);
    }
    if (task.predicted_finish_clk > m_clk) {
      next_clk = std::min(next_clk, task.predicted_finish_clk);
    }
  }

  if (next_clk == kInfClk) {
    return std::nullopt;
  }
  return next_clk;
}

bool AlternatingServingRuntime::try_fast_forward() {
  if ((m_memory_system != nullptr && m_memory_system->is_pending()) || !m_completed_pim_events.empty()) {
    return false;
  }

  if (m_active_task.has_value() && m_active_task->pim_job.has_value()) {
    const auto& job = *m_active_task->pim_job;
    if (!job.completed && m_clk >= m_active_task->start_clk) {
      return false;
    }
  }

  const auto next_clk = next_frontend_event_clk();
  if (!next_clk.has_value() || *next_clk <= (m_clk + 1)) {
    return false;
  }

  std::ostringstream oss;
  oss << "[dbg:fast_forward] clk=" << m_clk << " -> " << *next_clk;
  debug_write_line(oss.str());
  m_clk = *next_clk;
  maybe_log_progress("fast_forward");
  return true;
}

void AlternatingServingRuntime::tick() {
  if (!m_post_connected) {
    throw ConfigurationError("AlternatingServingFrontend: connect_memory_system() was not completed");
  }

  process_pim_callbacks();
  maybe_complete_active_task();
  admit_arrivals_up_to_now();
  maybe_start_next_task();
  pump_active_pim_job();
  maybe_complete_active_task();
  maybe_log_progress("tick");
  if (try_fast_forward()) {
    return;
  }
  m_clk++;
}

bool AlternatingServingRuntime::is_finished() const {
  if (m_arrival_idx < m_source_requests.size()) {
    return false;
  }
  if (m_active_task.has_value()) {
    return false;
  }
  if (!m_completed_pim_events.empty()) {
    return false;
  }
  for (const auto& req : m_requests) {
    if (req.state == RequestState::WaitingPrefill || req.state == RequestState::RunningPrefill ||
        req.state == RequestState::WaitingDecode || req.state == RequestState::RunningDecode) {
      return false;
    }
  }
  return true;
}

void AlternatingServingRuntime::finalize() {
  write_requests_csv();
  if (m_debug_log.is_open()) {
    m_debug_log.flush();
    m_debug_log.close();
  }
}

}  // namespace Ramulator::AlternatingServing
