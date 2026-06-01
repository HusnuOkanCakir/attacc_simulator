#include <algorithm>
#include <cmath>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>

#include "base/exception.h"
#include "frontend/impl/serving/runtime.h"

namespace Ramulator::Serving {

namespace fs = std::filesystem;

namespace {

std::string join_request_ids(const std::vector<int>& indices,
                             const std::vector<RuntimeRequest>& requests) {
  std::ostringstream oss;
  for (size_t i = 0; i < indices.size(); ++i) {
    if (i > 0) {
      oss << ';';
    }
    oss << requests[indices[i]].request_id;
  }
  return oss.str();
}

}  // namespace

void RealtimeServingRuntime::open_debug_log_if_needed() {
  if (!m_config.debug_enabled && !m_config.capture_debug_events) {
    return;
  }

  fs::path debug_path(m_config.debug_log_path);
  if (!debug_path.parent_path().empty()) {
    fs::create_directories(debug_path.parent_path());
  }
  m_debug_log.open(debug_path, std::ios::out | std::ios::trunc);
  if (!m_debug_log.is_open()) {
    throw ConfigurationError("RealtimeServingFrontend: cannot open debug_log_path {}", m_config.debug_log_path);
  }

  std::ostringstream oss;
  oss << "[dbg:init] requests=" << m_stats.total_requests
      << " gpu_cost_points=" << gpu_cost_point_count()
      << " hybrid_cost_points=" << hybrid_cost_point_count()
      << " debug_interval_clk=" << m_config.debug_interval_clk;
  debug_write_line(oss.str());
}

Clk_t RealtimeServingRuntime::ms_to_cycles(double ms) const {
  if (!(ms > 0.0)) {
    return 0;
  }
  const double raw = ms * m_cycles_per_ms;
  const auto cyc = static_cast<Clk_t>(std::ceil(raw - 1e-12));
  return std::max<Clk_t>(1, cyc);
}

double RealtimeServingRuntime::cycles_to_ms(Clk_t clk) const {
  if (clk <= 0) {
    return 0.0;
  }
  return static_cast<double>(clk) * static_cast<double>(m_tck_ns) / 1.0e6;
}

double RealtimeServingRuntime::quantize_export_ms(double ms) const {
  if (!std::isfinite(ms)) {
    return ms;
  }
  return cycles_to_ms(ms_to_cycles(ms));
}

std::optional<double> RealtimeServingRuntime::quantize_export_ms(
    const std::optional<double>& ms) const {
  if (!ms.has_value()) {
    return std::nullopt;
  }
  return quantize_export_ms(*ms);
}

void RealtimeServingRuntime::debug_write_line(const std::string& line) {
  if (!m_debug_log.is_open()) {
    return;
  }
  m_debug_log << line << '\n';
  m_debug_log.flush();
}

void RealtimeServingRuntime::record_debug_event(
    const std::string& event,
    const std::vector<std::pair<std::string, std::string>>& fields) {
  if (!m_config.capture_debug_events) {
    return;
  }
  std::ostringstream oss;
  oss << "[event:" << event << "]"
      << " sim_now_ms=" << cycles_to_ms(m_clk)
      << " gpu_free_ms=" << cycles_to_ms(m_gpu_busy_until)
      << " pim_free_ms=" << cycles_to_ms(m_pim_busy_until)
      << " waiting_admission_count=" << waiting_admission_request_indices().size();
  int waiting_prefill = 0;
  int waiting_decode = 0;
  for (const auto& req : m_requests) {
    if (req.state == RequestState::WaitingPrefill) {
      waiting_prefill++;
    } else if (req.state == RequestState::WaitingDecode && req.remaining_decode_tokens > 0) {
      waiting_decode++;
    }
  }
  oss << " waiting_prefill_count=" << waiting_prefill
      << " waiting_decode_count=" << waiting_decode
      << " arrival_index=" << m_arrival_idx;
  if (m_address_space) {
    oss << " allocator_regions=" << m_address_space->region_pressure_debug_string()
        << " allocator_hot_buckets=" << m_address_space->hot_bucket_pressure_debug_string(3);
  }
  for (const auto& [key, value] : fields) {
    oss << ' ' << key << '=' << value;
  }
  debug_write_line(oss.str());
}

void RealtimeServingRuntime::maybe_log_progress(const char* reason) {
  if (!m_config.debug_enabled || m_clk < m_next_debug_clk) {
    return;
  }

  const bool mem_pending = (m_memory_system != nullptr) && m_memory_system->is_pending();
  std::ostringstream oss;
  oss << "[dbg:" << reason << "] clk=" << m_clk
      << " arrival_idx=" << m_arrival_idx << "/" << m_source_requests.size()
      << " mem_pending=" << mem_pending
      << " completed_pim_events=" << m_completed_pim_events.size();
  if (m_active_task.has_value()) {
    const auto& task = *m_active_task;
    oss << " active_phase=" << phase_name(task.phase)
        << " active_route=" << task.route
        << " active_batch_size=" << task.batch_size
        << " active_req_ids=" << join_request_ids(task.request_indices, m_requests)
        << " start=" << task.start_clk
        << " finish=" << task.predicted_finish_clk
        << " active_pim_job_idx=" << task.active_pim_job_idx
        << " pim_jobs=" << task.pim_jobs.size();
  }
  if (m_address_space) {
    oss << " allocator_regions=" << m_address_space->region_pressure_debug_string()
        << " allocator_hot_buckets=" << m_address_space->hot_bucket_pressure_debug_string(3);
  }
  debug_write_line(oss.str());
  m_next_debug_clk = m_clk + m_config.debug_interval_clk;
}

std::optional<Clk_t> RealtimeServingRuntime::next_frontend_event_clk() const {
  Clk_t next_clk = kInfClk;

  if (m_arrival_idx < m_source_requests.size()) {
    const Clk_t arrival_clk = m_source_requests[m_arrival_idx].arrival_clk;
    if (arrival_clk > m_clk) {
      next_clk = std::min(next_clk, arrival_clk);
    }
  }
  if (use_waiting_admission_queue()) {
    const auto retry_clk = min_waiting_admission_retry_clk();
    if (retry_clk.has_value() && *retry_clk > m_clk) {
      next_clk = std::min(next_clk, *retry_clk);
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

bool RealtimeServingRuntime::try_fast_forward() {
  if ((m_memory_system != nullptr && m_memory_system->is_pending()) || !m_completed_pim_events.empty()) {
    return false;
  }

  if (m_active_task.has_value() && !active_task_pim_done() && m_clk >= m_active_task->start_clk) {
    return false;
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

}  // namespace Ramulator::Serving
