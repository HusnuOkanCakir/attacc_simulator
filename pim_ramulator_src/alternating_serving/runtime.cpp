#include <sstream>

#include "base/exception.h"
#include "frontend/impl/alternating_serving/inputs.h"
#include "frontend/impl/alternating_serving/runtime.h"

namespace Ramulator::AlternatingServing {

namespace {

std::string preferred_reason(bool prefer_hybrid) {
  return prefer_hybrid ? "alternating_hybrid" : "alternating_gpu";
}

}  // namespace

void AlternatingServingRuntime::admit_arrivals_up_to_now() {
  while (m_arrival_idx < m_source_requests.size() && m_source_requests[m_arrival_idx].arrival_clk <= m_clk) {
    const SourceRequest& src = m_source_requests[m_arrival_idx++];
    RuntimeRequest req;
    req.request_id = src.request_id;
    req.arrival_ms = src.arrival_ms;
    req.arrival_clk = src.arrival_clk;
    req.lin = src.context_tokens;
    req.lout = src.generated_tokens;
    req.ready_clk = src.arrival_clk;

    if (src.context_tokens <= 0 || src.generated_tokens <= 0) {
      req.state = RequestState::Dropped;
      req.dropped_reason = "invalid_shape";
      m_requests.push_back(req);
      m_stats.dropped_requests++;
      continue;
    }

    const bool prefer_hybrid = ((m_next_route_slot % 2) == 0);
    m_next_route_slot++;
    req.admission_route = prefer_hybrid ? m_config.hybrid_route_name : m_config.gpu_route_name;
    req.route_decision_reason = preferred_reason(prefer_hybrid);

    std::optional<ServiceEstimate> chosen_svc;
    std::string chosen_route;

    if (prefer_hybrid) {
      auto hybrid_svc = estimate_request_for_route(m_config.hybrid_route_name, req.lin, req.lout, m_config.batch_size);
      const bool missing_cost = !hybrid_svc.has_value();
      const bool missing_template = hybrid_svc.has_value() && hybrid_svc->uses_pim &&
                                    !has_template_for_lin(m_config.hybrid_route_name, hybrid_svc->mapped_lin);

      if (hybrid_svc.has_value() && !missing_template) {
        chosen_route = m_config.hybrid_route_name;
        chosen_svc = std::move(hybrid_svc);
      } else {
        auto gpu_svc = estimate_request_for_route(m_config.gpu_route_name, req.lin, req.lout, m_config.batch_size);
        if (gpu_svc.has_value()) {
          chosen_route = m_config.gpu_route_name;
          chosen_svc = std::move(gpu_svc);
          req.fallback_reason = missing_template ? "missing_template" : "missing_cost";
          req.route_decision_reason = missing_template ? "fallback_missing_template" : "fallback_missing_cost";
          m_stats.route_fallback_count++;
        } else {
          req.state = RequestState::Dropped;
          req.dropped_reason = "missing_gpu_cost";
          req.fallback_reason = missing_template ? "missing_template" : "missing_cost";
          req.route_decision_reason = missing_template ? "fallback_missing_template" : "fallback_missing_cost";
          m_requests.push_back(req);
          m_stats.dropped_requests++;
          continue;
        }
      }
    } else {
      auto gpu_svc = estimate_request_for_route(m_config.gpu_route_name, req.lin, req.lout, m_config.batch_size);
      if (!gpu_svc.has_value()) {
        req.state = RequestState::Dropped;
        req.dropped_reason = "missing_gpu_cost";
        m_requests.push_back(req);
        m_stats.dropped_requests++;
        continue;
      }
      chosen_route = m_config.gpu_route_name;
      chosen_svc = std::move(gpu_svc);
    }

    req.route = chosen_route;
    req.svc = chosen_svc;
    req.remaining_decode_tokens = req.svc->decode_tokens;
    req.state = RequestState::WaitingPrefill;

    if (req.svc->clipped_lin) {
      m_stats.clipped_lin_count++;
    }
    if (req.svc->clipped_lout) {
      m_stats.clipped_lout_count++;
    }

    m_stats.admitted_requests++;
    if (req.route == m_config.gpu_route_name) {
      m_stats.route_gpu++;
    } else {
      m_stats.route_hybrid++;
    }

    if (m_config.debug_enabled) {
      std::ostringstream oss;
      oss << "[dbg:admit] req=" << req.request_id
          << " arrival_clk=" << req.arrival_clk
          << " lin=" << req.lin
          << " lout=" << req.lout
          << " admission_route=" << req.admission_route
          << " route=" << req.route
          << " mapped_lin=" << req.svc->mapped_lin
          << " mapped_lout=" << req.svc->mapped_lout
          << " reason=" << req.route_decision_reason;
      if (!req.fallback_reason.empty()) {
        oss << " fallback_reason=" << req.fallback_reason;
      }
      debug_write_line(oss.str());
    }

    m_requests.push_back(std::move(req));
  }
}

int AlternatingServingRuntime::next_serial_request_index() const {
  for (int i = 0; i < static_cast<int>(m_requests.size()); ++i) {
    if (m_requests[i].state == RequestState::WaitingPrefill || m_requests[i].state == RequestState::WaitingDecode) {
      return i;
    }
    if (m_requests[i].state == RequestState::RunningPrefill || m_requests[i].state == RequestState::RunningDecode) {
      return -1;
    }
  }
  return -1;
}

void AlternatingServingRuntime::maybe_start_next_task() {
  if (m_active_task.has_value()) {
    return;
  }

  const int req_idx = next_serial_request_index();
  if (req_idx < 0) {
    return;
  }

  auto& req = m_requests[req_idx];
  if (!req.svc.has_value()) {
    throw ConfigurationError("AlternatingServingFrontend: request {} selected without service estimate",
                             req.request_id);
  }
  if (req.ready_clk > m_clk) {
    return;
  }

  const TaskPhase phase =
      (req.state == RequestState::WaitingPrefill) ? TaskPhase::Prefill : TaskPhase::Decode;
  const auto& svc = *req.svc;
  const Clk_t gpu_clk = (phase == TaskPhase::Prefill) ? svc.prefill_gpu_clk : svc.decode_gpu_clk;
  const Clk_t link_clk = (phase == TaskPhase::Prefill) ? svc.prefill_link_clk : svc.decode_link_clk;
  const Clk_t pim_clk = (phase == TaskPhase::Prefill) ? svc.prefill_pim_clk : svc.decode_pim_clk;
  const Clk_t e2e_clk = (phase == TaskPhase::Prefill) ? svc.prefill_e2e_clk : svc.decode_e2e_clk;
  const Clk_t start_clk = std::max(req.ready_clk, m_clk);

  ActiveTask task;
  task.task_id = m_next_task_id++;
  task.request_idx = req_idx;
  task.phase = phase;
  task.route = req.route;
  task.start_clk = start_clk;
  task.predicted_finish_clk = start_clk + e2e_clk;
  task.gpu_done_clk = start_clk + gpu_clk;
  task.link_done_clk = start_clk + link_clk;
  task.prefill_energy_nj = (phase == TaskPhase::Prefill) ? svc.prefill_energy_nj : 0.0;
  task.decode_energy_nj = (phase == TaskPhase::Decode) ? svc.decode_energy_nj : 0.0;

  if (gpu_clk > 0) {
    m_stats.gpu_busy_cycles += gpu_clk;
  }
  if (link_clk > 0) {
    m_stats.link_busy_cycles += link_clk;
  }
  if (pim_clk > 0) {
    m_stats.pim_busy_cycles += pim_clk;
  }

  if (phase == TaskPhase::Prefill) {
    req.state = RequestState::RunningPrefill;
    req.prefill_start_clk = start_clk;
  } else {
    req.state = RequestState::RunningDecode;
    if (!req.first_decode_start_clk.has_value()) {
      req.first_decode_start_clk = start_clk;
    }

    if (svc.uses_pim) {
      const TemplateTrace* tpl = get_template_for_lin(req.route, svc.mapped_lin);
      if (tpl == nullptr) {
        throw ConfigurationError("AlternatingServingFrontend: missing template for route={} lin={}",
                                 req.route,
                                 svc.mapped_lin);
      }
      PimJob job;
      job.job_id = m_next_pim_job_id++;
      job.request_idx = req_idx;
      job.token_idx = req.decoded_tokens_done;
      job.trace = tpl;
      task.pim_job = job;
      m_stats.real_pim_jobs_enqueued++;
    }
  }

  m_active_task = std::move(task);
  if (m_config.debug_enabled) {
    const auto& active = *m_active_task;
    std::ostringstream oss;
    oss << "[dbg:start] req=" << req.request_id
        << " phase=" << phase_name(active.phase)
        << " route=" << req.route
        << " start_clk=" << active.start_clk
        << " finish_clk=" << active.predicted_finish_clk
        << " uses_pim=" << active.pim_job.has_value();
    debug_write_line(oss.str());
  }
}

}  // namespace Ramulator::AlternatingServing
