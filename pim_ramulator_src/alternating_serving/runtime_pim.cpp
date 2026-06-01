#include <functional>
#include <sstream>

#include "base/exception.h"
#include "base/request.h"
#include "frontend/impl/alternating_serving/runtime.h"

namespace Ramulator::AlternatingServing {

void AlternatingServingRuntime::pump_active_pim_job() {
  if (!m_active_task.has_value() || !m_active_task->pim_job.has_value()) {
    return;
  }

  auto& task = *m_active_task;
  auto& job = *task.pim_job;
  if (m_memory_system == nullptr || job.completed || job.trace == nullptr || m_clk < task.start_clk) {
    return;
  }

  size_t issued = 0;
  while (job.next_cmd_idx < job.trace->commands.size()) {
    const auto& cmd = job.trace->commands[job.next_cmd_idx];
    const bool expects_callback = (cmd.type_id != Request::Type::PIM_BARRIER);
    Request req(cmd.addr,
                cmd.type_id,
                m_requests[job.request_idx].request_id,
                expects_callback ? std::function<void(Request&)>(
                                       [this, job_id = job.job_id](Request&) {
                                         m_completed_pim_events.push_back({job_id});
                                       })
                                 : std::function<void(Request&)>{});
    if (!m_memory_system->send(req)) {
      break;
    }
    if (expects_callback) {
      job.outstanding_cmds++;
    }
    job.next_cmd_idx++;
    issued++;
  }

  if (m_config.debug_enabled && issued > 0) {
    std::ostringstream oss;
    oss << "[dbg:pump] req=" << m_requests[job.request_idx].request_id
        << " phase=" << phase_name(task.phase)
        << " issued=" << issued
        << " next_cmd_idx=" << job.next_cmd_idx << "/" << job.trace->commands.size()
        << " outstanding=" << job.outstanding_cmds;
    debug_write_line(oss.str());
  }

  if (job.next_cmd_idx == job.trace->commands.size() && !job.issue_done) {
    job.issue_done = true;
    if (m_config.debug_enabled) {
      std::ostringstream oss;
      oss << "[dbg:pump_done] req=" << m_requests[job.request_idx].request_id
          << " phase=" << phase_name(task.phase)
          << " outstanding=" << job.outstanding_cmds
          << " completed=" << job.completed;
      debug_write_line(oss.str());
    }
    if (job.outstanding_cmds == 0 && !job.completed) {
      job.completed = true;
      m_stats.real_pim_jobs_completed++;
      if (m_config.debug_enabled) {
        std::ostringstream oss;
        oss << "[dbg:pim_complete] req=" << m_requests[job.request_idx].request_id
            << " phase=" << phase_name(task.phase)
            << " job_id=" << job.job_id
            << " token_idx=" << job.token_idx
            << " source=issue";
        debug_write_line(oss.str());
      }
    }
  }
}

void AlternatingServingRuntime::process_pim_callbacks() {
  while (!m_completed_pim_events.empty()) {
    const CompletedPimEvent event = m_completed_pim_events.front();
    m_completed_pim_events.pop_front();
    if (!m_active_task.has_value() || !m_active_task->pim_job.has_value()) {
      continue;
    }

    auto& job = *m_active_task->pim_job;
    if (job.job_id != event.job_id) {
      continue;
    }

    if (job.outstanding_cmds > 0) {
      job.outstanding_cmds--;
    }
    if (job.issue_done && job.outstanding_cmds == 0 && !job.completed) {
      job.completed = true;
      m_stats.real_pim_jobs_completed++;
      if (m_config.debug_enabled) {
        std::ostringstream oss;
        oss << "[dbg:pim_complete] req=" << m_requests[job.request_idx].request_id
            << " phase=" << phase_name(m_active_task->phase)
            << " job_id=" << job.job_id
            << " token_idx=" << job.token_idx;
        debug_write_line(oss.str());
      }
    }
  }
}

bool AlternatingServingRuntime::active_task_gpu_done() const {
  return !m_active_task.has_value() || m_clk >= m_active_task->gpu_done_clk;
}

bool AlternatingServingRuntime::active_task_link_done() const {
  return !m_active_task.has_value() || m_clk >= m_active_task->link_done_clk;
}

bool AlternatingServingRuntime::active_task_pim_done() const {
  if (!m_active_task.has_value() || !m_active_task->pim_job.has_value()) {
    return true;
  }
  return m_active_task->pim_job->completed;
}

bool AlternatingServingRuntime::active_task_e2e_done() const {
  return !m_active_task.has_value() || m_clk >= m_active_task->predicted_finish_clk;
}

void AlternatingServingRuntime::maybe_complete_active_task() {
  if (!m_active_task.has_value()) {
    return;
  }
  if (!active_task_gpu_done() || !active_task_link_done() || !active_task_pim_done() || !active_task_e2e_done()) {
    return;
  }

  ActiveTask task = *m_active_task;
  m_active_task.reset();
  auto& req = m_requests[task.request_idx];
  if (!req.svc.has_value()) {
    throw ConfigurationError("AlternatingServingFrontend: active request {} missing service estimate at completion",
                             req.request_id);
  }

  const Clk_t finish_clk = m_clk;
  if (task.phase == TaskPhase::Prefill) {
    req.prefill_end_clk = finish_clk;
    req.first_token_clk = finish_clk;
    req.last_token_completion_clk = finish_clk;
    m_stats.prefill_energy_total_nj += task.prefill_energy_nj;

    if (req.remaining_decode_tokens > 0) {
      req.state = RequestState::WaitingDecode;
      req.ready_clk = finish_clk;
    } else {
      req.state = RequestState::Done;
      req.completion_clk = finish_clk;
      m_stats.completed_requests++;
    }

    if (m_config.debug_enabled) {
      std::ostringstream oss;
      oss << "[dbg:complete] req=" << req.request_id
          << " phase=prefill"
          << " finish_clk=" << finish_clk
          << " remaining_decode_tokens=" << req.remaining_decode_tokens;
      debug_write_line(oss.str());
    }
    return;
  }

  if (req.last_token_completion_clk.has_value()) {
    req.tbt_intervals_clk.push_back(finish_clk - *req.last_token_completion_clk);
  }
  req.last_token_completion_clk = finish_clk;
  req.remaining_decode_tokens -= 1;
  req.decoded_tokens_done += 1;
  m_stats.decode_energy_total_nj += task.decode_energy_nj;
  m_stats.decode_token_count++;

  if (req.remaining_decode_tokens <= 0) {
    req.state = RequestState::Done;
    req.completion_clk = finish_clk;
    m_stats.completed_requests++;
  } else {
    req.state = RequestState::WaitingDecode;
    req.ready_clk = finish_clk;
  }

  if (m_config.debug_enabled) {
    std::ostringstream oss;
    oss << "[dbg:complete] req=" << req.request_id
        << " phase=decode"
        << " finish_clk=" << finish_clk
        << " decoded_tokens_done=" << req.decoded_tokens_done
        << " remaining_decode_tokens=" << req.remaining_decode_tokens;
    debug_write_line(oss.str());
  }
}

}  // namespace Ramulator::AlternatingServing
