#include <functional>

#include "base/request.h"
#include "frontend/impl/serving/pim_executor.h"

namespace Ramulator::Serving {

PimPumpResult pump_pim_job(IMemorySystem* memory_system,
                           PimJob& job,
                           int request_id,
                           Clk_t now_clk,
                           Clk_t start_clk,
                           const PimEventSink& event_sink) {
  PimPumpResult result;
  if (memory_system == nullptr || job.completed || job.trace == nullptr || now_clk < start_clk) {
    return result;
  }

  while (job.next_cmd_idx < job.trace->commands.size()) {
    const auto& cmd = job.trace->commands[job.next_cmd_idx];
    const bool expects_callback = (cmd.type_id != Request::Type::PIM_BARRIER);
    Request req(cmd.addr,
                cmd.type_id,
                request_id,
                expects_callback ? std::function<void(Request&)>(
                                       [event_sink, job_id = job.job_id](Request&) {
                                         if (event_sink) {
                                           event_sink({job_id});
                                         }
                                       })
                                 : std::function<void(Request&)>{});
    if (!memory_system->send(req)) {
      break;
    }
    if (expects_callback) {
      job.outstanding_cmds++;
    }
    job.next_cmd_idx++;
    result.issued++;
  }

  if (job.next_cmd_idx == job.trace->commands.size() && !job.issue_done) {
    job.issue_done = true;
    result.issue_done_changed = true;
    if (job.outstanding_cmds == 0 && !job.completed) {
      job.completed = true;
      result.completed_changed = true;
    }
  }

  return result;
}

PimCallbackResult apply_pim_completion(PimJob& job, const CompletedPimEvent& event) {
  PimCallbackResult result;
  if (job.job_id != event.job_id) {
    return result;
  }

  result.matched = true;
  if (job.outstanding_cmds > 0) {
    job.outstanding_cmds--;
  }
  if (job.issue_done && job.outstanding_cmds == 0 && !job.completed) {
    job.completed = true;
    result.completed_changed = true;
  }
  return result;
}

}  // namespace Ramulator::Serving
