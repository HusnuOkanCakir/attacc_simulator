#include <functional>

#include "base/request.h"
#include "frontend/impl/serving/command_generator.h"
#include "frontend/impl/serving/pim_executor.h"

namespace Ramulator::Serving {

PimPumpResult pump_pim_job(IMemorySystem* memory_system,
                           ITranslation* translation,
                           PimJob& job,
                           int request_id,
                           Clk_t now_clk,
                           Clk_t start_clk,
                           const PimEventSink& event_sink) {
  PimPumpResult result;
  if (memory_system == nullptr || job.completed || now_clk < start_clk) {
    return result;
  }

  while (true) {
    GeneratedCommand generated;
    if (job.command_stream) {
      if (job.has_pending_generated_cmd) {
        generated.type_id = job.pending_generated_type_id;
        generated.addr = job.pending_generated_addr;
        generated.expects_callback = job.pending_generated_expects_callback;
      } else {
        if (!job.command_stream->next(generated)) {
          break;
        }
        job.pending_generated_type_id = generated.type_id;
        job.pending_generated_addr = generated.addr;
        job.pending_generated_expects_callback = generated.expects_callback;
        job.has_pending_generated_cmd = true;
      }
    } else {
      if (job.trace == nullptr || job.next_cmd_idx >= job.trace->commands.size()) {
        break;
      }
      const auto& cmd = job.trace->commands[job.next_cmd_idx];
      generated.type_id = cmd.type_id;
      generated.addr = cmd.addr;
      generated.expects_callback = (cmd.type_id != Request::Type::PIM_BARRIER);
    }

    Request req(generated.addr,
                generated.type_id,
                request_id,
                generated.expects_callback ? std::function<void(Request&)>(
                                       [event_sink, job_id = job.job_id](Request&) {
                                         if (event_sink) {
                                           event_sink({job_id});
                                         }
                                       })
                                 : std::function<void(Request&)>{});
    if (translation != nullptr && !translation->translate(req)) {
      break;
    }
    if (!memory_system->send(req)) {
      break;
    }
    if (job.command_stream) {
      job.has_pending_generated_cmd = false;
    }
    if (generated.expects_callback) {
      job.outstanding_cmds++;
    }
    job.next_cmd_idx++;
    result.issued++;
  }

  const bool stream_done =
      job.command_stream ? (job.command_stream->done() && !job.has_pending_generated_cmd)
                         : (job.trace != nullptr && job.next_cmd_idx == job.trace->commands.size());
  if (stream_done && !job.issue_done) {
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
