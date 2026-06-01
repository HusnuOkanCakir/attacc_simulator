#ifndef RAMULATOR_FRONTEND_IMPL_SERVING_PIM_EXECUTOR_H
#define RAMULATOR_FRONTEND_IMPL_SERVING_PIM_EXECUTOR_H

#include <functional>

#include "memory_system/memory_system.h"
#include "frontend/impl/serving/types.h"
#include "translation/translation.h"

namespace Ramulator::Serving {

struct PimPumpResult {
  size_t issued = 0;
  bool issue_done_changed = false;
  bool completed_changed = false;
};

struct PimCallbackResult {
  bool matched = false;
  bool completed_changed = false;
};

using PimEventSink = std::function<void(CompletedPimEvent)>;

PimPumpResult pump_pim_job(IMemorySystem* memory_system,
                           ITranslation* translation,
                           PimJob& job,
                           int request_id,
                           Clk_t now_clk,
                           Clk_t start_clk,
                           const PimEventSink& event_sink);

PimCallbackResult apply_pim_completion(PimJob& job, const CompletedPimEvent& event);

}  // namespace Ramulator::Serving

#endif  // RAMULATOR_FRONTEND_IMPL_SERVING_PIM_EXECUTOR_H
