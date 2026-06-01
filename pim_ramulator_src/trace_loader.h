#ifndef RAMULATOR_FRONTEND_IMPL_SERVING_TRACE_LOADER_H
#define RAMULATOR_FRONTEND_IMPL_SERVING_TRACE_LOADER_H

#include <string>
#include <vector>

#include "frontend/impl/serving/types.h"

namespace Ramulator::Serving {

int parse_trace_op(const std::string& op, bool allow_load_store = true);

std::vector<TraceCommand> load_trace_commands(const std::string& file_path, bool allow_load_store = true);

}  // namespace Ramulator::Serving

#endif  // RAMULATOR_FRONTEND_IMPL_SERVING_TRACE_LOADER_H
