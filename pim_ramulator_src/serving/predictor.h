#ifndef RAMULATOR_FRONTEND_IMPL_SERVING_PREDICTOR_H
#define RAMULATOR_FRONTEND_IMPL_SERVING_PREDICTOR_H

#include <optional>
#include <string>
#include <vector>

#include "frontend/impl/serving/types.h"

namespace Ramulator::Serving {

RouteDecision choose_route(const std::vector<RouteCandidate>& candidates,
                           const FrontendConfig& config,
                           const std::optional<double>& deadline_ms);

std::string best_drop_reason(const std::vector<RouteCandidate>& candidates);

}  // namespace Ramulator::Serving

#endif  // RAMULATOR_FRONTEND_IMPL_SERVING_PREDICTOR_H
