#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

#include "frontend/impl/serving/predictor.h"

namespace Ramulator::Serving {

namespace {

double effective_finish_ms(const RouteCandidate& cand, const FrontendConfig& config) {
  return cand.predicted_finish_ms +
         config.route_load_balance_ms_per_active_request *
             static_cast<double>(cand.route_active_requests);
}

auto finish_sort_key(const RouteCandidate& cand, const FrontendConfig& config) {
  int tie_gpu_bias = 0;
  if (config.prefer_gpu_on_tie) {
    tie_gpu_bias = cand.uses_pim ? 1 : 0;
  }
  return std::make_tuple(effective_finish_ms(cand, config),
                         cand.predicted_finish_ms,
                         tie_gpu_bias,
                         cand.route);
}

auto slack_sort_key(const RouteCandidate& cand, const FrontendConfig& config) {
  const double slack = cand.slack_ms.has_value() ? *cand.slack_ms : -std::numeric_limits<double>::infinity();
  return std::make_tuple(-slack,
                         effective_finish_ms(cand, config),
                         cand.predicted_finish_ms,
                         cand.route);
}

auto energy_sort_key(const RouteCandidate& cand, const FrontendConfig& config) {
  double energy = cand.predicted_incremental_energy_nj;
  if (!std::isfinite(energy)) {
    energy = cand.predicted_incremental_decode_energy_nj;
  }
  if (!std::isfinite(energy)) {
    energy = std::numeric_limits<double>::infinity();
  }
  return std::make_tuple(energy,
                         effective_finish_ms(cand, config),
                         cand.predicted_finish_ms,
                         cand.route);
}

RouteDecision choose_guarded_energy(const std::vector<RouteCandidate>& candidates,
                                    const FrontendConfig& config,
                                    const std::string& reason) {
  const auto* fastest = &(*std::min_element(
      candidates.begin(), candidates.end(),
      [&](const RouteCandidate& lhs, const RouteCandidate& rhs) {
        return effective_finish_ms(lhs, config) < effective_finish_ms(rhs, config);
      }));
  const double fastest_finish = effective_finish_ms(*fastest, config);

  std::vector<const RouteCandidate*> guarded;
  for (const auto& cand : candidates) {
    if (effective_finish_ms(cand, config) <=
        fastest_finish + config.energy_latency_guard_ms + 1e-9) {
      guarded.push_back(&cand);
    }
  }
  if (guarded.empty()) {
    return {.route = std::nullopt, .dropped = true, .reason = "no_eligible_route"};
  }

  const auto* best = *std::min_element(
      guarded.begin(), guarded.end(),
      [&](const RouteCandidate* lhs, const RouteCandidate* rhs) {
        return energy_sort_key(*lhs, config) < energy_sort_key(*rhs, config);
      });
  return {.route = best->route, .dropped = false, .reason = reason};
}

}  // namespace

RouteDecision choose_route(const std::vector<RouteCandidate>& candidates,
                           const FrontendConfig& config,
                           const std::optional<double>& deadline_ms) {
  std::vector<const RouteCandidate*> eligible;
  eligible.reserve(candidates.size());
  for (const auto& cand : candidates) {
    if (cand.eligible) {
      eligible.push_back(&cand);
    }
  }
  if (eligible.empty()) {
    return {.route = std::nullopt, .dropped = true, .reason = "no_eligible_route"};
  }

  std::vector<const RouteCandidate*> filtered;
  filtered.reserve(eligible.size());
  for (const auto* cand : eligible) {
    if (cand->uses_pim && config.pim_wait_threshold_ms > 0.0 &&
        cand->predicted_pim_wait_ms > config.pim_wait_threshold_ms) {
      continue;
    }
    filtered.push_back(cand);
  }

  if (filtered.empty()) {
    for (const auto* cand : eligible) {
      if (!cand->uses_pim) {
        return {.route = cand->route, .dropped = false, .reason = "fallback_gpu_due_to_pim_threshold"};
      }
    }
    return {.route = std::nullopt, .dropped = true, .reason = "all_pim_routes_rejected_by_threshold"};
  }

  auto min_finish = [&]() -> RouteDecision {
    const auto* best = *std::min_element(
        filtered.begin(), filtered.end(),
        [&](const RouteCandidate* lhs, const RouteCandidate* rhs) {
          return finish_sort_key(*lhs, config) < finish_sort_key(*rhs, config);
        });
    return {.route = best->route, .dropped = false, .reason = "min_pred_finish"};
  };

  if (config.route_policy == "min_finish") {
    return min_finish();
  }

  if (config.route_policy == "slack_then_finish") {
    if (!deadline_ms.has_value()) {
      return min_finish();
    }
    std::vector<const RouteCandidate*> on_time;
    for (const auto* cand : filtered) {
      if (cand->slack_ms.has_value() && *cand->slack_ms >= 0.0) {
        on_time.push_back(cand);
      }
    }
    if (!on_time.empty()) {
      const auto* best = *std::min_element(
          on_time.begin(), on_time.end(),
          [&](const RouteCandidate* lhs, const RouteCandidate* rhs) {
            return finish_sort_key(*lhs, config) < finish_sort_key(*rhs, config);
          });
      return {.route = best->route, .dropped = false, .reason = "deadline_feasible_min_finish"};
    }
    const auto* best = *std::min_element(
        filtered.begin(), filtered.end(),
        [&](const RouteCandidate* lhs, const RouteCandidate* rhs) {
          return slack_sort_key(*lhs, config) < slack_sort_key(*rhs, config);
        });
    return {.route = best->route, .dropped = false, .reason = "max_slack"};
  }

  if (deadline_ms.has_value()) {
    std::vector<RouteCandidate> on_time;
    for (const auto* cand : filtered) {
      if (cand->slack_ms.has_value() && *cand->slack_ms >= 0.0) {
        on_time.push_back(*cand);
      }
    }
    if (!on_time.empty()) {
      return choose_guarded_energy(on_time, config, "deadline_feasible_latency_guarded_energy");
    }
    const auto* best = *std::min_element(
        filtered.begin(), filtered.end(),
        [&](const RouteCandidate* lhs, const RouteCandidate* rhs) {
          return slack_sort_key(*lhs, config) < slack_sort_key(*rhs, config);
        });
    return {.route = best->route, .dropped = false, .reason = "max_slack"};
  }

  std::vector<RouteCandidate> filtered_values;
  filtered_values.reserve(filtered.size());
  for (const auto* cand : filtered) {
    filtered_values.push_back(*cand);
  }
  return choose_guarded_energy(filtered_values, config, "latency_guarded_energy");
}

std::string best_drop_reason(const std::vector<RouteCandidate>& candidates) {
  for (const auto& cand : candidates) {
    if (!cand.reason.empty()) {
      return cand.reason;
    }
  }
  return "no_eligible_route";
}

}  // namespace Ramulator::Serving
