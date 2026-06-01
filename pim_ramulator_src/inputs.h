#ifndef RAMULATOR_FRONTEND_IMPL_SERVING_INPUTS_H
#define RAMULATOR_FRONTEND_IMPL_SERVING_INPUTS_H

#include <optional>
#include <string>
#include <vector>

#include "frontend/impl/serving/types.h"

namespace Ramulator::Serving {

void validate_static_config(const FrontendConfig& config);

std::vector<SourceRequest> load_requests_csv(const FrontendConfig& config);

CostTableMap load_cost_tables(const FrontendConfig& config);

TemplateMap load_templates(const FrontendConfig& config);

std::optional<LookupResult> lookup_cost(const CostTableMap& cost_points,
                                        const FrontendConfig& config,
                                        const std::string& route,
                                        int lin,
                                        int lout,
                                        int bs);

std::optional<ServiceEstimate> estimate_request(const CostTableMap& cost_points,
                                                const FrontendConfig& config,
                                                double cycles_per_ms,
                                                const std::string& route,
                                                int context_tokens,
                                                int generated_tokens,
                                                int bs);

}  // namespace Ramulator::Serving

#endif  // RAMULATOR_FRONTEND_IMPL_SERVING_INPUTS_H
