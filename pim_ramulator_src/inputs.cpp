#include <algorithm>
#include <filesystem>
#include <fstream>
#include <vector>

#include "base/exception.h"
#include "frontend/impl/serving/inputs.h"
#include "frontend/impl/serving/inputs_internal.h"

namespace Ramulator::Serving {

namespace fs = std::filesystem;

void validate_static_config(const FrontendConfig& config) {
  if (config.route_policy != "min_finish" &&
      config.route_policy != "slack_then_finish" &&
      config.route_policy != "latency_guarded_energy") {
    throw ConfigurationError("RealtimeServingFrontend: unsupported route_policy {}",
                             config.route_policy);
  }
  if (config.unsupported_policy != "clip" &&
      config.unsupported_policy != "nearest" &&
      config.unsupported_policy != "drop") {
    throw ConfigurationError("RealtimeServingFrontend: unsupported_policy must be clip/nearest/drop, got {}",
                             config.unsupported_policy);
  }
  if (config.clock_ratio != 1) {
    throw ConfigurationError("RealtimeServingFrontend: Frontend.clock_ratio must be 1");
  }
  if (config.batch_size < 1) {
    throw ConfigurationError("RealtimeServingFrontend: batch_size must be >= 1, got {}",
                             config.batch_size);
  }
  if (config.max_prefill_batch_size < 1 || config.max_decode_batch_size < 1) {
    throw ConfigurationError("RealtimeServingFrontend: max batch sizes must be >= 1");
  }
  if (config.local_scheduling_policy != "priority" &&
      config.local_scheduling_policy != "fcfs_strict" &&
      config.local_scheduling_policy != "prefill_priority_fcfs_decode") {
    throw ConfigurationError("RealtimeServingFrontend: invalid local_scheduling_policy {}",
                             config.local_scheduling_policy);
  }
  if (config.fcfs_decode_prediction_mode != "shadow" &&
      config.fcfs_decode_prediction_mode != "legacy" &&
      config.fcfs_decode_prediction_mode != "incremental" &&
      config.fcfs_decode_prediction_mode != "heuristic_refresh") {
    throw ConfigurationError("RealtimeServingFrontend: invalid fcfs_decode_prediction_mode {}",
                             config.fcfs_decode_prediction_mode);
  }
  if (config.enable_predictive_admission && config.enable_slo_guarded_admission) {
    throw ConfigurationError("RealtimeServingFrontend: predictive admission and slo_guarded_admission are mutually exclusive");
  }
  if (config.local_scheduling_policy == "prefill_priority_fcfs_decode") {
    if (config.enable_slo_guarded_admission && config.fcfs_decode_prediction_mode != "incremental") {
      throw ConfigurationError("RealtimeServingFrontend: slo_guarded_admission requires fcfs_decode_prediction_mode=incremental");
    }
  } else {
    if (config.fcfs_decode_prediction_mode != "shadow") {
      throw ConfigurationError("RealtimeServingFrontend: fcfs_decode_prediction_mode only applies to prefill_priority_fcfs_decode");
    }
    if (config.enable_slo_guarded_admission) {
      throw ConfigurationError("RealtimeServingFrontend: slo_guarded_admission requires prefill_priority_fcfs_decode");
    }
  }
  if (config.enable_slo_guarded_admission) {
    if (config.slo_e2e_ms < 0.0) {
      throw ConfigurationError("RealtimeServingFrontend: slo_guarded_admission requires slo_e2e_ms");
    }
    if (config.slo_tbt_ms < 0.0) {
      throw ConfigurationError("RealtimeServingFrontend: slo_guarded_admission requires slo_tbt_ms");
    }
  }
  if (config.route_policy == "slack_then_finish" && config.slo_e2e_ms < 0.0) {
    throw ConfigurationError("RealtimeServingFrontend: slack_then_finish requires slo_e2e_ms");
  }
  if (config.route_policy == "latency_guarded_energy") {
    if (config.local_scheduling_policy != "prefill_priority_fcfs_decode") {
      throw ConfigurationError("RealtimeServingFrontend: latency_guarded_energy requires prefill_priority_fcfs_decode");
    }
    if (config.fcfs_decode_prediction_mode != "incremental") {
      throw ConfigurationError("RealtimeServingFrontend: latency_guarded_energy requires fcfs_decode_prediction_mode=incremental");
    }
  }
}

std::vector<SourceRequest> load_requests_csv(const FrontendConfig& config) {
  fs::path p(config.requests_csv);
  if (!fs::exists(p)) {
    throw ConfigurationError("RealtimeServingFrontend: requests_csv {} does not exist", config.requests_csv);
  }
  std::ifstream in(p);
  if (!in.is_open()) {
    throw ConfigurationError("RealtimeServingFrontend: cannot open requests_csv {}", config.requests_csv);
  }

  std::string header_line;
  if (!std::getline(in, header_line)) {
    throw ConfigurationError("RealtimeServingFrontend: empty requests_csv {}", config.requests_csv);
  }

  const auto header = detail::split_csv_line(header_line);
  std::unordered_map<std::string, int> col_idx;
  for (int i = 0; i < static_cast<int>(header.size()); ++i) {
    col_idx[detail::lower(header[i])] = i;
  }

  const int c_ts = detail::find_col(col_idx, {"TIMESTAMP", "timestamp"});
  const int c_arrival_ms = detail::find_col(col_idx, {"arrival_ms", "ArrivalMs"});
  const int c_lin = detail::find_col(col_idx, {"ContextTokens", "context_tokens", "lin", "Lin"});
  const int c_lout = detail::find_col(col_idx, {"GeneratedTokens", "generated_tokens", "lout", "Lout"});

  if ((c_ts < 0 && c_arrival_ms < 0) || c_lin < 0 || c_lout < 0) {
    throw ConfigurationError("RealtimeServingFrontend: requests_csv missing required columns");
  }

  std::vector<SourceRequest> parsed;
  std::string line;
  int next_request_id = 0;
  while (std::getline(in, line)) {
    if (detail::trim(line).empty()) {
      continue;
    }
    const auto fields = detail::split_csv_line(line);
    auto get_field = [&](int idx) -> std::string {
      if (idx < 0 || idx >= static_cast<int>(fields.size())) {
        return "";
      }
      return fields[idx];
    };

    double arrival_raw_ms = 0.0;
    if (c_arrival_ms >= 0) {
      arrival_raw_ms = detail::to_double(get_field(c_arrival_ms), 0.0);
    } else if (!detail::parse_timestamp_ms(get_field(c_ts), arrival_raw_ms)) {
      continue;
    }

    SourceRequest req;
    req.request_id = next_request_id++;
    req.arrival_ms = arrival_raw_ms;
    req.context_tokens = detail::to_int(get_field(c_lin), 0);
    req.generated_tokens = detail::to_int(get_field(c_lout), 0);
    parsed.push_back(req);
  }

  if (parsed.empty()) {
    throw ConfigurationError("RealtimeServingFrontend: requests_csv {} has no valid rows", config.requests_csv);
  }

  std::sort(parsed.begin(), parsed.end(), [](const SourceRequest& a, const SourceRequest& b) {
    if (a.arrival_ms == b.arrival_ms) {
      return a.request_id < b.request_id;
    }
    return a.arrival_ms < b.arrival_ms;
  });

  const double base_ms = parsed.front().arrival_ms;
  for (auto& req : parsed) {
    req.arrival_ms = (req.arrival_ms - base_ms) * static_cast<double>(config.arrival_time_scale);
    if (req.arrival_ms < 0.0) {
      req.arrival_ms = 0.0;
    }
  }
  return parsed;
}

}  // namespace Ramulator::Serving
