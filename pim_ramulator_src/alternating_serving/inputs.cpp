#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/exception.h"
#include "frontend/impl/alternating_serving/inputs.h"
#include "frontend/impl/serving/inputs_internal.h"
#include "frontend/impl/serving/trace_loader.h"

namespace Ramulator::AlternatingServing {

namespace fs = std::filesystem;
namespace csv = Ramulator::Serving::detail;

void validate_static_config(const FrontendConfig& config) {
  if (config.route_policy != "min_finish") {
    throw ConfigurationError("AlternatingServingFrontend: only route_policy=min_finish is supported, got {}",
                             config.route_policy);
  }
  if (config.unsupported_policy != "clip" &&
      config.unsupported_policy != "nearest" &&
      config.unsupported_policy != "drop") {
    throw ConfigurationError("AlternatingServingFrontend: unsupported_policy must be clip/nearest/drop, got {}",
                             config.unsupported_policy);
  }
  if (config.clock_ratio != 1) {
    throw ConfigurationError("AlternatingServingFrontend: Frontend.clock_ratio must be 1, got {}",
                             config.clock_ratio);
  }
  if (config.batch_size != 1) {
    throw ConfigurationError("AlternatingServingFrontend: batch_size must be 1, got {}",
                             config.batch_size);
  }
}

std::vector<SourceRequest> load_requests_csv(const FrontendConfig& config) {
  fs::path p(config.requests_csv);
  if (!fs::exists(p)) {
    throw ConfigurationError("AlternatingServingFrontend: requests_csv {} does not exist", config.requests_csv);
  }

  std::ifstream in(p);
  if (!in.is_open()) {
    throw ConfigurationError("AlternatingServingFrontend: cannot open requests_csv {}", config.requests_csv);
  }

  std::string header_line;
  if (!std::getline(in, header_line)) {
    throw ConfigurationError("AlternatingServingFrontend: empty requests_csv {}", config.requests_csv);
  }

  const auto header = csv::split_csv_line(header_line);
  std::unordered_map<std::string, int> col_idx;
  for (int i = 0; i < static_cast<int>(header.size()); ++i) {
    col_idx[csv::lower(header[i])] = i;
  }

  const int c_ts = csv::find_col(col_idx, {"TIMESTAMP", "timestamp"});
  const int c_arrival_ms = csv::find_col(col_idx, {"arrival_ms", "ArrivalMs"});
  const int c_lin = csv::find_col(col_idx, {"ContextTokens", "context_tokens", "lin", "Lin"});
  const int c_lout = csv::find_col(col_idx, {"GeneratedTokens", "generated_tokens", "lout", "Lout"});

  if ((c_ts < 0 && c_arrival_ms < 0) || c_lin < 0 || c_lout < 0) {
    throw ConfigurationError("AlternatingServingFrontend: requests_csv missing required columns");
  }

  std::vector<SourceRequest> parsed;
  std::string line;
  int next_request_id = 0;
  while (std::getline(in, line)) {
    if (csv::trim(line).empty()) {
      continue;
    }

    const auto fields = csv::split_csv_line(line);
    auto get_field = [&](int idx) -> std::string {
      if (idx < 0 || idx >= static_cast<int>(fields.size())) {
        return "";
      }
      return fields[idx];
    };

    double arrival_raw_ms = 0.0;
    if (c_arrival_ms >= 0) {
      arrival_raw_ms = csv::to_double(get_field(c_arrival_ms), 0.0);
    } else if (!csv::parse_timestamp_ms(get_field(c_ts), arrival_raw_ms)) {
      continue;
    }

    SourceRequest req;
    req.request_id = next_request_id++;
    req.arrival_ms = arrival_raw_ms;
    req.context_tokens = csv::to_int(get_field(c_lin), 0);
    req.generated_tokens = csv::to_int(get_field(c_lout), 0);
    parsed.push_back(req);
  }

  if (parsed.empty()) {
    throw ConfigurationError("AlternatingServingFrontend: requests_csv {} has no valid rows", config.requests_csv);
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

namespace {

std::vector<CostPoint> load_cost_table_rows(const FrontendConfig& config,
                                            const std::string& route_name,
                                            const std::string& csv_path) {
  fs::path p(csv_path);
  if (!fs::exists(p)) {
    throw ConfigurationError("AlternatingServingFrontend: cost CSV {} does not exist", csv_path);
  }

  std::ifstream in(p);
  if (!in.is_open()) {
    throw ConfigurationError("AlternatingServingFrontend: cannot open cost CSV {}", csv_path);
  }

  std::string header_line;
  if (!std::getline(in, header_line)) {
    throw ConfigurationError("AlternatingServingFrontend: empty cost CSV {}", csv_path);
  }

  const auto header = csv::split_csv_line(header_line);
  std::unordered_map<std::string, int> col_idx;
  for (int i = 0; i < static_cast<int>(header.size()); ++i) {
    col_idx[csv::lower(header[i])] = i;
  }

  auto get_col = [&](const std::string& key) -> int {
    auto it = col_idx.find(csv::lower(key));
    return (it == col_idx.end()) ? -1 : it->second;
  };

  const int c_lin = get_col("Lin");
  const int c_lout = get_col("Lout");
  const int c_bs = get_col("bs");
  const int c_s_time = get_col("s_time");
  const int c_s_energy = get_col("s_energy (nJ)");
  const int c_g_time = get_col("g_time (ms)");
  const int c_g_energy = get_col("g_energy (nJ)");
  const int c_g_matmul = get_col("g_matmul");
  const int c_g_softmax = get_col("g_softmax");
  const int c_g_etc = get_col("g_etc");
  const int c_g2g = get_col("g2g_comm");
  const int c_c2g = get_col("c2g_comm");
  const int c_g_qkv = get_col("g_qkv_time");
  const int c_g_prj = get_col("g_prj_time");
  const int c_g_ff = get_col("g_ff_time");
  const int c_g_fc = get_col("g_fc");

  if (c_lin < 0 || c_lout < 0 || c_bs < 0 || c_s_time < 0 || c_g_time < 0 ||
      c_g_matmul < 0 || c_g_softmax < 0 || c_g_etc < 0 || c_g2g < 0 || c_c2g < 0 ||
      ((c_g_qkv < 0 || c_g_prj < 0 || c_g_ff < 0) && c_g_fc < 0)) {
    throw ConfigurationError("AlternatingServingFrontend: missing required columns in {}", csv_path);
  }

  auto get_field = [](const std::vector<std::string>& fields, int idx) -> std::string {
    if (idx < 0 || idx >= static_cast<int>(fields.size())) {
      return "";
    }
    return fields[idx];
  };

  std::vector<CostPoint> points;
  std::string line;
  while (std::getline(in, line)) {
    if (csv::trim(line).empty()) {
      continue;
    }

    const auto fields = csv::split_csv_line(line);
    CostPoint pnt;
    pnt.lin = csv::to_int(get_field(fields, c_lin), 0);
    pnt.lout = csv::to_int(get_field(fields, c_lout), 0);
    pnt.bs = csv::to_int(get_field(fields, c_bs), 1);
    pnt.prefill_e2e_ms = csv::to_double(get_field(fields, c_s_time), 0.0);
    pnt.prefill_gpu_ms = pnt.prefill_e2e_ms;
    pnt.prefill_link_ms = 0.0;
    pnt.prefill_pim_ms = 0.0;
    pnt.prefill_energy_nj = csv::to_double(get_field(fields, c_s_energy), 0.0);
    pnt.decode_e2e_ms = csv::to_double(get_field(fields, c_g_time), 0.0);
    pnt.decode_energy_nj = csv::to_double(get_field(fields, c_g_energy), 0.0);

    if (route_name == config.gpu_route_name) {
      pnt.decode_gpu_ms = pnt.decode_e2e_ms;
      pnt.decode_link_ms = 0.0;
      pnt.decode_pim_ms = 0.0;
    } else {
      const double qkv_ms = (c_g_qkv >= 0) ? csv::to_double(get_field(fields, c_g_qkv), 0.0)
                                           : csv::to_double(get_field(fields, c_g_fc), 0.0);
      const double prj_ms = (c_g_prj >= 0) ? csv::to_double(get_field(fields, c_g_prj), 0.0) : 0.0;
      const double ff_ms = (c_g_ff >= 0) ? csv::to_double(get_field(fields, c_g_ff), 0.0) : 0.0;
      const double coarse_fc_ms = (c_g_fc >= 0) ? csv::to_double(get_field(fields, c_g_fc), 0.0) : 0.0;
      pnt.decode_gpu_ms = qkv_ms + prj_ms + ff_ms + csv::to_double(get_field(fields, c_g_etc), 0.0) +
                          csv::to_double(get_field(fields, c_g2g), 0.0);
      if (pnt.decode_gpu_ms <= 0.0 && coarse_fc_ms > 0.0) {
        pnt.decode_gpu_ms = coarse_fc_ms + csv::to_double(get_field(fields, c_g_etc), 0.0) +
                            csv::to_double(get_field(fields, c_g2g), 0.0);
      }
      pnt.decode_link_ms = csv::to_double(get_field(fields, c_c2g), 0.0);
      pnt.decode_pim_ms = csv::to_double(get_field(fields, c_g_matmul), 0.0) +
                          csv::to_double(get_field(fields, c_g_softmax), 0.0);
      if (pnt.decode_gpu_ms <= 0.0 && pnt.decode_link_ms <= 0.0 && pnt.decode_pim_ms <= 0.0) {
        pnt.decode_gpu_ms = pnt.decode_e2e_ms;
      }
    }

    if (pnt.lin > 0 && pnt.lout > 0 && pnt.bs > 0) {
      points.push_back(pnt);
    }
  }

  if (points.empty()) {
    throw ConfigurationError("AlternatingServingFrontend: no valid rows in {}", csv_path);
  }
  return points;
}

}  // namespace

CostTableMap load_cost_tables(const FrontendConfig& config) {
  CostTableMap tables;
  tables[config.gpu_route_name] = load_cost_table_rows(config, config.gpu_route_name, config.cost_gpu_csv);
  tables[config.hybrid_route_name] = load_cost_table_rows(config, config.hybrid_route_name, config.cost_hybrid_csv);
  return tables;
}

TemplateMap load_templates(const FrontendConfig& config) {
  fs::path root(config.template_dir);
  if (!fs::exists(root)) {
    throw ConfigurationError("AlternatingServingFrontend: template_dir {} does not exist", config.template_dir);
  }

  TemplateMap templates;
  std::regex lin_regex("lin[_-]?([0-9]+)");
  for (const auto& entry : fs::directory_iterator(root)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".trace") {
      continue;
    }

    std::smatch match;
    const std::string filename = entry.path().filename().string();
    if (!std::regex_search(filename, match, lin_regex) || match.size() < 2) {
      continue;
    }

    const int lin = csv::to_int(match[1].str(), -1);
    if (lin <= 0) {
      continue;
    }

    TemplateTrace trace;
    trace.lin = lin;
    for (const auto& cmd : Ramulator::Serving::load_trace_commands(entry.path().string(), false)) {
      trace.commands.push_back({cmd.type_id, cmd.addr});
    }
    if (!trace.commands.empty()) {
      templates[config.hybrid_route_name][lin] = std::move(trace);
    }
  }
  return templates;
}

std::optional<LookupResult> lookup_cost(const CostTableMap& cost_points,
                                        const FrontendConfig& config,
                                        const std::string& route,
                                        int lin,
                                        int lout,
                                        int bs) {
  auto rit = cost_points.find(route);
  if (rit == cost_points.end() || rit->second.empty()) {
    return std::nullopt;
  }

  const auto& points = rit->second;
  int target_lin = csv::bucket_value(std::max(1, lin), config.lin_bucket);
  int target_lout = csv::bucket_value(std::max(1, lout), config.lout_bucket);
  const int target_bs = std::max(1, bs);

  bool clipped_lin = false;
  bool clipped_lout = false;
  if (config.unsupported_policy == "clip") {
    int min_lin = points.front().lin;
    int max_lin = points.front().lin;
    int min_lout = points.front().lout;
    int max_lout = points.front().lout;
    for (const auto& point : points) {
      min_lin = std::min(min_lin, point.lin);
      max_lin = std::max(max_lin, point.lin);
      min_lout = std::min(min_lout, point.lout);
      max_lout = std::max(max_lout, point.lout);
    }
    const int new_lin = std::min(std::max(target_lin, min_lin), max_lin);
    const int new_lout = std::min(std::max(target_lout, min_lout), max_lout);
    clipped_lin = (new_lin != target_lin);
    clipped_lout = (new_lout != target_lout);
    target_lin = new_lin;
    target_lout = new_lout;
  }

  for (const auto& point : points) {
    if (point.lin == target_lin && point.lout == target_lout && point.bs == target_bs) {
      LookupResult out;
      out.point = point;
      out.mapped_lin = point.lin;
      out.mapped_lout = point.lout;
      out.mapped_bs = point.bs;
      out.clipped_lin = clipped_lin;
      out.clipped_lout = clipped_lout;
      return out;
    }
  }

  if (config.unsupported_policy == "drop") {
    return std::nullopt;
  }

  const CostPoint* best = nullptr;
  int best_d = std::numeric_limits<int>::max();
  int best_bs_d = std::numeric_limits<int>::max();
  int best_sum = std::numeric_limits<int>::max();
  for (const auto& point : points) {
    const int d = std::abs(point.lin - target_lin) + std::abs(point.lout - target_lout);
    const int bs_d = std::abs(point.bs - target_bs);
    const int sum = point.lin + point.lout;
    if (d < best_d || (d == best_d && bs_d < best_bs_d) ||
        (d == best_d && bs_d == best_bs_d && sum < best_sum)) {
      best = &point;
      best_d = d;
      best_bs_d = bs_d;
      best_sum = sum;
    }
  }

  if (best == nullptr) {
    return std::nullopt;
  }

  LookupResult out;
  out.point = *best;
  out.mapped_lin = best->lin;
  out.mapped_lout = best->lout;
  out.mapped_bs = best->bs;
  out.clipped_lin = clipped_lin || (best->lin != target_lin);
  out.clipped_lout = clipped_lout || (best->lout != target_lout);
  return out;
}

std::optional<ServiceEstimate> estimate_request(const CostTableMap& cost_points,
                                                const FrontendConfig& config,
                                                double cycles_per_ms,
                                                const std::string& route,
                                                int context_tokens,
                                                int generated_tokens,
                                                int bs) {
  if (generated_tokens <= 0 || context_tokens <= 0) {
    return std::nullopt;
  }

  const auto lookup = lookup_cost(cost_points, config, route, context_tokens, generated_tokens, bs);
  if (!lookup.has_value()) {
    return std::nullopt;
  }

  const auto& point = lookup->point;
  ServiceEstimate est;
  est.route = route;
  est.lin = context_tokens;
  est.lout = generated_tokens;
  est.generated_tokens = generated_tokens;
  est.decode_tokens = std::max(0, generated_tokens - 1);
  est.mapped_lin = lookup->mapped_lin;
  est.mapped_lout = lookup->mapped_lout;
  est.mapped_bs = lookup->mapped_bs;
  est.clipped_lin = lookup->clipped_lin;
  est.clipped_lout = lookup->clipped_lout;

  est.prefill_gpu_clk = csv::ms_to_cycles(cycles_per_ms, point.prefill_gpu_ms);
  est.prefill_link_clk = csv::ms_to_cycles(cycles_per_ms, point.prefill_link_ms);
  est.prefill_pim_clk = csv::ms_to_cycles(cycles_per_ms, point.prefill_pim_ms);
  est.prefill_e2e_clk =
      std::max(csv::ms_to_cycles(cycles_per_ms, point.prefill_e2e_ms),
               std::max(est.prefill_gpu_clk, std::max(est.prefill_link_clk, est.prefill_pim_clk)));
  est.prefill_energy_nj = point.prefill_energy_nj;

  est.decode_gpu_clk = csv::ms_to_cycles(cycles_per_ms, point.decode_gpu_ms);
  est.decode_link_clk = csv::ms_to_cycles(cycles_per_ms, point.decode_link_ms);
  est.decode_pim_clk = csv::ms_to_cycles(cycles_per_ms, point.decode_pim_ms);
  est.decode_e2e_clk =
      std::max(csv::ms_to_cycles(cycles_per_ms, point.decode_e2e_ms),
               std::max(est.decode_gpu_clk, std::max(est.decode_link_clk, est.decode_pim_clk)));
  est.decode_energy_nj = point.decode_energy_nj;
  est.uses_pim = (route == config.hybrid_route_name) && est.decode_tokens > 0 &&
                 (est.decode_pim_clk > 0 || est.decode_link_clk > 0);
  return est;
}

}  // namespace Ramulator::AlternatingServing
