#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <vector>

#include "base/exception.h"
#include "frontend/impl/alternating_serving/runtime.h"

namespace Ramulator::AlternatingServing {

namespace fs = std::filesystem;

double AlternatingServingRuntime::request_ttft_ms(const RuntimeRequest& req) const {
  if (!req.first_token_clk.has_value()) {
    return 0.0;
  }
  return cycles_to_ms(*req.first_token_clk - req.arrival_clk);
}

double AlternatingServingRuntime::request_e2e_ms(const RuntimeRequest& req) const {
  if (!req.completion_clk.has_value()) {
    return 0.0;
  }
  return cycles_to_ms(*req.completion_clk - req.arrival_clk);
}

double AlternatingServingRuntime::request_mean_tbt_ms(const RuntimeRequest& req) const {
  if (!req.completion_clk.has_value()) {
    return 0.0;
  }
  if (req.tbt_intervals_clk.empty()) {
    return request_ttft_ms(req);
  }
  double sum_ms = 0.0;
  for (Clk_t interval : req.tbt_intervals_clk) {
    sum_ms += cycles_to_ms(interval);
  }
  return sum_ms / static_cast<double>(req.tbt_intervals_clk.size());
}

std::vector<double> AlternatingServingRuntime::collect_completed_request_metrics(
    const std::function<double(const RuntimeRequest&)>& metric_fn) const {
  std::vector<double> vals;
  vals.reserve(m_requests.size());
  for (const auto& req : m_requests) {
    if (req.state == RequestState::Done) {
      vals.push_back(metric_fn(req));
    }
  }
  return vals;
}

double AlternatingServingRuntime::mean(const std::vector<double>& vals) {
  if (vals.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  for (double v : vals) {
    sum += v;
  }
  return sum / static_cast<double>(vals.size());
}

double AlternatingServingRuntime::percentile(std::vector<double> vals, double q) {
  if (vals.empty()) {
    return 0.0;
  }
  std::sort(vals.begin(), vals.end());
  if (vals.size() == 1) {
    return vals.front();
  }
  const double pos = q * static_cast<double>(vals.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  if (lo == hi) {
    return vals[lo];
  }
  const double w = pos - static_cast<double>(lo);
  return vals[lo] * (1.0 - w) + vals[hi] * w;
}

double AlternatingServingRuntime::compute_makespan_ms() const {
  Clk_t makespan_clk = 0;
  for (const auto& req : m_requests) {
    if (req.completion_clk.has_value()) {
      makespan_clk = std::max(makespan_clk, *req.completion_clk);
    } else {
      makespan_clk = std::max(makespan_clk, req.arrival_clk);
    }
  }
  return cycles_to_ms(makespan_clk);
}

int AlternatingServingRuntime::total_completed_output_tokens() const {
  int total = 0;
  for (const auto& req : m_requests) {
    if (req.state == RequestState::Done) {
      total += req.lout;
    }
  }
  return total;
}

void AlternatingServingRuntime::print_stats(YAML::Emitter& emitter,
                                            const std::string& ifce_name,
                                            const std::string& impl_name) const {
  const double makespan_ms = compute_makespan_ms();
  const int completed = m_stats.completed_requests;
  const int total_tokens = total_completed_output_tokens();
  const double ttft_mean_ms = mean(collect_completed_request_metrics(
      [this](const RuntimeRequest& req) { return request_ttft_ms(req); }));
  const double ttft_p95_ms = percentile(collect_completed_request_metrics(
      [this](const RuntimeRequest& req) { return request_ttft_ms(req); }), 0.95);
  const double ttft_p99_ms = percentile(collect_completed_request_metrics(
      [this](const RuntimeRequest& req) { return request_ttft_ms(req); }), 0.99);
  const double e2e_mean_ms = mean(collect_completed_request_metrics(
      [this](const RuntimeRequest& req) { return request_e2e_ms(req); }));
  const double e2e_p95_ms = percentile(collect_completed_request_metrics(
      [this](const RuntimeRequest& req) { return request_e2e_ms(req); }), 0.95);
  const double e2e_p99_ms = percentile(collect_completed_request_metrics(
      [this](const RuntimeRequest& req) { return request_e2e_ms(req); }), 0.99);
  const double tbt_mean_ms = mean(collect_completed_request_metrics(
      [this](const RuntimeRequest& req) { return request_mean_tbt_ms(req); }));
  const double tbt_p95_ms = percentile(collect_completed_request_metrics(
      [this](const RuntimeRequest& req) { return request_mean_tbt_ms(req); }), 0.95);
  const double tbt_p99_ms = percentile(collect_completed_request_metrics(
      [this](const RuntimeRequest& req) { return request_mean_tbt_ms(req); }), 0.99);
  const double throughput_rps =
      (makespan_ms > 0.0) ? (static_cast<double>(completed) * 1000.0 / makespan_ms) : 0.0;
  const double throughput_tokps =
      (makespan_ms > 0.0) ? (static_cast<double>(total_tokens) * 1000.0 / makespan_ms) : 0.0;
  const double gpu_util =
      (makespan_ms > 0.0) ? std::min(1.0, cycles_to_ms(m_stats.gpu_busy_cycles) / makespan_ms) : 0.0;
  const double link_util =
      (makespan_ms > 0.0) ? std::min(1.0, cycles_to_ms(m_stats.link_busy_cycles) / makespan_ms) : 0.0;
  const double pim_util =
      (makespan_ms > 0.0) ? std::min(1.0, cycles_to_ms(m_stats.pim_busy_cycles) / makespan_ms) : 0.0;

  emitter << YAML::Key << ifce_name;
  emitter << YAML::Value;
  emitter << YAML::BeginMap;

  emitter << YAML::Key << "impl" << YAML::Value << impl_name;
  emitter << YAML::Key << "total_requests" << YAML::Value << m_stats.total_requests;
  emitter << YAML::Key << "admitted_requests" << YAML::Value << m_stats.admitted_requests;
  emitter << YAML::Key << "completed_requests" << YAML::Value << m_stats.completed_requests;
  emitter << YAML::Key << "dropped_requests" << YAML::Value << m_stats.dropped_requests;

  emitter << YAML::Key << "route_counts" << YAML::Value << YAML::BeginMap;
  emitter << YAML::Key << m_config.gpu_route_name << YAML::Value << m_stats.route_gpu;
  emitter << YAML::Key << m_config.hybrid_route_name << YAML::Value << m_stats.route_hybrid;
  emitter << YAML::EndMap;

  emitter << YAML::Key << "route_fallback_count" << YAML::Value << m_stats.route_fallback_count;
  emitter << YAML::Key << "mapping_clipped_lin_count" << YAML::Value << m_stats.clipped_lin_count;
  emitter << YAML::Key << "mapping_clipped_lout_count" << YAML::Value << m_stats.clipped_lout_count;
  emitter << YAML::Key << "real_pim_jobs_enqueued" << YAML::Value << m_stats.real_pim_jobs_enqueued;
  emitter << YAML::Key << "real_pim_jobs_completed" << YAML::Value << m_stats.real_pim_jobs_completed;

  emitter << YAML::Key << "makespan_ms" << YAML::Value << makespan_ms;
  emitter << YAML::Key << "throughput_rps" << YAML::Value << throughput_rps;
  emitter << YAML::Key << "throughput_tokps" << YAML::Value << throughput_tokps;
  emitter << YAML::Key << "gpu_util" << YAML::Value << gpu_util;
  emitter << YAML::Key << "link_util" << YAML::Value << link_util;
  emitter << YAML::Key << "pim_util" << YAML::Value << pim_util;

  emitter << YAML::Key << "ttft_mean_ms" << YAML::Value << ttft_mean_ms;
  emitter << YAML::Key << "ttft_p95_ms" << YAML::Value << ttft_p95_ms;
  emitter << YAML::Key << "ttft_p99_ms" << YAML::Value << ttft_p99_ms;
  emitter << YAML::Key << "e2e_mean_ms" << YAML::Value << e2e_mean_ms;
  emitter << YAML::Key << "e2e_p95_ms" << YAML::Value << e2e_p95_ms;
  emitter << YAML::Key << "e2e_p99_ms" << YAML::Value << e2e_p99_ms;
  emitter << YAML::Key << "tbt_mean_ms" << YAML::Value << tbt_mean_ms;
  emitter << YAML::Key << "tbt_p95_ms" << YAML::Value << tbt_p95_ms;
  emitter << YAML::Key << "tbt_p99_ms" << YAML::Value << tbt_p99_ms;

  emitter << YAML::Key << "prefill_energy_total_nj" << YAML::Value << m_stats.prefill_energy_total_nj;
  emitter << YAML::Key << "decode_energy_total_nj" << YAML::Value << m_stats.decode_energy_total_nj;
  emitter << YAML::Key << "energy_total_nj"
          << YAML::Value << (m_stats.prefill_energy_total_nj + m_stats.decode_energy_total_nj);

  emitter << YAML::EndMap;
  emitter << YAML::Newline;
}

void AlternatingServingRuntime::write_requests_csv() const {
  fs::path out(m_config.requests_out_csv);
  if (!out.parent_path().empty()) {
    fs::create_directories(out.parent_path());
  }

  std::ofstream of(out);
  if (!of.is_open()) {
    throw ConfigurationError("AlternatingServingFrontend: cannot open requests_out_csv {}", m_config.requests_out_csv);
  }

  of << "request_id,arrival_ms,lin,lout,route,admission_route,state,ttft_ms,e2e_ms,tbt_ms,pim_job_ms,link_job_ms,gpu_job_ms,decode_tokens,prefill_energy_nj,decode_energy_nj,mapped_lin,mapped_lout,clipped_lin,clipped_lout,route_decision_reason,fallback_reason,dropped_reason\n";
  of << std::fixed << std::setprecision(6);

  for (const auto& req : m_requests) {
    double total_gpu_job_ms = 0.0;
    double total_link_job_ms = 0.0;
    double total_pim_job_ms = 0.0;
    double prefill_energy_nj = 0.0;
    double decode_energy_nj = 0.0;
    int mapped_lin = 0;
    int mapped_lout = 0;
    int clipped_lin = 0;
    int clipped_lout = 0;
    int decode_tokens = 0;

    if (req.svc.has_value()) {
      const auto& svc = *req.svc;
      decode_tokens = svc.decode_tokens;
      total_gpu_job_ms = cycles_to_ms(svc.prefill_gpu_clk) + cycles_to_ms(svc.decode_gpu_clk) * svc.decode_tokens;
      total_link_job_ms = cycles_to_ms(svc.prefill_link_clk) + cycles_to_ms(svc.decode_link_clk) * svc.decode_tokens;
      total_pim_job_ms = cycles_to_ms(svc.prefill_pim_clk) + cycles_to_ms(svc.decode_pim_clk) * svc.decode_tokens;
      prefill_energy_nj = svc.prefill_energy_nj;
      decode_energy_nj = svc.decode_energy_nj;
      mapped_lin = svc.mapped_lin;
      mapped_lout = svc.mapped_lout;
      clipped_lin = svc.clipped_lin ? 1 : 0;
      clipped_lout = svc.clipped_lout ? 1 : 0;
    }

    of << req.request_id << ','
       << req.arrival_ms << ','
       << req.lin << ','
       << req.lout << ','
       << req.route << ','
       << req.admission_route << ','
       << state_to_string(req.state) << ','
       << request_ttft_ms(req) << ','
       << request_e2e_ms(req) << ','
       << request_mean_tbt_ms(req) << ','
       << total_pim_job_ms << ','
       << total_link_job_ms << ','
       << total_gpu_job_ms << ','
       << decode_tokens << ','
       << prefill_energy_nj << ','
       << decode_energy_nj << ','
       << mapped_lin << ','
       << mapped_lout << ','
       << clipped_lin << ','
       << clipped_lout << ','
       << req.route_decision_reason << ','
       << req.fallback_reason << ','
       << req.dropped_reason
       << '\n';
  }
}

}  // namespace Ramulator::AlternatingServing
