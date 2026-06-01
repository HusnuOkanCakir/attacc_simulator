#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <numeric>
#include <tuple>
#include <vector>

#include "base/exception.h"
#include "frontend/impl/serving/runtime.h"

namespace Ramulator::Serving {

namespace fs = std::filesystem;

double RealtimeServingRuntime::request_ttft_ms(const RuntimeRequest& req) const {
  if (!req.first_token_clk.has_value()) {
    return 0.0;
  }
  return cycles_to_ms(*req.first_token_clk - req.arrival_clk);
}

double RealtimeServingRuntime::request_prefill_wait_ms(const RuntimeRequest& req) const {
  if (!req.prefill_start_clk.has_value()) {
    return 0.0;
  }
  return cycles_to_ms(*req.prefill_start_clk - req.arrival_clk);
}

double RealtimeServingRuntime::request_e2e_ms(const RuntimeRequest& req) const {
  if (!req.completion_clk.has_value()) {
    return 0.0;
  }
  return cycles_to_ms(*req.completion_clk - req.arrival_clk);
}

double RealtimeServingRuntime::request_mean_tbt_ms(const RuntimeRequest& req) const {
  if (req.tbt_intervals_clk.empty()) {
    return request_ttft_ms(req);
  }
  double sum_ms = 0.0;
  for (Clk_t interval : req.tbt_intervals_clk) {
    sum_ms += cycles_to_ms(interval);
  }
  return sum_ms / static_cast<double>(req.tbt_intervals_clk.size());
}

std::vector<double> RealtimeServingRuntime::collect_completed_request_metrics(
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

double RealtimeServingRuntime::mean(const std::vector<double>& vals) {
  if (vals.empty()) {
    return 0.0;
  }
  return std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<double>(vals.size());
}

double RealtimeServingRuntime::percentile(std::vector<double> vals, double q) {
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

double RealtimeServingRuntime::compute_makespan_ms() const {
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

int RealtimeServingRuntime::total_completed_output_tokens() const {
  int total = 0;
  for (const auto& req : m_requests) {
    if (req.state == RequestState::Done) {
      total += req.lout;
    }
  }
  return total;
}

void RealtimeServingRuntime::print_stats(YAML::Emitter& emitter,
                                         const std::string& ifce_name,
                                         const std::string& impl_name) const {
  const double makespan_ms = compute_makespan_ms();
  const int completed = m_stats.completed_requests;
  const int total_tokens = total_completed_output_tokens();
  const auto ttft_vals = collect_completed_request_metrics(
      [this](const RuntimeRequest& req) { return request_ttft_ms(req); });
  const auto prefill_wait_vals = collect_completed_request_metrics(
      [this](const RuntimeRequest& req) { return request_prefill_wait_ms(req); });
  const auto e2e_vals = collect_completed_request_metrics(
      [this](const RuntimeRequest& req) { return request_e2e_ms(req); });
  const auto tbt_vals = collect_completed_request_metrics(
      [this](const RuntimeRequest& req) { return request_mean_tbt_ms(req); });
  const double throughput_rps =
      (makespan_ms > 0.0) ? (static_cast<double>(completed) * 1000.0 / makespan_ms) : 0.0;
  const double throughput_tokps =
      (makespan_ms > 0.0) ? (static_cast<double>(total_tokens) * 1000.0 / makespan_ms) : 0.0;

  int ttft_misses = 0;
  int e2e_misses = 0;
  int tbt_misses = 0;
  int ttft_count = 0;
  int e2e_count = 0;
  int tbt_count = 0;
  for (const auto& req : m_requests) {
    if (req.state != RequestState::Done) {
      continue;
    }
    if (m_config.slo_ttft_ms >= 0.0 && req.first_token_clk.has_value()) {
      ttft_count++;
      if (request_ttft_ms(req) > m_config.slo_ttft_ms) {
        ttft_misses++;
      }
    }
    if (m_config.slo_e2e_ms >= 0.0 && req.completion_clk.has_value()) {
      e2e_count++;
      if (request_e2e_ms(req) > m_config.slo_e2e_ms) {
        e2e_misses++;
      }
    }
    if (m_config.slo_tbt_ms >= 0.0) {
      tbt_count++;
      if (request_mean_tbt_ms(req) > m_config.slo_tbt_ms) {
        tbt_misses++;
      }
    }
  }

  emitter << YAML::Key << ifce_name;
  emitter << YAML::Value << YAML::BeginMap;
  emitter << YAML::Key << "impl" << YAML::Value << impl_name;
  emitter << YAML::Key << "total_requests" << YAML::Value << m_stats.total_requests;
  emitter << YAML::Key << "admitted_requests" << YAML::Value << m_stats.admitted_requests;
  emitter << YAML::Key << "completed_requests" << YAML::Value << m_stats.completed_requests;
  emitter << YAML::Key << "dropped_requests" << YAML::Value << m_stats.dropped_requests;

  emitter << YAML::Key << "route_counts" << YAML::Value << YAML::BeginMap;
  for (const auto& [route, count] : m_route_counts) {
    emitter << YAML::Key << route << YAML::Value << count;
  }
  emitter << YAML::EndMap;

  emitter << YAML::Key << "route_fallback_count" << YAML::Value << m_stats.route_fallback_count;
  emitter << YAML::Key << "mapping_clipped_lin_count" << YAML::Value << m_stats.clipped_lin_count;
  emitter << YAML::Key << "mapping_clipped_lout_count" << YAML::Value << m_stats.clipped_lout_count;
  emitter << YAML::Key << "real_pim_jobs_enqueued" << YAML::Value << m_stats.real_pim_jobs_enqueued;
  emitter << YAML::Key << "real_pim_jobs_completed" << YAML::Value << m_stats.real_pim_jobs_completed;
  emitter << YAML::Key << "hybrid_command_source" << YAML::Value << m_config.hybrid_command_source;
  emitter << YAML::Key << "pim_memory_peak_bytes" << YAML::Value << m_stats.pim_memory_peak_bytes;
  emitter << YAML::Key << "allocator_hold_count" << YAML::Value << m_stats.allocator_hold_count;
  emitter << YAML::Key << "allocator_gpu_fallback_count" << YAML::Value << m_stats.allocator_gpu_fallback_count;
  emitter << YAML::Key << "allocator_drop_count" << YAML::Value << m_stats.allocator_drop_count;
  emitter << YAML::Key << "allocator_region_peaks_bytes" << YAML::Value << YAML::BeginMap;
  emitter << YAML::Key << "kv" << YAML::Value << m_stats.kv_region_peak_bytes;
  emitter << YAML::Key << "score" << YAML::Value << m_stats.score_region_peak_bytes;
  emitter << YAML::Key << "context" << YAML::Value << m_stats.context_region_peak_bytes;
  emitter << YAML::Key << "scratch" << YAML::Value << m_stats.scratch_region_peak_bytes;
  emitter << YAML::EndMap;
  if (m_address_space) {
    emitter << YAML::Key << "allocator_region_pages" << YAML::Value << YAML::BeginMap;
    for (const auto& row : m_address_space->collect_region_pressure_rows()) {
      emitter << YAML::Key << region_name(row.region) << YAML::Value << YAML::BeginMap;
      emitter << YAML::Key << "total_pages" << YAML::Value << row.total_pages;
      emitter << YAML::Key << "free_pages" << YAML::Value << row.free_pages;
      emitter << YAML::Key << "used_pages" << YAML::Value << row.used_pages;
      emitter << YAML::Key << "min_free_pages" << YAML::Value << row.min_free_pages;
      emitter << YAML::Key << "peak_used_pages" << YAML::Value << row.peak_used_pages;
      emitter << YAML::EndMap;
    }
    emitter << YAML::EndMap;

    auto bucket_rows = m_address_space->collect_bucket_pressure_rows();
    std::sort(bucket_rows.begin(),
              bucket_rows.end(),
              [](const BucketPressureRow& a, const BucketPressureRow& b) {
                const double a_peak_used_ratio = (a.total_pages == 0)
                                                     ? 0.0
                                                     : static_cast<double>(a.peak_used_pages) /
                                                           a.total_pages;
                const double b_peak_used_ratio = (b.total_pages == 0)
                                                     ? 0.0
                                                     : static_cast<double>(b.peak_used_pages) /
                                                           b.total_pages;
                if (a_peak_used_ratio != b_peak_used_ratio) {
                  return a_peak_used_ratio > b_peak_used_ratio;
                }
                if (a.peak_used_pages != b.peak_used_pages) {
                  return a.peak_used_pages > b.peak_used_pages;
                }
                return std::tie(a.region, a.local_channel, a.rank, a.bank_group, a.bank) <
                       std::tie(b.region, b.local_channel, b.rank, b.bank_group, b.bank);
              });

    emitter << YAML::Key << "allocator_hot_buckets" << YAML::Value << YAML::BeginSeq;
    const size_t bucket_limit = std::min<size_t>(8, bucket_rows.size());
    for (size_t idx = 0; idx < bucket_limit; ++idx) {
      const auto& row = bucket_rows[idx];
      emitter << YAML::BeginMap;
      emitter << YAML::Key << "region" << YAML::Value << region_name(row.region);
      emitter << YAML::Key << "local_channel" << YAML::Value << row.local_channel;
      emitter << YAML::Key << "rank" << YAML::Value << static_cast<int>(row.rank);
      emitter << YAML::Key << "bank_group" << YAML::Value << static_cast<int>(row.bank_group);
      emitter << YAML::Key << "bank" << YAML::Value << static_cast<int>(row.bank);
      emitter << YAML::Key << "total_pages" << YAML::Value << row.total_pages;
      emitter << YAML::Key << "free_pages" << YAML::Value << row.free_pages;
      emitter << YAML::Key << "used_pages" << YAML::Value << row.used_pages;
      emitter << YAML::Key << "min_free_pages" << YAML::Value << row.min_free_pages;
      emitter << YAML::Key << "peak_used_pages" << YAML::Value << row.peak_used_pages;
      emitter << YAML::EndMap;
    }
    emitter << YAML::EndSeq;
  }

  emitter << YAML::Key << "makespan_ms" << YAML::Value << makespan_ms;
  emitter << YAML::Key << "throughput_rps" << YAML::Value << throughput_rps;
  emitter << YAML::Key << "throughput_tokps" << YAML::Value << throughput_tokps;
  emitter << YAML::Key << "gpu_util"
          << YAML::Value << ((makespan_ms > 0.0) ? std::min(1.0, cycles_to_ms(m_stats.gpu_busy_cycles) / makespan_ms) : 0.0);
  emitter << YAML::Key << "link_util"
          << YAML::Value << ((makespan_ms > 0.0) ? std::min(1.0, cycles_to_ms(m_stats.link_busy_cycles) / makespan_ms) : 0.0);
  emitter << YAML::Key << "pim_util"
          << YAML::Value << ((makespan_ms > 0.0) ? std::min(1.0, cycles_to_ms(m_stats.pim_busy_cycles) / makespan_ms) : 0.0);

  emitter << YAML::Key << "ttft_mean_ms" << YAML::Value << mean(ttft_vals);
  emitter << YAML::Key << "ttft_p95_ms" << YAML::Value << percentile(ttft_vals, 0.95);
  emitter << YAML::Key << "ttft_p99_ms" << YAML::Value << percentile(ttft_vals, 0.99);
  emitter << YAML::Key << "prefill_wait_mean_ms" << YAML::Value << mean(prefill_wait_vals);
  emitter << YAML::Key << "e2e_mean_ms" << YAML::Value << mean(e2e_vals);
  emitter << YAML::Key << "e2e_p95_ms" << YAML::Value << percentile(e2e_vals, 0.95);
  emitter << YAML::Key << "e2e_p99_ms" << YAML::Value << percentile(e2e_vals, 0.99);
  emitter << YAML::Key << "tbt_mean_ms" << YAML::Value << mean(tbt_vals);
  emitter << YAML::Key << "tbt_p95_ms" << YAML::Value << percentile(tbt_vals, 0.95);
  emitter << YAML::Key << "tbt_p99_ms" << YAML::Value << percentile(tbt_vals, 0.99);
  emitter << YAML::Key << "slo_ttft_miss_rate"
          << YAML::Value << ((ttft_count > 0) ? static_cast<double>(ttft_misses) / ttft_count : 0.0);
  emitter << YAML::Key << "slo_e2e_miss_rate"
          << YAML::Value << ((e2e_count > 0) ? static_cast<double>(e2e_misses) / e2e_count : 0.0);
  emitter << YAML::Key << "slo_tbt_miss_rate"
          << YAML::Value << ((tbt_count > 0) ? static_cast<double>(tbt_misses) / tbt_count : 0.0);

  emitter << YAML::Key << "prefill_batching" << YAML::Value << YAML::BeginMap;
  emitter << YAML::Key << "enabled" << YAML::Value << m_config.enable_prefill_batching;
  emitter << YAML::Key << "mean_batch_size"
          << YAML::Value << (m_prefill_batch_sizes.empty() ? 0.0
                                                           : static_cast<double>(std::accumulate(m_prefill_batch_sizes.begin(),
                                                                                                 m_prefill_batch_sizes.end(),
                                                                                                 0)) /
                                                                 static_cast<double>(m_prefill_batch_sizes.size()));
  emitter << YAML::Key << "max_batch_size"
          << YAML::Value << (m_prefill_batch_sizes.empty() ? 0 : *std::max_element(m_prefill_batch_sizes.begin(), m_prefill_batch_sizes.end()));
  emitter << YAML::Key << "num_prefill_steps" << YAML::Value << m_prefill_batch_sizes.size();
  emitter << YAML::EndMap;

  emitter << YAML::Key << "decode_batching" << YAML::Value << YAML::BeginMap;
  emitter << YAML::Key << "enabled" << YAML::Value << m_config.enable_decode_batching;
  emitter << YAML::Key << "mean_batch_size"
          << YAML::Value << (m_decode_batch_sizes.empty() ? 0.0
                                                          : static_cast<double>(std::accumulate(m_decode_batch_sizes.begin(),
                                                                                                m_decode_batch_sizes.end(),
                                                                                                0)) /
                                                                static_cast<double>(m_decode_batch_sizes.size()));
  emitter << YAML::Key << "max_batch_size"
          << YAML::Value << (m_decode_batch_sizes.empty() ? 0 : *std::max_element(m_decode_batch_sizes.begin(), m_decode_batch_sizes.end()));
  emitter << YAML::Key << "num_decode_steps" << YAML::Value << m_decode_batch_sizes.size();
  emitter << YAML::EndMap;

  emitter << YAML::Key << "prefill_energy_total_nj" << YAML::Value << m_stats.prefill_energy_total_nj;
  emitter << YAML::Key << "decode_energy_total_nj" << YAML::Value << m_stats.decode_energy_total_nj;
  emitter << YAML::Key << "energy_total_nj"
          << YAML::Value << (m_stats.prefill_energy_total_nj + m_stats.decode_energy_total_nj);

  emitter << YAML::Key << "admission_queue" << YAML::Value << YAML::BeginMap;
  emitter << YAML::Key << "max_len" << YAML::Value << m_stats.admission_queue_max_len;
  emitter << YAML::Key << "mean_wait_ms"
          << YAML::Value << (m_admission_queue_wait_samples_ms.empty()
                                 ? 0.0
                                 : std::accumulate(m_admission_queue_wait_samples_ms.begin(),
                                                   m_admission_queue_wait_samples_ms.end(),
                                                   0.0) /
                                       static_cast<double>(m_admission_queue_wait_samples_ms.size()));
  emitter << YAML::EndMap;

  emitter << YAML::Key << "admission" << YAML::Value << YAML::BeginMap;
  emitter << YAML::Key << "reject_count" << YAML::Value << m_stats.admission_reject_count;
  emitter << YAML::Key << "lost_count" << YAML::Value << m_stats.admission_lost_count;
  emitter << YAML::Key << "shadow_timeout_count" << YAML::Value << m_stats.admission_shadow_timeout_count;
  emitter << YAML::Key << "held_count" << YAML::Value << m_stats.admission_held_count;
  emitter << YAML::Key << "admitted_from_queue_count" << YAML::Value << m_stats.admission_admitted_from_queue_count;
  emitter << YAML::Key << "bypassed_count" << YAML::Value << m_stats.admission_bypassed_count;
  emitter << YAML::Key << "best_effort_count" << YAML::Value << m_stats.admission_best_effort_count;
  emitter << YAML::Key << "blocked_count" << YAML::Value << m_admission_blocked_request_ids.size();
  emitter << YAML::Key << "blocked_due_to_tbt_count" << YAML::Value << m_admission_tbt_blocked_request_ids.size();
  emitter << YAML::EndMap;

  emitter << YAML::Key << "local_scheduling" << YAML::Value << YAML::BeginMap;
  emitter << YAML::Key << "route_policy" << YAML::Value << m_config.route_policy;
  emitter << YAML::Key << "local_scheduling_policy" << YAML::Value << m_config.local_scheduling_policy;
  emitter << YAML::Key << "fcfs_decode_prediction_mode" << YAML::Value << m_config.fcfs_decode_prediction_mode;
  emitter << YAML::Key << "prefill_guard_ms" << YAML::Value << m_config.prefill_guard_ms;
  emitter << YAML::Key << "max_consecutive_decode_batches" << YAML::Value << m_config.max_consecutive_decode_batches;
  emitter << YAML::Key << "decode_batch_cap_with_prefill" << YAML::Value << m_config.decode_batch_cap_with_prefill;
  emitter << YAML::Key << "enable_decode_rebind" << YAML::Value << m_config.enable_decode_rebind;
  emitter << YAML::Key << "decode_rebind_margin_ms" << YAML::Value << m_config.decode_rebind_margin_ms;
  emitter << YAML::Key << "decode_rebind_attempt_count" << YAML::Value << m_stats.decode_rebind_attempt_count;
  emitter << YAML::Key << "decode_rebound_count" << YAML::Value << m_stats.decode_rebound_count;
  emitter << YAML::Key << "enable_predictive_admission" << YAML::Value << m_config.enable_predictive_admission;
  emitter << YAML::Key << "enable_slo_guarded_admission" << YAML::Value << m_config.enable_slo_guarded_admission;
  emitter << YAML::Key << "slo_tbt_ms" << YAML::Value << m_config.slo_tbt_ms;
  emitter << YAML::Key << "admission_shadow_max_steps" << YAML::Value << m_config.admission_shadow_max_steps;
  emitter << YAML::Key << "admission_retry_interval_ms" << YAML::Value << m_config.admission_retry_interval_ms;
  emitter << YAML::Key << "admission_max_harmed_requests" << YAML::Value << m_config.admission_max_harmed_requests;
  emitter << YAML::Key << "admission_max_total_harm_ms" << YAML::Value << m_config.admission_max_total_harm_ms;
  emitter << YAML::Key << "admission_max_single_harm_ms" << YAML::Value << m_config.admission_max_single_harm_ms;
  emitter << YAML::Key << "admission_max_own_miss_ms" << YAML::Value << m_config.admission_max_own_miss_ms;
  emitter << YAML::Key << "admission_check_own_slo" << YAML::Value << m_config.admission_check_own_slo;
  emitter << YAML::Key << "admission_max_bypass_count" << YAML::Value << m_config.admission_max_bypass_count;
  emitter << YAML::Key << "admission_max_wait_ms" << YAML::Value << m_config.admission_max_wait_ms;
  emitter << YAML::Key << "admission_aging_harm_ms_per_ms" << YAML::Value << m_config.admission_aging_harm_ms_per_ms;
  emitter << YAML::Key << "admission_aging_harmed_requests_per_ms" << YAML::Value << m_config.admission_aging_harmed_requests_per_ms;
  emitter << YAML::Key << "share_gpu_prefill_across_routes" << YAML::Value << m_config.share_gpu_prefill_across_routes;
  emitter << YAML::Key << "prefill_guard_trigger_count" << YAML::Value << m_stats.prefill_guard_trigger_count;
  emitter << YAML::Key << "decode_limit_trigger_count" << YAML::Value << m_stats.decode_limit_trigger_count;
  emitter << YAML::EndMap;

  emitter << YAML::Key << "cache_stats" << YAML::Value << YAML::BeginMap;
  emitter << YAML::Key << "estimate_cache_hits" << YAML::Value << m_estimate_cache_hits;
  emitter << YAML::Key << "estimate_cache_misses" << YAML::Value << m_estimate_cache_misses;
  emitter << YAML::Key << "estimate_prefetch_populated" << YAML::Value << m_estimate_prefetch_populated;
  emitter << YAML::Key << "incremental_active_forecast_cache_hits" << YAML::Value << m_incremental_active_forecast_cache_hits;
  emitter << YAML::Key << "incremental_active_forecast_cache_misses" << YAML::Value << m_incremental_active_forecast_cache_misses;
  emitter << YAML::Key << "projection_cache_hits" << YAML::Value << m_projection_cache_hits;
  emitter << YAML::Key << "projection_cache_misses" << YAML::Value << m_projection_cache_misses;
  emitter << YAML::EndMap;

  emitter << YAML::EndMap;
  emitter << YAML::Newline;
}

void RealtimeServingRuntime::write_requests_csv() const {
  fs::path out(m_config.requests_out_csv);
  if (!out.parent_path().empty()) {
    fs::create_directories(out.parent_path());
  }

  std::ofstream of(out);
  if (!of.is_open()) {
    throw ConfigurationError("RealtimeServingFrontend: cannot open requests_out_csv {}", m_config.requests_out_csv);
  }

  of << "request_id,arrival_ms,lin,lout,route,admission_route,state,route_decision_reason,dropped_reason,admission_time_ms,admission_queue_wait_ms,admission_attempts,admission_bypass_count,admission_last_reject_reason,admission_last_harmed_count,admission_last_total_harm_ms,admission_last_single_harm_ms,admission_last_own_miss_ms,admission_forced_best_effort,is_lost,ttft_ms,prefill_wait_ms,e2e_ms,tbt_ms,prefill_energy_nj,decode_energy_nj,decode_tokens,mapped_lin,mapped_lout,mapped_bs,clipped_lin,clipped_lout,deadline_ms,chosen_predicted_finish_ms,latest_predicted_finish_ms,chosen_predicted_gpu_wait_ms,chosen_predicted_pim_wait_ms,chosen_predicted_incremental_energy_nj,chosen_predicted_incremental_decode_energy_nj,chosen_queue_pressure_ms,chosen_slack_ms,executed_prefill_batch_size,max_executed_decode_batch_size,last_executed_decode_batch_size,decode_bind_time_ms,decode_rebind_attempted,decode_rebound,decode_rebind_reason,decode_enqueue_seq,prediction_refresh_count,decoded_tokens_done,fallback_reason,pim_kv_bytes,pim_transient_peak_bytes,allocator_fallback_reason,allocator_wait_ms\n";
  of << std::fixed << std::setprecision(12);

  for (const auto& req : m_requests) {
    of << req.request_id << ','
       << req.arrival_ms << ','
       << req.lin << ','
       << req.lout << ','
       << req.route << ','
       << req.admission_route << ','
       << state_to_string(req.state) << ','
       << req.route_decision_reason << ','
       << req.dropped_reason << ','
       << (req.admission_time_ms.has_value() ? *req.admission_time_ms : 0.0) << ','
       << (req.admission_queue_wait_ms.has_value() ? *req.admission_queue_wait_ms : 0.0) << ','
       << req.admission_attempts << ','
       << req.admission_bypass_count << ','
       << req.admission_last_reject_reason << ','
       << req.admission_last_harmed_count << ','
       << req.admission_last_total_harm_ms << ','
       << req.admission_last_single_harm_ms << ','
       << req.admission_last_own_miss_ms << ','
       << (req.admission_forced_best_effort ? 1 : 0) << ','
       << (req.is_lost ? 1 : 0) << ','
       << request_ttft_ms(req) << ','
       << request_prefill_wait_ms(req) << ','
       << request_e2e_ms(req) << ','
       << request_mean_tbt_ms(req) << ','
       << (req.svc.has_value() ? req.svc->prefill_energy_nj : 0.0) << ','
       << (req.svc.has_value() ? req.svc->decode_energy_nj : 0.0) << ','
       << (req.svc.has_value() ? req.svc->decode_tokens : 0) << ','
       << (req.svc.has_value() ? req.svc->mapped_lin : 0) << ','
       << (req.svc.has_value() ? req.svc->mapped_lout : 0) << ','
       << (req.svc.has_value() ? req.svc->mapped_bs : 0) << ','
       << (req.svc.has_value() ? (req.svc->clipped_lin ? 1 : 0) : 0) << ','
       << (req.svc.has_value() ? (req.svc->clipped_lout ? 1 : 0) : 0) << ','
       << (req.deadline_ms.has_value() ? *req.deadline_ms : 0.0) << ','
       << (req.chosen_predicted_finish_ms.has_value() ? *req.chosen_predicted_finish_ms : 0.0) << ','
       << (req.latest_predicted_finish_ms.has_value() ? *req.latest_predicted_finish_ms : 0.0) << ','
       << (req.chosen_predicted_gpu_wait_ms.has_value() ? *req.chosen_predicted_gpu_wait_ms : 0.0) << ','
       << (req.chosen_predicted_pim_wait_ms.has_value() ? *req.chosen_predicted_pim_wait_ms : 0.0) << ','
       << (req.chosen_predicted_incremental_energy_nj.has_value() ? *req.chosen_predicted_incremental_energy_nj : 0.0) << ','
       << (req.chosen_predicted_incremental_decode_energy_nj.has_value() ? *req.chosen_predicted_incremental_decode_energy_nj : 0.0) << ','
       << (req.chosen_queue_pressure_ms.has_value() ? *req.chosen_queue_pressure_ms : 0.0) << ','
       << (req.chosen_slack_ms.has_value() ? *req.chosen_slack_ms : 0.0) << ','
       << req.executed_prefill_batch_size << ','
       << req.max_executed_decode_batch_size << ','
       << req.last_executed_decode_batch_size << ','
       << (req.decode_bind_time_ms.has_value() ? *req.decode_bind_time_ms : 0.0) << ','
       << (req.decode_rebind_attempted ? 1 : 0) << ','
       << (req.decode_rebound ? 1 : 0) << ','
       << req.decode_rebind_reason << ','
       << (req.decode_enqueue_seq.has_value() ? std::to_string(*req.decode_enqueue_seq) : "") << ','
       << req.prediction_refresh_count << ','
       << req.decoded_tokens_done << ','
       << req.fallback_reason << ','
       << req.pim_kv_bytes << ','
       << req.pim_transient_peak_bytes << ','
       << req.allocator_fallback_reason << ','
       << req.allocator_wait_ms
       << '\n';
  }
}

void RealtimeServingRuntime::write_kv_placement_csv() const {
  if (m_config.kv_placement_csv.empty() || !m_address_space) {
    return;
  }

  fs::path out(m_config.kv_placement_csv);
  if (!out.parent_path().empty()) {
    fs::create_directories(out.parent_path());
  }

  std::ofstream of(out);
  if (!of.is_open()) {
    throw ConfigurationError("RealtimeServingFrontend: cannot open kv_placement_csv {}", m_config.kv_placement_csv);
  }

  of << "request_id,region,object_id,logical_page_index,logical_request_slot,logical_layer,logical_head_itr,logical_local_channel,logical_rank,logical_bank_group,logical_bank,logical_page_slot,logical_row,physical_page,physical_addr,physical_channel,physical_rank,physical_bank_group,physical_bank,physical_row,bucket_match\n";

  for (const auto& row : m_address_space->collect_placement_rows()) {
    of << row.request_id << ','
       << region_name(row.region) << ','
       << static_cast<int>(row.object_id) << ','
       << row.logical_page_index << ','
       << row.logical_request_slot << ','
       << row.logical_layer << ','
       << row.logical_head_itr << ','
       << row.logical_local_channel << ','
       << static_cast<int>(row.logical_rank) << ','
       << static_cast<int>(row.logical_bank_group) << ','
       << static_cast<int>(row.logical_bank) << ','
       << row.logical_page_slot << ','
       << row.logical_row << ','
       << row.physical_page << ','
       << row.physical_addr << ','
       << row.physical_channel << ','
       << static_cast<int>(row.physical_rank) << ','
       << static_cast<int>(row.physical_bank_group) << ','
       << static_cast<int>(row.physical_bank) << ','
       << row.physical_row << ','
       << (row.bucket_match ? 1 : 0)
       << '\n';
  }
}

void RealtimeServingRuntime::write_allocator_pressure_csv() const {
  if (m_config.allocator_pressure_csv.empty() || !m_address_space) {
    return;
  }

  fs::path out(m_config.allocator_pressure_csv);
  if (!out.parent_path().empty()) {
    fs::create_directories(out.parent_path());
  }

  std::ofstream of(out);
  if (!of.is_open()) {
    throw ConfigurationError("RealtimeServingFrontend: cannot open allocator_pressure_csv {}",
                             m_config.allocator_pressure_csv);
  }

  of << "scope,region,local_channel,rank,bank_group,bank,total_pages,free_pages,used_pages,min_free_pages,peak_used_pages,used_ratio,peak_used_ratio\n";
  for (const auto& row : m_address_space->collect_region_pressure_rows()) {
    const double used_ratio =
        (row.total_pages == 0) ? 0.0 : static_cast<double>(row.used_pages) / row.total_pages;
    const double peak_used_ratio = (row.total_pages == 0)
                                       ? 0.0
                                       : static_cast<double>(row.peak_used_pages) / row.total_pages;
    of << "region,"
       << region_name(row.region) << ",,,,,"
       << row.total_pages << ','
       << row.free_pages << ','
       << row.used_pages << ','
       << row.min_free_pages << ','
       << row.peak_used_pages << ','
       << used_ratio << ','
       << peak_used_ratio << '\n';
  }
  for (const auto& row : m_address_space->collect_bucket_pressure_rows()) {
    const double used_ratio =
        (row.total_pages == 0) ? 0.0 : static_cast<double>(row.used_pages) / row.total_pages;
    const double peak_used_ratio = (row.total_pages == 0)
                                       ? 0.0
                                       : static_cast<double>(row.peak_used_pages) / row.total_pages;
    of << "bucket,"
       << region_name(row.region) << ','
       << row.local_channel << ','
       << static_cast<int>(row.rank) << ','
       << static_cast<int>(row.bank_group) << ','
       << static_cast<int>(row.bank) << ','
       << row.total_pages << ','
       << row.free_pages << ','
       << row.used_pages << ','
       << row.min_free_pages << ','
       << row.peak_used_pages << ','
       << used_ratio << ','
       << peak_used_ratio << '\n';
  }
}

}  // namespace Ramulator::Serving
