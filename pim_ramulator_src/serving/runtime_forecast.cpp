#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "frontend/impl/serving/predictor.h"
#include "frontend/impl/serving/runtime.h"

namespace Ramulator::Serving {

namespace {

double opt_or_nan(const std::optional<double>& value) {
  return value.has_value() ? *value : std::numeric_limits<double>::quiet_NaN();
}

double opt_or_inf(const std::optional<double>& value) {
  return value.has_value() ? *value : std::numeric_limits<double>::infinity();
}

bool is_active_forecast_state(RequestState state, int remaining_decode_tokens) {
  return state == RequestState::WaitingPrefill ||
         (state == RequestState::WaitingDecode && remaining_decode_tokens > 0);
}

std::string format_ms(double value) {
  if (!std::isfinite(value)) {
    return "inf";
  }
  std::ostringstream oss;
  oss << value;
  return oss.str();
}

}  // namespace

ForecastRequest RealtimeServingRuntime::shadow_request_from_runtime_request(const RuntimeRequest& req) const {
  ForecastRequest out;
  out.request_id = req.request_id;
  out.arrival_ms = req.arrival_ms;
  out.context_tokens = req.lin;
  out.generated_tokens = req.lout;
  out.route = req.route;
  out.svc = *req.svc;
  out.state = req.state;
  out.ready_ms = cycles_to_ms(req.ready_clk);
  out.remaining_decode_tokens = req.remaining_decode_tokens;
  out.decoded_tokens_done = req.decoded_tokens_done;
  out.decode_enqueue_seq = req.decode_enqueue_seq;
  out.prefill_start_ms = req.prefill_start_clk.has_value() ? cycles_to_ms(*req.prefill_start_clk) : std::optional<double>{};
  out.prefill_end_ms = req.prefill_end_clk.has_value() ? cycles_to_ms(*req.prefill_end_clk) : std::optional<double>{};
  out.completion_time_ms = req.completion_clk.has_value() ? cycles_to_ms(*req.completion_clk) : std::optional<double>{};
  out.last_token_completion_ms =
      req.last_token_completion_clk.has_value() ? cycles_to_ms(*req.last_token_completion_clk) : std::optional<double>{};
  out.deadline_ms = req.deadline_ms;
  out.is_lost = req.is_lost;
  for (Clk_t clk : req.tbt_intervals_clk) {
    out.tbt_intervals_ms.push_back(cycles_to_ms(clk));
  }
  return out;
}

ForecastRequest RealtimeServingRuntime::shadow_request_for_candidate(const SourceRequest& req,
                                                                     const std::string& route,
                                                                     const ServiceEstimate& est) const {
  ForecastRequest out;
  out.request_id = req.request_id;
  out.arrival_ms = req.arrival_ms;
  out.context_tokens = req.context_tokens;
  out.generated_tokens = req.generated_tokens;
  out.route = route;
  out.svc = est;
  out.state = RequestState::WaitingPrefill;
  out.ready_ms = cycles_to_ms(m_clk);
  out.remaining_decode_tokens = est.decode_tokens;
  out.decoded_tokens_done = 0;
  return out;
}

ForecastRequest RealtimeServingRuntime::incremental_request_for_waiting_request(const RuntimeRequest& req,
                                                                                const std::string& route,
                                                                                const ServiceEstimate& est) const {
  ForecastRequest out;
  out.request_id = req.request_id;
  out.arrival_ms = req.arrival_ms;
  out.context_tokens = req.lin;
  out.generated_tokens = req.lout;
  out.route = route;
  out.svc = est;
  out.state = RequestState::WaitingPrefill;
  out.ready_ms = cycles_to_ms(m_clk);
  out.remaining_decode_tokens = est.decode_tokens;
  out.decoded_tokens_done = 0;
  return out;
}

ForecastRun RealtimeServingRuntime::simulate_prefill_priority_fcfs_decode_incremental(
    std::vector<ForecastRequest> forecast) const {
  ForecastRun run;
  double now = cycles_to_ms(m_clk);
  double gpu_free = cycles_to_ms(m_gpu_busy_until);
  double pim_free = cycles_to_ms(m_pim_busy_until);
  uint64_t next_decode_seq = m_next_decode_enqueue_seq;
  for (const auto& req : forecast) {
    if (req.decode_enqueue_seq.has_value()) {
      next_decode_seq = std::max(next_decode_seq, *req.decode_enqueue_seq + 1);
    }
  }

  auto waiting_prefills = [&]() {
    std::vector<int> idxs;
    for (int i = 0; i < static_cast<int>(forecast.size()); ++i) {
      if (forecast[i].state == RequestState::WaitingPrefill) {
        idxs.push_back(i);
      }
    }
    std::sort(idxs.begin(), idxs.end(), [&](int lhs, int rhs) {
      return std::tie(forecast[lhs].ready_ms, forecast[lhs].arrival_ms, forecast[lhs].request_id) <
             std::tie(forecast[rhs].ready_ms, forecast[rhs].arrival_ms, forecast[rhs].request_id);
    });
    return idxs;
  };

  auto waiting_decodes = [&]() {
    std::vector<int> idxs;
    for (int i = 0; i < static_cast<int>(forecast.size()); ++i) {
      if (forecast[i].state == RequestState::WaitingDecode &&
          forecast[i].remaining_decode_tokens > 0) {
        if (!forecast[i].decode_enqueue_seq.has_value()) {
          forecast[i].decode_enqueue_seq = next_decode_seq++;
        }
        idxs.push_back(i);
      }
    }
    std::sort(idxs.begin(), idxs.end(), [&](int lhs, int rhs) {
      return std::tie(*forecast[lhs].decode_enqueue_seq, forecast[lhs].request_id) <
             std::tie(*forecast[rhs].decode_enqueue_seq, forecast[rhs].request_id);
    });
    return idxs;
  };

  while (true) {
    auto prefills = waiting_prefills();
    if (!prefills.empty()) {
      auto& rr = forecast[prefills.front()];
      const double start = std::max(
          rr.ready_ms,
          std::max(rr.svc.prefill_gpu_ms > 0.0 ? gpu_free : rr.ready_ms,
                   rr.svc.prefill_pim_ms > 0.0 ? pim_free : rr.ready_ms));
      if (!std::isfinite(run.first_task_start_ms)) {
        run.first_task_start_ms = start;
      }
      const double finish = start + rr.svc.prefill_e2e_ms;
      now = std::max(now, start);
      run.total_prefill_energy_nj += std::max(0.0, rr.svc.prefill_energy_nj);
      if (rr.svc.prefill_gpu_ms > 0.0) {
        gpu_free = std::max(gpu_free, start) + rr.svc.prefill_gpu_ms;
      }
      if (rr.svc.prefill_pim_ms > 0.0) {
        pim_free = std::max(pim_free, start) + rr.svc.prefill_pim_ms;
      }
      if (!rr.prefill_start_ms.has_value()) {
        rr.prefill_start_ms = start;
      }
      rr.prefill_end_ms = finish;
      rr.last_token_completion_ms = finish;
      if (rr.remaining_decode_tokens > 0) {
        rr.state = RequestState::WaitingDecode;
        rr.ready_ms = finish;
        if (!rr.decode_enqueue_seq.has_value()) {
          rr.decode_enqueue_seq = next_decode_seq++;
        }
      } else {
        rr.state = RequestState::Done;
        rr.completion_time_ms = finish;
      }
      continue;
    }

    auto decodes = waiting_decodes();
    if (decodes.empty()) {
      break;
    }

    const auto& head = forecast[decodes.front()];
    const double cutoff = std::max(now, head.ready_ms);
    std::vector<int> batch = {decodes.front()};
    if (m_config.enable_decode_batching) {
      for (size_t i = 1; i < decodes.size(); ++i) {
        const auto& rr = forecast[decodes[i]];
        if (static_cast<int>(batch.size()) >= m_config.max_decode_batch_size) {
          break;
        }
        if (rr.route != head.route || rr.ready_ms > cutoff) {
          break;
        }
        batch.push_back(decodes[i]);
      }
    }

    std::vector<int> batch_reqs;
    double gpu_ms = 0.0;
    double pim_ms = 0.0;
    double e2e_ms = 0.0;
    double decode_energy_nj = 0.0;
    for (int batch_size = static_cast<int>(batch.size()); batch_size >= 1; --batch_size) {
      batch_reqs.assign(batch.begin(), batch.begin() + batch_size);
      if (batch_size == 1) {
        const auto& svc = forecast[batch_reqs.front()].svc;
        gpu_ms = svc.decode_gpu_ms;
        pim_ms = svc.decode_pim_ms;
        e2e_ms = svc.decode_e2e_ms;
        decode_energy_nj = svc.decode_energy_nj;
        break;
      }
      int batch_lin = 0;
      for (int idx : batch_reqs) {
        batch_lin = std::max(batch_lin, request_decode_context_tokens(forecast[idx]));
      }
      auto batch_est = const_cast<RealtimeServingRuntime*>(this)->estimate_request_cached(
          head.route, batch_lin, 2, batch_size);
      if (!batch_est.has_value()) {
        continue;
      }
      gpu_ms = batch_est->decode_gpu_ms;
      pim_ms = batch_est->decode_pim_ms;
      e2e_ms = batch_est->decode_e2e_ms;
      decode_energy_nj = batch_est->decode_energy_nj;
      break;
    }
    if (batch_reqs.empty()) {
      break;
    }

    const double start = std::max(cutoff,
                                  std::max(gpu_ms > 0.0 ? gpu_free : cutoff,
                                           pim_ms > 0.0 ? pim_free : cutoff));
    if (!std::isfinite(run.first_task_start_ms)) {
      run.first_task_start_ms = start;
    }
    const double finish = start + e2e_ms;
    now = std::max(now, start);
    run.total_decode_energy_nj += std::max(0.0, decode_energy_nj);
    if (gpu_ms > 0.0) {
      gpu_free = std::max(gpu_free, start) + gpu_ms;
    }
    if (pim_ms > 0.0) {
      pim_free = std::max(pim_free, start) + pim_ms;
    }
    for (int idx : batch_reqs) {
      auto& rr = forecast[idx];
      if (!rr.first_decode_start_ms.has_value()) {
        rr.first_decode_start_ms = start;
      }
      if (rr.last_token_completion_ms.has_value()) {
        rr.tbt_intervals_ms.push_back(finish - *rr.last_token_completion_ms);
      }
      rr.last_token_completion_ms = finish;
      rr.remaining_decode_tokens -= 1;
      rr.decoded_tokens_done += 1;
      if (rr.remaining_decode_tokens <= 0) {
        rr.state = RequestState::Done;
        rr.completion_time_ms = finish;
      } else {
        rr.state = RequestState::WaitingDecode;
        rr.ready_ms = finish;
      }
    }
  }

  for (const auto& rr : forecast) {
    run.requests[rr.request_id] = {
        {"completion_time_ms", rr.completion_time_ms.has_value() ? *rr.completion_time_ms
                                                                  : std::numeric_limits<double>::infinity()},
        {"prefill_start_ms", opt_or_nan(rr.prefill_start_ms)},
        {"prefill_end_ms", opt_or_nan(rr.prefill_end_ms)},
        {"first_decode_start_ms", opt_or_nan(rr.first_decode_start_ms)},
        {"mean_tbt_ms", rr.tbt_intervals_ms.empty()
                            ? rr.svc.decode_e2e_ms
                            : std::accumulate(rr.tbt_intervals_ms.begin(), rr.tbt_intervals_ms.end(), 0.0) /
                                  static_cast<double>(rr.tbt_intervals_ms.size())},
    };
  }
  return run;
}

std::unordered_map<int, std::unordered_map<std::string, double>>
RealtimeServingRuntime::simulate_prefill_priority_fcfs_decode_shadow(std::vector<ForecastRequest> shadow) const {
  return simulate_prefill_priority_fcfs_decode_incremental(std::move(shadow)).requests;
}

std::unordered_map<int, std::unordered_map<std::string, double>>
RealtimeServingRuntime::simulate_prefill_priority_fcfs_decode_heuristic(std::vector<ForecastRequest> forecast) const {
  double now = cycles_to_ms(m_clk);
  double gpu_free = cycles_to_ms(m_gpu_busy_until);
  double pim_free = cycles_to_ms(m_pim_busy_until);
  uint64_t next_decode_seq = m_next_decode_enqueue_seq;
  for (const auto& req : forecast) {
    if (req.decode_enqueue_seq.has_value()) {
      next_decode_seq = std::max(next_decode_seq, *req.decode_enqueue_seq + 1);
    }
  }

  std::vector<int> prefills;
  for (int i = 0; i < static_cast<int>(forecast.size()); ++i) {
    if (forecast[i].state == RequestState::WaitingPrefill) {
      prefills.push_back(i);
    }
  }
  std::sort(prefills.begin(), prefills.end(), [&](int lhs, int rhs) {
    return std::tie(forecast[lhs].ready_ms, forecast[lhs].arrival_ms, forecast[lhs].request_id) <
           std::tie(forecast[rhs].ready_ms, forecast[rhs].arrival_ms, forecast[rhs].request_id);
  });

  for (int idx : prefills) {
    auto& rr = forecast[idx];
    const double start = std::max(
        rr.ready_ms,
        std::max(rr.svc.prefill_gpu_ms > 0.0 ? gpu_free : rr.ready_ms,
                 rr.svc.prefill_pim_ms > 0.0 ? pim_free : rr.ready_ms));
    const double finish = start + rr.svc.prefill_e2e_ms;
    now = std::max(now, start);
    if (rr.svc.prefill_gpu_ms > 0.0) {
      gpu_free = std::max(gpu_free, start) + rr.svc.prefill_gpu_ms;
    }
    if (rr.svc.prefill_pim_ms > 0.0) {
      pim_free = std::max(pim_free, start) + rr.svc.prefill_pim_ms;
    }
    rr.prefill_start_ms = start;
    rr.prefill_end_ms = finish;
    rr.last_token_completion_ms = finish;
    if (rr.remaining_decode_tokens > 0) {
      rr.state = RequestState::WaitingDecode;
      rr.ready_ms = finish;
      if (!rr.decode_enqueue_seq.has_value()) {
        rr.decode_enqueue_seq = next_decode_seq++;
      }
    } else {
      rr.state = RequestState::Done;
      rr.completion_time_ms = finish;
    }
  }

  std::vector<int> decodes;
  for (int i = 0; i < static_cast<int>(forecast.size()); ++i) {
    if (forecast[i].state == RequestState::WaitingDecode &&
        forecast[i].remaining_decode_tokens > 0) {
      if (!forecast[i].decode_enqueue_seq.has_value()) {
        forecast[i].decode_enqueue_seq = next_decode_seq++;
      }
      decodes.push_back(i);
    }
  }
  std::sort(decodes.begin(), decodes.end(), [&](int lhs, int rhs) {
    return std::tie(*forecast[lhs].decode_enqueue_seq, forecast[lhs].request_id) <
           std::tie(*forecast[rhs].decode_enqueue_seq, forecast[rhs].request_id);
  });

  double current_time = now;
  size_t pos = 0;
  while (pos < decodes.size()) {
    std::vector<int> block = {decodes[pos++]};
    while (pos < decodes.size() && forecast[decodes[pos]].route == forecast[block.front()].route) {
      block.push_back(decodes[pos++]);
    }

    auto heuristic_step = [&](const std::vector<int>& idxs) {
      if (!m_config.enable_decode_batching) {
        const auto& svc = forecast[idxs.front()].svc;
        return std::make_tuple(svc.decode_gpu_ms, svc.decode_pim_ms, svc.decode_e2e_ms, 1);
      }
      const int max_batch = std::min(m_config.max_decode_batch_size, static_cast<int>(idxs.size()));
      int batch_lin = 0;
      for (int idx : idxs) {
        batch_lin = std::max(batch_lin, request_decode_context_tokens(forecast[idx]));
      }
      for (int batch_size = max_batch; batch_size >= 1; --batch_size) {
        if (batch_size == 1) {
          const auto& svc = forecast[idxs.front()].svc;
          return std::make_tuple(svc.decode_gpu_ms, svc.decode_pim_ms, svc.decode_e2e_ms, 1);
        }
        auto batch_est = const_cast<RealtimeServingRuntime*>(this)->estimate_request_cached(
            forecast[idxs.front()].route, batch_lin, 2, batch_size);
        if (batch_est.has_value()) {
          return std::make_tuple(batch_est->decode_gpu_ms,
                                 batch_est->decode_pim_ms,
                                 batch_est->decode_e2e_ms,
                                 batch_size);
        }
      }
      const auto& svc = forecast[idxs.front()].svc;
      return std::make_tuple(svc.decode_gpu_ms, svc.decode_pim_ms, svc.decode_e2e_ms, 1);
    };

    auto [step_gpu_ms, step_pim_ms, step_e2e_ms, step_batch_size] = heuristic_step(block);
    const double head_ready_ms = forecast[block.front()].ready_ms;
    const double block_start_ms = std::max(
        std::max(current_time, head_ready_ms),
        std::max(step_gpu_ms > 0.0 ? gpu_free : head_ready_ms,
                 step_pim_ms > 0.0 ? pim_free : head_ready_ms));

    int busy_rounds = 0;
    double block_finish_ms = block_start_ms;
    for (int i = 0; i < static_cast<int>(block.size()); ++i) {
      auto& rr = forecast[block[i]];
      const int excess_rounds = (step_batch_size <= 0) ? 0 : std::max(0, i - step_batch_size + 1);
      const double first_decode_start_ms = std::max(block_start_ms + excess_rounds * step_e2e_ms, rr.ready_ms);
      rr.first_decode_start_ms = first_decode_start_ms;
      const double finish_ms = first_decode_start_ms + rr.remaining_decode_tokens * step_e2e_ms;
      rr.completion_time_ms = finish_ms;
      rr.last_token_completion_ms = finish_ms;
      busy_rounds = std::max(busy_rounds, rr.remaining_decode_tokens + excess_rounds);
      block_finish_ms = std::max(block_finish_ms, finish_ms);
    }
    if (step_gpu_ms > 0.0) {
      gpu_free = std::max(gpu_free, block_start_ms) + busy_rounds * step_gpu_ms;
    }
    if (step_pim_ms > 0.0) {
      pim_free = std::max(pim_free, block_start_ms) + busy_rounds * step_pim_ms;
    }
    current_time = std::max(current_time, block_finish_ms);
  }

  std::unordered_map<int, std::unordered_map<std::string, double>> results;
  for (const auto& rr : forecast) {
    results[rr.request_id] = {
        {"completion_time_ms", rr.completion_time_ms.has_value() ? *rr.completion_time_ms
                                                                  : std::numeric_limits<double>::infinity()},
        {"prefill_start_ms", opt_or_nan(rr.prefill_start_ms)},
        {"prefill_end_ms", opt_or_nan(rr.prefill_end_ms)},
        {"first_decode_start_ms", opt_or_nan(rr.first_decode_start_ms)},
    };
  }
  return results;
}

ForecastRun RealtimeServingRuntime::incremental_active_forecast_results() {
  std::vector<RuntimeRequest> active;
  for (const auto& req : m_requests) {
    if (req.svc.has_value() && !req.route.empty() &&
        (req.state == RequestState::WaitingPrefill ||
         (req.state == RequestState::WaitingDecode && req.remaining_decode_tokens > 0))) {
      active.push_back(req);
    }
  }
  if (active.empty()) {
    return {};
  }

  std::ostringstream sig;
  sig << cycles_to_ms(m_gpu_busy_until) << '|' << cycles_to_ms(m_pim_busy_until) << '|'
      << m_next_decode_enqueue_seq;
  for (const auto& req : active) {
    sig << ';' << req.request_id << ':' << req.arrival_ms << ':' << req.lin << ':' << req.lout
        << ':' << req.route << ':' << state_to_string(req.state) << ':' << cycles_to_ms(req.ready_clk)
        << ':' << req.remaining_decode_tokens << ':' << req.decoded_tokens_done << ':';
    if (req.decode_enqueue_seq.has_value()) {
      sig << *req.decode_enqueue_seq;
    }
  }
  const std::string key = sig.str();
  if (m_incremental_active_forecast_cache_key.has_value() &&
      *m_incremental_active_forecast_cache_key == key &&
      m_incremental_active_forecast_cache_run.has_value()) {
    m_incremental_active_forecast_cache_hits++;
    return *m_incremental_active_forecast_cache_run;
  }
  m_incremental_active_forecast_cache_misses++;
  std::vector<ForecastRequest> forecast;
  forecast.reserve(active.size());
  for (const auto& req : active) {
    forecast.push_back(shadow_request_from_runtime_request(req));
  }
  ForecastRun run = simulate_prefill_priority_fcfs_decode_incremental(std::move(forecast));
  m_incremental_active_forecast_cache_key = key;
  m_incremental_active_forecast_cache_run = run;
  return run;
}

std::unordered_map<std::string, double>
RealtimeServingRuntime::project_prefill_priority_fcfs_decode_candidate(const SourceRequest& req,
                                                                       const std::string& route,
                                                                       const ServiceEstimate& est) {
  std::vector<ForecastRequest> shadow;
  for (const auto& rr : m_requests) {
    if (rr.svc.has_value() && !rr.route.empty() &&
        (rr.state == RequestState::WaitingPrefill ||
         (rr.state == RequestState::WaitingDecode && rr.remaining_decode_tokens > 0))) {
      shadow.push_back(shadow_request_from_runtime_request(rr));
    }
  }
  shadow.push_back(shadow_request_for_candidate(req, route, est));
  const auto results = simulate_prefill_priority_fcfs_decode_shadow(std::move(shadow));
  const auto it = results.find(req.request_id);
  const double prefill_start = (it != results.end()) ? it->second.at("prefill_start_ms") : std::numeric_limits<double>::quiet_NaN();
  const double prefill_end = (it != results.end()) ? it->second.at("prefill_end_ms") : std::numeric_limits<double>::quiet_NaN();
  const double first_decode_start = (it != results.end()) ? it->second.at("first_decode_start_ms") : std::numeric_limits<double>::quiet_NaN();
  const double finish = (it != results.end()) ? it->second.at("completion_time_ms") : std::numeric_limits<double>::infinity();
  return {
      {"candidate_finish_ms", finish},
      {"candidate_pred_gpu_wait_ms", std::isnan(prefill_start) ? 0.0 : std::max(0.0, prefill_start - req.arrival_ms)},
      {"candidate_pred_pim_wait_ms",
       (est.uses_pim && !std::isnan(prefill_end) && !std::isnan(first_decode_start))
           ? std::max(0.0, first_decode_start - prefill_end)
           : 0.0},
  };
}

std::unordered_map<std::string, double>
RealtimeServingRuntime::project_prefill_priority_fcfs_decode_incremental_candidate(
    const SourceRequest& req,
    const std::string& route,
    const ServiceEstimate& est,
    std::optional<double> baseline_total_energy_nj,
    std::optional<double> baseline_total_decode_energy_nj) {
  m_projection_cache_misses++;
  std::vector<ForecastRequest> forecast;
  for (const auto& rr : m_requests) {
    if (rr.svc.has_value() && !rr.route.empty() &&
        (rr.state == RequestState::WaitingPrefill ||
         (rr.state == RequestState::WaitingDecode && rr.remaining_decode_tokens > 0))) {
      forecast.push_back(shadow_request_from_runtime_request(rr));
    }
  }
  forecast.push_back(shadow_request_for_candidate(req, route, est));
  const ForecastRun run = simulate_prefill_priority_fcfs_decode_incremental(std::move(forecast));
  const auto it = run.requests.find(req.request_id);
  const double prefill_start = (it != run.requests.end()) ? it->second.at("prefill_start_ms") : std::numeric_limits<double>::quiet_NaN();
  const double prefill_end = (it != run.requests.end()) ? it->second.at("prefill_end_ms") : std::numeric_limits<double>::quiet_NaN();
  const double first_decode_start = (it != run.requests.end()) ? it->second.at("first_decode_start_ms") : std::numeric_limits<double>::quiet_NaN();
  const double finish = (it != run.requests.end()) ? it->second.at("completion_time_ms") : std::numeric_limits<double>::infinity();
  const double base_energy = baseline_total_energy_nj.value_or(0.0);
  const double base_decode_energy = baseline_total_decode_energy_nj.value_or(0.0);
  return {
      {"candidate_finish_ms", finish},
      {"candidate_pred_gpu_wait_ms", std::isnan(prefill_start) ? 0.0 : std::max(0.0, prefill_start - req.arrival_ms)},
      {"candidate_pred_pim_wait_ms",
       (est.uses_pim && !std::isnan(prefill_end) && !std::isnan(first_decode_start))
           ? std::max(0.0, first_decode_start - prefill_end)
           : 0.0},
      {"candidate_incremental_energy_nj",
       std::max(0.0, run.total_prefill_energy_nj + run.total_decode_energy_nj - base_energy)},
      {"candidate_incremental_decode_energy_nj",
       std::max(0.0, run.total_decode_energy_nj - base_decode_energy)},
  };
}

std::unordered_map<std::string, double>
RealtimeServingRuntime::project_prefill_priority_fcfs_decode_heuristic_candidate(
    const SourceRequest& req, const std::string& route, const ServiceEstimate& est) {
  std::vector<ForecastRequest> forecast;
  for (const auto& rr : m_requests) {
    if (rr.svc.has_value() && !rr.route.empty() &&
        (rr.state == RequestState::WaitingPrefill ||
         (rr.state == RequestState::WaitingDecode && rr.remaining_decode_tokens > 0))) {
      forecast.push_back(shadow_request_from_runtime_request(rr));
    }
  }
  forecast.push_back(shadow_request_for_candidate(req, route, est));
  const auto results = simulate_prefill_priority_fcfs_decode_heuristic(std::move(forecast));
  const auto it = results.find(req.request_id);
  const double prefill_start = (it != results.end()) ? it->second.at("prefill_start_ms") : std::numeric_limits<double>::quiet_NaN();
  const double prefill_end = (it != results.end()) ? it->second.at("prefill_end_ms") : std::numeric_limits<double>::quiet_NaN();
  const double first_decode_start = (it != results.end()) ? it->second.at("first_decode_start_ms") : std::numeric_limits<double>::quiet_NaN();
  double finish = (it != results.end()) ? it->second.at("completion_time_ms") : std::numeric_limits<double>::infinity();
  finish += queue_pressure_adjustment_ms(est.prefill_gpu_ms,
                                         est.prefill_pim_ms,
                                         est.prefill_e2e_ms,
                                         est.decode_gpu_ms,
                                         est.decode_pim_ms,
                                         est.decode_e2e_ms,
                                         est.decode_tokens);
  return {
      {"candidate_finish_ms", finish},
      {"candidate_pred_gpu_wait_ms", std::isnan(prefill_start) ? 0.0 : std::max(0.0, prefill_start - req.arrival_ms)},
      {"candidate_pred_pim_wait_ms",
       (est.uses_pim && !std::isnan(prefill_end) && !std::isnan(first_decode_start))
           ? std::max(0.0, first_decode_start - prefill_end)
           : 0.0},
  };
}

std::unordered_map<std::string, double>
RealtimeServingRuntime::project_prefill_priority_fcfs_decode_heuristic_decode_candidate(
    const RuntimeRequest& req, const std::string& route, const ServiceEstimate& est, double decision_time_ms) {
  std::vector<ForecastRequest> forecast;
  for (const auto& other : m_requests) {
    if (other.request_id == req.request_id || !other.svc.has_value() || other.route.empty()) {
      continue;
    }
    if (other.state == RequestState::WaitingPrefill ||
        (other.state == RequestState::WaitingDecode && other.remaining_decode_tokens > 0)) {
      forecast.push_back(shadow_request_from_runtime_request(other));
    }
  }
  ForecastRequest cand;
  cand.request_id = req.request_id;
  cand.arrival_ms = req.arrival_ms;
  cand.context_tokens = req.lin;
  cand.generated_tokens = req.lout;
  cand.route = route;
  cand.svc = est;
  cand.state = RequestState::WaitingDecode;
  cand.ready_ms = decision_time_ms;
  cand.remaining_decode_tokens = req.remaining_decode_tokens;
  cand.decoded_tokens_done = req.decoded_tokens_done;
  cand.decode_enqueue_seq = req.decode_enqueue_seq;
  cand.prefill_start_ms = req.prefill_start_clk.has_value() ? cycles_to_ms(*req.prefill_start_clk) : std::optional<double>{};
  cand.prefill_end_ms = req.prefill_end_clk.has_value() ? cycles_to_ms(*req.prefill_end_clk) : std::optional<double>{};
  forecast.push_back(cand);
  const auto results = simulate_prefill_priority_fcfs_decode_heuristic(std::move(forecast));
  const auto it = results.find(req.request_id);
  const double first_decode_start = (it != results.end()) ? it->second.at("first_decode_start_ms") : std::numeric_limits<double>::quiet_NaN();
  double finish = (it != results.end()) ? it->second.at("completion_time_ms") : std::numeric_limits<double>::infinity();
  finish += queue_pressure_adjustment_ms(0.0,
                                         0.0,
                                         0.0,
                                         est.decode_gpu_ms,
                                         est.decode_pim_ms,
                                         est.decode_e2e_ms,
                                         req.remaining_decode_tokens);
  const double decode_wait = std::isnan(first_decode_start) ? 0.0 : std::max(0.0, first_decode_start - decision_time_ms);
  return {
      {"candidate_finish_ms", finish},
      {"candidate_pred_gpu_wait_ms", est.decode_gpu_ms > 0.0 ? decode_wait : 0.0},
      {"candidate_pred_pim_wait_ms", est.uses_pim ? decode_wait : 0.0},
  };
}

std::unordered_map<std::string, double>
RealtimeServingRuntime::effective_slo_guard_budgets(const RuntimeRequest& req) const {
  const double queue_wait_ms = std::max(0.0, cycles_to_ms(m_clk) - req.arrival_ms);
  return {
      {"max_harmed_requests",
       static_cast<double>(m_config.admission_max_harmed_requests) +
           queue_wait_ms * m_config.admission_aging_harmed_requests_per_ms},
      {"max_total_harm_ms",
       m_config.admission_max_total_harm_ms +
           queue_wait_ms * m_config.admission_aging_harm_ms_per_ms},
      {"max_single_harm_ms",
       m_config.admission_max_single_harm_ms +
           queue_wait_ms * m_config.admission_aging_harm_ms_per_ms},
      {"max_own_miss_ms", m_config.admission_max_own_miss_ms},
  };
}

RouteCandidate RealtimeServingRuntime::evaluate_slo_guarded_route(
    RuntimeRequest& req,
    const std::string& route,
    const ServiceEstimate& est,
    const ForecastRun& baseline_run,
    const std::unordered_map<int, std::unordered_map<std::string, double>>& baseline_results) {
  std::vector<ForecastRequest> forecast;
  for (const auto& rr : m_requests) {
    if (rr.svc.has_value() && !rr.route.empty() &&
        (rr.state == RequestState::WaitingPrefill ||
         (rr.state == RequestState::WaitingDecode && rr.remaining_decode_tokens > 0))) {
      forecast.push_back(shadow_request_from_runtime_request(rr));
    }
  }
  ForecastRequest candidate = incremental_request_for_waiting_request(req, route, est);
  forecast.push_back(candidate);
  const ForecastRun projected_run = simulate_prefill_priority_fcfs_decode_incremental(std::move(forecast));
  const auto projected = projected_run.requests;
  const auto out = projected.at(req.request_id);
  const double finish = out.at("completion_time_ms");
  const double prefill_start = out.at("prefill_start_ms");
  const double prefill_end = out.at("prefill_end_ms");
  const double first_decode_start = out.at("first_decode_start_ms");
  const double candidate_mean_tbt = out.at("mean_tbt_ms");
  const double pred_gpu_wait = std::isnan(prefill_start) ? 0.0 : std::max(0.0, prefill_start - cycles_to_ms(m_clk));
  const double pred_pim_wait =
      (est.uses_pim && !std::isnan(prefill_end) && !std::isnan(first_decode_start))
          ? std::max(0.0, first_decode_start - prefill_end)
          : 0.0;

  double own_miss_ms = 0.0;
  std::optional<double> slack_ms;
  if (req.deadline_ms.has_value()) {
    own_miss_ms = std::max(0.0, finish - *req.deadline_ms);
    slack_ms = *req.deadline_ms - finish;
  }

  int harmed_count = 0;
  double total_harm_ms = 0.0;
  double max_single_harm_ms = 0.0;
  for (const auto& active : m_requests) {
    if (!(active.state == RequestState::WaitingPrefill ||
          (active.state == RequestState::WaitingDecode && active.remaining_decode_tokens > 0)) ||
        !active.svc.has_value() || active.route.empty() || !active.deadline_ms.has_value() || active.is_lost) {
      continue;
    }
    const double before_finish = baseline_results.count(active.request_id)
                                     ? baseline_results.at(active.request_id).at("completion_time_ms")
                                     : std::numeric_limits<double>::infinity();
    const double after_finish = projected.count(active.request_id)
                                    ? projected.at(active.request_id).at("completion_time_ms")
                                    : std::numeric_limits<double>::infinity();
    const double before_miss = std::max(0.0, before_finish - *active.deadline_ms);
    const double after_miss = std::max(0.0, after_finish - *active.deadline_ms);
    const double harm_ms = std::max(0.0, after_miss - before_miss);
    if (harm_ms > 0.0) {
      harmed_count++;
      total_harm_ms += harm_ms;
      max_single_harm_ms = std::max(max_single_harm_ms, harm_ms);
    }
  }

  const auto budgets = effective_slo_guard_budgets(req);
  const bool own_checks_enabled = m_config.admission_check_own_slo;
  const bool own_ok = !own_checks_enabled || own_miss_ms <= budgets.at("max_own_miss_ms");
  const bool tbt_ok = !own_checks_enabled || m_config.slo_tbt_ms < 0.0 || candidate_mean_tbt <= m_config.slo_tbt_ms;
  const bool harmed_ok = harmed_count <= static_cast<int>(std::floor(budgets.at("max_harmed_requests") + 1e-9));
  const bool total_harm_ok = total_harm_ms <= budgets.at("max_total_harm_ms");
  const bool single_harm_ok = max_single_harm_ms <= budgets.at("max_single_harm_ms");
  const bool eligible = own_ok && tbt_ok && harmed_ok && total_harm_ok && single_harm_ok;

  RouteCandidate cand;
  cand.route = route;
  cand.uses_pim = est.uses_pim;
  cand.svc = est;
  cand.predicted_finish_ms = finish;
  cand.predicted_gpu_wait_ms = pred_gpu_wait;
  cand.predicted_pim_wait_ms = pred_pim_wait;
  cand.predicted_incremental_energy_nj =
      std::max(0.0, projected_run.total_prefill_energy_nj + projected_run.total_decode_energy_nj -
                        (baseline_run.total_prefill_energy_nj + baseline_run.total_decode_energy_nj));
  cand.predicted_incremental_decode_energy_nj =
      std::max(0.0, projected_run.total_decode_energy_nj - baseline_run.total_decode_energy_nj);
  cand.deadline_ms = req.deadline_ms;
  cand.slack_ms = slack_ms;
  cand.candidate_mean_tbt_ms = candidate_mean_tbt;
  cand.harmed_count = harmed_count;
  cand.total_harm_ms = total_harm_ms;
  cand.max_single_harm_ms = max_single_harm_ms;
  cand.own_miss_ms = own_miss_ms;
  cand.eligible = eligible;
  if (cand.uses_pim && eligible) {
    const auto fit = predict_hybrid_request_memory_fit(
        static_cast<uint32_t>(req.request_id), req.lin, est);
    if (!fit.success) {
      cand.eligible = false;
      cand.reason = fit.reason;
      annotate_route_candidate(cand);
      return cand;
    }
  }
  if (own_checks_enabled && !own_ok) {
    cand.reason = "candidate_e2e_slo_miss";
  } else if (own_checks_enabled && !tbt_ok) {
    cand.reason = "candidate_tbt_slo_miss";
  } else if (!harmed_ok) {
    cand.reason = "existing_harmed_count_exceeded";
  } else if (!total_harm_ok) {
    cand.reason = "existing_total_harm_exceeded";
  } else if (!single_harm_ok) {
    cand.reason = "existing_single_harm_exceeded";
  }
  annotate_route_candidate(cand);
  return cand;
}

std::unordered_map<std::string, double>
RealtimeServingRuntime::project_admission_candidate(RuntimeRequest& req,
                                                    const std::string& route,
                                                    const ServiceEstimate& est) const {
  std::vector<ForecastRequest> shadow;
  for (const auto& rr : m_requests) {
    if ((rr.state != RequestState::WaitingPrefill &&
         !(rr.state == RequestState::WaitingDecode && rr.remaining_decode_tokens > 0)) ||
        !rr.svc.has_value() || rr.route.empty()) {
      continue;
    }
    shadow.push_back(shadow_request_from_runtime_request(rr));
  }
  ForecastRequest target = incremental_request_for_waiting_request(req, route, est);
  target.route = route;
  target.svc = est;
  shadow.push_back(target);
  const int target_id = req.request_id;

  double now = cycles_to_ms(m_clk);
  double gpu_free = cycles_to_ms(m_gpu_busy_until);
  double pim_free = cycles_to_ms(m_pim_busy_until);
  int steps = 0;
  int consecutive_decode_batches = 0;
  bool timeout = false;
  std::optional<int> shadow_fcfs_active_request_id;

  auto has_waiting_prefill = [&]() {
    for (const auto& r : shadow) {
      if (r.state == RequestState::WaitingPrefill) {
        return true;
      }
    }
    return false;
  };
  auto shadow_is_active = [&](const ForecastRequest& r) {
    return r.state == RequestState::WaitingPrefill ||
           (r.state == RequestState::WaitingDecode && r.remaining_decode_tokens > 0);
  };
  auto shadow_fcfs_active = [&]() -> std::optional<int> {
    if (shadow_fcfs_active_request_id.has_value()) {
      for (int i = 0; i < static_cast<int>(shadow.size()); ++i) {
        if (shadow[i].request_id == *shadow_fcfs_active_request_id && shadow_is_active(shadow[i])) {
          return i;
        }
      }
      shadow_fcfs_active_request_id.reset();
    }
    std::vector<int> active;
    for (int i = 0; i < static_cast<int>(shadow.size()); ++i) {
      if (shadow_is_active(shadow[i])) {
        active.push_back(i);
      }
    }
    if (active.empty()) {
      return std::nullopt;
    }
    std::sort(active.begin(), active.end(), [&](int lhs, int rhs) {
      const double lhs_admit = (shadow[lhs].completion_time_ms.has_value() ? *shadow[lhs].completion_time_ms
                                                                           : shadow[lhs].arrival_ms);
      const double rhs_admit = (shadow[rhs].completion_time_ms.has_value() ? *shadow[rhs].completion_time_ms
                                                                           : shadow[rhs].arrival_ms);
      return std::tie(lhs_admit, shadow[lhs].request_id) < std::tie(rhs_admit, shadow[rhs].request_id);
    });
    shadow_fcfs_active_request_id = shadow[active.front()].request_id;
    return active.front();
  };
  auto build_prefill_tasks = [&]() {
    std::vector<TaskCandidate> tasks;
    for (int i = 0; i < static_cast<int>(shadow.size()); ++i) {
      if (shadow[i].state != RequestState::WaitingPrefill) {
        continue;
      }
      TaskCandidate tc;
      tc.phase = TaskPhase::Prefill;
      tc.request_indices = {i};
      tc.route = shadow[i].route;
      tc.ready_clk = ms_to_cycles(shadow[i].ready_ms);
      tc.batch_size = 1;
      tc.earliest_start_clk = ms_to_cycles(std::max(
          shadow[i].ready_ms,
          std::max(shadow[i].svc.prefill_gpu_ms > 0.0 ? gpu_free : shadow[i].ready_ms,
                   shadow[i].svc.prefill_pim_ms > 0.0 ? pim_free : shadow[i].ready_ms)));
      tc.gpu_clk = ms_to_cycles(shadow[i].svc.prefill_gpu_ms);
      tc.pim_clk = ms_to_cycles(shadow[i].svc.prefill_pim_ms);
      tc.e2e_clk = ms_to_cycles(shadow[i].svc.prefill_e2e_ms);
      tasks.push_back(tc);
    }
    return tasks;
  };
  auto build_decode_tasks = [&]() {
    std::vector<TaskCandidate> tasks;
    if (m_config.enable_decode_batching) {
      std::unordered_map<std::string, std::vector<int>> grouped;
      for (int i = 0; i < static_cast<int>(shadow.size()); ++i) {
        if (shadow[i].state == RequestState::WaitingDecode &&
            shadow[i].remaining_decode_tokens > 0) {
          grouped[shadow[i].route].push_back(i);
        }
      }
      for (auto& [route_name, idxs] : grouped) {
        std::sort(idxs.begin(), idxs.end(), [&](int lhs, int rhs) {
          return std::tie(shadow[lhs].ready_ms, shadow[lhs].arrival_ms, shadow[lhs].request_id) <
                 std::tie(shadow[rhs].ready_ms, shadow[rhs].arrival_ms, shadow[rhs].request_id);
        });
        const double cutoff = std::max(now, shadow[idxs.front()].ready_ms);
        std::vector<int> cohort;
        for (int idx : idxs) {
          if (shadow[idx].ready_ms <= cutoff) {
            cohort.push_back(idx);
          }
        }
        int max_batch = std::min(static_cast<int>(cohort.size()), m_config.max_decode_batch_size);
        if (has_waiting_prefill() && m_config.decode_batch_cap_with_prefill > 0) {
          max_batch = std::min(max_batch, m_config.decode_batch_cap_with_prefill);
        }
        for (int bs = max_batch; bs >= 1; --bs) {
          const std::vector<int> batch(cohort.begin(), cohort.begin() + bs);
          ServiceEstimate batch_est;
          if (bs == 1) {
            batch_est = shadow[batch.front()].svc;
          } else {
            int batch_lin = 0;
            for (int idx : batch) {
              batch_lin = std::max(batch_lin, request_decode_context_tokens(shadow[idx]));
            }
            auto est_opt = const_cast<RealtimeServingRuntime*>(this)->estimate_request_cached(route_name, batch_lin, 2, bs);
            if (!est_opt.has_value()) {
              continue;
            }
            batch_est = *est_opt;
          }
          TaskCandidate tc;
          tc.phase = TaskPhase::Decode;
          tc.request_indices = batch;
          tc.route = route_name;
          tc.ready_clk = ms_to_cycles(cutoff);
          tc.batch_size = bs;
          tc.earliest_start_clk = ms_to_cycles(std::max(
              cutoff,
              std::max(batch_est.decode_gpu_ms > 0.0 ? gpu_free : cutoff,
                       batch_est.decode_pim_ms > 0.0 ? pim_free : cutoff)));
          tc.gpu_clk = ms_to_cycles(batch_est.decode_gpu_ms);
          tc.pim_clk = ms_to_cycles(batch_est.decode_pim_ms);
          tc.e2e_clk = ms_to_cycles(batch_est.decode_e2e_ms);
          tasks.push_back(tc);
          break;
        }
      }
    } else {
      for (int i = 0; i < static_cast<int>(shadow.size()); ++i) {
        if (shadow[i].state != RequestState::WaitingDecode || shadow[i].remaining_decode_tokens <= 0) {
          continue;
        }
        TaskCandidate tc;
        tc.phase = TaskPhase::Decode;
        tc.request_indices = {i};
        tc.route = shadow[i].route;
        tc.ready_clk = ms_to_cycles(shadow[i].ready_ms);
        tc.batch_size = 1;
        tc.earliest_start_clk = ms_to_cycles(std::max(
            shadow[i].ready_ms,
            std::max(shadow[i].svc.decode_gpu_ms > 0.0 ? gpu_free : shadow[i].ready_ms,
                     shadow[i].svc.decode_pim_ms > 0.0 ? pim_free : shadow[i].ready_ms)));
        tc.gpu_clk = ms_to_cycles(shadow[i].svc.decode_gpu_ms);
        tc.pim_clk = ms_to_cycles(shadow[i].svc.decode_pim_ms);
        tc.e2e_clk = ms_to_cycles(shadow[i].svc.decode_e2e_ms);
        tasks.push_back(tc);
      }
    }
    return tasks;
  };

  while (steps < m_config.admission_shadow_max_steps) {
    auto target_it = std::find_if(shadow.begin(), shadow.end(),
                                  [&](const ForecastRequest& r) { return r.request_id == target_id; });
    if (target_it == shadow.end()) {
      break;
    }
    bool pending_existing = false;
    for (const auto& r : shadow) {
      if (r.request_id == target_id || r.is_lost || !r.deadline_ms.has_value()) {
        continue;
      }
      if (!r.completion_time_ms.has_value()) {
        pending_existing = true;
        break;
      }
    }
    if (target_it->completion_time_ms.has_value() && !pending_existing) {
      break;
    }

    std::optional<TaskCandidate> chosen_task;
    if (is_fcfs_strict()) {
      const auto active_idx = shadow_fcfs_active();
      if (active_idx.has_value()) {
        TaskCandidate tc;
        tc.phase = (shadow[*active_idx].state == RequestState::WaitingPrefill) ? TaskPhase::Prefill : TaskPhase::Decode;
        tc.request_indices = {*active_idx};
        tc.route = shadow[*active_idx].route;
        tc.ready_clk = ms_to_cycles(shadow[*active_idx].ready_ms);
        tc.batch_size = 1;
        const auto& svc = shadow[*active_idx].svc;
        tc.gpu_clk = ms_to_cycles(tc.phase == TaskPhase::Prefill ? svc.prefill_gpu_ms : svc.decode_gpu_ms);
        tc.pim_clk = ms_to_cycles(tc.phase == TaskPhase::Prefill ? svc.prefill_pim_ms : svc.decode_pim_ms);
        tc.e2e_clk = ms_to_cycles(tc.phase == TaskPhase::Prefill ? svc.prefill_e2e_ms : svc.decode_e2e_ms);
        tc.earliest_start_clk = ms_to_cycles(std::max(
            shadow[*active_idx].ready_ms,
            std::max((tc.phase == TaskPhase::Prefill ? svc.prefill_gpu_ms : svc.decode_gpu_ms) > 0.0 ? gpu_free : shadow[*active_idx].ready_ms,
                     (tc.phase == TaskPhase::Prefill ? svc.prefill_pim_ms : svc.decode_pim_ms) > 0.0 ? pim_free : shadow[*active_idx].ready_ms)));
        chosen_task = tc;
      }
    } else {
      auto prefill_tasks = build_prefill_tasks();
      auto decode_tasks = build_decode_tasks();
      std::vector<TaskCandidate> cands = prefill_tasks;
      cands.insert(cands.end(), decode_tasks.begin(), decode_tasks.end());
      if (is_prefill_priority_fcfs_decode() && !prefill_tasks.empty()) {
        chosen_task = *std::min_element(prefill_tasks.begin(), prefill_tasks.end(),
                                        [](const TaskCandidate& lhs, const TaskCandidate& rhs) {
                                          return lhs.earliest_start_clk < rhs.earliest_start_clk;
                                        });
      } else if (!cands.empty()) {
        if (!prefill_tasks.empty()) {
          if (m_config.prefill_guard_ms >= 0.0) {
            double max_prefill_wait_ms = 0.0;
            for (const auto& tc : prefill_tasks) {
              max_prefill_wait_ms =
                  std::max(max_prefill_wait_ms, std::max(0.0, now - cycles_to_ms(tc.ready_clk)));
            }
            if (max_prefill_wait_ms >= m_config.prefill_guard_ms) {
              chosen_task = *std::min_element(prefill_tasks.begin(), prefill_tasks.end(),
                                              [](const TaskCandidate& lhs, const TaskCandidate& rhs) {
                                                return lhs.earliest_start_clk < rhs.earliest_start_clk;
                                              });
            }
          }
          if (!chosen_task.has_value() && m_config.max_consecutive_decode_batches > 0 &&
              consecutive_decode_batches >= m_config.max_consecutive_decode_batches) {
            chosen_task = *std::min_element(prefill_tasks.begin(), prefill_tasks.end(),
                                            [](const TaskCandidate& lhs, const TaskCandidate& rhs) {
                                              return lhs.earliest_start_clk < rhs.earliest_start_clk;
                                            });
          }
        }
        if (!chosen_task.has_value()) {
          auto key = [&](const TaskCandidate& tc) {
            int phase_prio = 0;
            if (m_config.prompt_priority) {
              phase_prio = (tc.phase == TaskPhase::Prefill) ? 0 : 1;
            }
            return std::make_tuple(cycles_to_ms(tc.earliest_start_clk),
                                   phase_prio,
                                   cycles_to_ms(tc.ready_clk),
                                   shadow[tc.request_indices.front()].request_id);
          };
          chosen_task = *std::min_element(cands.begin(), cands.end(),
                                          [&](const TaskCandidate& lhs, const TaskCandidate& rhs) {
                                            return key(lhs) < key(rhs);
                                          });
        }
      }
    }

    if (!chosen_task.has_value()) {
      break;
    }

    steps++;
    const double start = cycles_to_ms(chosen_task->earliest_start_clk);
    const double gpu_ms = cycles_to_ms(chosen_task->gpu_clk);
    const double pim_ms = cycles_to_ms(chosen_task->pim_clk);
    const double e2e_ms = cycles_to_ms(chosen_task->e2e_clk);
    const double finish = start + e2e_ms;
    now = std::max(now, start);
    if (gpu_ms > 0.0) {
      gpu_free = std::max(gpu_free, start) + gpu_ms;
    }
    if (pim_ms > 0.0) {
      pim_free = std::max(pim_free, start) + pim_ms;
    }

    if (chosen_task->phase == TaskPhase::Prefill) {
      consecutive_decode_batches = 0;
      for (int idx : chosen_task->request_indices) {
        auto& r = shadow[idx];
        if (!r.prefill_start_ms.has_value()) {
          r.prefill_start_ms = start;
        }
        r.prefill_end_ms = finish;
        r.last_token_completion_ms = finish;
        if (r.remaining_decode_tokens > 0) {
          r.state = RequestState::WaitingDecode;
          r.ready_ms = finish;
        } else {
          r.state = RequestState::Done;
          r.completion_time_ms = finish;
        }
      }
    } else {
      consecutive_decode_batches++;
      for (int idx : chosen_task->request_indices) {
        auto& r = shadow[idx];
        if (r.request_id == target_id && !r.first_decode_start_ms.has_value()) {
          r.first_decode_start_ms = start;
        }
        if (r.last_token_completion_ms.has_value()) {
          r.tbt_intervals_ms.push_back(finish - *r.last_token_completion_ms);
        }
        r.last_token_completion_ms = finish;
        r.remaining_decode_tokens -= 1;
        r.decoded_tokens_done += 1;
        if (r.remaining_decode_tokens <= 0) {
          r.state = RequestState::Done;
          r.completion_time_ms = finish;
        } else {
          r.state = RequestState::WaitingDecode;
          r.ready_ms = finish;
        }
      }
      if (is_fcfs_strict() && shadow_fcfs_active_request_id.has_value()) {
        bool still_live = false;
        for (const auto& r : shadow) {
          if (r.request_id == *shadow_fcfs_active_request_id && shadow_is_active(r)) {
            still_live = true;
            break;
          }
        }
        if (!still_live) {
          shadow_fcfs_active_request_id.reset();
        }
      }
    }
  }

  if (steps >= m_config.admission_shadow_max_steps) {
    timeout = true;
  }
  auto target_it = std::find_if(shadow.begin(), shadow.end(),
                                [&](const ForecastRequest& r) { return r.request_id == target_id; });
  const double cand_finish = (target_it != shadow.end() && target_it->completion_time_ms.has_value())
                                 ? *target_it->completion_time_ms
                                 : std::numeric_limits<double>::infinity();
  if (!std::isfinite(cand_finish)) {
    timeout = true;
  }
  const double prefill_start = (target_it != shadow.end()) ? opt_or_nan(target_it->prefill_start_ms)
                                                            : std::numeric_limits<double>::quiet_NaN();
  const double prefill_end = (target_it != shadow.end()) ? opt_or_nan(target_it->prefill_end_ms)
                                                          : std::numeric_limits<double>::quiet_NaN();
  const double first_decode_start = (target_it != shadow.end()) ? opt_or_nan(target_it->first_decode_start_ms)
                                                                 : std::numeric_limits<double>::quiet_NaN();
  double mean_tbt = est.decode_e2e_ms;
  if (target_it != shadow.end() && !target_it->tbt_intervals_ms.empty()) {
    mean_tbt = std::accumulate(target_it->tbt_intervals_ms.begin(),
                               target_it->tbt_intervals_ms.end(), 0.0) /
               static_cast<double>(target_it->tbt_intervals_ms.size());
  }

  bool existing_ok = true;
  for (const auto& r : shadow) {
    if (r.request_id == target_id || r.is_lost || !r.deadline_ms.has_value()) {
      continue;
    }
    if (!r.completion_time_ms.has_value() || *r.completion_time_ms > *r.deadline_ms) {
      existing_ok = false;
      break;
    }
  }
  if (existing_ok && m_config.slo_tbt_ms >= 0.0) {
    for (const auto& r : shadow) {
      if (r.request_id == target_id || r.is_lost) {
        continue;
      }
      if (r.remaining_decode_tokens <= 0 && r.tbt_intervals_ms.empty()) {
        continue;
      }
      const double shadow_mean_tbt = r.tbt_intervals_ms.empty()
                                         ? r.svc.decode_e2e_ms
                                         : std::accumulate(r.tbt_intervals_ms.begin(),
                                                           r.tbt_intervals_ms.end(),
                                                           0.0) / static_cast<double>(r.tbt_intervals_ms.size());
      if (shadow_mean_tbt > m_config.slo_tbt_ms) {
        existing_ok = false;
        break;
      }
    }
  }

  return {
      {"timeout", timeout ? 1.0 : 0.0},
      {"candidate_finish_ms", cand_finish},
      {"candidate_mean_tbt_ms", mean_tbt},
      {"candidate_pred_gpu_wait_ms", std::isnan(prefill_start) ? 0.0 : std::max(0.0, prefill_start - cycles_to_ms(m_clk))},
      {"candidate_pred_pim_wait_ms",
       (est.uses_pim && !std::isnan(prefill_end) && !std::isnan(first_decode_start))
           ? std::max(0.0, first_decode_start - prefill_end)
           : 0.0},
      {"existing_non_lost_feasible", existing_ok ? 1.0 : 0.0},
  };
}

void RealtimeServingRuntime::process_admission_queue() {
  while (true) {
    const auto q = waiting_admission_request_indices();
    if (q.empty()) {
      return;
    }
    m_stats.admission_queue_max_len = std::max(m_stats.admission_queue_max_len, static_cast<int>(q.size()));
    prefetch_request_estimates(q);
    auto& rr = m_requests[q.front()];
    if (rr.admission_next_retry_clk > m_clk) {
      return;
    }
    rr.admission_attempts++;

    std::vector<RouteCandidate> route_candidates;
    std::unordered_map<std::string, ServiceEstimate> est_by_route;
    std::vector<bool> own_feasible_flags;

    for (const auto& route : {m_config.gpu_route_name, m_config.hybrid_route_name}) {
      auto est = estimate_request_cached(route, rr.lin, rr.lout, predicted_lookup_batch_size(route));
      if (!est.has_value()) {
        RouteCandidate cand;
        cand.route = route;
        cand.eligible = false;
        cand.uses_pim = (route == m_config.hybrid_route_name);
        cand.deadline_ms = rr.deadline_ms;
        cand.reason = "unsupported_length";
        annotate_route_candidate(cand);
        route_candidates.push_back(cand);
        continue;
      }
      const auto proj = project_admission_candidate(rr, route, *est);
      if (proj.at("timeout") > 0.5) {
        m_stats.admission_shadow_timeout_count++;
        RouteCandidate cand;
        cand.route = route;
        cand.eligible = false;
        cand.uses_pim = est->uses_pim;
        cand.deadline_ms = rr.deadline_ms;
        cand.reason = "shadow_timeout";
        annotate_route_candidate(cand);
        route_candidates.push_back(cand);
        continue;
      }
      const double finish = proj.at("candidate_finish_ms");
      const double pred_gpu_wait = proj.at("candidate_pred_gpu_wait_ms");
      const double pred_pim_wait = proj.at("candidate_pred_pim_wait_ms");
      const bool own_checks_enabled = m_config.admission_check_own_slo;
      const bool own_e2e_ok = (!own_checks_enabled) || !rr.deadline_ms.has_value() || finish <= *rr.deadline_ms;
      const bool own_tbt_ok = (!own_checks_enabled) || m_config.slo_tbt_ms < 0.0 ||
                              proj.at("candidate_mean_tbt_ms") <= m_config.slo_tbt_ms;
      const bool existing_ok = proj.at("existing_non_lost_feasible") > 0.5;
      own_feasible_flags.push_back(own_e2e_ok);
      RouteCandidate cand;
      cand.route = route;
      cand.eligible = own_e2e_ok && own_tbt_ok && existing_ok;
      cand.uses_pim = est->uses_pim;
      cand.svc = *est;
      cand.predicted_finish_ms = finish;
      cand.predicted_gpu_wait_ms = pred_gpu_wait;
      cand.predicted_pim_wait_ms = pred_pim_wait;
      cand.deadline_ms = rr.deadline_ms;
      cand.slack_ms = rr.deadline_ms.has_value() ? std::optional<double>(*rr.deadline_ms - finish) : std::optional<double>{};
      if (cand.uses_pim && cand.eligible) {
        const auto fit = predict_hybrid_request_memory_fit(
            static_cast<uint32_t>(rr.request_id), rr.lin, *est);
        if (!fit.success) {
          cand.eligible = false;
          cand.reason = fit.reason;
        }
      }
      if (cand.reason.empty() && own_checks_enabled && !own_e2e_ok) {
        cand.reason = "candidate_e2e_slo_miss";
      } else if (cand.reason.empty() && own_checks_enabled && !own_tbt_ok) {
        cand.reason = "candidate_tbt_slo_miss";
      } else if (cand.reason.empty() && !existing_ok) {
        cand.reason = "existing_slo_violation";
      }
      annotate_route_candidate(cand);
      route_candidates.push_back(cand);
      est_by_route[route] = *est;
    }

    const RouteDecision decision = choose_route(route_candidates, m_config, rr.deadline_ms);
    const auto chosen_it =
        std::find_if(route_candidates.begin(), route_candidates.end(),
                     [&](const RouteCandidate& cand) {
                       return decision.route.has_value() && cand.route == *decision.route;
                     });
    if (!decision.dropped && decision.route.has_value() &&
        chosen_it != route_candidates.end() && chosen_it->eligible &&
        est_by_route.find(*decision.route) != est_by_route.end()) {
      std::string final_route = *decision.route;
      ServiceEstimate final_est = est_by_route[*decision.route];
      RouteCandidate final_cand = *chosen_it;
      std::string final_reason = decision.reason;
      if (final_route == m_config.hybrid_route_name) {
        std::string allocator_reason;
        if (!reserve_hybrid_request_memory(rr, final_est, allocator_reason)) {
          const auto gpu_cand = find_candidate(route_candidates, m_config.gpu_route_name);
          if (gpu_cand.has_value() && gpu_cand->eligible && gpu_cand->svc.has_value()) {
            final_route = m_config.gpu_route_name;
            final_est = *gpu_cand->svc;
            final_cand = *gpu_cand;
            final_reason = "fallback_gpu_due_to_" + allocator_reason;
            rr.fallback_reason = allocator_reason;
            m_stats.allocator_gpu_fallback_count++;
          } else {
            rr.admission_last_reject_reason = allocator_reason;
            rr.admission_next_retry_clk = m_clk + ms_to_cycles(std::max(1e-6, m_config.admission_retry_interval_ms));
            m_stats.admission_held_count++;
            m_stats.allocator_hold_count++;
            return;
          }
        }
      }
      admit_waiting_request(rr, final_route, final_est, final_cand, final_reason, false);
      m_stats.admission_admitted_from_queue_count++;
      continue;
    }

    const bool own_infeasible_all = !own_feasible_flags.empty() &&
                                    std::none_of(own_feasible_flags.begin(), own_feasible_flags.end(),
                                                 [](bool ok) { return ok; });
    if (own_infeasible_all) {
      rr.is_lost = true;
      m_stats.admission_lost_count++;
      std::vector<RouteCandidate> admitted;
      for (const auto& cand : route_candidates) {
        if (cand.svc.has_value()) {
          admitted.push_back(cand);
        }
      }
      if (admitted.empty()) {
        rr.state = RequestState::Dropped;
        rr.dropped_reason = "no_route_for_lost_request";
        rr.route_decision_reason = rr.dropped_reason;
        m_stats.dropped_requests++;
        return;
      }
      const auto best = *std::min_element(admitted.begin(), admitted.end(),
                                          [](const RouteCandidate& lhs, const RouteCandidate& rhs) {
                                            return std::tie(lhs.predicted_finish_ms, lhs.route) <
                                                   std::tie(rhs.predicted_finish_ms, rhs.route);
                                          });
      std::string final_route = best.route;
      ServiceEstimate final_est = *best.svc;
      RouteCandidate final_cand = best;
      std::string final_reason = "admit_lost_best_effort";
      if (final_route == m_config.hybrid_route_name) {
        std::string allocator_reason;
        if (!reserve_hybrid_request_memory(rr, final_est, allocator_reason)) {
          const auto gpu_cand = find_candidate(route_candidates, m_config.gpu_route_name);
          if (gpu_cand.has_value() && gpu_cand->svc.has_value()) {
            final_route = m_config.gpu_route_name;
            final_est = *gpu_cand->svc;
            final_cand = *gpu_cand;
            final_reason = "fallback_gpu_due_to_" + allocator_reason;
            rr.fallback_reason = allocator_reason;
            m_stats.allocator_gpu_fallback_count++;
          } else {
            rr.admission_last_reject_reason = allocator_reason;
            rr.admission_next_retry_clk = m_clk + ms_to_cycles(std::max(1e-6, m_config.admission_retry_interval_ms));
            m_stats.admission_held_count++;
            m_stats.allocator_hold_count++;
            return;
          }
        }
      }
      admit_waiting_request(rr, final_route, final_est, final_cand, final_reason, true);
      m_stats.admission_admitted_from_queue_count++;
      continue;
    }

    rr.admission_last_reject_reason = decision.reason;
    double retry_dt = std::max(0.0, m_config.admission_retry_interval_ms);
    if (retry_dt == 0.0) {
      retry_dt = 1e-6;
    }
    rr.admission_next_retry_clk = m_clk + ms_to_cycles(retry_dt);
    m_stats.admission_reject_count++;
    return;
  }
}

void RealtimeServingRuntime::process_slo_guarded_admission_queue() {
  while (true) {
    const auto q = waiting_admission_request_indices();
    if (q.empty()) {
      return;
    }
    m_stats.admission_queue_max_len = std::max(m_stats.admission_queue_max_len, static_cast<int>(q.size()));
    prefetch_request_estimates(q);
    const ForecastRun baseline_run = incremental_active_forecast_results();
    const auto baseline_results = baseline_run.requests;
    bool admitted_any = false;

    for (size_t qpos = 0; qpos < q.size(); ++qpos) {
      auto& rr = m_requests[q[qpos]];
      if (rr.admission_next_retry_clk > m_clk) {
        continue;
      }
      rr.admission_attempts++;
      std::vector<RouteCandidate> route_candidates;
      std::unordered_map<std::string, ServiceEstimate> est_by_route;
      for (const auto& route : {m_config.gpu_route_name, m_config.hybrid_route_name}) {
        auto est = estimate_request_cached(route, rr.lin, rr.lout, predicted_lookup_batch_size(route));
        if (!est.has_value()) {
          RouteCandidate cand;
          cand.route = route;
          cand.eligible = false;
          cand.uses_pim = (route == m_config.hybrid_route_name);
          cand.deadline_ms = rr.deadline_ms;
          cand.reason = "unsupported_length";
          annotate_route_candidate(cand);
          route_candidates.push_back(cand);
          continue;
        }
        auto cand = evaluate_slo_guarded_route(rr, route, *est, baseline_run, baseline_results);
        route_candidates.push_back(cand);
        est_by_route[route] = *est;
      }

      const RouteDecision decision = choose_route(route_candidates, m_config, rr.deadline_ms);
      const auto chosen_it =
          std::find_if(route_candidates.begin(), route_candidates.end(),
                       [&](const RouteCandidate& cand) {
                         return decision.route.has_value() && cand.route == *decision.route;
                       });
      if (!decision.dropped && decision.route.has_value() &&
          chosen_it != route_candidates.end() && chosen_it->eligible &&
          est_by_route.find(*decision.route) != est_by_route.end()) {
        std::string final_route = *decision.route;
        ServiceEstimate final_est = est_by_route[*decision.route];
        RouteCandidate final_cand = *chosen_it;
        std::string final_reason = decision.reason;
        if (final_route == m_config.hybrid_route_name) {
          std::string allocator_reason;
          if (!reserve_hybrid_request_memory(rr, final_est, allocator_reason)) {
            const auto gpu_cand = find_candidate(route_candidates, m_config.gpu_route_name);
            if (gpu_cand.has_value() && gpu_cand->eligible && gpu_cand->svc.has_value()) {
              final_route = m_config.gpu_route_name;
              final_est = *gpu_cand->svc;
              final_cand = *gpu_cand;
              final_reason = "fallback_gpu_due_to_" + allocator_reason;
              rr.fallback_reason = allocator_reason;
              m_stats.allocator_gpu_fallback_count++;
            } else {
              rr.admission_last_reject_reason = allocator_reason;
              rr.admission_next_retry_clk = m_clk + ms_to_cycles(std::max(1e-6, m_config.admission_retry_interval_ms));
              m_stats.admission_held_count++;
              m_stats.allocator_hold_count++;
              continue;
            }
          }
        }
        admit_waiting_request(rr, final_route, final_est, final_cand, final_reason, false);
        m_stats.admission_admitted_from_queue_count++;
        for (size_t i = 0; i < qpos; ++i) {
          auto& blocked = m_requests[q[i]];
          if (blocked.state != RequestState::WaitingAdmission) {
            continue;
          }
          blocked.admission_bypass_count++;
          m_stats.admission_bypassed_count++;
        }
        refresh_active_predictions("new_admission");
        admitted_any = true;
        break;
      }

      const double queue_wait_ms = std::max(0.0, cycles_to_ms(m_clk) - rr.arrival_ms);
      const bool force_best_effort =
          (m_config.admission_max_wait_ms > 0.0 && queue_wait_ms >= m_config.admission_max_wait_ms) ||
          (m_config.admission_max_bypass_count > 0 && rr.admission_bypass_count >= m_config.admission_max_bypass_count);
      if (force_best_effort) {
        std::vector<RouteCandidate> admitted;
        for (const auto& cand : route_candidates) {
          if (cand.svc.has_value()) {
            admitted.push_back(cand);
          }
        }
        if (admitted.empty()) {
          rr.state = RequestState::Dropped;
          rr.dropped_reason = "no_route_for_best_effort";
          rr.route_decision_reason = rr.dropped_reason;
          m_stats.dropped_requests++;
          admitted_any = true;
          break;
        }
        rr.is_lost = true;
        rr.admission_forced_best_effort = true;
        m_stats.admission_lost_count++;
        m_stats.admission_best_effort_count++;
        const auto best = *std::min_element(admitted.begin(), admitted.end(),
                                            [](const RouteCandidate& lhs, const RouteCandidate& rhs) {
                                              return std::tie(lhs.predicted_finish_ms, lhs.route) <
                                                     std::tie(rhs.predicted_finish_ms, rhs.route);
                                            });
        std::string final_route = best.route;
        ServiceEstimate final_est = *best.svc;
        RouteCandidate final_cand = best;
        std::string final_reason = "admit_best_effort_due_to_wait_or_bypass";
        if (final_route == m_config.hybrid_route_name) {
          std::string allocator_reason;
          if (!reserve_hybrid_request_memory(rr, final_est, allocator_reason)) {
            const auto gpu_cand = find_candidate(route_candidates, m_config.gpu_route_name);
            if (gpu_cand.has_value() && gpu_cand->svc.has_value()) {
              final_route = m_config.gpu_route_name;
              final_est = *gpu_cand->svc;
              final_cand = *gpu_cand;
              final_reason = "fallback_gpu_due_to_" + allocator_reason;
              rr.fallback_reason = allocator_reason;
              m_stats.allocator_gpu_fallback_count++;
            } else {
              rr.admission_last_reject_reason = allocator_reason;
              rr.admission_next_retry_clk = m_clk + ms_to_cycles(std::max(1e-6, m_config.admission_retry_interval_ms));
              m_stats.admission_held_count++;
              m_stats.allocator_hold_count++;
              continue;
            }
          }
        }
        admit_waiting_request(rr, final_route, final_est, final_cand, final_reason, true);
        m_stats.admission_admitted_from_queue_count++;
        for (size_t i = 0; i < qpos; ++i) {
          auto& blocked = m_requests[q[i]];
          if (blocked.state != RequestState::WaitingAdmission) {
            continue;
          }
          blocked.admission_bypass_count++;
          m_stats.admission_bypassed_count++;
        }
        refresh_active_predictions("new_admission");
        admitted_any = true;
        break;
      }

      rr.admission_last_reject_reason = decision.reason;
      if (chosen_it != route_candidates.end()) {
        rr.admission_last_harmed_count = chosen_it->harmed_count;
        rr.admission_last_total_harm_ms = chosen_it->total_harm_ms;
        rr.admission_last_single_harm_ms = chosen_it->max_single_harm_ms;
        rr.admission_last_own_miss_ms = chosen_it->own_miss_ms;
      }
      double retry_dt = std::max(0.0, m_config.admission_retry_interval_ms);
      if (retry_dt == 0.0) {
        retry_dt = 1e-6;
      }
      rr.admission_next_retry_clk = m_clk + ms_to_cycles(retry_dt);
      m_stats.admission_held_count++;
      m_admission_blocked_request_ids.insert(rr.request_id);
      bool all_tbt = true;
      bool any_estimated = false;
      for (const auto& cand : route_candidates) {
        if (cand.svc.has_value()) {
          any_estimated = true;
          all_tbt = all_tbt && cand.reason == "candidate_tbt_slo_miss";
        }
      }
      if (any_estimated && all_tbt) {
        m_admission_tbt_blocked_request_ids.insert(rr.request_id);
      }
    }

    if (!admitted_any) {
      return;
    }
  }
}

RouteCandidate RealtimeServingRuntime::predict_route_candidate(
    const SourceRequest& req,
    const std::string& route,
    const std::optional<double>& deadline_ms,
    std::optional<double> baseline_total_energy_nj,
    std::optional<double> baseline_total_decode_energy_nj) {
  auto est = estimate_request_cached(route,
                                     req.context_tokens,
                                     req.generated_tokens,
                                     predicted_lookup_batch_size(route));
  RouteCandidate cand;
  cand.route = route;
  cand.deadline_ms = deadline_ms;
  cand.uses_pim = (route == m_config.hybrid_route_name);
  if (!est.has_value()) {
    cand.reason = "unsupported_length";
    annotate_route_candidate(cand);
    return cand;
  }
  cand.uses_pim = est->uses_pim;
  cand.svc = *est;
  if (cand.uses_pim && !has_template_for_lin(route, est->mapped_lin)) {
    cand.reason = "template_missing";
    annotate_route_candidate(cand);
    return cand;
  }

  if (use_shadow_prediction_for_fcfs_decode()) {
    const auto proj = project_prefill_priority_fcfs_decode_candidate(req, route, *est);
    cand.eligible = true;
    cand.predicted_finish_ms = proj.at("candidate_finish_ms");
    cand.predicted_gpu_wait_ms = proj.at("candidate_pred_gpu_wait_ms");
    cand.predicted_pim_wait_ms = proj.at("candidate_pred_pim_wait_ms");
    cand.predicted_incremental_energy_nj =
        std::max(0.0, est->prefill_energy_nj + est->decode_energy_nj * est->decode_tokens);
    cand.slack_ms = deadline_ms.has_value() ? std::optional<double>(*deadline_ms - cand.predicted_finish_ms)
                                            : std::optional<double>{};
    if (cand.uses_pim) {
      const auto fit = predict_hybrid_request_memory_fit(
          static_cast<uint32_t>(req.request_id), req.context_tokens, *est);
      if (!fit.success) {
        cand.eligible = false;
        cand.reason = fit.reason;
      }
    }
    annotate_route_candidate(cand);
    return cand;
  }
  if (use_incremental_prediction_for_fcfs_decode()) {
    if ((!baseline_total_energy_nj.has_value() || !baseline_total_decode_energy_nj.has_value()) &&
        uses_latency_guarded_energy_policy()) {
      const auto baseline_run = incremental_active_forecast_results();
      baseline_total_energy_nj = baseline_run.total_prefill_energy_nj + baseline_run.total_decode_energy_nj;
      baseline_total_decode_energy_nj = baseline_run.total_decode_energy_nj;
    }
    const auto proj = project_prefill_priority_fcfs_decode_incremental_candidate(
        req,
        route,
        *est,
        baseline_total_energy_nj,
        baseline_total_decode_energy_nj);
    cand.eligible = true;
    cand.predicted_finish_ms = proj.at("candidate_finish_ms");
    cand.predicted_gpu_wait_ms = proj.at("candidate_pred_gpu_wait_ms");
    cand.predicted_pim_wait_ms = proj.at("candidate_pred_pim_wait_ms");
    cand.predicted_incremental_energy_nj = proj.at("candidate_incremental_energy_nj");
    cand.predicted_incremental_decode_energy_nj = proj.at("candidate_incremental_decode_energy_nj");
    cand.slack_ms = deadline_ms.has_value() ? std::optional<double>(*deadline_ms - cand.predicted_finish_ms)
                                            : std::optional<double>{};
    if (cand.uses_pim) {
      const auto fit = predict_hybrid_request_memory_fit(
          static_cast<uint32_t>(req.request_id), req.context_tokens, *est);
      if (!fit.success) {
        cand.eligible = false;
        cand.reason = fit.reason;
      }
    }
    annotate_route_candidate(cand);
    return cand;
  }
  if (use_heuristic_refresh_prediction_for_fcfs_decode()) {
    const auto proj = project_prefill_priority_fcfs_decode_heuristic_candidate(req, route, *est);
    cand.eligible = true;
    cand.predicted_finish_ms = proj.at("candidate_finish_ms");
    cand.predicted_gpu_wait_ms = proj.at("candidate_pred_gpu_wait_ms");
    cand.predicted_pim_wait_ms = proj.at("candidate_pred_pim_wait_ms");
    cand.slack_ms = deadline_ms.has_value() ? std::optional<double>(*deadline_ms - cand.predicted_finish_ms)
                                            : std::optional<double>{};
    if (cand.uses_pim) {
      const auto fit = predict_hybrid_request_memory_fit(
          static_cast<uint32_t>(req.request_id), req.context_tokens, *est);
      if (!fit.success) {
        cand.eligible = false;
        cand.reason = fit.reason;
      }
    }
    annotate_route_candidate(cand);
    return cand;
  }

  const double now_ms = req.arrival_ms;
  const double prefill_start = std::max(
      now_ms,
      std::max(est->prefill_gpu_ms > 0.0 ? cycles_to_ms(m_gpu_busy_until) : now_ms,
               est->prefill_pim_ms > 0.0 ? cycles_to_ms(m_pim_busy_until) : now_ms));
  const double prefill_end = prefill_start + est->prefill_e2e_ms;
  double gpu_free_after_prefill = cycles_to_ms(m_gpu_busy_until);
  double pim_free_after_prefill = cycles_to_ms(m_pim_busy_until);
  if (est->prefill_gpu_ms > 0.0) {
    gpu_free_after_prefill = std::max(gpu_free_after_prefill, prefill_start) + est->prefill_gpu_ms;
  }
  if (est->prefill_pim_ms > 0.0) {
    pim_free_after_prefill = std::max(pim_free_after_prefill, prefill_start) + est->prefill_pim_ms;
  }
  const double decode_gpu_total = est->decode_gpu_ms * est->decode_tokens;
  const double decode_pim_total = est->decode_pim_ms * est->decode_tokens;
  const double decode_e2e_total = est->decode_e2e_ms * est->decode_tokens;
  const double decode_start = std::max(
      prefill_end,
      std::max(decode_gpu_total > 0.0 ? gpu_free_after_prefill : prefill_end,
               decode_pim_total > 0.0 ? pim_free_after_prefill : prefill_end));
  cand.eligible = true;
  cand.predicted_finish_ms = decode_start + decode_e2e_total;
  cand.predicted_gpu_wait_ms = std::max(0.0, prefill_start - req.arrival_ms);
  cand.predicted_pim_wait_ms =
      (est->uses_pim && est->decode_tokens > 0 && est->decode_pim_ms > 0.0)
          ? std::max(0.0, pim_free_after_prefill - prefill_end)
          : 0.0;
  cand.predicted_incremental_energy_nj =
      std::max(0.0, est->prefill_energy_nj + est->decode_energy_nj * est->decode_tokens);
  cand.queue_pressure_ms = queue_pressure_adjustment_ms(est->prefill_gpu_ms,
                                                        est->prefill_pim_ms,
                                                        est->prefill_e2e_ms,
                                                        est->decode_gpu_ms,
                                                        est->decode_pim_ms,
                                                        est->decode_e2e_ms,
                                                        est->decode_tokens);
  cand.predicted_finish_ms += cand.queue_pressure_ms;
  cand.slack_ms = deadline_ms.has_value() ? std::optional<double>(*deadline_ms - cand.predicted_finish_ms)
                                          : std::optional<double>{};
  if (cand.uses_pim) {
    const auto fit = predict_hybrid_request_memory_fit(
        static_cast<uint32_t>(req.request_id), req.context_tokens, *est);
    if (!fit.success) {
      cand.eligible = false;
      cand.reason = fit.reason;
    }
  }
  annotate_route_candidate(cand);
  return cand;
}

RouteCandidate RealtimeServingRuntime::predict_decode_route_candidate(const RuntimeRequest& req,
                                                                     const std::string& route,
                                                                     double decision_time_ms) {
  auto est = estimate_request_cached(route, req.lin, req.lout, predicted_lookup_batch_size(route));
  RouteCandidate cand;
  cand.route = route;
  cand.deadline_ms = req.deadline_ms;
  cand.uses_pim = (route == m_config.hybrid_route_name);
  if (!est.has_value()) {
    cand.reason = "unsupported_length";
    annotate_route_candidate(cand);
    return cand;
  }
  cand.uses_pim = est->uses_pim;
  cand.svc = *est;
  if (cand.uses_pim && !has_template_for_lin(route, est->mapped_lin)) {
    cand.reason = "template_missing";
    annotate_route_candidate(cand);
    return cand;
  }

  if (use_heuristic_refresh_prediction_for_fcfs_decode()) {
    const auto proj = project_prefill_priority_fcfs_decode_heuristic_decode_candidate(req, route, *est, decision_time_ms);
    cand.eligible = true;
    cand.predicted_finish_ms = proj.at("candidate_finish_ms");
    cand.predicted_gpu_wait_ms = proj.at("candidate_pred_gpu_wait_ms");
    cand.predicted_pim_wait_ms = proj.at("candidate_pred_pim_wait_ms");
    cand.predicted_incremental_energy_nj =
        std::max(0.0, est->prefill_energy_nj + est->decode_energy_nj * req.remaining_decode_tokens);
    cand.predicted_incremental_decode_energy_nj =
        std::max(0.0, est->decode_energy_nj * req.remaining_decode_tokens);
    cand.slack_ms = req.deadline_ms.has_value() ? std::optional<double>(*req.deadline_ms - cand.predicted_finish_ms)
                                                : std::optional<double>{};
    annotate_route_candidate(cand);
    return cand;
  }

  const double decode_gpu_total = est->decode_gpu_ms * req.remaining_decode_tokens;
  const double decode_pim_total = est->decode_pim_ms * req.remaining_decode_tokens;
  const double decode_e2e_total = est->decode_e2e_ms * req.remaining_decode_tokens;
  const double decode_start = std::max(
      decision_time_ms,
      std::max(decode_gpu_total > 0.0 ? cycles_to_ms(m_gpu_busy_until) : decision_time_ms,
               decode_pim_total > 0.0 ? cycles_to_ms(m_pim_busy_until) : decision_time_ms));
  cand.eligible = true;
  cand.predicted_finish_ms = decode_start + decode_e2e_total;
  cand.predicted_gpu_wait_ms = (decode_gpu_total > 0.0) ? std::max(0.0, cycles_to_ms(m_gpu_busy_until) - decision_time_ms) : 0.0;
  cand.predicted_pim_wait_ms = (decode_pim_total > 0.0) ? std::max(0.0, cycles_to_ms(m_pim_busy_until) - decision_time_ms) : 0.0;
  cand.predicted_incremental_energy_nj =
      std::max(0.0, est->prefill_energy_nj + est->decode_energy_nj * req.remaining_decode_tokens);
  cand.predicted_incremental_decode_energy_nj =
      std::max(0.0, est->decode_energy_nj * req.remaining_decode_tokens);
  cand.slack_ms = req.deadline_ms.has_value() ? std::optional<double>(*req.deadline_ms - cand.predicted_finish_ms)
                                              : std::optional<double>{};
  annotate_route_candidate(cand);
  return cand;
}

void RealtimeServingRuntime::refresh_active_predictions(const std::string& cause) {
  if (!is_prefill_priority_fcfs_decode()) {
    return;
  }
  std::vector<RuntimeRequest*> active;
  for (auto& req : m_requests) {
    if (req.svc.has_value() && !req.route.empty() &&
        (req.state == RequestState::WaitingPrefill ||
         (req.state == RequestState::WaitingDecode && req.remaining_decode_tokens > 0))) {
      active.push_back(&req);
    }
  }
  if (active.empty()) {
    return;
  }

  std::unordered_map<int, std::unordered_map<std::string, double>> results;
  if (use_shadow_prediction_for_fcfs_decode()) {
    std::vector<ForecastRequest> shadow;
    for (const auto* req : active) {
      shadow.push_back(shadow_request_from_runtime_request(*req));
    }
    results = simulate_prefill_priority_fcfs_decode_shadow(std::move(shadow));
  } else if (use_incremental_prediction_for_fcfs_decode()) {
    results = incremental_active_forecast_results().requests;
  } else if (use_heuristic_refresh_prediction_for_fcfs_decode()) {
    std::vector<ForecastRequest> forecast;
    for (const auto* req : active) {
      forecast.push_back(shadow_request_from_runtime_request(*req));
    }
    results = simulate_prefill_priority_fcfs_decode_heuristic(std::move(forecast));
  } else {
    return;
  }

  for (auto* req : active) {
    const double old_finish = req->latest_predicted_finish_ms.value_or(std::numeric_limits<double>::infinity());
    double new_finish = results.count(req->request_id) ? results[req->request_id]["completion_time_ms"]
                                                       : old_finish;
    if (use_heuristic_refresh_prediction_for_fcfs_decode() && std::isfinite(new_finish)) {
      const double prefill_gpu_ms = (req->state == RequestState::WaitingPrefill) ? req->svc->prefill_gpu_ms : 0.0;
      const double prefill_pim_ms = (req->state == RequestState::WaitingPrefill) ? req->svc->prefill_pim_ms : 0.0;
      const double prefill_e2e_ms = (req->state == RequestState::WaitingPrefill) ? req->svc->prefill_e2e_ms : 0.0;
      new_finish += queue_pressure_adjustment_ms(prefill_gpu_ms,
                                                 prefill_pim_ms,
                                                 prefill_e2e_ms,
                                                 req->svc->decode_gpu_ms,
                                                 req->svc->decode_pim_ms,
                                                 req->svc->decode_e2e_ms,
                                                 req->remaining_decode_tokens);
    }
    req->latest_predicted_finish_ms =
        std::isfinite(new_finish) ? quantize_export_ms(new_finish) : old_finish;
    req->prediction_refresh_count++;
    record_debug_event("prediction_refresh",
                       {{"request_id", std::to_string(req->request_id)},
                        {"cause", cause},
                        {"old_latest_predicted_finish_ms", format_ms(old_finish)},
                        {"new_latest_predicted_finish_ms", format_ms(*req->latest_predicted_finish_ms)}});
  }
}

void RealtimeServingRuntime::maybe_rebind_decode_route(RuntimeRequest& req, Clk_t decision_time_clk) {
  if (!m_config.enable_decode_rebind || !req.svc.has_value() || req.route.empty() ||
      req.remaining_decode_tokens <= 0 || req.decoded_tokens_done > 0) {
    return;
  }
  req.decode_bind_clk = decision_time_clk;
  req.decode_bind_time_ms = cycles_to_ms(decision_time_clk);
  if (!req.svc->uses_pim) {
    req.decode_rebind_reason = "skip_non_pim_route";
    return;
  }

  req.decode_rebind_attempted = true;
  m_stats.decode_rebind_attempt_count++;
  const auto current_cand = predict_decode_route_candidate(req, req.route, *req.decode_bind_time_ms);
  if (!current_cand.svc.has_value()) {
    req.decode_rebind_reason = "current_route_estimate_unavailable";
    return;
  }

  std::vector<RouteCandidate> candidates = {current_cand};
  if (current_cand.uses_pim) {
    auto gpu_cand = predict_decode_route_candidate(req, m_config.gpu_route_name, *req.decode_bind_time_ms);
    if (gpu_cand.svc.has_value()) {
      candidates.push_back(gpu_cand);
    }
  }
  if (candidates.size() == 1) {
    req.latest_predicted_finish_ms = quantize_export_ms(current_cand.predicted_finish_ms);
    req.decode_rebind_reason = "no_gpu_candidate";
    return;
  }

  const RouteDecision decision = choose_route(candidates, m_config, req.deadline_ms);
  auto chosen_it = std::find_if(candidates.begin(), candidates.end(),
                                [&](const RouteCandidate& cand) {
                                  return decision.route.has_value() && cand.route == *decision.route;
                                });
  if (chosen_it == candidates.end()) {
    req.decode_rebind_reason = "no_route_choice";
    return;
  }
  req.latest_predicted_finish_ms = quantize_export_ms(chosen_it->predicted_finish_ms);
  const double improvement_ms = current_cand.predicted_finish_ms - chosen_it->predicted_finish_ms;
  const std::string old_route = req.route;
  if (decision.route.has_value() &&
      *decision.route == m_config.gpu_route_name &&
      old_route != m_config.gpu_route_name &&
      improvement_ms > std::max(0.0, m_config.decode_rebind_margin_ms) &&
      chosen_it->svc.has_value()) {
    release_hybrid_request_memory(req);
    req.route = m_config.gpu_route_name;
    req.svc = *chosen_it->svc;
    req.decode_rebound = true;
    req.decode_rebind_reason = "rebound_to_gpu:" + decision.reason;
    move_route_count(old_route, req.route);
    m_stats.decode_rebound_count++;
  } else {
    req.decode_rebind_reason = "keep_route:" + decision.reason;
  }
}

}  // namespace Ramulator::Serving
