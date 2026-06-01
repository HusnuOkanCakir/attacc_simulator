#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "base/exception.h"
#include "frontend/impl/serving/inputs.h"
#include "frontend/impl/serving/pim_executor.h"
#include "frontend/impl/serving/predictor.h"
#include "frontend/impl/serving/runtime.h"
#include "translation/impl/attacc_serving_translation.h"

namespace Ramulator::Serving {

namespace fs = std::filesystem;

namespace {

double as_ms_or_inf(const std::optional<double>& value) {
  return value.has_value() ? *value : std::numeric_limits<double>::infinity();
}

int first_request_id(const std::vector<int>& indices,
                     const std::vector<RuntimeRequest>& requests) {
  int best = std::numeric_limits<int>::max();
  for (int idx : indices) {
    best = std::min(best, requests[idx].request_id);
  }
  return (best == std::numeric_limits<int>::max()) ? -1 : best;
}

std::string join_request_ids(const std::vector<int>& indices,
                             const std::vector<RuntimeRequest>& requests) {
  std::ostringstream oss;
  for (size_t i = 0; i < indices.size(); ++i) {
    if (i > 0) {
      oss << ';';
    }
    oss << requests[indices[i]].request_id;
  }
  return oss.str();
}

std::string format_ms(double value) {
  if (!std::isfinite(value)) {
    return "inf";
  }
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6) << value;
  return oss.str();
}

}  // namespace

RealtimeServingRuntime::RealtimeServingRuntime(Logger_t logger) : m_logger(std::move(logger)) {}

void RealtimeServingRuntime::initialize(const FrontendConfig& config) {
  m_config = config;
  validate_static_config(m_config);
  m_source_requests = load_requests_csv(m_config);
  m_cost_points = load_cost_tables(m_config);
  m_templates = load_templates(m_config);
  if (use_runtime_generated_hybrid()) {
    // ---- Legacy allocator (handles transient: Score/Context/Scratch) ----
    ServingAllocatorConfig allocator_config;
    allocator_config.max_addr = m_config.translation_max_addr;
    allocator_config.page_size_bytes = m_config.translation_page_size_bytes;
    allocator_config.kv_pool_bytes = m_config.kv_pool_bytes;
    allocator_config.score_pool_bytes = m_config.score_pool_bytes;
    allocator_config.context_pool_bytes = m_config.context_pool_bytes;
    allocator_config.scratch_pool_bytes = m_config.scratch_pool_bytes;
    allocator_config.placement_channel_count = m_config.generator_channel_count;
    allocator_config.placement_rank_count = 1;
    allocator_config.placement_bank_group_count = 4;
    allocator_config.placement_bank_count = 4;
    m_address_space = std::make_shared<ServingAddressSpace>(allocator_config);
    m_allocator = std::make_unique<ServingAllocatorFacade>(m_address_space, generator_geometry());

    // ---- New clean allocator (handles KV: KvKey/KvValue, and VA→PA translation) ----
    PimAllocator::Config pim_cfg;
    pim_cfg.page_size_bytes    = m_config.translation_page_size_bytes;
    pim_cfg.kv_pool_bytes      = m_config.kv_pool_bytes;
    pim_cfg.score_pool_bytes   = m_config.score_pool_bytes;
    pim_cfg.context_pool_bytes = m_config.context_pool_bytes;
    pim_cfg.scratch_pool_bytes = m_config.scratch_pool_bytes;
    pim_cfg.num_channels       = m_config.generator_channel_count;
    m_pim_allocator = std::make_shared<PimAllocator>(pim_cfg);
  }
  m_stats.total_requests = static_cast<int>(m_source_requests.size());
  open_debug_log_if_needed();
}

void RealtimeServingRuntime::attach_translation(ITranslation* translation) {
  m_translation = translation;
  // Wire the new PimAllocator into the translation layer so that virtual
  // addresses emitted by the command generator are resolved at issue time.
  if (m_pim_allocator) {
    if (auto* ctrl = dynamic_cast<IAttAccServingTranslationControl*>(translation)) {
      ctrl->attach_pim_allocator(m_pim_allocator);
    }
  }
}

GeneratorGeometry RealtimeServingRuntime::generator_geometry() const {
  GeneratorGeometry geometry;
  geometry.backend = m_config.generator_backend;
  geometry.dhead = m_config.generator_dhead;
  geometry.heads_per_hbm = m_config.generator_heads_per_hbm;
  geometry.dtype_bytes = m_config.generator_dtype_bytes;
  geometry.num_layers = m_config.generator_num_layers;
  geometry.channel_count = m_config.generator_channel_count;
  return geometry;
}

bool RealtimeServingRuntime::use_runtime_generated_hybrid() const {
  return m_config.hybrid_command_source == "runtime_generated";
}

void RealtimeServingRuntime::connect_memory_system(IMemorySystem* memory_system) {
  m_memory_system = memory_system;
  finalize_post_connect_setup();
}

void RealtimeServingRuntime::finalize_post_connect_setup() {
  if (m_memory_system == nullptr) {
    throw ConfigurationError("RealtimeServingFrontend: memory system is not connected");
  }
  if (m_memory_system->get_clock_ratio() != 1) {
    throw ConfigurationError("RealtimeServingFrontend: MemorySystem.clock_ratio must be 1");
  }

  m_tck_ns = m_memory_system->get_tCK();
  if (!(m_tck_ns > 0.0f)) {
    throw ConfigurationError("RealtimeServingFrontend: memory system returned invalid tCK {}", m_tck_ns);
  }

  m_cycles_per_ms = 1.0e6 / static_cast<double>(m_tck_ns);
  if (!(m_cycles_per_ms > 0.0)) {
    throw ConfigurationError("RealtimeServingFrontend: failed to derive cycles_per_ms from tCK {}", m_tck_ns);
  }

  for (auto& req : m_source_requests) {
    req.arrival_clk = ms_to_cycles(req.arrival_ms);
  }
  prefetch_arrival_request_estimates();
  m_post_connected = true;
}

bool RealtimeServingRuntime::use_waiting_admission_queue() const {
  return m_config.enable_predictive_admission || m_config.enable_slo_guarded_admission;
}

bool RealtimeServingRuntime::uses_slo_guarded_admission() const {
  return m_config.enable_slo_guarded_admission;
}

bool RealtimeServingRuntime::is_fcfs_strict() const {
  return m_config.local_scheduling_policy == "fcfs_strict";
}

bool RealtimeServingRuntime::is_prefill_priority_fcfs_decode() const {
  return m_config.local_scheduling_policy == "prefill_priority_fcfs_decode";
}

bool RealtimeServingRuntime::use_shadow_prediction_for_fcfs_decode() const {
  return is_prefill_priority_fcfs_decode() &&
         m_config.fcfs_decode_prediction_mode == "shadow";
}

bool RealtimeServingRuntime::use_incremental_prediction_for_fcfs_decode() const {
  return is_prefill_priority_fcfs_decode() &&
         m_config.fcfs_decode_prediction_mode == "incremental";
}

bool RealtimeServingRuntime::use_heuristic_refresh_prediction_for_fcfs_decode() const {
  return is_prefill_priority_fcfs_decode() &&
         m_config.fcfs_decode_prediction_mode == "heuristic_refresh";
}

bool RealtimeServingRuntime::uses_latency_guarded_energy_policy() const {
  return m_config.route_policy == "latency_guarded_energy";
}

void RealtimeServingRuntime::tick() {
  if (!m_post_connected) {
    throw ConfigurationError("RealtimeServingFrontend: connect_memory_system() was not completed");
  }

  process_pim_callbacks();
  maybe_complete_active_task();

  if (use_waiting_admission_queue()) {
    enqueue_arrivals_up_to_now();
    if (uses_slo_guarded_admission()) {
      process_slo_guarded_admission_queue();
    } else {
      process_admission_queue();
    }
  } else {
    admit_arrivals_up_to_now();
    // If any requests were held because PIM KV memory was full, retry them
    // via the admission queue path even when predictive admission is off.
    if (has_kv_held_requests()) {
      process_admission_queue();
    }
  }

  maybe_start_next_task();
  pump_active_pim_job();
  maybe_complete_active_task();
  maybe_log_progress("tick");
  if (try_fast_forward()) {
    return;
  }
  m_clk++;
}

bool RealtimeServingRuntime::is_finished() const {
  if (m_arrival_idx < m_source_requests.size()) {
    return false;
  }
  if (m_active_task.has_value()) {
    return false;
  }
  if (!m_completed_pim_events.empty()) {
    return false;
  }
  for (const auto& req : m_requests) {
    if (req.state == RequestState::WaitingAdmission ||
        req.state == RequestState::WaitingPrefill ||
        req.state == RequestState::RunningPrefill ||
        req.state == RequestState::WaitingDecode ||
        req.state == RequestState::RunningDecode) {
      return false;
    }
  }
  return true;
}

void RealtimeServingRuntime::finalize() {
  write_requests_csv();
  write_kv_placement_csv();
  write_allocator_pressure_csv();
  if (m_debug_log.is_open()) {
    m_debug_log.flush();
    m_debug_log.close();
  }
}

size_t RealtimeServingRuntime::gpu_cost_point_count() const {
  auto it = m_cost_points.find(m_config.gpu_route_name);
  return (it == m_cost_points.end()) ? 0 : it->second.size();
}

size_t RealtimeServingRuntime::hybrid_cost_point_count() const {
  auto it = m_cost_points.find(m_config.hybrid_route_name);
  return (it == m_cost_points.end()) ? 0 : it->second.size();
}

std::optional<ServiceEstimate> RealtimeServingRuntime::estimate_request_for_route(const std::string& route,
                                                                                  int context_tokens,
                                                                                  int generated_tokens,
                                                                                  int bs) const {
  return estimate_request(m_cost_points, m_config, m_cycles_per_ms, route, context_tokens, generated_tokens, bs);
}

std::string RealtimeServingRuntime::estimate_cache_key(const std::string& route,
                                                       int context_tokens,
                                                       int generated_tokens,
                                                       int bs) const {
  std::ostringstream oss;
  oss << route << ':' << context_tokens << ':' << generated_tokens << ':' << bs << ':'
      << m_config.unsupported_policy << ':' << m_config.lin_bucket << ':' << m_config.lout_bucket;
  return oss.str();
}

std::optional<ServiceEstimate> RealtimeServingRuntime::shared_gpu_prefill_estimate(
    const std::string& route,
    int context_tokens,
    int generated_tokens,
    int bs,
    const std::optional<ServiceEstimate>& est) {
  if (!m_config.share_gpu_prefill_across_routes || !est.has_value() || route == m_config.gpu_route_name) {
    return est;
  }
  auto gpu_est = estimate_request_for_route(m_config.gpu_route_name, context_tokens, generated_tokens, bs);
  if (!gpu_est.has_value()) {
    return est;
  }
  ServiceEstimate out = *est;
  out.prefill_e2e_ms = gpu_est->prefill_e2e_ms;
  out.prefill_gpu_ms = gpu_est->prefill_gpu_ms;
  out.prefill_link_ms = gpu_est->prefill_link_ms;
  out.prefill_pim_ms = gpu_est->prefill_pim_ms;
  out.prefill_e2e_clk = gpu_est->prefill_e2e_clk;
  out.prefill_gpu_clk = gpu_est->prefill_gpu_clk;
  out.prefill_link_clk = gpu_est->prefill_link_clk;
  out.prefill_pim_clk = gpu_est->prefill_pim_clk;
  out.prefill_energy_nj = gpu_est->prefill_energy_nj;
  return out;
}

std::optional<ServiceEstimate> RealtimeServingRuntime::estimate_request_cached(const std::string& route,
                                                                               int context_tokens,
                                                                               int generated_tokens,
                                                                               int bs) {
  const std::string key = estimate_cache_key(route, context_tokens, generated_tokens, bs);
  auto it = m_estimate_cache.find(key);
  if (it != m_estimate_cache.end()) {
    m_estimate_cache_hits++;
    return it->second;
  }

  m_estimate_cache_misses++;
  auto est = shared_gpu_prefill_estimate(route,
                                         context_tokens,
                                         generated_tokens,
                                         bs,
                                         estimate_request_for_route(route, context_tokens, generated_tokens, bs));
  m_estimate_cache.emplace(key, est);
  return est;
}

const GeneratedHybridArtifacts& RealtimeServingRuntime::generated_artifacts_for_lin(int lin) {
  auto it = m_generated_artifacts.find(lin);
  if (it != m_generated_artifacts.end()) {
    return it->second;
  }
  auto [inserted_it, _] = m_generated_artifacts.emplace(
      lin,
      build_lpddr5_bank_generated_artifacts(
          generator_geometry(), lin, m_config.translation_page_size_bytes));
  return inserted_it->second;
}

const std::vector<LogicalCommand>& RealtimeServingRuntime::generated_logical_trace_for_lin(int lin) {
  return generated_artifacts_for_lin(lin).commands;
}

void RealtimeServingRuntime::prefetch_request_estimates(const std::vector<int>& request_indices,
                                                        const std::vector<std::string>& routes) {
  std::vector<std::string> route_list = routes;
  if (route_list.empty()) {
    route_list = {m_config.gpu_route_name, m_config.hybrid_route_name};
  }
  if (m_config.share_gpu_prefill_across_routes &&
      std::find(route_list.begin(), route_list.end(), m_config.gpu_route_name) == route_list.end()) {
    route_list.push_back(m_config.gpu_route_name);
  }
  for (int idx : request_indices) {
    const auto& req = m_requests[idx];
    for (const auto& route : route_list) {
      const int bs = predicted_lookup_batch_size(route);
      const std::string key = estimate_cache_key(route, req.lin, req.lout, bs);
      if (m_estimate_cache.find(key) != m_estimate_cache.end()) {
        continue;
      }
      m_estimate_cache.emplace(key,
                               shared_gpu_prefill_estimate(
                                   route,
                                   req.lin,
                                   req.lout,
                                   bs,
                                   estimate_request_for_route(route, req.lin, req.lout, bs)));
      m_estimate_prefetch_populated++;
    }
  }
}

void RealtimeServingRuntime::prefetch_arrival_request_estimates() {
  if (!(use_incremental_prediction_for_fcfs_decode() || uses_slo_guarded_admission() ||
        uses_latency_guarded_energy_policy())) {
    return;
  }
  for (const auto& req : m_source_requests) {
    (void)estimate_request_cached(m_config.gpu_route_name,
                                  req.context_tokens,
                                  req.generated_tokens,
                                  predicted_lookup_batch_size(m_config.gpu_route_name));
    (void)estimate_request_cached(m_config.hybrid_route_name,
                                  req.context_tokens,
                                  req.generated_tokens,
                                  predicted_lookup_batch_size(m_config.hybrid_route_name));
    m_estimate_prefetch_populated += 2;
  }
}

int RealtimeServingRuntime::predicted_lookup_batch_size(const std::string& route) const {
  if (is_fcfs_strict()) {
    return 1;
  }
  if (use_shadow_prediction_for_fcfs_decode() ||
      use_incremental_prediction_for_fcfs_decode() ||
      use_heuristic_refresh_prediction_for_fcfs_decode()) {
    return 1;
  }
  int predicted_bs = std::max(1, m_config.batch_size);
  if (!m_config.enable_decode_batching) {
    return predicted_bs;
  }
  int same_route_decode = 0;
  for (const auto& req : m_requests) {
    if (req.route == route && req.state == RequestState::WaitingDecode && req.svc.has_value()) {
      same_route_decode++;
    }
  }
  predicted_bs = std::min(m_config.max_decode_batch_size, same_route_decode + 1);
  if (!is_prefill_priority_fcfs_decode()) {
    bool has_prefill = false;
    for (const auto& req : m_requests) {
      if (req.state == RequestState::WaitingPrefill && req.svc.has_value()) {
        has_prefill = true;
        break;
      }
    }
    if (has_prefill && m_config.decode_batch_cap_with_prefill > 0) {
      predicted_bs = std::min(predicted_bs, m_config.decode_batch_cap_with_prefill);
    }
  }
  return std::max(std::max(1, m_config.batch_size), predicted_bs);
}

int RealtimeServingRuntime::request_decode_context_tokens(const RuntimeRequest& req) const {
  return req.lin + 1 + req.decoded_tokens_done;
}

int RealtimeServingRuntime::request_decode_context_tokens(const ForecastRequest& req) const {
  return req.context_tokens + 1 + req.decoded_tokens_done;
}

bool RealtimeServingRuntime::has_template_for_lin(const std::string& route, int lin) const {
  if (route == m_config.hybrid_route_name && use_runtime_generated_hybrid()) {
    return lin > 0;
  }
  auto route_it = m_templates.find(route);
  if (route_it == m_templates.end()) {
    return false;
  }
  return route_it->second.find(lin) != route_it->second.end();
}

const TemplateTrace* RealtimeServingRuntime::get_template_for_lin(const std::string& route, int lin) const {
  if (use_runtime_generated_hybrid()) {
    return nullptr;
  }
  auto route_it = m_templates.find(route);
  if (route_it == m_templates.end()) {
    return nullptr;
  }
  auto it = route_it->second.find(lin);
  return (it == route_it->second.end()) ? nullptr : &it->second;
}

bool RealtimeServingRuntime::reserve_hybrid_request_memory(RuntimeRequest& req,
                                                           const ServiceEstimate& svc,
                                                           std::string& reason) {
  if (!use_runtime_generated_hybrid() || !m_pim_allocator) {
    return true;
  }

  const auto fp = m_allocator->request_footprint(req.lin);
  const uint32_t rid = static_cast<uint32_t>(req.request_id);

  // Allocate KvKey first, then KvValue. If KvValue fails, roll back KvKey.
  const bool key_ok = m_pim_allocator->allocate(PimRegion::KvKey,   rid, fp.kv_key_bytes);
  const bool val_ok = key_ok &&
                      m_pim_allocator->allocate(PimRegion::KvValue, rid, fp.kv_value_bytes);

  if (!key_ok || !val_ok) {
    if (key_ok) m_pim_allocator->free(PimRegion::KvKey, rid);  // rollback
    reason = "kv_capacity";
    req.allocator_fallback_reason = "kv_capacity";
    if (!req.allocator_first_block_clk.has_value()) {
      req.allocator_first_block_clk = m_clk;
    }
    return false;
  }

  req.pim_kv_bytes = fp.total_bytes();
  sync_allocator_stats();
  return true;
}

AllocationOutcome RealtimeServingRuntime::predict_hybrid_request_memory_fit(
    uint32_t request_id,
    int capacity_context_tokens,
    const ServiceEstimate& svc) {
  if (!use_runtime_generated_hybrid() || !m_allocator) {
    return {true, ""};
  }
  const auto& artifacts = generated_artifacts_for_lin(svc.mapped_lin);
  return m_allocator->can_fit_request_kv(
      request_id, capacity_context_tokens, *artifacts.kv_layout);
}

void RealtimeServingRuntime::release_hybrid_request_memory(RuntimeRequest& req) {
  if (!use_runtime_generated_hybrid() || req.route != m_config.hybrid_route_name) {
    return;
  }
  if (m_pim_allocator) {
    const uint32_t rid = static_cast<uint32_t>(req.request_id);
    m_pim_allocator->free(PimRegion::KvKey,   rid);
    m_pim_allocator->free(PimRegion::KvValue, rid);
    // Wake any requests that were held because KV memory was full.
    // They will be reconsidered at the next admission tick.
    wake_kv_held_requests();
  }
  sync_allocator_stats();
}

void RealtimeServingRuntime::wake_kv_held_requests() {
  // Reset the retry clock for all requests currently blocked on KV capacity,
  // so process_admission_queue() picks them up at the very next tick.
  for (auto& req : m_requests) {
    if (req.state == RequestState::WaitingAdmission &&
        req.admission_last_reject_reason == "kv_capacity") {
      req.admission_next_retry_clk = m_clk;
    }
  }
}

bool RealtimeServingRuntime::has_kv_held_requests() const {
  for (const auto& req : m_requests) {
    if (req.state == RequestState::WaitingAdmission &&
        req.admission_last_reject_reason == "kv_capacity") {
      return true;
    }
  }
  return false;
}

bool RealtimeServingRuntime::reserve_hybrid_task_memory(const TaskCandidate& tc, std::string& reason) {
  if (!use_runtime_generated_hybrid() || !m_allocator ||
      tc.route != m_config.hybrid_route_name || tc.phase != TaskPhase::Decode) {
    return true;
  }
  int batch_lin = 0;
  for (int idx : tc.request_indices) {
    batch_lin = std::max(batch_lin, request_decode_context_tokens(m_requests[idx]));
  }
  const auto outcome = m_allocator->reserve_decode_task(tc.request_indices, batch_lin, tc.batch_size);
  if (!outcome.success) {
    reason = outcome.reason;
    return false;
  }
  const auto fp = m_allocator->task_footprint(batch_lin, tc.batch_size);
  for (int idx : tc.request_indices) {
    m_requests[idx].pim_transient_peak_bytes =
        std::max<uint64_t>(m_requests[idx].pim_transient_peak_bytes, fp.total_bytes());
  }
  sync_allocator_stats();
  return true;
}

AllocationOutcome RealtimeServingRuntime::predict_hybrid_task_memory_fit(
    const std::vector<int>& request_indices,
    int decode_context_tokens,
    int batch_size) const {
  if (!use_runtime_generated_hybrid() || !m_allocator) {
    return {true, ""};
  }
  return m_allocator->can_fit_decode_task(request_indices, decode_context_tokens, batch_size);
}

void RealtimeServingRuntime::release_hybrid_task_memory(const ActiveTask& task) {
  if (!use_runtime_generated_hybrid() || !m_allocator ||
      task.route != m_config.hybrid_route_name || task.phase != TaskPhase::Decode) {
    return;
  }
  m_allocator->release_decode_task(task.request_indices);
  sync_allocator_stats();
}

void RealtimeServingRuntime::sync_allocator_stats() {
  if (!m_address_space) {
    return;
  }
  const auto& stats = m_address_space->stats();
  const auto key_peak = stats.peak_region_bytes.count(MemoryRegionKind::KvKey)
                            ? stats.peak_region_bytes.at(MemoryRegionKind::KvKey)
                            : 0;
  const auto value_peak = stats.peak_region_bytes.count(MemoryRegionKind::KvValue)
                              ? stats.peak_region_bytes.at(MemoryRegionKind::KvValue)
                              : 0;
  m_stats.pim_memory_peak_bytes = std::max(m_stats.pim_memory_peak_bytes, stats.peak_total_bytes);
  m_stats.kv_region_peak_bytes =
      std::max<uint64_t>(m_stats.kv_region_peak_bytes, key_peak + value_peak);
  m_stats.score_region_peak_bytes =
      std::max<uint64_t>(m_stats.score_region_peak_bytes,
                         stats.peak_region_bytes.count(MemoryRegionKind::Score)
                             ? stats.peak_region_bytes.at(MemoryRegionKind::Score)
                             : 0);
  m_stats.context_region_peak_bytes =
      std::max<uint64_t>(m_stats.context_region_peak_bytes,
                         stats.peak_region_bytes.count(MemoryRegionKind::Context)
                             ? stats.peak_region_bytes.at(MemoryRegionKind::Context)
                             : 0);
  m_stats.scratch_region_peak_bytes =
      std::max<uint64_t>(m_stats.scratch_region_peak_bytes,
                         stats.peak_region_bytes.count(MemoryRegionKind::Scratch)
                             ? stats.peak_region_bytes.at(MemoryRegionKind::Scratch)
                             : 0);
}

std::optional<RouteCandidate> RealtimeServingRuntime::find_candidate(
    const std::vector<RouteCandidate>& candidates, const std::string& route) const {
  auto it = std::find_if(candidates.begin(), candidates.end(), [&](const RouteCandidate& cand) {
    return cand.route == route;
  });
  return (it == candidates.end()) ? std::optional<RouteCandidate>{} : std::optional<RouteCandidate>(*it);
}

void RealtimeServingRuntime::inc_route_count(const std::string& route, int delta) {
  if (route.empty() || delta == 0) {
    return;
  }
  const int next = m_route_counts[route] + delta;
  if (next <= 0) {
    m_route_counts.erase(route);
  } else {
    m_route_counts[route] = next;
  }
}

void RealtimeServingRuntime::move_route_count(const std::string& old_route, const std::string& new_route) {
  if (old_route == new_route) {
    return;
  }
  inc_route_count(old_route, -1);
  inc_route_count(new_route, 1);
}

QueuePressureSnapshot RealtimeServingRuntime::queue_pressure_snapshot() const {
  QueuePressureSnapshot snap;
  for (const auto& req : m_requests) {
    if (!req.svc.has_value()) {
      continue;
    }
    if (req.state != RequestState::WaitingPrefill &&
        req.state != RequestState::WaitingDecode) {
      continue;
    }
    snap.active_requests++;
    if (req.state == RequestState::WaitingPrefill) {
      snap.pending_gpu_work_ms += req.svc->prefill_gpu_ms;
      snap.pending_pim_work_ms += req.svc->prefill_pim_ms;
    }
    if (req.remaining_decode_tokens > 0) {
      snap.pending_decode_tokens += req.remaining_decode_tokens;
      snap.pending_gpu_work_ms += req.remaining_decode_tokens * req.svc->decode_gpu_ms;
      snap.pending_pim_work_ms += req.remaining_decode_tokens * req.svc->decode_pim_ms;
    }
  }
  return snap;
}

double RealtimeServingRuntime::queue_pressure_adjustment_ms(double prefill_gpu_ms,
                                                            double prefill_pim_ms,
                                                            double prefill_e2e_ms,
                                                            double decode_gpu_ms,
                                                            double decode_pim_ms,
                                                            double decode_e2e_ms,
                                                            int decode_tokens) const {
  if (m_config.gpu_queue_alpha <= 0.0 &&
      m_config.pim_queue_alpha <= 0.0 &&
      m_config.active_request_alpha <= 0.0 &&
      m_config.decode_token_alpha <= 0.0) {
    return 0.0;
  }

  const QueuePressureSnapshot snap = queue_pressure_snapshot();
  const double decode_gpu_total = decode_gpu_ms * static_cast<double>(decode_tokens);
  const double decode_pim_total = decode_pim_ms * static_cast<double>(decode_tokens);
  const double decode_e2e_total = decode_e2e_ms * static_cast<double>(decode_tokens);
  const double req_gpu_work_ms = prefill_gpu_ms + decode_gpu_total;
  const double req_pim_work_ms = prefill_pim_ms + decode_pim_total;
  const double req_total_ms = std::max(prefill_e2e_ms + decode_e2e_total, 1e-9);
  const double gpu_share = (req_gpu_work_ms > 0.0) ? (req_gpu_work_ms / req_total_ms) : 0.0;
  const double pim_share = (req_pim_work_ms > 0.0) ? (req_pim_work_ms / req_total_ms) : 0.0;

  double queue_pressure_ms = 0.0;
  queue_pressure_ms += m_config.gpu_queue_alpha * snap.pending_gpu_work_ms * gpu_share;
  queue_pressure_ms += m_config.pim_queue_alpha * snap.pending_pim_work_ms * pim_share;
  queue_pressure_ms += m_config.active_request_alpha * snap.active_requests;
  queue_pressure_ms += m_config.decode_token_alpha *
                       snap.pending_decode_tokens *
                       std::max(decode_e2e_ms, 0.0);
  return queue_pressure_ms;
}

void RealtimeServingRuntime::annotate_route_candidate(RouteCandidate& cand) const {
  int waiting_prefill_count = 0;
  int waiting_decode_count = 0;
  int pending_decode_tokens = 0;
  for (const auto& req : m_requests) {
    if (req.route != cand.route) {
      continue;
    }
    if (req.state == RequestState::WaitingPrefill) {
      waiting_prefill_count++;
    } else if (req.state == RequestState::WaitingDecode) {
      waiting_decode_count++;
      pending_decode_tokens += std::max(0, req.remaining_decode_tokens);
    }
  }
  cand.route_waiting_prefill_count = waiting_prefill_count;
  cand.route_waiting_decode_count = waiting_decode_count;
  cand.route_pending_decode_tokens = pending_decode_tokens;
  cand.route_active_requests = waiting_prefill_count + waiting_decode_count;
}

void RealtimeServingRuntime::enqueue_arrivals_up_to_now() {
  while (m_arrival_idx < m_source_requests.size() && m_source_requests[m_arrival_idx].arrival_clk <= m_clk) {
    const auto& src = m_source_requests[m_arrival_idx++];
    RuntimeRequest req;
    req.request_id = src.request_id;
    req.arrival_ms = src.arrival_ms;
    req.arrival_clk = src.arrival_clk;
    req.lin = src.context_tokens;
    req.lout = src.generated_tokens;
    req.state = RequestState::WaitingAdmission;
    req.ready_clk = src.arrival_clk;
    req.admission_next_retry_clk = m_clk;
    if (m_config.slo_e2e_ms >= 0.0) {
      req.deadline_ms = req.arrival_ms + m_config.slo_e2e_ms;
      req.deadline_clk = ms_to_cycles(*req.deadline_ms);
    }
    m_requests.push_back(req);
    record_debug_event("arrival_enqueued",
                       {{"request_id", std::to_string(req.request_id)},
                        {"request_arrival_ms", format_ms(req.arrival_ms)},
                        {"context_tokens", std::to_string(req.lin)},
                        {"generated_tokens", std::to_string(req.lout)}});
  }
}

void RealtimeServingRuntime::admit_waiting_request(RuntimeRequest& req,
                                                   const std::string& route,
                                                   const ServiceEstimate& est,
                                                   const RouteCandidate& cand,
                                                   const std::string& reason,
                                                   bool forced_best_effort) {
  const double predicted_finish_ms = quantize_export_ms(cand.predicted_finish_ms);
  const double predicted_gpu_wait_ms = quantize_export_ms(cand.predicted_gpu_wait_ms);
  const double predicted_pim_wait_ms = quantize_export_ms(cand.predicted_pim_wait_ms);
  const auto chosen_queue_pressure_ms = quantize_export_ms(cand.queue_pressure_ms);
  const auto chosen_slack_ms =
      req.deadline_ms.has_value() ? std::optional<double>(*req.deadline_ms - predicted_finish_ms)
                                  : quantize_export_ms(cand.slack_ms);

  req.route = route;
  req.admission_route = route;
  req.route_decision_reason = reason;
  req.admission_forced_best_effort = forced_best_effort;
  req.svc = est;
  req.remaining_decode_tokens = est.decode_tokens;
  req.state = RequestState::WaitingPrefill;
  req.ready_clk = m_clk;
  req.admission_time_clk = m_clk;
  req.admission_time_ms = cycles_to_ms(m_clk);
  req.admission_queue_wait_ms = req.admission_time_ms.value() - req.arrival_ms;
  if (req.allocator_first_block_clk.has_value()) {
    req.allocator_wait_ms = cycles_to_ms(m_clk - *req.allocator_first_block_clk);
  }
  req.admission_last_reject_reason.clear();
  req.admission_last_harmed_count = cand.harmed_count;
  req.admission_last_total_harm_ms = cand.total_harm_ms;
  req.admission_last_single_harm_ms = cand.max_single_harm_ms;
  req.admission_last_own_miss_ms = cand.own_miss_ms;
  req.chosen_predicted_finish_ms = predicted_finish_ms;
  req.latest_predicted_finish_ms = predicted_finish_ms;
  req.chosen_predicted_gpu_wait_ms = predicted_gpu_wait_ms;
  req.chosen_predicted_pim_wait_ms = predicted_pim_wait_ms;
  req.chosen_predicted_incremental_energy_nj = cand.predicted_incremental_energy_nj;
  req.chosen_predicted_incremental_decode_energy_nj = cand.predicted_incremental_decode_energy_nj;
  req.chosen_queue_pressure_ms = chosen_queue_pressure_ms;
  req.chosen_slack_ms = chosen_slack_ms;

  if (req.svc->clipped_lin) {
    m_stats.clipped_lin_count++;
  }
  if (req.svc->clipped_lout) {
    m_stats.clipped_lout_count++;
  }
  inc_route_count(req.route, 1);
  m_admission_queue_wait_samples_ms.push_back(std::max(0.0, *req.admission_queue_wait_ms));
  if (reason.find("fallback_gpu") != std::string::npos) {
    m_stats.route_fallback_count++;
  }
  m_stats.admitted_requests++;
  record_debug_event("admit_route",
                     {{"request_id", std::to_string(req.request_id)},
                      {"request_arrival_ms", format_ms(req.arrival_ms)},
                      {"context_tokens", std::to_string(req.lin)},
                      {"generated_tokens", std::to_string(req.lout)},
                      {"admission_route", req.admission_route},
                      {"chosen_route", req.route},
                      {"decision_reason", req.route_decision_reason},
                      {"predicted_finish_ms", format_ms(predicted_finish_ms)},
                      {"predicted_gpu_wait_ms", format_ms(predicted_gpu_wait_ms)},
                      {"predicted_pim_wait_ms", format_ms(predicted_pim_wait_ms)},
                      {"is_lost", req.is_lost ? "1" : "0"},
                      {"forced_best_effort", req.admission_forced_best_effort ? "1" : "0"},
                      {"admission_attempts", std::to_string(req.admission_attempts)},
                      {"admission_bypass_count", std::to_string(req.admission_bypass_count)},
                      {"admission_queue_wait_ms", format_ms(std::max(0.0, *req.admission_queue_wait_ms))}});
}

void RealtimeServingRuntime::admit_arrivals_up_to_now() {
  while (m_arrival_idx < m_source_requests.size() && m_source_requests[m_arrival_idx].arrival_clk <= m_clk) {
    const auto& src = m_source_requests[m_arrival_idx++];
    RuntimeRequest req;
    req.request_id = src.request_id;
    req.arrival_ms = src.arrival_ms;
    req.arrival_clk = src.arrival_clk;
    req.lin = src.context_tokens;
    req.lout = src.generated_tokens;
    req.ready_clk = src.arrival_clk;
    if (m_config.slo_e2e_ms >= 0.0) {
      req.deadline_ms = req.arrival_ms + m_config.slo_e2e_ms;
      req.deadline_clk = ms_to_cycles(*req.deadline_ms);
    }

    if (src.context_tokens <= 0 || src.generated_tokens <= 0) {
      req.state = RequestState::Dropped;
      req.dropped_reason = "invalid_shape";
      req.route_decision_reason = req.dropped_reason;
      req.fallback_reason = req.dropped_reason;
      m_requests.push_back(req);
      m_stats.dropped_requests++;
      continue;
    }

    std::optional<double> baseline_total_energy_nj;
    std::optional<double> baseline_total_decode_energy_nj;
    if (uses_latency_guarded_energy_policy()) {
      const auto baseline_run = incremental_active_forecast_results();
      baseline_total_energy_nj = baseline_run.total_prefill_energy_nj + baseline_run.total_decode_energy_nj;
      baseline_total_decode_energy_nj = baseline_run.total_decode_energy_nj;
    }

    std::vector<RouteCandidate> candidates;
    candidates.push_back(predict_route_candidate(src,
                                                 m_config.gpu_route_name,
                                                 req.deadline_ms,
                                                 baseline_total_energy_nj,
                                                 baseline_total_decode_energy_nj));
    candidates.push_back(predict_route_candidate(src,
                                                 m_config.hybrid_route_name,
                                                 req.deadline_ms,
                                                 baseline_total_energy_nj,
                                                 baseline_total_decode_energy_nj));

    const RouteDecision decision = choose_route(candidates, m_config, req.deadline_ms);
    const auto* chosen = [&]() -> const RouteCandidate* {
      if (!decision.route.has_value()) {
        return nullptr;
      }
      for (const auto& cand : candidates) {
        if (cand.route == *decision.route) {
          return &cand;
        }
      }
      return nullptr;
    }();
    if (chosen == nullptr || !chosen->svc.has_value() || decision.dropped) {
      req.state = RequestState::Dropped;
      req.dropped_reason = decision.reason.empty() ? best_drop_reason(candidates) : decision.reason;
      req.route_decision_reason = req.dropped_reason;
      req.fallback_reason = req.dropped_reason;
      m_requests.push_back(req);
      m_stats.dropped_requests++;
      record_debug_event("admit_drop",
                         {{"request_id", std::to_string(req.request_id)},
                          {"request_arrival_ms", format_ms(req.arrival_ms)},
                          {"context_tokens", std::to_string(req.lin)},
                          {"generated_tokens", std::to_string(req.lout)},
                          {"decision_reason", req.dropped_reason}});
      continue;
    }

    if (*decision.route == m_config.gpu_route_name) {
      for (const auto& cand : candidates) {
        if (cand.route == m_config.hybrid_route_name && !cand.eligible) {
          req.fallback_reason = cand.reason;
          break;
        }
      }
    }

    m_requests.push_back(req);
    auto& admitted = m_requests.back();
    std::string final_route = *decision.route;
    ServiceEstimate final_svc = *chosen->svc;
    RouteCandidate final_cand = *chosen;
    std::string final_reason =
        (!admitted.fallback_reason.empty() && *decision.route == m_config.gpu_route_name)
            ? ("fallback_gpu_due_to_" + admitted.fallback_reason)
            : decision.reason;

    if (final_route == m_config.hybrid_route_name) {
      std::string allocator_reason;
      if (!reserve_hybrid_request_memory(admitted, final_svc, allocator_reason)) {

        // ---- KV OOM: hold the request and retry after memory is freed ----
        // Check if we timed out first. If admission_max_wait_ms is configured
        // and has elapsed, skip the hold and fall through to GPU fallback/drop.
        const bool timed_out =
            m_config.admission_max_wait_ms > 0.0 &&
            cycles_to_ms(m_clk - admitted.arrival_clk) > m_config.admission_max_wait_ms;

        if (allocator_reason == "kv_capacity" && !timed_out) {
          // Leave request in WaitingAdmission. wake_kv_held_requests() will
          // reset this clock when another request finishes and frees KV pages.
          admitted.state = RequestState::WaitingAdmission;
          admitted.admission_last_reject_reason = "kv_capacity";
          admitted.admission_next_retry_clk =
              m_clk + ms_to_cycles(std::max(1.0, m_config.admission_retry_interval_ms));
          m_stats.allocator_hold_count++;
          continue;
        }

        // ---- Fallback: try GPU route or drop ----
        const auto gpu_cand = find_candidate(candidates, m_config.gpu_route_name);
        if (gpu_cand.has_value() && gpu_cand->eligible && gpu_cand->svc.has_value()) {
          final_route = m_config.gpu_route_name;
          final_svc = *gpu_cand->svc;
          final_cand = *gpu_cand;
          final_reason = "fallback_gpu_due_to_" + allocator_reason;
          admitted.fallback_reason = allocator_reason;
          m_stats.allocator_gpu_fallback_count++;
        } else {
          admitted.state = RequestState::Dropped;
          admitted.dropped_reason = allocator_reason;
          admitted.route_decision_reason = allocator_reason;
          admitted.allocator_fallback_reason = allocator_reason;
          m_stats.allocator_drop_count++;
          m_stats.dropped_requests++;
          record_debug_event("admit_drop",
                             {{"request_id", std::to_string(admitted.request_id)},
                              {"decision_reason", allocator_reason}});
          continue;
        }
      }
    }

    admit_waiting_request(admitted, final_route, final_svc, final_cand, final_reason, false);
  }
}

std::vector<int> RealtimeServingRuntime::waiting_admission_request_indices() const {
  std::vector<int> out;
  for (int i = 0; i < static_cast<int>(m_requests.size()); ++i) {
    if (m_requests[i].state == RequestState::WaitingAdmission) {
      out.push_back(i);
    }
  }
  std::sort(out.begin(), out.end(), [&](int lhs, int rhs) {
    return std::tie(m_requests[lhs].arrival_clk, m_requests[lhs].request_id) <
           std::tie(m_requests[rhs].arrival_clk, m_requests[rhs].request_id);
  });
  return out;
}

std::optional<Clk_t> RealtimeServingRuntime::min_waiting_admission_retry_clk() const {
  Clk_t best = kInfClk;
  bool found = false;
  for (const auto& req : m_requests) {
    if (req.state != RequestState::WaitingAdmission) {
      continue;
    }
    best = std::min(best, req.admission_next_retry_clk);
    found = true;
  }
  return found ? std::optional<Clk_t>(best) : std::nullopt;
}

// Remaining forecasting, admission, task-selection, and task-execution
// helpers are implemented in runtime_forecast.cpp and below in this file.

std::optional<int> RealtimeServingRuntime::fcfs_active_request_index() {
  if (!is_fcfs_strict()) {
    return std::nullopt;
  }
  if (m_fcfs_active_request_idx.has_value()) {
    const auto& req = m_requests[*m_fcfs_active_request_idx];
    if ((req.state == RequestState::WaitingPrefill && req.svc.has_value()) ||
        (req.state == RequestState::WaitingDecode && req.svc.has_value() && req.remaining_decode_tokens > 0)) {
      return m_fcfs_active_request_idx;
    }
    m_fcfs_active_request_idx.reset();
  }

  int best = -1;
  for (int i = 0; i < static_cast<int>(m_requests.size()); ++i) {
    const auto& req = m_requests[i];
    if (!req.svc.has_value()) {
      continue;
    }
    if (req.state != RequestState::WaitingPrefill &&
        !(req.state == RequestState::WaitingDecode && req.remaining_decode_tokens > 0)) {
      continue;
    }
    const double lhs_admit = req.admission_time_ms.has_value() ? *req.admission_time_ms : req.arrival_ms;
    const double rhs_admit = (best >= 0 && m_requests[best].admission_time_ms.has_value())
                                 ? *m_requests[best].admission_time_ms
                                 : (best >= 0 ? m_requests[best].arrival_ms : std::numeric_limits<double>::infinity());
    if (best < 0 || std::tie(lhs_admit, req.request_id) <
                         std::tie(rhs_admit, m_requests[best].request_id)) {
      best = i;
    }
  }
  if (best >= 0) {
    m_fcfs_active_request_idx = best;
    return best;
  }
  return std::nullopt;
}

void RealtimeServingRuntime::refresh_fcfs_active_request() {
  if (!is_fcfs_strict() || !m_fcfs_active_request_idx.has_value()) {
    return;
  }
  const auto& req = m_requests[*m_fcfs_active_request_idx];
  if ((req.state == RequestState::WaitingPrefill && req.svc.has_value()) ||
      (req.state == RequestState::WaitingDecode && req.svc.has_value() && req.remaining_decode_tokens > 0)) {
    return;
  }
  m_fcfs_active_request_idx.reset();
}

std::optional<int> RealtimeServingRuntime::pick_oldest_prefill_request_index() const {
  int best = -1;
  for (int i = 0; i < static_cast<int>(m_requests.size()); ++i) {
    const auto& req = m_requests[i];
    if (req.state != RequestState::WaitingPrefill || !req.svc.has_value()) {
      continue;
    }
    if (best < 0 ||
        std::tie(req.ready_clk, req.arrival_clk, req.request_id) <
            std::tie(m_requests[best].ready_clk, m_requests[best].arrival_clk, m_requests[best].request_id)) {
      best = i;
    }
  }
  return (best >= 0) ? std::optional<int>(best) : std::nullopt;
}

void RealtimeServingRuntime::transition_to_waiting_decode(RuntimeRequest& req, Clk_t ready_clk) {
  req.state = RequestState::WaitingDecode;
  req.ready_clk = ready_clk;
  if (is_prefill_priority_fcfs_decode() && !req.decode_enqueue_seq.has_value()) {
    req.decode_enqueue_seq = m_next_decode_enqueue_seq++;
    record_debug_event("decode_enqueue",
                       {{"request_id", std::to_string(req.request_id)},
                        {"route", req.route},
                        {"decode_enqueue_seq", std::to_string(*req.decode_enqueue_seq)},
                        {"ready_ms", format_ms(cycles_to_ms(ready_clk))}});
  }
}

std::vector<int> RealtimeServingRuntime::fcfs_decode_queue_indices() {
  std::vector<int> out;
  for (int i = 0; i < static_cast<int>(m_requests.size()); ++i) {
    auto& req = m_requests[i];
    if (req.state != RequestState::WaitingDecode || !req.svc.has_value() || req.remaining_decode_tokens <= 0) {
      continue;
    }
    if (!req.decode_enqueue_seq.has_value()) {
      req.decode_enqueue_seq = m_next_decode_enqueue_seq++;
    }
    out.push_back(i);
  }
  std::sort(out.begin(), out.end(), [&](int lhs, int rhs) {
    return std::tie(*m_requests[lhs].decode_enqueue_seq, m_requests[lhs].request_id) <
           std::tie(*m_requests[rhs].decode_enqueue_seq, m_requests[rhs].request_id);
  });
  return out;
}

std::optional<TaskCandidate> RealtimeServingRuntime::prefill_task_for_request(int request_idx) const {
  const auto& req = m_requests[request_idx];
  if (req.state != RequestState::WaitingPrefill || !req.svc.has_value()) {
    return std::nullopt;
  }
  TaskCandidate tc;
  tc.request_indices = {request_idx};
  tc.route = req.route;
  tc.phase = TaskPhase::Prefill;
  tc.ready_clk = req.ready_clk;
  tc.gpu_clk = req.svc->prefill_gpu_clk;
  tc.link_clk = req.svc->prefill_link_clk;
  tc.pim_clk = req.svc->prefill_pim_clk;
  tc.e2e_clk = req.svc->prefill_e2e_clk;
  tc.prefill_energy_nj = req.svc->prefill_energy_nj;
  tc.decode_energy_nj = 0.0;
  tc.batch_svc = *req.svc;
  tc.batch_size = 1;
  tc.earliest_start_clk = predict_stage_start(req.ready_clk,
                                              tc.gpu_clk,
                                              tc.link_clk,
                                              tc.pim_clk,
                                              m_gpu_busy_until,
                                              m_link_busy_until,
                                              m_pim_busy_until);
  return tc;
}

std::optional<ServiceEstimate> RealtimeServingRuntime::estimate_prefill_batch(
    const std::string& route, const std::vector<int>& request_indices) {
  if (request_indices.empty()) {
    return std::nullopt;
  }
  int batch_lin = 0;
  int batch_lout = 0;
  for (int idx : request_indices) {
    batch_lin = std::max(batch_lin, m_requests[idx].lin);
    batch_lout = std::max(batch_lout, m_requests[idx].lout);
  }
  return estimate_request_cached(route, batch_lin, batch_lout, static_cast<int>(request_indices.size()));
}

std::vector<TaskCandidate> RealtimeServingRuntime::build_prefill_batch_candidates() const {
  std::unordered_map<std::string, std::vector<int>> grouped;
  for (int i = 0; i < static_cast<int>(m_requests.size()); ++i) {
    const auto& req = m_requests[i];
    if (req.state != RequestState::WaitingPrefill || !req.svc.has_value() || req.route.empty()) {
      continue;
    }
    grouped[req.route].push_back(i);
  }

  std::vector<TaskCandidate> candidates;
  for (auto& [route, idxs] : grouped) {
    std::sort(idxs.begin(), idxs.end(), [&](int lhs, int rhs) {
      return std::tie(m_requests[lhs].ready_clk, m_requests[lhs].arrival_clk, m_requests[lhs].request_id) <
             std::tie(m_requests[rhs].ready_clk, m_requests[rhs].arrival_clk, m_requests[rhs].request_id);
    });
    const Clk_t cutoff = std::max(m_clk, m_requests[idxs.front()].ready_clk);
    std::vector<int> cohort;
    for (int idx : idxs) {
      if (m_requests[idx].ready_clk <= cutoff) {
        cohort.push_back(idx);
      }
    }
    if (cohort.empty()) {
      continue;
    }
    cohort.resize(std::min<int>(cohort.size(), m_config.max_prefill_batch_size));
    auto batch_est = const_cast<RealtimeServingRuntime*>(this)->estimate_prefill_batch(route, cohort);
    if (!batch_est.has_value()) {
      continue;
    }
    TaskCandidate tc;
    tc.request_indices = cohort;
    tc.route = route;
    tc.phase = TaskPhase::Prefill;
    tc.ready_clk = cutoff;
    tc.gpu_clk = batch_est->prefill_gpu_clk;
    tc.link_clk = batch_est->prefill_link_clk;
    tc.pim_clk = batch_est->prefill_pim_clk;
    tc.e2e_clk = batch_est->prefill_e2e_clk;
    tc.prefill_energy_nj = batch_est->prefill_energy_nj;
    tc.batch_svc = *batch_est;
    tc.batch_size = static_cast<int>(cohort.size());
    tc.earliest_start_clk = predict_stage_start(cutoff,
                                                tc.gpu_clk,
                                                tc.link_clk,
                                                tc.pim_clk,
                                                m_gpu_busy_until,
                                                m_link_busy_until,
                                                m_pim_busy_until);
    candidates.push_back(std::move(tc));
  }
  return candidates;
}

std::optional<TaskCandidate> RealtimeServingRuntime::single_decode_task_for_request(int request_idx) const {
  const auto& req = m_requests[request_idx];
  if (req.state != RequestState::WaitingDecode || !req.svc.has_value() || req.remaining_decode_tokens <= 0) {
    return std::nullopt;
  }
  TaskCandidate tc;
  tc.request_indices = {request_idx};
  tc.route = req.route;
  tc.phase = TaskPhase::Decode;
  tc.ready_clk = req.ready_clk;
  tc.gpu_clk = req.svc->decode_gpu_clk;
  tc.link_clk = req.svc->decode_link_clk;
  tc.pim_clk = req.svc->decode_pim_clk;
  tc.e2e_clk = req.svc->decode_e2e_clk;
  tc.decode_energy_nj = req.svc->decode_energy_nj;
  tc.batch_svc = *req.svc;
  tc.batch_size = 1;
  tc.earliest_start_clk = predict_stage_start(req.ready_clk,
                                              tc.gpu_clk,
                                              tc.link_clk,
                                              tc.pim_clk,
                                              m_gpu_busy_until,
                                              m_link_busy_until,
                                              m_pim_busy_until);
  return tc;
}

std::vector<TaskCandidate> RealtimeServingRuntime::build_decode_batch_candidates() const {
  std::unordered_map<std::string, std::vector<int>> grouped;
  for (int i = 0; i < static_cast<int>(m_requests.size()); ++i) {
    const auto& req = m_requests[i];
    if (req.state != RequestState::WaitingDecode || !req.svc.has_value() ||
        req.remaining_decode_tokens <= 0 || req.route.empty()) {
      continue;
    }
    grouped[req.route].push_back(i);
  }

  bool has_waiting_prefill = false;
  for (const auto& req : m_requests) {
    if (req.state == RequestState::WaitingPrefill && req.svc.has_value()) {
      has_waiting_prefill = true;
      break;
    }
  }

  std::vector<TaskCandidate> candidates;
  for (auto& [route, idxs] : grouped) {
    std::sort(idxs.begin(), idxs.end(), [&](int lhs, int rhs) {
      return std::tie(m_requests[lhs].ready_clk, m_requests[lhs].arrival_clk, m_requests[lhs].request_id) <
             std::tie(m_requests[rhs].ready_clk, m_requests[rhs].arrival_clk, m_requests[rhs].request_id);
    });
    const Clk_t cutoff = std::max(m_clk, m_requests[idxs.front()].ready_clk);
    std::vector<int> cohort;
    for (int idx : idxs) {
      if (m_requests[idx].ready_clk <= cutoff) {
        cohort.push_back(idx);
      }
    }
    if (cohort.empty()) {
      continue;
    }
    int max_batch = std::min<int>(cohort.size(), m_config.max_decode_batch_size);
    if (has_waiting_prefill && m_config.decode_batch_cap_with_prefill > 0) {
      max_batch = std::min(max_batch, m_config.decode_batch_cap_with_prefill);
    }
    std::optional<TaskCandidate> fallback_unfit;
    for (int bs = max_batch; bs >= 1; --bs) {
      std::vector<int> batch_indices(cohort.begin(), cohort.begin() + bs);
      int batch_lin = 0;
      for (int idx : batch_indices) {
        batch_lin = std::max(batch_lin,
                             const_cast<RealtimeServingRuntime*>(this)->request_decode_context_tokens(
                                 m_requests[idx]));
      }
      auto batch_est = (bs == 1)
                           ? m_requests[batch_indices.front()].svc
                           : const_cast<RealtimeServingRuntime*>(this)->estimate_request_cached(
                                 route,
                                 batch_lin,
                                 2,
                                 bs);
      if (!batch_est.has_value()) {
        continue;
      }
      TaskCandidate tc;
      tc.request_indices = batch_indices;
      tc.route = route;
      tc.phase = TaskPhase::Decode;
      tc.ready_clk = cutoff;
      tc.gpu_clk = batch_est->decode_gpu_clk;
      tc.link_clk = batch_est->decode_link_clk;
      tc.pim_clk = batch_est->decode_pim_clk;
      tc.e2e_clk = batch_est->decode_e2e_clk;
      tc.decode_energy_nj = batch_est->decode_energy_nj;
      tc.batch_svc = *batch_est;
      tc.batch_size = bs;
      tc.earliest_start_clk = predict_stage_start(cutoff,
                                                  tc.gpu_clk,
                                                  tc.link_clk,
                                                  tc.pim_clk,
                                                  m_gpu_busy_until,
                                                  m_link_busy_until,
                                                  m_pim_busy_until);
      if (route == m_config.hybrid_route_name) {
        const auto fit = const_cast<RealtimeServingRuntime*>(this)->predict_hybrid_task_memory_fit(
            batch_indices, batch_lin, bs);
        if (!fit.success) {
          if (!fallback_unfit.has_value() || bs == 1) {
            fallback_unfit = tc;
          }
          continue;
        }
      }
      candidates.push_back(std::move(tc));
      break;
    }
    if (fallback_unfit.has_value()) {
      candidates.push_back(*fallback_unfit);
    }
  }
  return candidates;
}

std::optional<TaskCandidate> RealtimeServingRuntime::build_fcfs_prefix_decode_task() const {
  auto decode_q = const_cast<RealtimeServingRuntime*>(this)->fcfs_decode_queue_indices();
  if (decode_q.empty()) {
    return std::nullopt;
  }
  const int head_idx = decode_q.front();
  const auto& head = m_requests[head_idx];
  const Clk_t cutoff = std::max(m_clk, head.ready_clk);
  std::vector<int> prefix = {head_idx};
  if (m_config.enable_decode_batching) {
    for (size_t i = 1; i < decode_q.size(); ++i) {
      if (static_cast<int>(prefix.size()) >= m_config.max_decode_batch_size) {
        break;
      }
      const auto& rr = m_requests[decode_q[i]];
      if (rr.route != head.route || rr.ready_clk > cutoff) {
        break;
      }
      prefix.push_back(decode_q[i]);
    }
  }
  if (!m_config.enable_decode_batching) {
    prefix.resize(1);
  }

  std::optional<TaskCandidate> fallback_unfit;
  for (int bs = static_cast<int>(prefix.size()); bs >= 1; --bs) {
    std::vector<int> batch_indices(prefix.begin(), prefix.begin() + bs);
    int batch_lin = 0;
    for (int idx : batch_indices) {
      batch_lin = std::max(batch_lin,
                           const_cast<RealtimeServingRuntime*>(this)->request_decode_context_tokens(
                               m_requests[idx]));
    }
    auto batch_est = (bs == 1)
                         ? m_requests[batch_indices.front()].svc
                         : const_cast<RealtimeServingRuntime*>(this)->estimate_request_cached(
                               head.route,
                               batch_lin,
                               2,
                               bs);
    if (!batch_est.has_value()) {
      continue;
    }
    TaskCandidate tc;
    tc.request_indices = batch_indices;
    tc.route = head.route;
    tc.phase = TaskPhase::Decode;
    tc.ready_clk = cutoff;
    tc.gpu_clk = batch_est->decode_gpu_clk;
    tc.link_clk = batch_est->decode_link_clk;
    tc.pim_clk = batch_est->decode_pim_clk;
    tc.e2e_clk = batch_est->decode_e2e_clk;
    tc.decode_energy_nj = batch_est->decode_energy_nj;
    tc.batch_svc = *batch_est;
    tc.batch_size = bs;
    tc.earliest_start_clk = predict_stage_start(cutoff,
                                                tc.gpu_clk,
                                                tc.link_clk,
                                                tc.pim_clk,
                                                m_gpu_busy_until,
                                                m_link_busy_until,
                                                m_pim_busy_until);
    if (head.route == m_config.hybrid_route_name) {
      const auto fit = const_cast<RealtimeServingRuntime*>(this)->predict_hybrid_task_memory_fit(
          batch_indices, batch_lin, bs);
      if (!fit.success) {
        if (!fallback_unfit.has_value() || bs == 1) {
          fallback_unfit = tc;
        }
        continue;
      }
    }
    return tc;
  }
  return fallback_unfit;
}

std::optional<TaskCandidate> RealtimeServingRuntime::pick_prefill_task(
    const std::vector<TaskCandidate>& prefill_tasks) const {
  if (prefill_tasks.empty()) {
    return std::nullopt;
  }
  return *std::min_element(prefill_tasks.begin(), prefill_tasks.end(),
                           [&](const TaskCandidate& lhs, const TaskCandidate& rhs) {
                             return std::make_tuple(lhs.earliest_start_clk,
                                                    lhs.ready_clk,
                                                    first_request_id(lhs.request_indices, m_requests)) <
                                    std::make_tuple(rhs.earliest_start_clk,
                                                    rhs.ready_clk,
                                                    first_request_id(rhs.request_indices, m_requests));
                           });
}

std::optional<TaskCandidate> RealtimeServingRuntime::choose_next_task() {
  if (is_fcfs_strict()) {
    const auto idx = fcfs_active_request_index();
    if (!idx.has_value()) {
      return std::nullopt;
    }
    if (m_requests[*idx].state == RequestState::WaitingPrefill) {
      return prefill_task_for_request(*idx);
    }
    if (m_requests[*idx].state == RequestState::WaitingDecode) {
      return single_decode_task_for_request(*idx);
    }
    m_fcfs_active_request_idx.reset();
    return std::nullopt;
  }

  if (is_prefill_priority_fcfs_decode()) {
    const auto prefill_idx = pick_oldest_prefill_request_index();
    if (prefill_idx.has_value()) {
      if (m_config.enable_prefill_batching) {
        auto prefill_tasks = build_prefill_batch_candidates();
        if (!prefill_tasks.empty()) {
          return pick_prefill_task(prefill_tasks);
        }
      }
      return prefill_task_for_request(*prefill_idx);
    }
    return build_fcfs_prefix_decode_task();
  }

  std::vector<TaskCandidate> prefill_tasks;
  if (m_config.enable_prefill_batching) {
    prefill_tasks = build_prefill_batch_candidates();
  } else {
    for (int i = 0; i < static_cast<int>(m_requests.size()); ++i) {
      auto task = prefill_task_for_request(i);
      if (task.has_value()) {
        prefill_tasks.push_back(*task);
      }
    }
  }
  std::vector<TaskCandidate> decode_tasks;
  if (m_config.enable_decode_batching) {
    decode_tasks = build_decode_batch_candidates();
  } else {
    for (int i = 0; i < static_cast<int>(m_requests.size()); ++i) {
      auto task = single_decode_task_for_request(i);
      if (task.has_value()) {
        decode_tasks.push_back(*task);
      }
    }
  }

  if (prefill_tasks.empty() && decode_tasks.empty()) {
    return std::nullopt;
  }

  if (!prefill_tasks.empty()) {
    if (m_config.prefill_guard_ms >= 0.0) {
      double max_prefill_wait_ms = 0.0;
      for (const auto& task : prefill_tasks) {
        max_prefill_wait_ms = std::max(max_prefill_wait_ms,
                                       std::max(0.0, cycles_to_ms(m_clk - task.ready_clk)));
      }
      if (max_prefill_wait_ms >= m_config.prefill_guard_ms) {
        m_stats.prefill_guard_trigger_count++;
        return pick_prefill_task(prefill_tasks);
      }
    }
    if (m_config.max_consecutive_decode_batches > 0 &&
        m_consecutive_decode_batches >= m_config.max_consecutive_decode_batches) {
      m_stats.decode_limit_trigger_count++;
      return pick_prefill_task(prefill_tasks);
    }
  }

  std::vector<TaskCandidate> all_tasks = prefill_tasks;
  all_tasks.insert(all_tasks.end(), decode_tasks.begin(), decode_tasks.end());
  auto key = [&](const TaskCandidate& tc) {
    int phase_prio = 0;
    if (m_config.prompt_priority) {
      phase_prio = (tc.phase == TaskPhase::Prefill) ? 0 : 1;
    }
    return std::make_tuple(tc.earliest_start_clk,
                           phase_prio,
                           tc.ready_clk,
                           first_request_id(tc.request_indices, m_requests));
  };
  return *std::min_element(all_tasks.begin(), all_tasks.end(),
                           [&](const TaskCandidate& lhs, const TaskCandidate& rhs) {
                             return key(lhs) < key(rhs);
                           });
}

void RealtimeServingRuntime::start_task(const TaskCandidate& tc) {
  if (m_active_task.has_value()) {
    return;
  }

  ActiveTask task;
  task.task_id = m_next_task_id++;
  task.request_indices = tc.request_indices;
  task.route = tc.route;
  task.phase = tc.phase;
  task.batch_size = tc.batch_size;
  task.start_clk = tc.earliest_start_clk;
  task.predicted_finish_clk = tc.earliest_start_clk + tc.e2e_clk;
  task.gpu_done_clk = tc.earliest_start_clk + tc.gpu_clk;
  task.link_done_clk = tc.earliest_start_clk + tc.link_clk;
  task.prefill_energy_nj = tc.prefill_energy_nj;
  task.decode_energy_nj = tc.decode_energy_nj;
  task.batch_svc = tc.batch_svc;

  if (tc.gpu_clk > 0) {
    m_gpu_busy_until = std::max(m_gpu_busy_until, tc.earliest_start_clk) + tc.gpu_clk;
    m_stats.gpu_busy_cycles += tc.gpu_clk;
  }
  if (tc.link_clk > 0) {
    m_link_busy_until = std::max(m_link_busy_until, tc.earliest_start_clk) + tc.link_clk;
    m_stats.link_busy_cycles += tc.link_clk;
  }
  if (tc.pim_clk > 0) {
    m_pim_busy_until = std::max(m_pim_busy_until, tc.earliest_start_clk) + tc.pim_clk;
    m_stats.pim_busy_cycles += tc.pim_clk;
  }

  for (int idx : tc.request_indices) {
    auto& req = m_requests[idx];
    if (tc.phase == TaskPhase::Prefill) {
      req.state = RequestState::RunningPrefill;
      req.executed_prefill_batch_size = std::max(req.executed_prefill_batch_size, tc.batch_size);
      if (!req.prefill_start_clk.has_value()) {
        req.prefill_start_clk = tc.earliest_start_clk;
      }
    } else {
      req.state = RequestState::RunningDecode;
      req.last_executed_decode_batch_size = tc.batch_size;
      req.max_executed_decode_batch_size =
          std::max(req.max_executed_decode_batch_size, tc.batch_size);
      if (!req.first_decode_start_clk.has_value()) {
        req.first_decode_start_clk = tc.earliest_start_clk;
      }
      if (req.route == m_config.hybrid_route_name &&
          req.svc.has_value() &&
          req.svc->uses_pim &&
          req.remaining_decode_tokens > 0) {
        PimJob job;
        job.job_id = m_next_pim_job_id++;
        job.request_idx = idx;
        job.phase = TaskPhase::Decode;
        job.token_idx = req.decoded_tokens_done;
        if (use_runtime_generated_hybrid()) {
          const auto& artifacts = generated_artifacts_for_lin(req.svc->mapped_lin);
          job.command_stream = std::make_shared<LogicalTemplateCommandStream>(
              &artifacts.commands,
              std::make_shared<VirtualAddressEncoder>(artifacts.kv_layout),
              req.request_id);
        } else {
          const TemplateTrace* tpl = get_template_for_lin(req.route, req.svc->mapped_lin);
          if (tpl == nullptr) {
            throw ConfigurationError("RealtimeServingFrontend: missing template for route={} lin={}",
                                     req.route,
                                     req.svc->mapped_lin);
          }
          job.trace = tpl;
        }
        task.pim_jobs.push_back(job);
        m_stats.real_pim_jobs_enqueued++;
      }
    }
  }

  m_active_task = std::move(task);
  record_debug_event("dispatch_task",
                     {{"phase", phase_name(tc.phase)},
                      {"route", tc.route},
                      {"batch_size", std::to_string(tc.batch_size)},
                      {"request_ids", join_request_ids(tc.request_indices, m_requests)},
                      {"task_ready_ms", format_ms(cycles_to_ms(tc.ready_clk))},
                      {"task_earliest_start_ms", format_ms(cycles_to_ms(tc.earliest_start_clk))},
                      {"task_gpu_ms", format_ms(cycles_to_ms(tc.gpu_clk))},
                      {"task_pim_ms", format_ms(cycles_to_ms(tc.pim_clk))},
                      {"task_e2e_ms", format_ms(cycles_to_ms(tc.e2e_clk))}});
}

void RealtimeServingRuntime::maybe_start_next_task() {
  if (m_active_task.has_value()) {
    return;
  }
  auto task = choose_next_task();
  if (!task.has_value()) {
    return;
  }
  std::string alloc_reason;
  if (!reserve_hybrid_task_memory(*task, alloc_reason)) {
    m_stats.allocator_hold_count++;
    bool rebound = false;
    for (int idx : task->request_indices) {
      auto& req = m_requests[idx];
      req.allocator_fallback_reason = alloc_reason;
      if (!req.allocator_first_block_clk.has_value()) {
        req.allocator_first_block_clk = m_clk;
      }
      const std::string old_route = req.route;
      maybe_rebind_decode_route(req, m_clk);
      rebound = rebound || (old_route != req.route);
      req.ready_clk = std::max(req.ready_clk, m_clk + 1);
    }
    if (rebound) {
      refresh_active_predictions("allocator_rebind");
    }
    return;
  }
  start_task(*task);
}

void RealtimeServingRuntime::pump_active_pim_job() {
  if (!m_active_task.has_value() || m_active_task->pim_jobs.empty()) {
    return;
  }
  auto& task = *m_active_task;
  while (task.active_pim_job_idx < task.pim_jobs.size() &&
         task.pim_jobs[task.active_pim_job_idx].completed) {
    task.active_pim_job_idx++;
  }
  if (task.active_pim_job_idx >= task.pim_jobs.size()) {
    return;
  }

  auto& job = task.pim_jobs[task.active_pim_job_idx];
  const auto result = pump_pim_job(m_memory_system,
                                   m_translation,
                                   job,
                                   m_requests[job.request_idx].request_id,
                                   m_clk,
                                   task.start_clk,
                                   [this](CompletedPimEvent event) { m_completed_pim_events.push_back(event); });
  if (result.completed_changed) {
    m_stats.real_pim_jobs_completed++;
    if (job.completed) {
      task.active_pim_job_idx++;
    }
  }
}

void RealtimeServingRuntime::process_pim_callbacks() {
  while (!m_completed_pim_events.empty()) {
    const CompletedPimEvent event = m_completed_pim_events.front();
    m_completed_pim_events.pop_front();
    if (!m_active_task.has_value()) {
      continue;
    }

    auto& task = *m_active_task;
    for (size_t i = 0; i < task.pim_jobs.size(); ++i) {
      auto result = apply_pim_completion(task.pim_jobs[i], event);
      if (!result.matched) {
        continue;
      }
      if (result.completed_changed) {
        m_stats.real_pim_jobs_completed++;
      }
      while (task.active_pim_job_idx < task.pim_jobs.size() &&
             task.pim_jobs[task.active_pim_job_idx].completed) {
        task.active_pim_job_idx++;
      }
      break;
    }
  }
}

bool RealtimeServingRuntime::active_task_gpu_done() const {
  return !m_active_task.has_value() || m_clk >= m_active_task->gpu_done_clk;
}

bool RealtimeServingRuntime::active_task_link_done() const {
  return !m_active_task.has_value() || m_clk >= m_active_task->link_done_clk;
}

bool RealtimeServingRuntime::active_task_pim_done() const {
  if (!m_active_task.has_value()) {
    return true;
  }
  const auto& task = *m_active_task;
  return task.active_pim_job_idx >= task.pim_jobs.size();
}

bool RealtimeServingRuntime::active_task_e2e_done() const {
  return !m_active_task.has_value() || m_clk >= m_active_task->predicted_finish_clk;
}

void RealtimeServingRuntime::maybe_complete_active_task() {
  if (!m_active_task.has_value()) {
    return;
  }
  if (!active_task_gpu_done() || !active_task_link_done() ||
      !active_task_pim_done() || !active_task_e2e_done()) {
    return;
  }

  ActiveTask task = *m_active_task;
  m_active_task.reset();
  const Clk_t finish_clk = m_clk;
  const double finish_ms = cycles_to_ms(finish_clk);
  release_hybrid_task_memory(task);

  if (task.phase == TaskPhase::Prefill) {
    m_prefill_batch_sizes.push_back(task.batch_size);
    m_consecutive_decode_batches = 0;
    m_stats.prefill_energy_total_nj += task.prefill_energy_nj;
    m_prefill_energy_by_route_nj[task.route] += std::max(0.0, task.prefill_energy_nj);
    bool rebound_changed = false;
    for (int idx : task.request_indices) {
      auto& req = m_requests[idx];
      req.prefill_end_clk = finish_clk;
      req.first_token_clk = finish_clk;
      req.last_token_completion_clk = finish_clk;
      if (req.remaining_decode_tokens > 0) {
        transition_to_waiting_decode(req, finish_clk);
        const std::string old_route = req.route;
        maybe_rebind_decode_route(req, finish_clk);
        rebound_changed = rebound_changed || (old_route != req.route);
      } else {
        req.state = RequestState::Done;
        req.completion_clk = finish_clk;
        release_hybrid_request_memory(req);
        m_stats.completed_requests++;
      }
    }
    if (rebound_changed) {
      refresh_active_predictions("decode_rebind");
    }
  } else {
    m_decode_batch_sizes.push_back(task.batch_size);
    m_consecutive_decode_batches++;
    m_stats.decode_energy_total_nj += task.decode_energy_nj;
    m_decode_energy_by_route_nj[task.route] += std::max(0.0, task.decode_energy_nj);
    m_stats.decode_token_count += task.batch_size;
    for (int idx : task.request_indices) {
      auto& req = m_requests[idx];
      if (req.last_token_completion_clk.has_value()) {
        req.tbt_intervals_clk.push_back(finish_clk - *req.last_token_completion_clk);
      }
      req.last_token_completion_clk = finish_clk;
      req.remaining_decode_tokens -= 1;
      req.decoded_tokens_done += 1;
      if (req.remaining_decode_tokens <= 0) {
        req.state = RequestState::Done;
        req.completion_clk = finish_clk;
        release_hybrid_request_memory(req);
        m_stats.completed_requests++;
      } else {
        req.state = RequestState::WaitingDecode;
        req.ready_clk = finish_clk;
      }
    }
  }

  record_debug_event("complete_task",
                     {{"phase", phase_name(task.phase)},
                      {"route", task.route},
                      {"batch_size", std::to_string(task.batch_size)},
                      {"request_ids", join_request_ids(task.request_indices, m_requests)},
                      {"finish_ms", format_ms(finish_ms)}});
  refresh_fcfs_active_request();
}

}  // namespace Ramulator::Serving
