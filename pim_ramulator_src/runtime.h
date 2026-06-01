#ifndef RAMULATOR_FRONTEND_IMPL_SERVING_RUNTIME_H
#define RAMULATOR_FRONTEND_IMPL_SERVING_RUNTIME_H

#include <deque>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "base/base.h"
#include "base/logging.h"
#include "frontend/impl/serving/allocator.h"
#include "frontend/impl/serving/command_generator.h"
#include "frontend/impl/serving/pim_allocator.h"
#include "memory_system/memory_system.h"
#include "frontend/impl/serving/types.h"
#include "translation/translation.h"

namespace Ramulator::Serving {

class RealtimeServingRuntime {
 public:
  explicit RealtimeServingRuntime(Logger_t logger);

  void initialize(const FrontendConfig& config);
  void attach_translation(ITranslation* translation);
  void connect_memory_system(IMemorySystem* memory_system);
  void tick();
  bool is_finished() const;
  void finalize();

  void print_stats(YAML::Emitter& emitter,
                   const std::string& ifce_name,
                   const std::string& impl_name) const;

  Clk_t clk() const { return m_clk; }
  size_t total_requests() const { return m_stats.total_requests; }
  size_t gpu_cost_point_count() const;
  size_t hybrid_cost_point_count() const;
  std::shared_ptr<ServingAddressSpace> shared_address_space() const { return m_address_space; }

 private:
  void open_debug_log_if_needed();
  void finalize_post_connect_setup();
  GeneratorGeometry generator_geometry() const;
  bool use_runtime_generated_hybrid() const;

  Clk_t ms_to_cycles(double ms) const;
  double cycles_to_ms(Clk_t clk) const;
  double quantize_export_ms(double ms) const;
  std::optional<double> quantize_export_ms(const std::optional<double>& ms) const;
  bool use_waiting_admission_queue() const;
  bool is_fcfs_strict() const;
  bool is_prefill_priority_fcfs_decode() const;
  bool use_shadow_prediction_for_fcfs_decode() const;
  bool use_incremental_prediction_for_fcfs_decode() const;
  bool use_heuristic_refresh_prediction_for_fcfs_decode() const;
  bool uses_latency_guarded_energy_policy() const;
  bool uses_slo_guarded_admission() const;

  std::optional<ServiceEstimate> estimate_request_for_route(const std::string& route,
                                                            int context_tokens,
                                                            int generated_tokens,
                                                            int bs) const;
  const GeneratedHybridArtifacts& generated_artifacts_for_lin(int lin);
  const std::vector<LogicalCommand>& generated_logical_trace_for_lin(int lin);
  std::optional<ServiceEstimate> estimate_request_cached(const std::string& route,
                                                         int context_tokens,
                                                         int generated_tokens,
                                                         int bs);
  void prefetch_request_estimates(const std::vector<int>& request_indices,
                                  const std::vector<std::string>& routes = {});
  void prefetch_arrival_request_estimates();
  std::string estimate_cache_key(const std::string& route,
                                 int context_tokens,
                                 int generated_tokens,
                                 int bs) const;
  std::optional<ServiceEstimate> shared_gpu_prefill_estimate(const std::string& route,
                                                             int context_tokens,
                                                             int generated_tokens,
                                                             int bs,
                                                             const std::optional<ServiceEstimate>& est);
  int predicted_lookup_batch_size(const std::string& route) const;
  int request_decode_context_tokens(const RuntimeRequest& req) const;
  int request_decode_context_tokens(const ForecastRequest& req) const;

  bool has_template_for_lin(const std::string& route, int lin) const;
  const TemplateTrace* get_template_for_lin(const std::string& route, int lin) const;
  bool reserve_hybrid_request_memory(RuntimeRequest& req,
                                     const ServiceEstimate& svc,
                                     std::string& reason);
  AllocationOutcome predict_hybrid_request_memory_fit(uint32_t request_id,
                                                      int capacity_context_tokens,
                                                      const ServiceEstimate& svc);
  void release_hybrid_request_memory(RuntimeRequest& req);
  bool reserve_hybrid_task_memory(const TaskCandidate& tc, std::string& reason);
  AllocationOutcome predict_hybrid_task_memory_fit(const std::vector<int>& request_indices,
                                                   int decode_context_tokens,
                                                   int batch_size) const;
  void release_hybrid_task_memory(const ActiveTask& task);
  void sync_allocator_stats();
  // Reset retry clocks for all requests currently held due to KV OOM, so they
  // are reconsidered at the next admission tick after memory has been freed.
  void wake_kv_held_requests();
  // Returns true if any request in m_requests is waiting solely due to KV OOM.
  bool has_kv_held_requests() const;
  std::optional<RouteCandidate> find_candidate(const std::vector<RouteCandidate>& candidates,
                                               const std::string& route) const;

  void debug_write_line(const std::string& line);
  void record_debug_event(const std::string& event,
                          const std::vector<std::pair<std::string, std::string>>& fields = {});
  void maybe_log_progress(const char* reason);
  std::optional<Clk_t> next_frontend_event_clk() const;
  bool try_fast_forward();

  void enqueue_arrivals_up_to_now();
  void admit_arrivals_up_to_now();
  std::vector<int> waiting_admission_request_indices() const;
  std::optional<Clk_t> min_waiting_admission_retry_clk() const;
  void admit_waiting_request(RuntimeRequest& req,
                             const std::string& route,
                             const ServiceEstimate& est,
                             const RouteCandidate& cand,
                             const std::string& reason,
                             bool forced_best_effort);
  void process_admission_queue();
  void process_slo_guarded_admission_queue();
  RouteCandidate evaluate_slo_guarded_route(RuntimeRequest& req,
                                            const std::string& route,
                                            const ServiceEstimate& est,
                                            const ForecastRun& baseline_run,
                                            const std::unordered_map<int, std::unordered_map<std::string, double>>& baseline_results);
  std::unordered_map<std::string, double> effective_slo_guard_budgets(const RuntimeRequest& req) const;
  std::unordered_map<std::string, double> project_admission_candidate(RuntimeRequest& req,
                                                                      const std::string& route,
                                                                      const ServiceEstimate& est) const;

  RouteCandidate predict_route_candidate(const SourceRequest& src,
                                         const std::string& route,
                                         const std::optional<double>& deadline_ms,
                                         std::optional<double> baseline_total_energy_nj = std::nullopt,
                                         std::optional<double> baseline_total_decode_energy_nj = std::nullopt);
  RouteCandidate predict_decode_route_candidate(const RuntimeRequest& req,
                                                const std::string& route,
                                                double decision_time_ms);
  void maybe_rebind_decode_route(RuntimeRequest& req, Clk_t decision_time_clk);
  void refresh_active_predictions(const std::string& cause);

  ForecastRequest shadow_request_from_runtime_request(const RuntimeRequest& req) const;
  ForecastRequest shadow_request_for_candidate(const SourceRequest& req,
                                              const std::string& route,
                                              const ServiceEstimate& est) const;
  ForecastRequest incremental_request_for_waiting_request(const RuntimeRequest& req,
                                                          const std::string& route,
                                                          const ServiceEstimate& est) const;
  ForecastRun incremental_active_forecast_results();
  ForecastRun simulate_prefill_priority_fcfs_decode_incremental(std::vector<ForecastRequest> forecast) const;
  std::unordered_map<int, std::unordered_map<std::string, double>>
  simulate_prefill_priority_fcfs_decode_shadow(std::vector<ForecastRequest> shadow) const;
  std::unordered_map<int, std::unordered_map<std::string, double>>
  simulate_prefill_priority_fcfs_decode_heuristic(std::vector<ForecastRequest> forecast) const;
  std::unordered_map<std::string, double> project_prefill_priority_fcfs_decode_candidate(
      const SourceRequest& req, const std::string& route, const ServiceEstimate& est);
  std::unordered_map<std::string, double> project_prefill_priority_fcfs_decode_incremental_candidate(
      const SourceRequest& req,
      const std::string& route,
      const ServiceEstimate& est,
      std::optional<double> baseline_total_energy_nj = std::nullopt,
      std::optional<double> baseline_total_decode_energy_nj = std::nullopt);
  std::unordered_map<std::string, double> project_prefill_priority_fcfs_decode_heuristic_candidate(
      const SourceRequest& req, const std::string& route, const ServiceEstimate& est);
  std::unordered_map<std::string, double> project_prefill_priority_fcfs_decode_heuristic_decode_candidate(
      const RuntimeRequest& req, const std::string& route, const ServiceEstimate& est, double decision_time_ms);

  QueuePressureSnapshot queue_pressure_snapshot() const;
  double queue_pressure_adjustment_ms(double prefill_gpu_ms,
                                      double prefill_pim_ms,
                                      double prefill_e2e_ms,
                                      double decode_gpu_ms,
                                      double decode_pim_ms,
                                      double decode_e2e_ms,
                                      int decode_tokens) const;
  void annotate_route_candidate(RouteCandidate& cand) const;
  void move_route_count(const std::string& old_route, const std::string& new_route);
  void inc_route_count(const std::string& route, int delta);

  std::optional<int> fcfs_active_request_index();
  void refresh_fcfs_active_request();
  std::optional<int> pick_oldest_prefill_request_index() const;
  std::vector<int> fcfs_decode_queue_indices();
  void transition_to_waiting_decode(RuntimeRequest& req, Clk_t ready_clk);
  std::optional<TaskCandidate> prefill_task_for_request(int request_idx) const;
  std::optional<ServiceEstimate> estimate_prefill_batch(const std::string& route,
                                                        const std::vector<int>& request_indices);
  std::vector<TaskCandidate> build_prefill_batch_candidates() const;
  std::optional<TaskCandidate> single_decode_task_for_request(int request_idx) const;
  std::vector<TaskCandidate> build_decode_batch_candidates() const;
  std::optional<TaskCandidate> build_fcfs_prefix_decode_task() const;
  std::optional<TaskCandidate> choose_next_task();
  std::optional<TaskCandidate> pick_prefill_task(const std::vector<TaskCandidate>& prefill_tasks) const;
  void start_task(const TaskCandidate& tc);
  void maybe_start_next_task();
  void pump_active_pim_job();
  void process_pim_callbacks();
  bool active_task_gpu_done() const;
  bool active_task_link_done() const;
  bool active_task_pim_done() const;
  bool active_task_e2e_done() const;
  void maybe_complete_active_task();

  double request_ttft_ms(const RuntimeRequest& req) const;
  double request_prefill_wait_ms(const RuntimeRequest& req) const;
  double request_e2e_ms(const RuntimeRequest& req) const;
  double request_mean_tbt_ms(const RuntimeRequest& req) const;
  std::vector<double> collect_completed_request_metrics(
      const std::function<double(const RuntimeRequest&)>& metric_fn) const;
  static double mean(const std::vector<double>& vals);
  static double percentile(std::vector<double> vals, double q);
  double compute_makespan_ms() const;
  int total_completed_output_tokens() const;
  void write_requests_csv() const;
  void write_kv_placement_csv() const;
  void write_allocator_pressure_csv() const;

  FrontendConfig m_config;
  Logger_t m_logger;
  IMemorySystem* m_memory_system = nullptr;
  ITranslation* m_translation = nullptr;

  float m_tck_ns = -1.0f;
  double m_cycles_per_ms = 0.0;
  bool m_post_connected = false;

  std::vector<SourceRequest> m_source_requests;
  CostTableMap m_cost_points;
  TemplateMap m_templates;
  std::shared_ptr<ServingAddressSpace> m_address_space;
  std::unique_ptr<ServingAllocatorFacade> m_allocator;
  // Clean allocator: handles KV (persistent) allocation and VA→PA translation
  // for the runtime_generated hybrid path. Replaces the legacy KV path in
  // ServingAllocatorFacade while leaving transient (Score/Context/Scratch) unchanged.
  std::shared_ptr<PimAllocator> m_pim_allocator;
  std::unordered_map<int, GeneratedHybridArtifacts> m_generated_artifacts;

  Clk_t m_clk = 0;
  size_t m_arrival_idx = 0;
  std::vector<RuntimeRequest> m_requests;
  std::optional<ActiveTask> m_active_task;
  std::deque<CompletedPimEvent> m_completed_pim_events;

  Clk_t m_gpu_busy_until = 0;
  Clk_t m_link_busy_until = 0;
  Clk_t m_pim_busy_until = 0;

  std::optional<int> m_fcfs_active_request_idx;
  uint64_t m_next_decode_enqueue_seq = 0;
  uint64_t m_next_task_id = 1;
  uint64_t m_next_pim_job_id = 1;
  Clk_t m_next_debug_clk = 0;

  std::unordered_map<std::string, int> m_route_counts;
  std::vector<int> m_prefill_batch_sizes;
  std::vector<int> m_decode_batch_sizes;
  int m_consecutive_decode_batches = 0;
  std::vector<double> m_admission_queue_wait_samples_ms;
  std::unordered_set<int> m_admission_blocked_request_ids;
  std::unordered_set<int> m_admission_tbt_blocked_request_ids;
  std::unordered_map<std::string, std::optional<ServiceEstimate>> m_estimate_cache;
  int m_estimate_cache_hits = 0;
  int m_estimate_cache_misses = 0;
  int m_estimate_prefetch_populated = 0;
  std::optional<std::string> m_incremental_active_forecast_cache_key;
  std::optional<ForecastRun> m_incremental_active_forecast_cache_run;
  int m_incremental_active_forecast_cache_hits = 0;
  int m_incremental_active_forecast_cache_misses = 0;
  std::optional<std::string> m_projection_cache_baseline_key;
  std::unordered_map<std::string, ForecastRun> m_incremental_projection_cache;
  int m_projection_cache_hits = 0;
  int m_projection_cache_misses = 0;
  std::unordered_map<std::string, double> m_prefill_energy_by_route_nj;
  std::unordered_map<std::string, double> m_decode_energy_by_route_nj;

  RuntimeStats m_stats;
  std::ofstream m_debug_log;
};

}  // namespace Ramulator::Serving

#endif  // RAMULATOR_FRONTEND_IMPL_SERVING_RUNTIME_H
