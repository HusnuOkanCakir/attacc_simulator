#ifndef RAMULATOR_FRONTEND_IMPL_ALTERNATING_SERVING_RUNTIME_H
#define RAMULATOR_FRONTEND_IMPL_ALTERNATING_SERVING_RUNTIME_H

#include <deque>
#include <fstream>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "base/base.h"
#include "base/logging.h"
#include "frontend/impl/alternating_serving/types.h"
#include "memory_system/memory_system.h"

namespace Ramulator::AlternatingServing {

class AlternatingServingRuntime {
 public:
  explicit AlternatingServingRuntime(Logger_t logger);

  void initialize(const FrontendConfig& config);
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

 private:
  void open_debug_log_if_needed();
  void finalize_post_connect_setup();

  Clk_t ms_to_cycles(double ms) const;
  double cycles_to_ms(Clk_t clk) const;
  std::optional<ServiceEstimate> estimate_request_for_route(const std::string& route,
                                                            int context_tokens,
                                                            int generated_tokens,
                                                            int bs) const;
  bool has_template_for_lin(const std::string& route, int lin) const;
  const TemplateTrace* get_template_for_lin(const std::string& route, int lin) const;

  void debug_write_line(const std::string& line);
  void maybe_log_progress(const char* reason);
  std::optional<Clk_t> next_frontend_event_clk() const;
  bool try_fast_forward();

  void admit_arrivals_up_to_now();
  int next_serial_request_index() const;
  void maybe_start_next_task();
  void pump_active_pim_job();
  void process_pim_callbacks();
  bool active_task_gpu_done() const;
  bool active_task_link_done() const;
  bool active_task_pim_done() const;
  bool active_task_e2e_done() const;
  void maybe_complete_active_task();

  double request_ttft_ms(const RuntimeRequest& req) const;
  double request_e2e_ms(const RuntimeRequest& req) const;
  double request_mean_tbt_ms(const RuntimeRequest& req) const;
  std::vector<double> collect_completed_request_metrics(
      const std::function<double(const RuntimeRequest&)>& metric_fn) const;
  static double mean(const std::vector<double>& vals);
  static double percentile(std::vector<double> vals, double q);
  double compute_makespan_ms() const;
  int total_completed_output_tokens() const;
  void write_requests_csv() const;

  FrontendConfig m_config;
  Logger_t m_logger;
  IMemorySystem* m_memory_system = nullptr;

  float m_tck_ns = -1.0f;
  double m_cycles_per_ms = 0.0;
  bool m_post_connected = false;

  std::vector<SourceRequest> m_source_requests;
  CostTableMap m_cost_points;
  TemplateMap m_templates;

  Clk_t m_clk = 0;
  size_t m_arrival_idx = 0;
  size_t m_next_route_slot = 0;
  std::vector<RuntimeRequest> m_requests;
  std::optional<ActiveTask> m_active_task;
  std::deque<CompletedPimEvent> m_completed_pim_events;

  uint64_t m_next_task_id = 1;
  uint64_t m_next_pim_job_id = 1;
  Clk_t m_next_debug_clk = 0;

  RuntimeStats m_stats;
  std::ofstream m_debug_log;
};

}  // namespace Ramulator::AlternatingServing

#endif  // RAMULATOR_FRONTEND_IMPL_ALTERNATING_SERVING_RUNTIME_H
