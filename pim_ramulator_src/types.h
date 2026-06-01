#ifndef RAMULATOR_FRONTEND_IMPL_SERVING_TYPES_H
#define RAMULATOR_FRONTEND_IMPL_SERVING_TYPES_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "base/base.h"

namespace Ramulator::Serving {

static constexpr Clk_t kInfClk = std::numeric_limits<Clk_t>::max() / 4;

enum class RequestState {
  WaitingAdmission,
  WaitingPrefill,
  RunningPrefill,
  WaitingDecode,
  RunningDecode,
  Done,
  Dropped,
};

enum class TaskPhase {
  Prefill,
  Decode,
};

struct FrontendConfig {
  std::string requests_csv;
  std::string requests_out_csv;
  std::string template_dir;
  std::string cost_gpu_csv;
  std::string cost_hybrid_csv;
  std::string gpu_route_name = "gpu_only";
  std::string hybrid_route_name = "lpddr5_pim_bank";

  std::string route_policy = "min_finish";
  bool prefer_gpu_on_tie = true;
  double energy_latency_guard_ms = 5.0;
  double route_load_balance_ms_per_active_request = 0.0;
  double pim_wait_threshold_ms = 0.0;

  std::string unsupported_policy = "clip";
  float arrival_time_scale = 1.0f;
  int lin_bucket = 1;
  int lout_bucket = 1;
  int batch_size = 1;

  bool enable_prefill_batching = false;
  int max_prefill_batch_size = 1;
  bool enable_decode_batching = false;
  int max_decode_batch_size = 1;
  double prefill_guard_ms = -1.0;
  int max_consecutive_decode_batches = 0;
  int decode_batch_cap_with_prefill = 0;

  std::string local_scheduling_policy = "prefill_priority_fcfs_decode";
  std::string fcfs_decode_prediction_mode = "incremental";
  bool prompt_priority = true;

  bool enable_decode_rebind = false;
  double decode_rebind_margin_ms = 0.0;

  bool enable_predictive_admission = false;
  bool enable_slo_guarded_admission = false;
  double slo_tbt_ms = -1.0;
  double slo_e2e_ms = -1.0;
  double slo_ttft_ms = -1.0;
  int admission_shadow_max_steps = 10000;
  double admission_retry_interval_ms = 0.0;
  int admission_max_harmed_requests = 0;
  double admission_max_total_harm_ms = 0.0;
  double admission_max_single_harm_ms = 0.0;
  double admission_max_own_miss_ms = 0.0;
  bool admission_check_own_slo = false;
  int admission_max_bypass_count = 0;
  double admission_max_wait_ms = 0.0;
  double admission_aging_harm_ms_per_ms = 0.0;
  double admission_aging_harmed_requests_per_ms = 0.0;

  bool share_gpu_prefill_across_routes = false;
  double gpu_queue_alpha = 0.0;
  double pim_queue_alpha = 0.0;
  double active_request_alpha = 0.0;
  double decode_token_alpha = 0.0;

  bool capture_debug_events = false;
  bool debug_enabled = false;
  std::string debug_log_path = "log/online_serving/frontend_debug.log";
  Clk_t debug_interval_clk = 1000000;
  uint clock_ratio = 1;
};

struct SourceRequest {
  int request_id = -1;
  double arrival_ms = 0.0;
  Clk_t arrival_clk = 0;
  int context_tokens = 0;
  int generated_tokens = 0;
};

struct CostPoint {
  int lin = 0;
  int lout = 0;
  int bs = 1;

  double prefill_e2e_ms = 0.0;
  double prefill_gpu_ms = 0.0;
  double prefill_link_ms = 0.0;
  double prefill_pim_ms = 0.0;
  double prefill_energy_nj = 0.0;

  double decode_e2e_ms = 0.0;
  double decode_gpu_ms = 0.0;
  double decode_link_ms = 0.0;
  double decode_pim_ms = 0.0;
  double decode_energy_nj = 0.0;
};

struct LookupResult {
  CostPoint point;
  int mapped_lin = 0;
  int mapped_lout = 0;
  int mapped_bs = 1;
  bool clipped_lin = false;
  bool clipped_lout = false;
};

struct ServiceEstimate {
  std::string route;
  bool uses_pim = false;

  int lin = 0;
  int lout = 0;
  int generated_tokens = 0;
  int decode_tokens = 0;

  int mapped_lin = 0;
  int mapped_lout = 0;
  int mapped_bs = 1;
  bool clipped_lin = false;
  bool clipped_lout = false;

  double prefill_e2e_ms = 0.0;
  double prefill_gpu_ms = 0.0;
  double prefill_link_ms = 0.0;
  double prefill_pim_ms = 0.0;
  double prefill_energy_nj = 0.0;

  double decode_e2e_ms = 0.0;
  double decode_gpu_ms = 0.0;
  double decode_link_ms = 0.0;
  double decode_pim_ms = 0.0;
  double decode_energy_nj = 0.0;

  Clk_t prefill_e2e_clk = 0;
  Clk_t prefill_gpu_clk = 0;
  Clk_t prefill_link_clk = 0;
  Clk_t prefill_pim_clk = 0;

  Clk_t decode_e2e_clk = 0;
  Clk_t decode_gpu_clk = 0;
  Clk_t decode_link_clk = 0;
  Clk_t decode_pim_clk = 0;
};

struct RouteCandidate {
  std::string route;
  bool eligible = false;
  bool uses_pim = false;
  std::string reason;
  std::optional<ServiceEstimate> svc;

  double predicted_finish_ms = std::numeric_limits<double>::infinity();
  double predicted_gpu_wait_ms = std::numeric_limits<double>::infinity();
  double predicted_pim_wait_ms = std::numeric_limits<double>::infinity();
  double predicted_incremental_energy_nj = 0.0;
  double predicted_incremental_decode_energy_nj = 0.0;
  double queue_pressure_ms = 0.0;
  std::optional<double> deadline_ms;
  std::optional<double> slack_ms;
  std::optional<double> candidate_mean_tbt_ms;
  int route_active_requests = 0;
  int route_waiting_prefill_count = 0;
  int route_waiting_decode_count = 0;
  int route_pending_decode_tokens = 0;
  int harmed_count = 0;
  double total_harm_ms = 0.0;
  double max_single_harm_ms = 0.0;
  double own_miss_ms = 0.0;
};

struct RouteDecision {
  std::optional<std::string> route;
  bool dropped = false;
  std::string reason;
};

struct RuntimeRequest {
  int request_id = -1;
  double arrival_ms = 0.0;
  Clk_t arrival_clk = 0;
  int lin = 0;
  int lout = 0;

  std::string route;
  std::string admission_route;
  std::string route_decision_reason;
  std::string fallback_reason;
  std::string dropped_reason;

  RequestState state = RequestState::WaitingPrefill;
  Clk_t ready_clk = 0;
  Clk_t admission_next_retry_clk = 0;
  std::optional<ServiceEstimate> svc;
  int remaining_decode_tokens = 0;
  int decoded_tokens_done = 0;
  int executed_prefill_batch_size = 0;
  int max_executed_decode_batch_size = 0;
  int last_executed_decode_batch_size = 0;
  std::optional<uint64_t> decode_enqueue_seq;

  std::optional<double> deadline_ms;
  std::optional<Clk_t> deadline_clk;
  std::optional<Clk_t> admission_time_clk;
  std::optional<double> admission_time_ms;
  std::optional<double> admission_queue_wait_ms;
  int admission_attempts = 0;
  int admission_bypass_count = 0;
  std::string admission_last_reject_reason;
  int admission_last_harmed_count = 0;
  double admission_last_total_harm_ms = 0.0;
  double admission_last_single_harm_ms = 0.0;
  double admission_last_own_miss_ms = 0.0;
  bool admission_forced_best_effort = false;
  bool is_lost = false;

  std::optional<double> chosen_predicted_finish_ms;
  std::optional<double> latest_predicted_finish_ms;
  std::optional<double> chosen_predicted_gpu_wait_ms;
  std::optional<double> chosen_predicted_pim_wait_ms;
  std::optional<double> chosen_predicted_incremental_energy_nj;
  std::optional<double> chosen_predicted_incremental_decode_energy_nj;
  std::optional<double> chosen_queue_pressure_ms;
  std::optional<double> chosen_slack_ms;

  std::optional<Clk_t> decode_bind_clk;
  std::optional<double> decode_bind_time_ms;
  bool decode_rebind_attempted = false;
  bool decode_rebound = false;
  std::string decode_rebind_reason;
  int prediction_refresh_count = 0;

  std::optional<Clk_t> prefill_start_clk;
  std::optional<Clk_t> prefill_end_clk;
  std::optional<Clk_t> first_token_clk;
  std::optional<Clk_t> first_decode_start_clk;
  std::optional<Clk_t> completion_clk;
  std::optional<Clk_t> last_token_completion_clk;
  std::vector<Clk_t> tbt_intervals_clk;
};

struct TraceCommand {
  int type_id = -1;
  Addr_t addr = 0;
};

struct TemplateTrace {
  int lin = 0;
  std::vector<TraceCommand> commands;
};

struct PimJob {
  uint64_t job_id = 0;
  int request_idx = -1;
  TaskPhase phase = TaskPhase::Decode;
  int token_idx = -1;
  const TemplateTrace* trace = nullptr;
  size_t next_cmd_idx = 0;
  size_t outstanding_cmds = 0;
  bool issue_done = false;
  bool completed = false;
};

struct CompletedPimEvent {
  uint64_t job_id = 0;
};

struct TaskCandidate {
  std::vector<int> request_indices;
  std::string route;
  TaskPhase phase = TaskPhase::Prefill;
  Clk_t ready_clk = 0;
  Clk_t earliest_start_clk = 0;
  Clk_t gpu_clk = 0;
  Clk_t link_clk = 0;
  Clk_t pim_clk = 0;
  Clk_t e2e_clk = 0;
  double prefill_energy_nj = 0.0;
  double decode_energy_nj = 0.0;
  ServiceEstimate batch_svc;
  int batch_size = 1;
};

struct ActiveTask {
  uint64_t task_id = 0;
  std::vector<int> request_indices;
  std::string route;
  TaskPhase phase = TaskPhase::Prefill;
  int batch_size = 1;
  Clk_t start_clk = 0;
  Clk_t predicted_finish_clk = 0;
  Clk_t gpu_done_clk = 0;
  Clk_t link_done_clk = 0;
  double prefill_energy_nj = 0.0;
  double decode_energy_nj = 0.0;
  ServiceEstimate batch_svc;
  std::vector<PimJob> pim_jobs;
  size_t active_pim_job_idx = 0;
};

struct QueuePressureSnapshot {
  int active_requests = 0;
  int pending_decode_tokens = 0;
  double pending_gpu_work_ms = 0.0;
  double pending_pim_work_ms = 0.0;
};

struct ForecastRequest {
  int request_id = -1;
  double arrival_ms = 0.0;
  int context_tokens = 0;
  int generated_tokens = 0;
  std::string route;
  ServiceEstimate svc;
  RequestState state = RequestState::WaitingPrefill;
  double ready_ms = 0.0;
  int remaining_decode_tokens = 0;
  int decoded_tokens_done = 0;
  std::optional<uint64_t> decode_enqueue_seq;
  std::optional<double> prefill_start_ms;
  std::optional<double> prefill_end_ms;
  std::optional<double> first_decode_start_ms;
  std::optional<double> completion_time_ms;
  std::optional<double> last_token_completion_ms;
  std::vector<double> tbt_intervals_ms;
  std::optional<double> deadline_ms;
  bool is_lost = false;
};

struct ForecastRun {
  std::unordered_map<int, std::unordered_map<std::string, double>> requests;
  double total_prefill_energy_nj = 0.0;
  double total_decode_energy_nj = 0.0;
  double first_task_start_ms = std::numeric_limits<double>::infinity();
};

using CostTableMap = std::unordered_map<std::string, std::vector<CostPoint>>;
using TemplateMap = std::unordered_map<std::string, std::unordered_map<int, TemplateTrace>>;

struct RuntimeStats {
  int total_requests = 0;
  int admitted_requests = 0;
  int completed_requests = 0;
  int dropped_requests = 0;
  int route_fallback_count = 0;
  int clipped_lin_count = 0;
  int clipped_lout_count = 0;
  int real_pim_jobs_enqueued = 0;
  int real_pim_jobs_completed = 0;

  int prefill_guard_trigger_count = 0;
  int decode_limit_trigger_count = 0;
  int admission_reject_count = 0;
  int admission_lost_count = 0;
  int admission_shadow_timeout_count = 0;
  int admission_queue_max_len = 0;
  int admission_held_count = 0;
  int admission_admitted_from_queue_count = 0;
  int admission_bypassed_count = 0;
  int admission_best_effort_count = 0;
  int decode_rebind_attempt_count = 0;
  int decode_rebound_count = 0;

  Clk_t gpu_busy_cycles = 0;
  Clk_t link_busy_cycles = 0;
  Clk_t pim_busy_cycles = 0;
  double prefill_energy_total_nj = 0.0;
  double decode_energy_total_nj = 0.0;
  int decode_token_count = 0;
};

inline const char* phase_name(TaskPhase phase) {
  switch (phase) {
    case TaskPhase::Prefill:
      return "prefill";
    case TaskPhase::Decode:
      return "decode";
  }
  return "unknown";
}

inline std::string state_to_string(RequestState state) {
  switch (state) {
    case RequestState::WaitingAdmission:
      return "waiting_admission";
    case RequestState::WaitingPrefill:
      return "waiting_prefill";
    case RequestState::RunningPrefill:
      return "running_prefill";
    case RequestState::WaitingDecode:
      return "waiting_decode";
    case RequestState::RunningDecode:
      return "running_decode";
    case RequestState::Done:
      return "done";
    case RequestState::Dropped:
      return "dropped";
  }
  return "unknown";
}

inline bool is_waiting_state(RequestState state) {
  return state == RequestState::WaitingAdmission ||
         state == RequestState::WaitingPrefill ||
         state == RequestState::WaitingDecode;
}

inline Clk_t predict_stage_start(Clk_t ready_clk,
                                 Clk_t gpu_clk,
                                 Clk_t link_clk,
                                 Clk_t pim_clk,
                                 Clk_t gpu_free_clk,
                                 Clk_t link_free_clk,
                                 Clk_t pim_free_clk) {
  Clk_t t = ready_clk;
  if (gpu_clk > 0) {
    t = std::max(t, gpu_free_clk);
  }
  if (link_clk > 0) {
    t = std::max(t, link_free_clk);
  }
  if (pim_clk > 0) {
    t = std::max(t, pim_free_clk);
  }
  return t;
}

inline bool is_valid_ms(double value) {
  return value >= 0.0 && std::isfinite(value);
}

}  // namespace Ramulator::Serving

#endif  // RAMULATOR_FRONTEND_IMPL_SERVING_TYPES_H
