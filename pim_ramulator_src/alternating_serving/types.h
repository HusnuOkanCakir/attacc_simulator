#ifndef RAMULATOR_FRONTEND_IMPL_ALTERNATING_SERVING_TYPES_H
#define RAMULATOR_FRONTEND_IMPL_ALTERNATING_SERVING_TYPES_H

#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/base.h"

namespace Ramulator::AlternatingServing {

static constexpr Clk_t kInfClk = std::numeric_limits<Clk_t>::max() / 4;

enum class RequestState {
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
  std::string unsupported_policy = "clip";
  float arrival_time_scale = 1.0f;
  float ignored_pim_wait_threshold_ms = 1.0f;
  int lin_bucket = 1;
  int lout_bucket = 1;
  int batch_size = 1;
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

  Clk_t prefill_e2e_clk = 0;
  Clk_t prefill_gpu_clk = 0;
  Clk_t prefill_link_clk = 0;
  Clk_t prefill_pim_clk = 0;
  double prefill_energy_nj = 0.0;

  Clk_t decode_e2e_clk = 0;
  Clk_t decode_gpu_clk = 0;
  Clk_t decode_link_clk = 0;
  Clk_t decode_pim_clk = 0;
  double decode_energy_nj = 0.0;
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
  std::optional<ServiceEstimate> svc;
  Clk_t ready_clk = 0;
  int remaining_decode_tokens = 0;
  int decoded_tokens_done = 0;

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

struct ActiveTask {
  uint64_t task_id = 0;
  int request_idx = -1;
  TaskPhase phase = TaskPhase::Prefill;
  std::string route;
  Clk_t start_clk = 0;
  Clk_t predicted_finish_clk = 0;
  Clk_t gpu_done_clk = 0;
  Clk_t link_done_clk = 0;
  double prefill_energy_nj = 0.0;
  double decode_energy_nj = 0.0;
  std::optional<PimJob> pim_job;
};

using CostTableMap = std::unordered_map<std::string, std::vector<CostPoint>>;
using TemplateMap = std::unordered_map<std::string, std::unordered_map<int, TemplateTrace>>;

struct RuntimeStats {
  int total_requests = 0;
  int admitted_requests = 0;
  int completed_requests = 0;
  int dropped_requests = 0;
  int route_gpu = 0;
  int route_hybrid = 0;
  int route_fallback_count = 0;
  int clipped_lin_count = 0;
  int clipped_lout_count = 0;
  int real_pim_jobs_enqueued = 0;
  int real_pim_jobs_completed = 0;

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

inline const char* state_to_string(RequestState state) {
  switch (state) {
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

}  // namespace Ramulator::AlternatingServing

#endif  // RAMULATOR_FRONTEND_IMPL_ALTERNATING_SERVING_TYPES_H
