#include <algorithm>
#include <memory>
#include <string>

#include "base/exception.h"
#include "frontend/frontend.h"
#include "frontend/impl/serving/runtime.h"

namespace Ramulator {

class RealtimeServingFrontend : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd,
                                    RealtimeServingFrontend,
                                    "RealtimeServingFrontend",
                                    "Realtime online LLM serving frontend with callback-backed PIM execution.");

 public:
  void init() override {
    Serving::FrontendConfig config;
    config.requests_csv = param<std::string>("requests_csv").desc("Input request CSV path.").required();
    config.requests_out_csv = param<std::string>("requests_out_csv")
                                  .desc("Per-request output CSV path.")
                                  .default_val("cluster_outputs/online_serving_requests.csv");
    config.template_dir = param<std::string>("template_dir")
                              .desc("Directory containing PIM template traces.")
                              .required();
    config.cost_gpu_csv =
        param<std::string>("cost_gpu_csv").desc("Cost table CSV for GPU-only route.").required();
    config.cost_hybrid_csv =
        param<std::string>("cost_hybrid_csv").desc("Cost table CSV for hybrid route.").required();
    config.gpu_route_name = param<std::string>("gpu_route_name").default_val("gpu_only");
    config.hybrid_route_name = param<std::string>("hybrid_route_name").default_val("lpddr5_pim_bank");
    config.route_policy = param<std::string>("route_policy").default_val("min_finish");
    config.prefer_gpu_on_tie = param<bool>("prefer_gpu_on_tie").default_val(true);
    config.energy_latency_guard_ms = param<float>("energy_latency_guard_ms").default_val(5.0f);
    config.route_load_balance_ms_per_active_request =
        param<float>("route_load_balance_ms_per_active_request").default_val(0.0f);
    config.unsupported_policy = param<std::string>("unsupported_policy").default_val("clip");
    config.pim_wait_threshold_ms = param<float>("pim_wait_threshold_ms").default_val(0.0f);
    config.arrival_time_scale = param<float>("arrival_time_scale").default_val(1.0f);
    config.lin_bucket = param<int>("lin_bucket").default_val(1);
    config.lout_bucket = param<int>("lout_bucket").default_val(1);
    config.batch_size = param<int>("batch_size").default_val(1);
    config.enable_prefill_batching = param<bool>("enable_prefill_batching").default_val(false);
    config.max_prefill_batch_size = param<int>("max_prefill_batch_size").default_val(1);
    config.enable_decode_batching = param<bool>("enable_decode_batching").default_val(false);
    config.max_decode_batch_size = param<int>("max_decode_batch_size").default_val(1);
    config.prefill_guard_ms = param<float>("prefill_guard_ms").default_val(-1.0f);
    config.max_consecutive_decode_batches = param<int>("max_consecutive_decode_batches").default_val(0);
    config.decode_batch_cap_with_prefill = param<int>("decode_batch_cap_with_prefill").default_val(0);
    config.local_scheduling_policy =
        param<std::string>("local_scheduling_policy").default_val("prefill_priority_fcfs_decode");
    config.fcfs_decode_prediction_mode =
        param<std::string>("fcfs_decode_prediction_mode").default_val("incremental");
    config.prompt_priority = param<bool>("prompt_priority").default_val(true);
    config.enable_decode_rebind = param<bool>("enable_decode_rebind").default_val(false);
    config.decode_rebind_margin_ms = param<float>("decode_rebind_margin_ms").default_val(0.0f);
    config.enable_predictive_admission = param<bool>("enable_predictive_admission").default_val(false);
    config.enable_slo_guarded_admission = param<bool>("enable_slo_guarded_admission").default_val(false);
    config.slo_tbt_ms = param<float>("slo_tbt_ms").default_val(-1.0f);
    config.slo_e2e_ms = param<float>("slo_e2e_ms").default_val(-1.0f);
    config.slo_ttft_ms = param<float>("slo_ttft_ms").default_val(-1.0f);
    config.admission_shadow_max_steps = param<int>("admission_shadow_max_steps").default_val(10000);
    config.admission_retry_interval_ms = param<float>("admission_retry_interval_ms").default_val(0.0f);
    config.admission_max_harmed_requests = param<int>("admission_max_harmed_requests").default_val(0);
    config.admission_max_total_harm_ms = param<float>("admission_max_total_harm_ms").default_val(0.0f);
    config.admission_max_single_harm_ms = param<float>("admission_max_single_harm_ms").default_val(0.0f);
    config.admission_max_own_miss_ms = param<float>("admission_max_own_miss_ms").default_val(0.0f);
    config.admission_check_own_slo = param<bool>("admission_check_own_slo").default_val(false);
    config.admission_max_bypass_count = param<int>("admission_max_bypass_count").default_val(0);
    config.admission_max_wait_ms = param<float>("admission_max_wait_ms").default_val(0.0f);
    config.admission_aging_harm_ms_per_ms =
        param<float>("admission_aging_harm_ms_per_ms").default_val(0.0f);
    config.admission_aging_harmed_requests_per_ms =
        param<float>("admission_aging_harmed_requests_per_ms").default_val(0.0f);
    config.share_gpu_prefill_across_routes =
        param<bool>("share_gpu_prefill_across_routes").default_val(false);
    config.gpu_queue_alpha = param<float>("gpu_queue_alpha").default_val(0.0f);
    config.pim_queue_alpha = param<float>("pim_queue_alpha").default_val(0.0f);
    config.active_request_alpha = param<float>("active_request_alpha").default_val(0.0f);
    config.decode_token_alpha = param<float>("decode_token_alpha").default_val(0.0f);
    config.capture_debug_events = param<bool>("capture_debug_events").default_val(false);
    config.debug_enabled = param<bool>("debug").default_val(false);
    config.debug_log_path = param<std::string>("debug_log_path").default_val("log/online_serving/frontend_debug.log");
    config.debug_interval_clk =
        std::max<Clk_t>(1, static_cast<Clk_t>(param<int>("debug_interval_cycles").default_val(1000000)));
    config.clock_ratio = param<uint>("clock_ratio").required();

    m_clock_ratio = config.clock_ratio;
    m_logger = Logging::create_logger("RealtimeServingFrontend");
    m_runtime = std::make_unique<Serving::RealtimeServingRuntime>(m_logger);
    m_runtime->initialize(config);

    m_logger->info("Loaded {} requests, {} GPU cost points, {} hybrid cost points.",
                   m_runtime->total_requests(),
                   m_runtime->gpu_cost_point_count(),
                   m_runtime->hybrid_cost_point_count());
    m_logger->info("Realtime serving config: route_policy={} local_policy={} batch={} decode_batching={} admission_queue={}",
                   config.route_policy,
                   config.local_scheduling_policy,
                   config.batch_size,
                   config.enable_decode_batching,
                   (config.enable_predictive_admission || config.enable_slo_guarded_admission));
  }

  void connect_memory_system(IMemorySystem* memory_system) override {
    IFrontEnd::connect_memory_system(memory_system);
    if (!m_runtime) {
      throw ConfigurationError("RealtimeServingFrontend: runtime was not initialized");
    }
    m_runtime->connect_memory_system(memory_system);
    m_clk = m_runtime->clk();
  }

  void tick() override {
    if (!m_runtime) {
      throw ConfigurationError("RealtimeServingFrontend: runtime was not initialized");
    }
    m_runtime->tick();
    m_clk = m_runtime->clk();
  }

  bool is_finished() override {
    if (!m_runtime) {
      return true;
    }
    return m_runtime->is_finished();
  }

  void finalize() override {
    if (m_runtime) {
      m_runtime->finalize();
      m_clk = m_runtime->clk();
    }
    IFrontEnd::finalize();
  }

  void print_stats(YAML::Emitter& emitter) override {
    if (!m_runtime) {
      throw ConfigurationError("RealtimeServingFrontend: runtime was not initialized");
    }
    m_runtime->print_stats(emitter, get_ifce_name(), get_name());
  }

 private:
  Logger_t m_logger;
  std::unique_ptr<Serving::RealtimeServingRuntime> m_runtime;
};

}  // namespace Ramulator
