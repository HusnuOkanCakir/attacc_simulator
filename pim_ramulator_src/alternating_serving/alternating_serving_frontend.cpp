#include <algorithm>
#include <memory>
#include <string>

#include "base/exception.h"
#include "frontend/frontend.h"
#include "frontend/impl/alternating_serving/runtime.h"

namespace Ramulator {

class AlternatingServingFrontend : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd,
                                    AlternatingServingFrontend,
                                    "AlternatingServingFrontend",
                                    "Simple online serving frontend that alternates between hybrid and GPU-only.");

 public:
  void init() override {
    AlternatingServing::FrontendConfig config;
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
    config.unsupported_policy = param<std::string>("unsupported_policy").default_val("clip");
    config.ignored_pim_wait_threshold_ms = param<float>("pim_wait_threshold_ms")
                                               .desc("Ignored; kept for wrapper compatibility.")
                                               .default_val(1.0f);
    config.arrival_time_scale = param<float>("arrival_time_scale").default_val(1.0f);
    config.lin_bucket = param<int>("lin_bucket").default_val(1);
    config.lout_bucket = param<int>("lout_bucket").default_val(1);
    config.batch_size = param<int>("batch_size").default_val(1);
    config.debug_enabled = param<bool>("debug").default_val(false);
    config.debug_log_path = param<std::string>("debug_log_path").default_val("log/online_serving/frontend_debug.log");
    config.debug_interval_clk =
        std::max<Clk_t>(1, static_cast<Clk_t>(param<int>("debug_interval_cycles").default_val(1000000)));
    config.clock_ratio = param<uint>("clock_ratio").required();

    m_clock_ratio = config.clock_ratio;
    m_logger = Logging::create_logger("AlternatingServingFrontend");
    m_runtime = std::make_unique<AlternatingServing::AlternatingServingRuntime>(m_logger);
    m_runtime->initialize(config);

    m_logger->info("Loaded {} requests, {} GPU cost points, {} hybrid cost points.",
                   m_runtime->total_requests(),
                   m_runtime->gpu_cost_point_count(),
                   m_runtime->hybrid_cost_point_count());
    m_logger->info("Alternating frontend keeps pim_wait_threshold_ms={} only for YAML compatibility.",
                   static_cast<double>(config.ignored_pim_wait_threshold_ms));
  }

  void connect_memory_system(IMemorySystem* memory_system) override {
    IFrontEnd::connect_memory_system(memory_system);
    if (!m_runtime) {
      throw ConfigurationError("AlternatingServingFrontend: runtime was not initialized");
    }
    m_runtime->connect_memory_system(memory_system);
    m_clk = m_runtime->clk();
  }

  void tick() override {
    if (!m_runtime) {
      throw ConfigurationError("AlternatingServingFrontend: runtime was not initialized");
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
      throw ConfigurationError("AlternatingServingFrontend: runtime was not initialized");
    }
    m_runtime->print_stats(emitter, get_ifce_name(), get_name());
  }

 private:
  Logger_t m_logger;
  std::unique_ptr<AlternatingServing::AlternatingServingRuntime> m_runtime;
};

}  // namespace Ramulator
