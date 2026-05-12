/**
 * serving_online/frontend.cpp
 *
 * ServingOnlineFrontend: clean IFrontEnd for pi0 PIM+GPU LLM serving.
 *
 * Architecture:
 *   - Scheduler admits arriving requests to routes (pim+gpu or gpu_only).
 *   - GPU work advances a wall-clock free-time estimate directly.
 *   - PIM work issues Request objects to the ramulator memory system, which
 *     drives the actual LPDDR5-PIM timing model.
 *   - The frontend ticks until all requests are Done or Dropped.
 *
 * File organisation:
 *   Runtime     (namespace Ramulator::ServingOnline) -- all simulation logic.
 *   ServingOnlineFrontend (namespace Ramulator) -- IFrontEnd wrapper, reads YAML params.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/exception.h"
#include "frontend/frontend.h"
#include "frontend/impl/serving_online/cost_table.h"
#include "frontend/impl/serving_online/csv_loader.h"
#include "frontend/impl/serving_online/pim_allocator.h"
#include "frontend/impl/serving_online/pim_commands.h"
#include "frontend/impl/serving_online/scheduler.h"
#include "translation/translation.h"
#include "translation/impl/serving_online_translation.h"

namespace {

template<typename... Args>
static std::string sfmt(const char* fmt_str, Args... args) {
  char buf[512];
  std::snprintf(buf, sizeof(buf), fmt_str, args...);
  return {buf};
}

// Decode a PIM VA into its constituent bit fields (mirrors pim_allocator.cpp constants).
struct VaFields {
  bool     is_pim_va;
  uint64_t region;      // 0=Key, 1=Value
  uint64_t request_id;
  uint64_t page_index;
  uint64_t page_offset;
};
static VaFields decode_va(uint64_t va) {
  VaFields f;
  f.is_pim_va   = (va >> 63) & 1;
  f.region      = (va >> 60) & 0x7;
  f.request_id  = (va >> 40) & 0xFFFFF;
  f.page_index  = (va >> 12) & 0xFFFFFFF;
  f.page_offset =  va        & 0xFFF;
  return f;
}

} // namespace

namespace Ramulator::ServingOnline {

// ─── Helpers ─────────────────────────────────────────────────────────────────

static std::string state_name(ReqState state) {
  switch (state) {
    case ReqState::WaitingAdmission: return "waiting_admission";
    case ReqState::WaitingPrefill:   return "waiting_prefill";
    case ReqState::WaitingDecode:    return "waiting_decode";
    case ReqState::Done:             return "done";
    case ReqState::Dropped:          return "dropped";
  }
  return "unknown";
}

static double mean_or_neg1(const std::vector<double>& v) {
  if (v.empty()) return -1.0;
  return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

// ─── PIM shape cache ─────────────────────────────────────────────────────────
//
// Per-shape DRAM-drain cycle cache. Each (route, ctx_bucket, batch_size)
// shape is measured inline on first sighting; subsequent decode tasks with
// the same shape reuse the cached cycle count and skip Ramulator entirely.

static constexpr int kCacheBucketTokens = 32;  // = kLineBytes/dtype_bytes (FP16)

struct PimShapeKey {
  std::string route;
  int         ctx_bucket;
  int         batch_size;

  bool operator==(const PimShapeKey& o) const {
    return ctx_bucket == o.ctx_bucket
        && batch_size == o.batch_size
        && route      == o.route;
  }
};

struct PimShapeKeyHash {
  size_t operator()(const PimShapeKey& k) const {
    size_t h = std::hash<std::string>{}(k.route);
    h ^= std::hash<int>{}(k.ctx_bucket) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(k.batch_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

struct PimShapeStats {
  Clk_t  drain_clk  = 0;   // simulated DRAM cycles to drain inline (first sighting)
  size_t cmds_total = 0;   // command count at first sighting (sanity / logging)
  size_t hits       = 0;   // cache-hit counter
};

// ─── Runtime ─────────────────────────────────────────────────────────────────
//
// Contains all simulation state. The outer IFrontEnd class delegates to this.

class Runtime {
 public:
  explicit Runtime(Logger_t logger) : m_logger(std::move(logger)) {}

  // ── Initialization ─────────────────────────────────────────────────────────

  void initialize(const ServingOnlineConfig& cfg) {
    m_cfg = cfg;

    // Load Azure inference CSV.
    m_arrivals = load_azure_csv(cfg.csv_path, cfg.arrival_limit, cfg.arrival_time_scale);
    m_total_requests = m_arrivals.size();

    // Load cost table(s). Both routes may share a file or have separate files.
    m_cost_table.load(cfg.cost_gpu_csv);
    if (!cfg.cost_pim_csv.empty() && cfg.cost_pim_csv != cfg.cost_gpu_csv) {
      m_cost_table.load(cfg.cost_pim_csv);
    }
    if (!cfg.cost_gpu_model.empty())    m_cost_table.load_ml(cfg.cost_gpu_model);
    if (!cfg.cost_hybrid_model.empty()) m_cost_table.load_ml(cfg.cost_hybrid_model);

    // Build KV allocator.
    PimAllocator::Config alloc_cfg;
    alloc_cfg.kv_pool_bytes   = cfg.kv_pool_bytes;
    alloc_cfg.page_size_bytes = cfg.page_size_bytes;
    alloc_cfg.num_channels    = cfg.num_channels;
    m_allocator = std::make_shared<PimAllocator>(alloc_cfg);

    // Build PIM command generator.
    PimCommandGen::Params cmd_params;
    cmd_params.num_layers  = cfg.num_layers;
    cmd_params.num_heads   = cfg.num_heads;
    cmd_params.d_head      = cfg.d_head;
    cmd_params.dtype_bytes = cfg.dtype_bytes;
    if (cfg.pim_command_mode == "simple") {
      cmd_params.mode = PimCommandGen::Mode::Simple;
    } else {
      cmd_params.mode = PimCommandGen::Mode::Realistic;
    }
    m_command_gen = std::make_unique<PimCommandGen>(cmd_params, m_allocator.get());

    // Initialize scheduler.
    m_scheduler.initialize(cfg, m_arrivals, m_cost_table, *m_allocator);

    // Open debug log if requested.
    if (!cfg.debug_log_path.empty()) {
      m_debug_log.open(cfg.debug_log_path);
      if (!m_debug_log.is_open()) {
        throw std::runtime_error("ServingOnline runtime: cannot open debug_log_path: "
                                 + cfg.debug_log_path);
      }
      m_debug_log << "# serving_online debug log\n"
                  << "# format: [wall_time_ms] EVENT  fields...\n"
                  << "# events: ARRIVE ADMIT KV_ALLOC KV_FREE KV_OOM_HOLD KV_OOM_FALLBACK "
                     "KV_WAKE TASK_START TRANSLATE TASK_DONE DONE PROGRESS TIMING "
                     "CACHE_HIT CACHE_MISS CACHE_FILL PIM_CACHE_STATS CMD_GEN\n"
                  << "#\n";
      // Forward scheduler events to the same log file.
      m_scheduler.set_log_fn([this](const std::string& line) {
        m_debug_log << line << '\n';
      });

      // Derive status.txt path next to debug.log.  status.txt is rewritten
      // atomically every ~1 wall-second so the user can `cat` it any time
      // to see live progress without waiting for the run to finish.
      const auto slash = cfg.debug_log_path.find_last_of('/');
      m_status_file_path = (slash == std::string::npos)
          ? std::string("status.txt")
          : (cfg.debug_log_path.substr(0, slash) + "/status.txt");
    }

    // Note: load_pim_cache() runs from connect_memory_system(), not here,
    // because the cache fingerprint includes m_memory_system->get_tCK()
    // and the memory system is attached after init().
  }

  void attach_translation(ITranslation* translation) {
    m_translation = translation;
  }

  std::shared_ptr<PimAllocator> allocator_shared() const { return m_allocator; }

  // ── Memory system attachment ───────────────────────────────────────────────

  void connect_memory_system(IMemorySystem* memory_system) {
    m_memory_system = memory_system;
    if (m_memory_system == nullptr) {
      throw ConfigurationError("ServingOnline runtime: memory system is null");
    }
    if (m_memory_system->get_tCK() <= 0.0f) {
      throw ConfigurationError("ServingOnline runtime: invalid tCK from memory system");
    }
    m_cycles_per_ms = 1.0e6 / static_cast<double>(m_memory_system->get_tCK());

    // Now that we know tCK we can compute the cache fingerprint and try
    // to load a persisted cache. No-op when pim_cache_file is empty.
    load_pim_cache();
  }

  // ── Simulation tick ────────────────────────────────────────────────────────

  void tick() {
    if (is_finished()) return;

    maybe_log_progress();

    process_callbacks();
    {
      auto t0 = wall_clock::now();
      complete_active_task_if_ready();
      m_t_complete += wall_clock::now() - t0;
    }

    if (!m_active_task.has_value()) {
      auto t0 = wall_clock::now();
      auto next = m_scheduler.tick(now_ms(), m_gpu_free_ms, m_pim_free_ms);
      m_t_sched += wall_clock::now() - t0;
      if (next) {
        start_task(std::move(*next));
      } else if (auto wake = m_scheduler.next_wakeup_ms(); wake.has_value()) {
        fast_forward_to_ms(*wake);
        return;
      }
    }

    if (m_active_task.has_value()) {
      if (m_active_task->phase == "prefill") {
        fast_forward_to_ms(m_active_task->start_ms + m_active_task->e2e_ms);
        auto t0 = wall_clock::now();
        complete_active_task_if_ready();
        m_t_complete += wall_clock::now() - t0;
        return;
      }
      if (m_active_task->phase == "decode"
          && m_active_task->route != m_cfg.pim_route_name) {
        fast_forward_to_ms(m_active_task->start_ms + m_active_task->gpu_ms);
        auto t0 = wall_clock::now();
        complete_active_task_if_ready();
        m_t_complete += wall_clock::now() - t0;
        return;
      }
    }

    pump_active_pim();
    process_callbacks();
    {
      auto t0 = wall_clock::now();
      complete_active_task_if_ready();
      m_t_complete += wall_clock::now() - t0;
    }

    // Once all PIM reads for a PIM-decode task have been issued and drained,
    // the task is only gated on wall-clock gpu/pim time. Fast-forward instead
    // of burning millions of empty cycles.
    if (m_active_task.has_value()
        && m_active_task->phase == "decode"
        && m_active_task->route == m_cfg.pim_route_name
        && m_issue_done
        && m_pim_outstanding == 0) {
      const double target = m_active_task->start_ms
                          + std::max({m_active_task->gpu_ms,
                                      m_active_task->pim_ms,
                                      m_active_task->e2e_ms});
      fast_forward_to_ms(target);
      auto t0 = wall_clock::now();
      complete_active_task_if_ready();
      m_t_complete += wall_clock::now() - t0;
      return;
    }

    if (!is_finished()) {
      ++m_clk;
    }
  }

  // Phase 1: print one PROGRESS line per ~1s wall-clock, gated on debug log.
  // Also rewrites status.txt atomically so the user can `cat` it for live
  // progress at any moment. Cheap: one steady_clock::now() + one comparison
  // per tick.
  void maybe_log_progress() {
    if (!m_debug_log.is_open()) return;
    const auto now = wall_clock::now();
    using namespace std::chrono;
    if (now - m_last_progress_wall < seconds(1)) return;
    const double dt_wall_s =
        duration<double>(now - m_last_progress_wall).count();
    m_last_progress_wall = now;

    const double wall_s = duration<double>(now - m_wall_start).count();
    const double sim_s  = now_ms() / 1000.0;
    const double ratio  = (wall_s > 1e-9) ? (sim_s / wall_s) : 0.0;

    // Delta-based throughput rates since the last PROGRESS emission.
    const size_t d_cmds = (m_next_pim_request >= m_last_progress_cmds_sent)
        ? (m_next_pim_request - m_last_progress_cmds_sent) : 0;
    const Clk_t  d_cyc  = (m_clk >= m_last_progress_clk)
        ? (m_clk - m_last_progress_clk) : 0;
    const double cmds_per_s = (dt_wall_s > 1e-9)
        ? (static_cast<double>(d_cmds) / dt_wall_s) : 0.0;
    const double cyc_per_s  = (dt_wall_s > 1e-9)
        ? (static_cast<double>(d_cyc) / dt_wall_s) : 0.0;
    m_last_progress_cmds_sent = m_next_pim_request;
    m_last_progress_clk       = m_clk;

    std::string task_str = "idle";
    if (m_active_task.has_value()) {
      task_str = m_active_task->phase + "/" + m_active_task->route;
    }
    dlog(sfmt("[%10.3fms] PROGRESS    cycle=%llu  task=%-30s  "
              "cmds_total=%zu  cmds_sent=%zu  cmds_outstanding=%zu  "
              "cmds_per_s=%.2e  cycles_per_s=%.2e  "
              "bp_task=%zu  bp_total=%zu  "
              "wall_elapsed=%.3fs  sim_per_wall=%.3fx",
              now_ms(), (unsigned long long)m_clk,
              task_str.c_str(),
              m_pending_pim_requests.size(),
              m_next_pim_request,
              m_pim_outstanding,
              cmds_per_s, cyc_per_s,
              m_send_backpressure_events_task,
              m_send_backpressure_events_total,
              wall_s, ratio));
    // Flush so `tail -f debug.log` shows PROGRESS lines in real time
    // instead of in bursts.
    m_debug_log.flush();

    emit_status_file(wall_s, ratio, cmds_per_s, cyc_per_s);
  }

  // Atomically write a small status.txt next to debug.log so the user can
  // `cat <run_dir>/<policy>/status.txt` any time without parsing logs.
  // Write to status.txt.tmp then rename — a concurrent reader either sees
  // the previous snapshot or the new one, never a half-written file.
  void emit_status_file(double wall_s, double sim_per_wall,
                        double cmds_per_s, double cyc_per_s) {
    if (m_status_file_path.empty()) return;
    const std::string tmp = m_status_file_path + ".tmp";
    std::ofstream out(tmp);
    if (!out.is_open()) return;

    std::string task_str = "idle";
    int    task_ctx = 0;
    int    task_bs  = 0;
    std::string task_reqs = "";
    if (m_active_task.has_value()) {
      task_str = m_active_task->phase + "/" + m_active_task->route;
      task_ctx = m_active_task->context_tokens;
      task_bs  = m_active_task->batch_size;
      for (int idx : m_active_task->request_indices) {
        if (!task_reqs.empty()) task_reqs += ',';
        task_reqs += std::to_string(m_scheduler.requests()[idx].id);
      }
    }

    // Scheduler state summary.
    size_t s_waiting = 0, s_done = 0, s_dropped = 0;
    for (const auto& r : m_scheduler.requests()) {
      switch (r.state) {
        case ReqState::WaitingAdmission:
        case ReqState::WaitingPrefill:
        case ReqState::WaitingDecode:  s_waiting++; break;
        case ReqState::Done:           s_done++;    break;
        case ReqState::Dropped:        s_dropped++; break;
      }
    }

    const size_t cmds_total = m_pending_pim_requests.size();
    const double pct = (cmds_total > 0)
        ? (100.0 * static_cast<double>(m_next_pim_request) / cmds_total) : 0.0;

    // ETAs derived from observed rates. Print "N/A" when the input rate is
    // zero (e.g. no in-flight task, no completed requests yet) so we never
    // emit a misleading huge number.
    const size_t cmds_remaining_in_task = (cmds_total > m_next_pim_request)
        ? (cmds_total - m_next_pim_request) : 0;
    std::string eta_task_str = "N/A";
    if (m_active_task.has_value() && cmds_per_s > 1.0 && cmds_remaining_in_task > 0) {
      const double eta_task_s =
          static_cast<double>(cmds_remaining_in_task) / cmds_per_s;
      eta_task_str = sfmt("%.1f s", eta_task_s);
    }
    std::string eta_run_str = "N/A";
    if (s_done > 0 && wall_s > 1.0) {
      const double done_rate = static_cast<double>(s_done) / wall_s;
      const size_t remaining = (m_total_requests > s_done)
          ? (m_total_requests - s_done) : 0;
      if (remaining > 0 && done_rate > 0.0) {
        const double eta_run_s = static_cast<double>(remaining) / done_rate;
        eta_run_str = sfmt("%.1f s  (~%.1f min)", eta_run_s, eta_run_s / 60.0);
      } else {
        eta_run_str = "0 s";
      }
    }

    out << "# live status snapshot — rewritten every ~1 wall second\n";
    out << sfmt("sim_time_ms              : %.3f\n",  now_ms());
    out << sfmt("wall_elapsed_s           : %.3f\n",  wall_s);
    out << sfmt("sim_per_wall             : %.4fx\n", sim_per_wall);
    out << sfmt("cycle                    : %llu\n",  (unsigned long long)m_clk);
    out << sfmt("cmds_per_wall_s          : %.2e\n",  cmds_per_s);
    out << sfmt("cycles_per_wall_s        : %.2e\n",  cyc_per_s);
    out << "\n";
    out << sfmt("active_task              : %s\n",    task_str.c_str());
    out << sfmt("active_reqs              : [%s]\n",  task_reqs.c_str());
    out << sfmt("active_ctx_tokens        : %d\n",    task_ctx);
    out << sfmt("active_batch_size        : %d\n",    task_bs);
    out << sfmt("cmds_sent_in_task        : %zu / %zu  (%.1f%%)\n",
                m_next_pim_request, cmds_total, pct);
    out << sfmt("cmds_outstanding         : %zu\n",   m_pim_outstanding);
    out << sfmt("backpressure_task        : %zu\n",   m_send_backpressure_events_task);
    out << sfmt("backpressure_total       : %zu\n",   m_send_backpressure_events_total);
    out << sfmt("eta_current_task         : %s\n",    eta_task_str.c_str());
    out << "\n";
    out << sfmt("cache_hits_total         : %zu\n",   m_pim_cache_total_hits);
    out << sfmt("cache_misses_total       : %zu\n",   m_pim_cache_total_misses);
    out << sfmt("cache_distinct_shapes    : %zu\n",   m_pim_shape_cache.size());
    out << "\n";
    out << sfmt("scheduler_waiting        : %zu\n",   s_waiting);
    out << sfmt("scheduler_done           : %zu\n",   s_done);
    out << sfmt("scheduler_dropped        : %zu\n",   s_dropped);
    out << sfmt("scheduler_total          : %zu\n",   m_total_requests);
    out << sfmt("eta_full_run             : %s\n",    eta_run_str.c_str());
    out.close();
    if (out.fail()) {
      std::remove(tmp.c_str());
      return;
    }
    std::rename(tmp.c_str(), m_status_file_path.c_str());
  }

  // ── Status ────────────────────────────────────────────────────────────────

  bool is_finished() const {
    return m_scheduler.all_done()
        && !m_active_task.has_value()
        && m_completed_callback_events == 0
        && m_pim_outstanding == 0;
  }

  // ── Output ────────────────────────────────────────────────────────────────

  // Emit a PIM_CACHE_STATS summary line plus a per-shape breakdown to the
  // debug log. Called from finalize(). No-op if debug log is disabled or
  // the cache is off.
  void log_pim_cache_stats() {
    if (!m_debug_log.is_open()) return;
    if (!m_cfg.pim_cache_enabled) {
      m_debug_log << "[          ] PIM_CACHE_STATS  enabled=false" << std::endl;
      return;
    }
    const size_t total = m_pim_cache_total_hits + m_pim_cache_total_misses;
    const double hit_rate = (total > 0)
        ? (100.0 * static_cast<double>(m_pim_cache_total_hits) / total) : 0.0;
    m_debug_log << sfmt("[          ] PIM_CACHE_STATS  enabled=true  hits=%zu  misses=%zu  "
                        "hit_rate=%.2f%%  distinct_shapes=%zu",
                        m_pim_cache_total_hits, m_pim_cache_total_misses,
                        hit_rate, m_pim_shape_cache.size())
                << std::endl;
    for (const auto& [key, stats] : m_pim_shape_cache) {
      const double drain_ms = (m_cycles_per_ms > 0.0)
          ? (static_cast<double>(stats.drain_clk) / m_cycles_per_ms) : 0.0;
      m_debug_log << sfmt("[          ]   SHAPE  route=%-20s  ctx_bucket=%4d  bs=%d  "
                          "hits=%zu  drain_clk=%llu  drain_ms=%.3f  cmds=%zu",
                          key.route.c_str(), key.ctx_bucket, key.batch_size,
                          stats.hits,
                          (unsigned long long)stats.drain_clk, drain_ms,
                          stats.cmds_total)
                  << std::endl;
    }
    m_debug_log.flush();
  }

  // ── Cross-run PIM-shape cache I/O ──────────────────────────────────────────
  //
  // Schema fingerprint: model architecture fields from ServingOnlineConfig
  // plus the Ramulator clock period (tCK). Any of those changing means a
  // previously-measured drain_clk is no longer valid for the same logical
  // (route, ctx_bucket, batch_size) shape — different command count or
  // different DRAM timing.

  std::string pim_cache_fingerprint() const {
    return sfmt("num_layers=%d num_heads=%d d_head=%d dtype_bytes=%d "
                "pim_route=%s gpu_route=%s pim_cmd_mode=%s vision_prefix=%d "
                "tCK=%.4f",
                m_cfg.num_layers, m_cfg.num_heads, m_cfg.d_head, m_cfg.dtype_bytes,
                m_cfg.pim_route_name.c_str(), m_cfg.gpu_route_name.c_str(),
                m_cfg.pim_command_mode.c_str(),
                m_cfg.vision_prefix_tokens,
                (m_memory_system != nullptr) ? m_memory_system->get_tCK() : 0.0f);
  }

  // Populate m_pim_shape_cache from disk if pim_cache_file is set and the
  // file exists. Header line must match the current run's fingerprint;
  // otherwise the load is rejected (cache starts empty) and a warning is
  // emitted to debug.log.
  void load_pim_cache() {
    if (m_cfg.pim_cache_file.empty()) return;
    std::ifstream in(m_cfg.pim_cache_file);
    if (!in.is_open()) {
      if (m_debug_log.is_open()) {
        dlog(sfmt("[          ] PIM_CACHE_LOAD  path=%s  status=missing  "
                  "(no preload, cache starts empty)",
                  m_cfg.pim_cache_file.c_str()));
      }
      return;
    }
    const std::string expected_fp = pim_cache_fingerprint();
    std::string line;
    bool header_ok = false;
    size_t loaded = 0;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      if (line[0] == '#') {
        // Look for the fingerprint line — first '#' line that contains '='.
        if (!header_ok && line.find('=') != std::string::npos) {
          // Strip leading "# " and compare.
          const auto pos = line.find_first_not_of("# \t");
          const std::string got = (pos == std::string::npos) ? "" : line.substr(pos);
          if (got != expected_fp) {
            if (m_debug_log.is_open()) {
              dlog(sfmt("[          ] PIM_CACHE_LOAD_REJECT  path=%s  "
                        "reason=header_mismatch",
                        m_cfg.pim_cache_file.c_str()));
              dlog(sfmt("[          ]   expected: %s", expected_fp.c_str()));
              dlog(sfmt("[          ]   got     : %s", got.c_str()));
            }
            return;  // start fresh, do NOT load any rows
          }
          header_ok = true;
        }
        continue;
      }
      // Data row: route,ctx_bucket,bs,drain_clk,cmds_total
      // We tolerate header_ok==false silently here (file might not have a
      // header at all — old format). But for safety, require header_ok.
      if (!header_ok) {
        if (m_debug_log.is_open()) {
          dlog(sfmt("[          ] PIM_CACHE_LOAD_REJECT  path=%s  "
                    "reason=no_header_line",
                    m_cfg.pim_cache_file.c_str()));
        }
        return;
      }
      const auto c1 = line.find(',');
      const auto c2 = (c1 == std::string::npos) ? std::string::npos : line.find(',', c1 + 1);
      const auto c3 = (c2 == std::string::npos) ? std::string::npos : line.find(',', c2 + 1);
      const auto c4 = (c3 == std::string::npos) ? std::string::npos : line.find(',', c3 + 1);
      if (c4 == std::string::npos) continue;  // malformed
      try {
        const std::string route = line.substr(0, c1);
        const int ctx = std::stoi(line.substr(c1 + 1, c2 - c1 - 1));
        const int bs  = std::stoi(line.substr(c2 + 1, c3 - c2 - 1));
        const Clk_t drain = static_cast<Clk_t>(std::stoull(line.substr(c3 + 1, c4 - c3 - 1)));
        const size_t cmds = static_cast<size_t>(std::stoull(line.substr(c4 + 1)));
        m_pim_shape_cache[PimShapeKey{route, ctx, bs}] =
            PimShapeStats{drain, cmds, 0};
        loaded++;
      } catch (const std::exception&) {
        // Skip malformed line.
      }
    }
    if (m_debug_log.is_open()) {
      dlog(sfmt("[          ] PIM_CACHE_LOAD  path=%s  status=loaded  "
                "entries=%zu",
                m_cfg.pim_cache_file.c_str(), loaded));
    }
  }

  // Persist m_pim_shape_cache to disk. Atomic via .tmp + rename so a
  // concurrent reader (or a crash) never sees a half-written file.
  void save_pim_cache() {
    if (m_cfg.pim_cache_file.empty()) return;
    if (m_pim_shape_cache.empty()) {
      // Nothing learned this run; do not stomp an existing file with
      // an empty body.
      if (m_debug_log.is_open()) {
        dlog(sfmt("[          ] PIM_CACHE_SAVE  path=%s  status=skipped  "
                  "reason=empty_cache",
                  m_cfg.pim_cache_file.c_str()));
      }
      return;
    }
    const std::string tmp = m_cfg.pim_cache_file + ".tmp";
    std::ofstream out(tmp);
    if (!out.is_open()) {
      if (m_debug_log.is_open()) {
        dlog(sfmt("[          ] PIM_CACHE_SAVE  path=%s  status=open_failed",
                  m_cfg.pim_cache_file.c_str()));
      }
      return;
    }
    out << "# pim_shape_cache v1\n";
    out << "# " << pim_cache_fingerprint() << '\n';
    out << "# columns: route,ctx_bucket,batch_size,drain_clk,cmds_total\n";
    for (const auto& [key, stats] : m_pim_shape_cache) {
      out << key.route << ','
          << key.ctx_bucket << ','
          << key.batch_size << ','
          << stats.drain_clk << ','
          << stats.cmds_total << '\n';
    }
    out.close();
    if (out.fail()) {
      std::remove(tmp.c_str());
      if (m_debug_log.is_open()) {
        dlog(sfmt("[          ] PIM_CACHE_SAVE  path=%s  status=write_failed",
                  m_cfg.pim_cache_file.c_str()));
      }
      return;
    }
    if (std::rename(tmp.c_str(), m_cfg.pim_cache_file.c_str()) != 0) {
      std::remove(tmp.c_str());
      if (m_debug_log.is_open()) {
        dlog(sfmt("[          ] PIM_CACHE_SAVE  path=%s  status=rename_failed",
                  m_cfg.pim_cache_file.c_str()));
      }
      return;
    }
    if (m_debug_log.is_open()) {
      dlog(sfmt("[          ] PIM_CACHE_SAVE  path=%s  status=saved  "
                "entries=%zu",
                m_cfg.pim_cache_file.c_str(), m_pim_shape_cache.size()));
      m_debug_log.flush();
    }
  }

  void finalize() {
    log_pim_cache_stats();
    save_pim_cache();
    if (m_cfg.requests_out_csv.empty()) return;

    std::ofstream out(m_cfg.requests_out_csv);
    if (!out.is_open()) {
      throw std::runtime_error(
          "ServingOnline runtime: cannot open requests_out_csv: " + m_cfg.requests_out_csv);
    }

    out << "request_id,arrival_ms,context_tokens,generated_tokens,state,route,is_lost,"
           "admission_attempts,admission_last_reject_reason,dropped_reason,"
           "admitted_ms,scheduler_predicted_start_ms,scheduler_predicted_finish_ms,"
           "scheduler_predicted_e2e_ms,scheduler_predicted_prefill_ms,"
           "scheduler_predicted_decode_total_ms,"
           "prefill_start_ms,prefill_end_ms,first_token_ms,completion_ms,"
           "ttft_ms,e2e_ms,tbt_mean_ms,preempt_count,tail_trim_tokens\n";

    for (const auto& req : m_scheduler.requests()) {
      const double ttft_ms =
          req.first_token_ms >= 0.0 ? (req.first_token_ms - req.arrival_ms) : -1.0;
      const double e2e_ms =
          req.completion_ms >= 0.0 ? (req.completion_ms - req.arrival_ms) : -1.0;

      out << req.id << ','
          << req.arrival_ms << ','
          << req.context_tokens << ','
          << req.generated_tokens << ','
          << state_name(req.state) << ','
          << req.route << ','
          << (req.is_lost ? 1 : 0) << ','
          << req.admission_attempts << ','
          << req.admission_last_reject_reason << ','
          << req.dropped_reason << ','
          << req.admitted_ms << ','
          << req.scheduler_predicted_start_ms << ','
          << req.scheduler_predicted_finish_ms << ','
          << req.scheduler_predicted_e2e_ms << ','
          << req.scheduler_predicted_prefill_ms << ','
          << req.scheduler_predicted_decode_total_ms << ','
          << req.prefill_start_ms << ','
          << req.prefill_end_ms << ','
          << req.first_token_ms << ','
          << req.completion_ms << ','
          << ttft_ms << ','
          << e2e_ms << ','
          << mean_or_neg1(req.tbt_ms) << ','
          << req.preempt_count << ','
          << req.tail_trim_tokens << '\n';
    }
  }

  void print_stats(YAML::Emitter& emitter,
                   const std::string& ifce_name,
                   const std::string& impl_name) const {
    size_t completed = 0;
    size_t dropped   = 0;
    size_t pim_routes = 0;
    size_t gpu_routes = 0;
    for (const auto& req : m_scheduler.requests()) {
      if (req.state == ReqState::Done)    completed++;
      if (req.state == ReqState::Dropped) dropped++;
      if (req.route == m_cfg.pim_route_name) pim_routes++;
      if (req.route == m_cfg.gpu_route_name) gpu_routes++;
    }

    emitter << YAML::Key << ifce_name << YAML::Value << YAML::BeginMap;
    emitter << YAML::Key << "impl"              << YAML::Value << impl_name;
    emitter << YAML::Key << "total_requests"    << YAML::Value << m_total_requests;
    emitter << YAML::Key << "admitted_requests" << YAML::Value << m_scheduler.admitted_count();
    emitter << YAML::Key << "completed_requests"<< YAML::Value << completed;
    emitter << YAML::Key << "dropped_requests"  << YAML::Value << dropped;
    emitter << YAML::Key << "kv_oom_holds"      << YAML::Value << m_scheduler.kv_hold_count();
    emitter << YAML::Key << "preempt_count"     << YAML::Value << m_scheduler.preempt_count();
    emitter << YAML::Key << "tail_trim_count"   << YAML::Value << m_scheduler.tail_trim_count();
    emitter << YAML::Key << "full_evict_count"  << YAML::Value << m_scheduler.full_evict_count();
    emitter << YAML::Key << "tokens_recomputed" << YAML::Value << m_scheduler.tokens_recomputed();
    emitter << YAML::Key << "predictive_holds"  << YAML::Value << m_scheduler.predictive_holds();
    emitter << YAML::Key << "kv_scheduler_policy" << YAML::Value << m_cfg.kv_scheduler_policy;
    emitter << YAML::Key << "route_counts"      << YAML::Value << YAML::BeginMap;
    emitter << YAML::Key << m_cfg.gpu_route_name << YAML::Value << gpu_routes;
    emitter << YAML::Key << m_cfg.pim_route_name << YAML::Value << pim_routes;
    emitter << YAML::EndMap;
    emitter << YAML::Key << "frontend_cycles"   << YAML::Value << m_clk;
    emitter << YAML::Key << "gpu_free_ms"       << YAML::Value << m_gpu_free_ms;
    emitter << YAML::Key << "pim_free_ms"       << YAML::Value << m_pim_free_ms;
    emitter << YAML::Key << "requests_out_csv"  << YAML::Value << m_cfg.requests_out_csv;
    // PIM shape-cache stats (Lever 1).
    emitter << YAML::Key << "pim_cache_enabled"      << YAML::Value
            << (m_cfg.pim_cache_enabled ? "true" : "false");
    emitter << YAML::Key << "pim_cache_hits"         << YAML::Value << m_pim_cache_total_hits;
    emitter << YAML::Key << "pim_cache_misses"       << YAML::Value << m_pim_cache_total_misses;
    emitter << YAML::Key << "pim_cache_distinct_shapes" << YAML::Value
            << m_pim_shape_cache.size();
    emitter << YAML::EndMap << YAML::Newline;
  }

  // ── Accessors ─────────────────────────────────────────────────────────────

  size_t total_requests()   const { return m_total_requests; }
  size_t cost_point_count() const { return m_cost_table.total_points(); }
  Clk_t  clk()              const { return m_clk; }

 private:
  // ── Time conversion ────────────────────────────────────────────────────────

  double now_ms() const {
    if (m_cycles_per_ms <= 0.0) return 0.0;
    return static_cast<double>(m_clk) / m_cycles_per_ms;
  }

  Clk_t clk_for_ms(double time_ms) const {
    if (m_cycles_per_ms <= 0.0 || time_ms <= 0.0) return 0;
    return static_cast<Clk_t>(std::ceil(time_ms * m_cycles_per_ms));
  }

  void fast_forward_to_ms(double target_ms) {
    const Clk_t target_clk = clk_for_ms(target_ms);
    if (target_clk > m_clk) {
      m_clk = target_clk;
    }
  }

  // Effective decode-time context depth: prompt + the first token produced by
  // prefill + completed decode tokens.
  int current_context_tokens(const RuntimeRequest& req) const {
    const int completed = std::max(0, req.svc.decode_tokens - req.remaining_decode);
    const int prefill_token = req.generated_tokens > 0 ? 1 : 0;
    return req.context_tokens + prefill_token + completed;
  }

  // ── Task lifecycle ─────────────────────────────────────────────────────────

  void start_task(ActiveTask task) {
    const double start_ms = now_ms();
    task.start_ms = start_ms;

    // Stamp prefill_start_ms for every request in this task.
    if (task.phase == "prefill") {
      for (const int idx : task.request_indices) {
        auto& req = m_scheduler.mutable_requests()[idx];
        if (req.prefill_start_ms < 0.0) {
          req.prefill_start_ms = start_ms;
        }
      }
    }

    // Advance the GPU and PIM free-time clocks.
    m_gpu_free_ms = std::max(m_gpu_free_ms, start_ms) + task.gpu_ms;
    m_pim_free_ms = std::max(m_pim_free_ms, start_ms) + task.pim_ms;

    // Reset PIM command state.
    m_pending_pim_requests.clear();
    m_next_pim_request          = 0;
    m_pim_outstanding           = 0;
    m_completed_callback_events = 0;
    m_issue_done                = true;
    m_send_backpressure_events_task = 0;

    // Reset section timers and remember wall-clock start of this task so
    // TASK_DONE can print a per-section breakdown.
    m_t_cmd_gen = m_t_translate = m_t_send = m_t_sched = m_t_complete =
        std::chrono::nanoseconds{0};
    m_task_wall_start              = wall_clock::now();
    m_clk_at_task_start            = m_clk;
    m_active_task_was_cache_hit    = false;

    // For PIM decode tasks: consult the per-shape cache before generating
    // commands. On hit, fast-forward by max(cost_table, cached_drain) and
    // skip Ramulator entirely. On miss, fall through to the inline path
    // and capture drain_clk in complete_active_task_if_ready().
    if (task.phase == "decode" && task.route == m_cfg.pim_route_name) {
      if (m_cfg.pim_cache_enabled) {
        const PimShapeKey key{task.route,
                              ctx_bucket(task.context_tokens),
                              task.batch_size};
        auto it = m_pim_shape_cache.find(key);
        if (it != m_pim_shape_cache.end()) {
          // Cache hit: skip command generation and Ramulator drain.
          it->second.hits++;
          m_pim_cache_total_hits++;
          m_active_task_was_cache_hit = true;
          if (m_debug_log.is_open()) {
            const double drain_ms = (m_cycles_per_ms > 0.0)
                ? (static_cast<double>(it->second.drain_clk) / m_cycles_per_ms)
                : 0.0;
            dlog(sfmt("[%10.3fms] CACHE_HIT   route=%-20s  ctx_bucket=%4d  bs=%d  "
                      "ctx=%5d  drain_clk=%llu  drain_ms=%.3f  hits=%zu",
                      start_ms, task.route.c_str(),
                      key.ctx_bucket, key.batch_size,
                      task.context_tokens,
                      (unsigned long long)it->second.drain_clk,
                      drain_ms, it->second.hits));
          }
          // Hit path: m_issue_done stays true, no commands pumped, no
          // outstanding callbacks. complete_active_task_if_ready() will
          // see pim_done=true immediately and finish the task on the
          // existing fast-forward in tick().
          m_active_task = std::move(task);
          return;
        }
        // Cache miss: counted now; drain_clk recorded on completion.
        m_pim_cache_total_misses++;
        if (m_debug_log.is_open()) {
          dlog(sfmt("[%10.3fms] CACHE_MISS  route=%-20s  ctx_bucket=%4d  bs=%d  "
                    "ctx=%5d  (will measure inline)",
                    start_ms, task.route.c_str(),
                    key.ctx_bucket, key.batch_size,
                    task.context_tokens));
        }
      }
      m_issue_done = false;
      for (const int idx : task.request_indices) {
        const auto& req = m_scheduler.requests()[idx];
        const int ctx_now = current_context_tokens(req);
        auto on_complete = [this](Request&) { m_completed_callback_events++; };
        auto t0 = wall_clock::now();
        auto cmds = m_command_gen->decode_step(
            req.id, ctx_now, on_complete);
        const auto elapsed = wall_clock::now() - t0;
        m_t_cmd_gen += elapsed;

        // Per-call CMD_GEN diagnostic: how much wall-time did decode_step()
        // spend constructing this request's command vector, and what's the
        // per-element cost? Lets us directly test the hypothesis that
        // online command generation is the wall-clock bottleneck.
        if (m_debug_log.is_open()) {
          const size_t ncmds = cmds.size();
          const double wall_ms =
              std::chrono::duration<double, std::milli>(elapsed).count();
          const double per_cmd_ns = (ncmds > 0)
              ? (std::chrono::duration<double, std::nano>(elapsed).count()
                 / static_cast<double>(ncmds))
              : 0.0;
          dlog(sfmt("[%10.3fms] CMD_GEN     req=%3d  ctx=%5d  cmds=%zu  "
                    "wall_ms=%.3f  per_cmd_ns=%.1f",
                    start_ms, req.id, ctx_now, ncmds, wall_ms, per_cmd_ns));
        }

        m_pending_pim_requests.insert(m_pending_pim_requests.end(),
                                      std::make_move_iterator(cmds.begin()),
                                      std::make_move_iterator(cmds.end()));
      }
    }

    // Log task start (before move — use local `task`, not m_active_task).
    m_translate_sample_count = 0;
    if (m_debug_log.is_open()) {
      std::string reqs_str;
      for (int idx : task.request_indices) {
        if (!reqs_str.empty()) reqs_str += ',';
        reqs_str += std::to_string(m_scheduler.requests()[idx].id);
      }
      if (task.phase == "decode" && task.route == m_cfg.pim_route_name) {
        dlog(sfmt("[%10.3fms] TASK_START  phase=decode   route=%-20s  reqs=[%s]"
                  "  ctx=%5d  bs=%d  gpu_ms=%6.2f  pim_ms=%6.2f  e2e_ms=%6.2f  cmds=%zu",
                  start_ms, task.route.c_str(), reqs_str.c_str(),
                  task.context_tokens, task.batch_size,
                  task.gpu_ms, task.pim_ms, task.e2e_ms,
                  m_pending_pim_requests.size()));
      } else {
        dlog(sfmt("[%10.3fms] TASK_START  phase=%-7s  route=%-20s  reqs=[%s]"
                  "  ctx=%5d  bs=%d  gpu_ms=%6.2f  pim_ms=%6.2f  e2e_ms=%6.2f",
                  start_ms, task.phase.c_str(), task.route.c_str(), reqs_str.c_str(),
                  task.context_tokens, task.batch_size,
                  task.gpu_ms, task.pim_ms, task.e2e_ms));
      }
    }

    m_active_task = std::move(task);
  }

  // Send pending PIM requests to the memory system one per tick (back-pressure).
  void pump_active_pim() {
    if (!m_active_task.has_value()
        || m_active_task->phase != "decode"
        || m_active_task->route != m_cfg.pim_route_name
        || m_memory_system == nullptr) {
      return;
    }

    while (m_next_pim_request < m_pending_pim_requests.size()) {
      Request req = m_pending_pim_requests[m_next_pim_request];

      // Log first 4 VA→PA translations per task so the address path is visible.
      const uint64_t orig_va = static_cast<uint64_t>(req.addr);

      // VA-to-PA translation (bit 63 indicates a KV virtual address).
      if (m_translation != nullptr) {
        auto t0 = wall_clock::now();
        const bool ok = m_translation->translate(req);
        m_t_translate += wall_clock::now() - t0;
        if (!ok) {
          throw std::runtime_error("ServingOnline runtime: request translation failed");
        }
      }

      if (m_debug_log.is_open() && m_translate_sample_count < 4) {
        const uint64_t pa = static_cast<uint64_t>(req.addr);
        const VaFields f  = decode_va(orig_va);
        dlog(sfmt("[%10.3fms] TRANSLATE  req=%3d  cmd=%4zu  "
                  "va=0x%016llx  [%s  req_id=%llu  page=%llu  off=%llu]  "
                  "pa=0x%016llx",
                  now_ms(), req.source_id,
                  m_next_pim_request,
                  (unsigned long long)orig_va,
                  f.region == 0 ? "Key" : "Val",
                  (unsigned long long)f.request_id,
                  (unsigned long long)f.page_index,
                  (unsigned long long)f.page_offset,
                  (unsigned long long)pa));
        m_translate_sample_count++;
        if (m_translate_sample_count == 4) {
          const size_t remaining = m_pending_pim_requests.size() - m_next_pim_request - 1;
          if (remaining > 0) {
            dlog(sfmt("[%10.3fms]   ... (%zu more cmds not logged)",
                      now_ms(), remaining));
          }
        }
      }

      bool sent;
      {
        auto t0 = wall_clock::now();
        sent = m_memory_system->send(req);
        m_t_send += wall_clock::now() - t0;
      }
      if (!sent) {
        // Memory system full; we'll retry next tick. Count this so PROGRESS
        // / status.txt can show how often the pump is stalled by Ramulator
        // backpressure vs running freely.
        m_send_backpressure_events_task++;
        m_send_backpressure_events_total++;
        break;
      }
      if (req.callback) {
        m_pim_outstanding++;
      }
      m_next_pim_request++;
    }
    m_issue_done = (m_next_pim_request == m_pending_pim_requests.size());
  }

  // Drain completed PIM callbacks.
  void process_callbacks() {
    while (m_completed_callback_events > 0 && m_pim_outstanding > 0) {
      m_completed_callback_events--;
      m_pim_outstanding--;
    }
  }

  // Check whether the active task is finished; if so, complete it.
  void complete_active_task_if_ready() {
    if (!m_active_task.has_value()) return;

    const double now = now_ms();

    if (m_active_task->phase == "prefill") {
      // Prefill: wall-clock latency gates completion.
      if (now < m_active_task->start_ms + m_active_task->e2e_ms) return;
    } else {
      // Decode: both GPU time and PIM commands must be done.
      const bool gpu_done = (now >= m_active_task->start_ms + m_active_task->gpu_ms);
      const bool pim_done = (m_active_task->route != m_cfg.pim_route_name)
                         || (m_issue_done && m_pim_outstanding == 0);
      if (!gpu_done || !pim_done) return;
    }

    if (m_debug_log.is_open()) {
      std::string reqs_str;
      for (int idx : m_active_task->request_indices) {
        if (!reqs_str.empty()) reqs_str += ',';
        reqs_str += std::to_string(m_scheduler.requests()[idx].id);
      }
      dlog(sfmt("[%10.3fms] TASK_DONE   phase=%-7s  route=%-20s  reqs=[%s]  duration_ms=%.3f",
                now, m_active_task->phase.c_str(), m_active_task->route.c_str(),
                reqs_str.c_str(), now - m_active_task->start_ms));
      // Phase 2: per-section wall-clock breakdown for this task.
      const double wall_total =
          std::chrono::duration<double>(wall_clock::now() - m_task_wall_start).count();
      const double t_cg = ns_to_s(m_t_cmd_gen);
      const double t_tr = ns_to_s(m_t_translate);
      const double t_sn = ns_to_s(m_t_send);
      const double t_sc = ns_to_s(m_t_sched);
      const double t_co = ns_to_s(m_t_complete);
      const double t_un = std::max(0.0, wall_total - t_cg - t_tr - t_sn - t_sc - t_co);
      dlog(sfmt("[%10.3fms]   TIMING    cmds=%zu  wall=%.3fs  "
                "t_cmd_gen=%.3fs  t_translate=%.3fs  t_send=%.3fs  "
                "t_sched=%.3fs  t_complete=%.3fs  t_unaccounted=%.3fs",
                now,
                m_pending_pim_requests.size(),
                wall_total, t_cg, t_tr, t_sn, t_sc, t_co, t_un));
    }

    // Miss-path: record drain_clk for this PIM-decode shape so future tasks
    // with the same (route, ctx_bucket, batch_size) hit the cache.
    if (m_cfg.pim_cache_enabled
        && m_active_task->phase == "decode"
        && m_active_task->route == m_cfg.pim_route_name
        && !m_active_task_was_cache_hit) {
      const PimShapeKey key{m_active_task->route,
                            ctx_bucket(m_active_task->context_tokens),
                            m_active_task->batch_size};
      const Clk_t drain_clk = (m_clk > m_clk_at_task_start)
                              ? (m_clk - m_clk_at_task_start) : 0;
      m_pim_shape_cache[key] = PimShapeStats{
          drain_clk, m_pending_pim_requests.size(), 0};
      if (m_debug_log.is_open()) {
        const double drain_ms = (m_cycles_per_ms > 0.0)
            ? (static_cast<double>(drain_clk) / m_cycles_per_ms) : 0.0;
        dlog(sfmt("[%10.3fms]   CACHE_FILL  route=%-20s  ctx_bucket=%4d  bs=%d  "
                  "drain_clk=%llu  drain_ms=%.3f  cmds=%zu",
                  now, m_active_task->route.c_str(),
                  key.ctx_bucket, key.batch_size,
                  (unsigned long long)drain_clk, drain_ms,
                  m_pending_pim_requests.size()));
      }
    }

    m_scheduler.complete_task(*m_active_task, now);
    m_active_task.reset();
    m_pending_pim_requests.clear();
    m_next_pim_request          = 0;
    m_issue_done                = true;
    m_pim_outstanding           = 0;
    m_completed_callback_events = 0;
  }

  // ── State ──────────────────────────────────────────────────────────────────

  Logger_t       m_logger;
  ServingOnlineConfig m_cfg;
  IMemorySystem* m_memory_system = nullptr;
  ITranslation*  m_translation   = nullptr;

  CostTable                       m_cost_table;
  std::vector<SourceRequest>      m_arrivals;
  std::shared_ptr<PimAllocator>   m_allocator;
  std::unique_ptr<PimCommandGen>  m_command_gen;
  Scheduler                       m_scheduler;

  size_t m_total_requests  = 0;
  double m_cycles_per_ms   = 0.0;
  Clk_t  m_clk             = 0;
  double m_gpu_free_ms     = 0.0;
  double m_pim_free_ms     = 0.0;

  std::optional<ActiveTask>  m_active_task;
  std::vector<Request>       m_pending_pim_requests;
  size_t m_next_pim_request          = 0;
  size_t m_pim_outstanding           = 0;
  size_t m_completed_callback_events = 0;
  bool   m_issue_done                = true;

  // Debug log (null stream when disabled).
  std::ofstream m_debug_log;
  std::string   m_status_file_path;   // <run_dir>/<policy>/status.txt; live heartbeat
  int m_translate_sample_count = 0;   // resets per task; cap TRANSLATE lines at 4

  // Live-instrumentation counters.
  //   *_task  — reset at every start_task; reflects the in-flight task only.
  //   *_total — never reset; lifetime totals for run summary.
  // Delta-tracking state is for PROGRESS rate computation between consecutive
  // ~1s emissions.
  size_t m_send_backpressure_events_task  = 0;
  size_t m_send_backpressure_events_total = 0;
  size_t m_last_progress_cmds_sent        = 0;
  Clk_t  m_last_progress_clk              = 0;

  // ── PIM shape cache ────────────────────────────────────────────────────────
  // Populated on cache miss in complete_active_task_if_ready(); consulted at
  // the top of start_task() for PIM decode tasks. Set m_active_task_was_cache_hit
  // so the miss-path knows whether to record drain_clk on completion.
  std::unordered_map<PimShapeKey, PimShapeStats, PimShapeKeyHash> m_pim_shape_cache;
  Clk_t  m_clk_at_task_start         = 0;
  bool   m_active_task_was_cache_hit = false;
  size_t m_pim_cache_total_hits      = 0;
  size_t m_pim_cache_total_misses    = 0;

  static int ctx_bucket(int context_tokens) {
    return (context_tokens + kCacheBucketTokens - 1) / kCacheBucketTokens;
  }

  // ── Wall-clock instrumentation ─────────────────────────────────────────────
  // Phase 1: periodic PROGRESS line every ~1 second of wall-clock during a
  // long-running task. Phase 2: per-section accumulators printed at TASK_DONE
  // so we can see whether time is going to cmd-gen / translate / send / sched
  // / complete-checks vs the unaccounted tail (which is the Ramulator DRAM
  // tick driven by the outer loop).
  using wall_clock = std::chrono::steady_clock;
  wall_clock::time_point m_wall_start         = wall_clock::now();
  wall_clock::time_point m_last_progress_wall = wall_clock::now();
  wall_clock::time_point m_task_wall_start    = wall_clock::now();
  std::chrono::nanoseconds m_t_cmd_gen{0};
  std::chrono::nanoseconds m_t_translate{0};
  std::chrono::nanoseconds m_t_send{0};
  std::chrono::nanoseconds m_t_sched{0};
  std::chrono::nanoseconds m_t_complete{0};

  static double ns_to_s(std::chrono::nanoseconds ns) {
    return std::chrono::duration<double>(ns).count();
  }

  void dlog(const std::string& line) {
    if (m_debug_log.is_open()) m_debug_log << line << '\n';
  }
};

}  // namespace Ramulator::ServingOnline

// ─── IFrontEnd wrapper ────────────────────────────────────────────────────────

namespace Ramulator {

class ServingOnlineFrontend : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(
      IFrontEnd,
      ServingOnlineFrontend,
      "ServingOnlineFrontend",
      "Clean pi0 PIM+GPU LLM serving with runtime LPDDR5 command generation.");

 public:
  void init() override {
    ServingOnline::ServingOnlineConfig cfg;

    // Input files
    cfg.csv_path        = param<std::string>("requests_csv").required();
    cfg.requests_out_csv = param<std::string>("requests_out_csv")
                              .default_val("cluster_outputs/serving_online_requests.csv");
    cfg.cost_gpu_csv    = param<std::string>("cost_gpu_csv").required();
    cfg.cost_pim_csv    = param<std::string>("cost_hybrid_csv").required();
    cfg.cost_gpu_model    = param<std::string>("cost_gpu_model").default_val("");
    cfg.cost_hybrid_model = param<std::string>("cost_hybrid_model").default_val("");

    // Route names
    cfg.gpu_route_name  = param<std::string>("gpu_route_name").default_val("gpu_only");
    cfg.pim_route_name  = param<std::string>("hybrid_route_name").default_val("pim+gpu");

    // Routing policy
    cfg.route_policy           = param<std::string>("route_policy").default_val("min_finish");
    cfg.energy_latency_guard_ms = param<float>("energy_latency_guard_ms").default_val(5.0f);

    // SLO thresholds
    cfg.slo_e2e_ms  = param<float>("slo_e2e_ms").default_val(-1.0f);
    cfg.slo_ttft_ms = param<float>("slo_ttft_ms").default_val(-1.0f);

    // Scheduling
    cfg.max_decode_batch_size          = param<int>("max_decode_batch_size").default_val(8);
    cfg.prompt_priority                = param<bool>("prompt_priority").default_val(true);
    cfg.max_consecutive_decode_batches = param<int>("max_consecutive_decode_batches").default_val(4);
    cfg.max_active_requests            = param<int>("max_active_requests").default_val(0);

    // Admission
    cfg.admission_max_wait_ms       = param<float>("admission_max_wait_ms").default_val(0.0f);
    cfg.admission_retry_interval_ms = param<float>("admission_retry_interval_ms").default_val(1.0f);
    cfg.kv_oom_policy               = param<std::string>("kv_oom_policy").default_val("hold");
    cfg.kv_oom_hold_limit_ms        = param<float>("kv_oom_hold_limit_ms").default_val(50.0f);
    cfg.kv_scheduler_policy         = param<std::string>("kv_scheduler_policy")
                                          .default_val("guaranteed_no_evict");

    // Fine-grained KV knobs (read only under kv_scheduler_policy=max_utilization).
    cfg.kv_decode_chunk_tokens         = param<int>("kv_decode_chunk_tokens").default_val(32);
    cfg.kv_evict_granularity           = param<std::string>("kv_evict_granularity")
                                            .default_val("tail");
    cfg.predictive_admission           = param<bool>("predictive_admission").default_val(false);
    cfg.predictive_pressure_threshold  = param<float>("predictive_pressure_threshold")
                                            .default_val(1.0f);
    cfg.predictive_hold_ms             = param<float>("predictive_hold_ms").default_val(50.0f);
    cfg.kv_tail_trim_max_per_request   = param<int>("kv_tail_trim_max_per_request").default_val(8);
    cfg.kv_tail_trim_multiplier        = param<int>("kv_tail_trim_multiplier").default_val(1);

    // Arrival
    cfg.arrival_time_scale = param<float>("arrival_time_scale").default_val(1.0f);
    cfg.arrival_limit      = param<int>("arrival_limit").default_val(-1);

    // Memory
    cfg.kv_pool_bytes   = param<uint64_t>("kv_pool_bytes").default_val(0);
    cfg.page_size_bytes = (param<Addr_t>("translation_pagesize_KB").default_val(4) << 10);
    cfg.num_channels    = param<int>("generator_channel_count").default_val(8);

    // Model architecture (Pi0 defaults; OpenVLA: 32/32/128)
    cfg.num_layers  = param<int>("generator_num_layers").default_val(18);
    cfg.num_heads   = param<int>("generator_num_heads").default_val(8);
    cfg.d_head      = param<int>("generator_dhead").default_val(256);
    cfg.dtype_bytes = param<int>("generator_dtype_bytes").default_val(2);

    // Vision-encoder prefix: pre-encoded image tokens added to every
    // request's context at load time. 0 = text-only model (Pi0). 256 is
    // typical for OpenVLA's SigLIP/DINO patch grid.
    cfg.vision_prefix_tokens = param<int>("vision_prefix_tokens").default_val(0);

    // PIM command stream complexity: "realistic" (default) or "simple".
    cfg.pim_command_mode = param<std::string>("pim_command_mode").default_val("realistic");

    // Per-shape PIM-decode cycle cache (Lever 1). True = skip Ramulator on
    // repeated (route, ctx_bucket, batch_size) shapes.
    cfg.pim_cache_enabled = param<bool>("pim_cache_enabled").default_val(true);

    // Cross-run cache persistence. Empty = disabled. When set, runtime
    // loads at init and rewrites at finalize. Mismatched header → rejected.
    cfg.pim_cache_file = param<std::string>("pim_cache_file").default_val("");

    // Debug log (empty string = disabled)
    cfg.debug_log_path = param<std::string>("debug_log_path").default_val("");

    // Validate route policy.
    if (cfg.route_policy != "min_finish" &&
        cfg.route_policy != "latency_guarded_energy") {
      throw ConfigurationError(
          "ServingOnlineFrontend: route_policy must be 'min_finish' or 'latency_guarded_energy'");
    }
    // Validate kv_oom_policy.
    if (cfg.kv_oom_policy != "hold" && cfg.kv_oom_policy != "fallback_gpu" &&
        cfg.kv_oom_policy != "hold_then_fallback") {
      throw ConfigurationError(
          "ServingOnlineFrontend: kv_oom_policy must be 'hold', 'fallback_gpu', or 'hold_then_fallback'");
    }
    // Validate pim_command_mode.
    if (cfg.pim_command_mode != "realistic" && cfg.pim_command_mode != "simple") {
      throw ConfigurationError(
          "ServingOnlineFrontend: pim_command_mode must be 'realistic' or 'simple'");
    }
    // Validate kv_scheduler_policy.
    if (cfg.kv_scheduler_policy != "guaranteed_no_evict" &&
        cfg.kv_scheduler_policy != "max_utilization") {
      throw ConfigurationError(
          "ServingOnlineFrontend: kv_scheduler_policy must be 'guaranteed_no_evict' or 'max_utilization'");
    }
    // Validate fine-grained KV knobs.
    if (cfg.kv_decode_chunk_tokens < 1) {
      throw ConfigurationError(
          "ServingOnlineFrontend: kv_decode_chunk_tokens must be >= 1");
    }
    if (cfg.kv_evict_granularity != "tail" && cfg.kv_evict_granularity != "full") {
      throw ConfigurationError(
          "ServingOnlineFrontend: kv_evict_granularity must be 'tail' or 'full'");
    }
    if (!(cfg.predictive_pressure_threshold > 0.0 &&
          cfg.predictive_pressure_threshold <= 4.0)) {
      throw ConfigurationError(
          "ServingOnlineFrontend: predictive_pressure_threshold must be in (0, 4]");
    }
    if (cfg.kv_tail_trim_max_per_request < 0) {
      throw ConfigurationError(
          "ServingOnlineFrontend: kv_tail_trim_max_per_request must be >= 0 "
          "(0 disables the livelock guard)");
    }
    if (cfg.kv_scheduler_policy == "max_utilization" &&
        cfg.kv_oom_policy != "hold") {
      // Not an error — surface as a warning so the user knows the kv_oom knob
      // is inert under max_utilization. The scheduler always holds FCFS when
      // even prefill bytes don't fit (per vLLM §4.5).
      std::fprintf(
          stderr,
          "[ServingOnlineFrontend] kv_scheduler_policy=max_utilization "
          "ignores kv_oom_policy='%s'. Admission holds FCFS; in-flight KV "
          "growth triggers LIFO preemption.\n",
          cfg.kv_oom_policy.c_str());
    }

    m_clock_ratio = param<uint>("clock_ratio").required();

    m_logger  = Logging::create_logger("ServingOnlineFrontend");
    m_runtime = std::make_unique<ServingOnline::Runtime>(m_logger);
    m_runtime->initialize(cfg);

    // Create and wire the VA-to-PA translation child.
    m_translation = create_child_ifce<ITranslation>();
    auto* control = dynamic_cast<IServingOnlineTranslationControl*>(m_translation);
    if (control == nullptr) {
      throw ConfigurationError(
          "ServingOnlineFrontend: Translation child must implement IServingOnlineTranslationControl "
          "(use Translation: impl: ServingOnlineTranslation)");
    }
    control->attach_pim_allocator(m_runtime->allocator_shared());
    m_runtime->attach_translation(m_translation);
  }

  void connect_memory_system(IMemorySystem* memory_system) override {
    IFrontEnd::connect_memory_system(memory_system);
    if (!m_runtime) {
      throw ConfigurationError("ServingOnlineFrontend: runtime not initialized");
    }
    m_runtime->connect_memory_system(memory_system);
    m_clk = m_runtime->clk();
  }

  void tick() override {
    if (!m_runtime) return;
    m_runtime->tick();
    m_clk = m_runtime->clk();
  }

  bool is_finished() override {
    return !m_runtime || m_runtime->is_finished();
  }

  void finalize() override {
    if (m_runtime) {
      m_runtime->finalize();
      m_clk = m_runtime->clk();
    }
    IFrontEnd::finalize();
  }

  void print_stats(YAML::Emitter& emitter) override {
    if (!m_runtime) return;
    m_runtime->print_stats(emitter, get_ifce_name(), get_name());
  }

 private:
  Logger_t m_logger;
  ITranslation* m_translation = nullptr;
  std::unique_ptr<ServingOnline::Runtime> m_runtime;
};

}  // namespace Ramulator
