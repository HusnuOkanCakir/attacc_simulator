/**
 * serving_online/cost_table.cpp
 *
 * Implementation of CostTable: CSV loading and nearest-neighbor lookup.
 *
 * Two CSV formats are accepted:
 *
 * 1. Explicit serving format (preferred):
 *    route,lin,lout,bs,prefill_gpu_ms,prefill_pim_ms,prefill_e2e_ms,prefill_energy_nj,
 *    decode_gpu_ms,decode_pim_ms,decode_e2e_ms,decode_energy_nj
 *
 * 2. Perf-model export format (from the HistGBR pipeline):
 *    Lin,Lout,bs,...,s_time,...,g_time (ms),...,g_matmul,...,g_softmax,...
 *    Route name is inferred from the filename stem.
 *    For gpu_only:   decode_pim_ms = 0, decode_gpu_ms = g_time
 *    For pim routes: decode_gpu_ms = FC+etc+comm components, decode_pim_ms = matmul+softmax
 */

#include "frontend/impl/serving_online/cost_table.h"
#include "frontend/impl/serving_online/ml_cost_model.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace Ramulator::ServingOnline {

namespace fs = std::filesystem;

namespace {

std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> parts;
  std::stringstream ss(s);
  std::string part;
  while (std::getline(ss, part, delim)) parts.push_back(part);
  return parts;
}

std::string trim(const std::string& s) {
  const size_t start = s.find_first_not_of(" \t\r\n\"");
  if (start == std::string::npos) return "";
  const size_t end = s.find_last_not_of(" \t\r\n\"");
  return s.substr(start, end - start + 1);
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

}  // namespace

CostTable::CostTable()  = default;
CostTable::~CostTable() = default;

// ─── load() ─────────────────────────────────────────────────────────────────

void CostTable::load(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error("cost_table: cannot open file: " + path);
  }

  std::string header_line;
  if (!std::getline(f, header_line)) {
    throw std::runtime_error("cost_table: empty file: " + path);
  }
  auto header = split(header_line, ',');
  for (auto& c : header) c = trim(c);

  // Case-insensitive column lookup; returns -1 if not found.
  auto find_col = [&](const std::string& name) -> int {
    const std::string key = lower(name);
    for (int i = 0; i < static_cast<int>(header.size()); ++i) {
      if (lower(header[i]) == key) return i;
    }
    return -1;
  };

  // Explicit serving-format columns.
  const int c_route  = find_col("route");
  const int c_lin    = find_col("lin");
  const int c_lout   = find_col("lout");
  const int c_bs     = find_col("bs");
  const int c_pfgpu  = find_col("prefill_gpu_ms");
  const int c_pfpim  = find_col("prefill_pim_ms");
  const int c_pfe2e  = find_col("prefill_e2e_ms");
  const int c_pfenj  = find_col("prefill_energy_nj");
  const int c_dcgpu  = find_col("decode_gpu_ms");
  const int c_dcpim  = find_col("decode_pim_ms");
  const int c_dce2e  = find_col("decode_e2e_ms");
  const int c_dcenj  = find_col("decode_energy_nj");

  // Perf-model export columns.
  const int c_s_time    = find_col("s_time");
  const int c_s_energy  = find_col("s_energy (nj)");
  const int c_g_time    = find_col("g_time (ms)");
  const int c_g_energy  = find_col("g_energy (nj)");
  const int c_g_matmul  = find_col("g_matmul");
  const int c_g_softmax = find_col("g_softmax");
  const int c_g_etc     = find_col("g_etc");
  const int c_g2g       = find_col("g2g_comm");
  const int c_c2g       = find_col("c2g_comm");
  const int c_g_qkv     = find_col("g_qkv_time");
  const int c_g_prj     = find_col("g_prj_time");
  const int c_g_ff      = find_col("g_ff_time");
  const int c_g_fc      = find_col("g_fc");

  const bool explicit_format =
      c_route >= 0 && c_lin >= 0 && c_lout >= 0 && c_bs >= 0
      && c_pfgpu >= 0 && c_pfpim >= 0 && c_pfe2e >= 0 && c_pfenj >= 0
      && c_dcgpu >= 0 && c_dcpim >= 0 && c_dce2e >= 0 && c_dcenj >= 0;

  const bool perf_format =
      c_lin >= 0 && c_lout >= 0 && c_bs >= 0
      && c_s_time >= 0 && c_g_time >= 0
      && c_g_matmul >= 0 && c_g_softmax >= 0 && c_g_etc >= 0
      && c_g2g >= 0 && c_c2g >= 0
      && ((c_g_qkv >= 0 && c_g_prj >= 0 && c_g_ff >= 0) || c_g_fc >= 0);

  if (!explicit_format && !perf_format) {
    throw std::runtime_error(
        "cost_table: unrecognized CSV schema in: " + path
        + " (need explicit serving columns or perf-model export columns)");
  }

  // Route name for perf-model files is inferred from the filename stem.
  const std::string inferred_route = fs::path(path).stem().string();

  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    auto cols = split(line, ',');

    auto field = [&](int idx) -> std::string {
      if (idx < 0 || idx >= static_cast<int>(cols.size())) return "";
      return trim(cols[idx]);
    };

    CostPoint p;
    try {
      p.lin  = std::stoi(field(c_lin));
      p.lout = std::stoi(field(c_lout));
      p.bs   = std::stoi(field(c_bs));

      if (explicit_format) {
        p.route            = field(c_route);
        p.prefill_gpu_ms   = std::stod(field(c_pfgpu));
        p.prefill_pim_ms   = std::stod(field(c_pfpim));
        p.prefill_e2e_ms   = std::stod(field(c_pfe2e));
        p.prefill_energy_nj = std::stod(field(c_pfenj));
        p.decode_gpu_ms    = std::stod(field(c_dcgpu));
        p.decode_pim_ms    = std::stod(field(c_dcpim));
        p.decode_e2e_ms    = std::stod(field(c_dce2e));
        p.decode_energy_nj = std::stod(field(c_dcenj));
      } else {
        // Perf-model export: map columns to serving fields.
        p.route            = inferred_route;
        p.prefill_e2e_ms   = std::stod(field(c_s_time));
        p.prefill_gpu_ms   = p.prefill_e2e_ms;
        p.prefill_pim_ms   = 0.0;
        p.prefill_energy_nj = (!field(c_s_energy).empty())
                                ? std::stod(field(c_s_energy)) : 0.0;
        p.decode_e2e_ms    = std::stod(field(c_g_time));
        p.decode_energy_nj = (!field(c_g_energy).empty())
                                ? std::stod(field(c_g_energy)) : 0.0;
        if (p.route == "gpu_only") {
          p.decode_gpu_ms = p.decode_e2e_ms;
          p.decode_pim_ms = 0.0;
        } else {
          // PIM route: GPU component = FC layers + misc; PIM = matmul + softmax.
          const double qkv_ms = c_g_qkv >= 0 ? std::stod(field(c_g_qkv))
                              : (c_g_fc  >= 0 ? std::stod(field(c_g_fc)) : 0.0);
          const double prj_ms = c_g_prj >= 0 ? std::stod(field(c_g_prj)) : 0.0;
          const double ff_ms  = c_g_ff  >= 0 ? std::stod(field(c_g_ff))  : 0.0;
          const double fc_ms  = c_g_fc  >= 0 ? std::stod(field(c_g_fc))  : 0.0;
          p.decode_gpu_ms =
              qkv_ms + prj_ms + ff_ms + std::stod(field(c_g_etc)) + std::stod(field(c_g2g));
          if (p.decode_gpu_ms <= 0.0 && fc_ms > 0.0) {
            p.decode_gpu_ms = fc_ms + std::stod(field(c_g_etc)) + std::stod(field(c_g2g));
          }
          p.decode_pim_ms = std::stod(field(c_g_matmul)) + std::stod(field(c_g_softmax));
        }
      }
    } catch (...) {
      continue;  // skip malformed rows silently
    }

    if (!p.route.empty() && p.lin > 0 && p.lout > 0 && p.bs > 0) {
      m_points[p.route].push_back(std::move(p));
    }
  }
}

// ─── load_ml() ───────────────────────────────────────────────────────────────

void CostTable::load_ml(const std::string& bin_path) {
  if (!m_ml) {
    m_ml = std::make_unique<Ramulator::MlCostModel>(
        Ramulator::MlCostModel::load(bin_path));
  } else {
    m_ml->merge(Ramulator::MlCostModel::load(bin_path));
  }
}

// ─── estimate() ─────────────────────────────────────────────────────────────

std::optional<CostEstimate> CostTable::estimate(const std::string& route,
                                                 int context_tokens,
                                                 int generated_tokens,
                                                 int bs) const {
  // ML model takes priority over nearest-neighbor when loaded.
  if (m_ml) {
    return m_ml->estimate(route, context_tokens, generated_tokens, bs);
  }

  auto it = m_points.find(route);
  if (it == m_points.end() || it->second.empty()) {
    return std::nullopt;
  }

  const auto& pts = it->second;

  // Nearest neighbor: minimize L1 distance in (lin, lout, bs) feature space.
  const CostPoint* best = &pts[0];
  int best_dist = std::abs(pts[0].lin  - context_tokens)
                + std::abs(pts[0].lout - generated_tokens)
                + std::abs(pts[0].bs   - bs);

  for (const auto& p : pts) {
    int d = std::abs(p.lin  - context_tokens)
          + std::abs(p.lout - generated_tokens)
          + std::abs(p.bs   - bs);
    if (d < best_dist) {
      best_dist = d;
      best      = &p;
    }
  }

  // Assemble a CostEstimate from the nearest point.
  // Clamp all times to >= 0 to defend against any CSV issues.
  auto nn0 = [](double v) { return v >= 0.0 ? v : 0.0; };

  CostEstimate est;
  est.route            = route;
  est.uses_pim         = (route.find("pim") != std::string::npos);
  est.prefill_gpu_ms   = nn0(best->prefill_gpu_ms);
  est.prefill_pim_ms   = nn0(best->prefill_pim_ms);
  // e2e must be >= max of its components
  est.prefill_e2e_ms   = std::max({nn0(best->prefill_e2e_ms),
                                   est.prefill_gpu_ms, est.prefill_pim_ms});
  est.prefill_energy_nj = nn0(best->prefill_energy_nj);
  est.decode_gpu_ms    = nn0(best->decode_gpu_ms);
  est.decode_pim_ms    = nn0(best->decode_pim_ms);
  est.decode_e2e_ms    = std::max({nn0(best->decode_e2e_ms),
                                   est.decode_gpu_ms, est.decode_pim_ms});
  est.decode_energy_nj = nn0(best->decode_energy_nj);
  est.decode_tokens    = std::max(0, generated_tokens - 1);
  return est;
}

// ─── routes() / total_points() ──────────────────────────────────────────────

std::vector<std::string> CostTable::routes() const {
  std::vector<std::string> r;
  for (const auto& kv : m_points) r.push_back(kv.first);
  return r;
}

size_t CostTable::total_points() const {
  size_t n = 0;
  for (const auto& kv : m_points) n += kv.second.size();
  return n;
}

}  // namespace Ramulator::ServingOnline
