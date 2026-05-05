#include "ml_cost_model.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace Ramulator {

// ── Binary file constants (must match export_cost_model_trees.py) ─────────────
static constexpr uint32_t MAGIC   = 0xC05774EEu;
static constexpr uint32_t VERSION = 1u;

// On-disk node layout: B(1) i(4) d(8) i(4) i(4) d(8) = 29 bytes, no padding.
#pragma pack(push, 1)
struct DiskNode {
  uint8_t  is_leaf;
  int32_t  feature_idx;
  double   threshold;
  int32_t  left;
  int32_t  right;
  double   value;
};
#pragma pack(pop)
static_assert(sizeof(DiskNode) == 29, "DiskNode size mismatch");

// ── Helpers ───────────────────────────────────────────────────────────────────

static void read_exact(std::ifstream& f, void* dst, size_t n,
                       const std::string& ctx) {
  if (!f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n)))
    throw std::runtime_error("MlCostModel: read error at " + ctx);
}

static std::string read_string(std::ifstream& f) {
  uint32_t len = 0;
  read_exact(f, &len, 4, "string length");
  std::string s(len, '\0');
  if (len > 0)
    read_exact(f, &s[0], len, "string data");
  return s;
}

// ── MlTree::predict ───────────────────────────────────────────────────────────

double MlTree::predict(double lin, double lout, double bs) const noexcept {
  const double feats[3] = {lin, lout, bs};
  int n = 0;
  while (!nodes[n].is_leaf) {
    n = (feats[nodes[n].feature_idx] <= nodes[n].threshold)
        ? nodes[n].left
        : nodes[n].right;
  }
  return nodes[n].value;
}

// ── MlEnsemble::predict ───────────────────────────────────────────────────────

double MlEnsemble::predict(double lin, double lout, double bs) const noexcept {
  double sum = base;
  for (const auto& t : trees)
    sum += t.predict(lin, lout, bs);
  return sum;
}

// ── MlCostModel::load ─────────────────────────────────────────────────────────

MlCostModel MlCostModel::load(const std::string& bin_path) {
  std::ifstream f(bin_path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("MlCostModel: cannot open " + bin_path);

  uint32_t magic = 0, version = 0, n_targets = 0;
  read_exact(f, &magic,     4, "magic");
  read_exact(f, &version,   4, "version");
  read_exact(f, &n_targets, 4, "n_targets");

  if (magic != MAGIC)
    throw std::runtime_error("MlCostModel: bad magic in " + bin_path);
  if (version != VERSION)
    throw std::runtime_error("MlCostModel: unsupported version in " + bin_path);

  // Infer route name from filename stem (e.g. "lpddr5_pim_bank_trees.bin" → not needed;
  // route is stored in the path the caller passes, but we need a key.
  // Use the filename stem stripped of "_trees".
  std::string stem = bin_path;
  auto slash = stem.rfind('/');
  if (slash != std::string::npos) stem = stem.substr(slash + 1);
  auto dot = stem.rfind('.');
  if (dot != std::string::npos) stem = stem.substr(0, dot);
  auto suffix = stem.rfind("_trees");
  if (suffix != std::string::npos) stem = stem.substr(0, suffix);
  const std::string route = stem;

  MlCostModel model;
  auto& ensembles = model.m_routes[route];

  for (uint32_t ti = 0; ti < n_targets; ++ti) {
    std::string tname = read_string(f);

    double   base    = 0.0;
    uint32_t n_trees = 0;
    read_exact(f, &base,    8, "base");
    read_exact(f, &n_trees, 4, "n_trees");

    MlEnsemble ens;
    ens.base = base;
    ens.trees.resize(n_trees);

    for (uint32_t tri = 0; tri < n_trees; ++tri) {
      uint32_t n_nodes = 0;
      read_exact(f, &n_nodes, 4, "n_nodes");

      MlTree& tree = ens.trees[tri];
      tree.nodes.resize(n_nodes);

      for (uint32_t ni = 0; ni < n_nodes; ++ni) {
        DiskNode dn{};
        read_exact(f, &dn, sizeof(DiskNode), "node");
        MlNode& mn = tree.nodes[ni];
        mn.is_leaf     = dn.is_leaf;
        mn.feature_idx = dn.feature_idx;
        mn.threshold   = dn.threshold;
        mn.left        = dn.left;
        mn.right       = dn.right;
        mn.value       = dn.value;
      }
    }
    ensembles[tname] = std::move(ens);
  }
  return model;
}

// ── MlCostModel::estimate ────────────────────────────────────────────────────

std::optional<ServingOnline::CostEstimate> MlCostModel::estimate(const std::string& route,
                                                                  int lin, int lout,
                                                                  int bs) const {
  auto rit = m_routes.find(route);
  if (rit == m_routes.end()) return std::nullopt;
  const auto& ens = rit->second;

  auto get = [&](const std::string& name) -> double {
    auto it = ens.find(name);
    if (it == ens.end()) return 0.0;
    double v = it->second.predict(static_cast<double>(lin),
                                  static_cast<double>(lout),
                                  static_cast<double>(bs));
    return v >= 0.0 ? v : 0.0;  // clamp negatives
  };

  ServingOnline::CostEstimate est;
  est.route            = route;
  est.uses_pim         = (route.find("pim") != std::string::npos);
  est.prefill_gpu_ms   = get("prefill_gpu_ms");
  est.prefill_pim_ms   = get("prefill_pim_ms");
  est.prefill_e2e_ms   = std::max({get("prefill_e2e_ms"),
                                   est.prefill_gpu_ms, est.prefill_pim_ms});
  est.prefill_energy_nj = get("prefill_energy_nj");
  est.decode_gpu_ms    = get("decode_gpu_ms");
  est.decode_pim_ms    = get("decode_pim_ms");
  est.decode_e2e_ms    = std::max({get("decode_e2e_ms"),
                                   est.decode_gpu_ms, est.decode_pim_ms});
  est.decode_energy_nj = get("decode_energy_nj");
  est.decode_tokens    = std::max(0, lout - 1);
  return est;
}

// ── MlCostModel::merge ───────────────────────────────────────────────────────

void MlCostModel::merge(MlCostModel other) {
  for (auto& [route, targets] : other.m_routes)
    m_routes[route] = std::move(targets);
}

} // namespace Ramulator
