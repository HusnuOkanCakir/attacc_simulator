#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include "cost_table.h"   // CostEstimate

namespace Ramulator {

// ── flat node in a HistGBR decision tree ─────────────────────────────────────
struct MlNode {
  uint8_t  is_leaf;
  int32_t  feature_idx;  // 0=Lin 1=Lout 2=bs; ignored when is_leaf
  double   threshold;
  int32_t  left;         // index into owning tree's node array
  int32_t  right;
  double   value;        // used only at leaf nodes
};

// ── single decision tree ─────────────────────────────────────────────────────
struct MlTree {
  std::vector<MlNode> nodes;
  double predict(double lin, double lout, double bs) const noexcept;
};

// ── ensemble of trees for one (route, target) pair ───────────────────────────
struct MlEnsemble {
  double              base;   // _baseline_prediction from sklearn
  std::vector<MlTree> trees;
  double predict(double lin, double lout, double bs) const noexcept;
};

// ── full ML cost model: route → target → ensemble ────────────────────────────
class MlCostModel {
public:
  // Load from a binary .bin file produced by tools/export_cost_model_trees.py.
  // Throws std::runtime_error on file or format errors.
  static MlCostModel load(const std::string& bin_path);

  // Predict all 8 cost targets and return a CostEstimate.
  // Returns nullopt if the route is unknown.
  std::optional<ServingOnline::CostEstimate> estimate(const std::string& route,
                                                      int lin, int lout, int bs) const;

  // Merge all routes from another model into this one (used by CostTable::load_ml).
  void merge(MlCostModel other);

private:
  // route_name → (target_name → ensemble)
  std::unordered_map<std::string,
    std::unordered_map<std::string, MlEnsemble>> m_routes;
};

} // namespace Ramulator
