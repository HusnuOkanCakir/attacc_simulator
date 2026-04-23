/**
 * serving_online/cost_table.h
 *
 * Cost model for serving_online: loads a CSV of pre-computed cost points (exported
 * from HistGradientBoostingRegressor models trained in Python) and answers
 * nearest-neighbor lookup queries at runtime.
 *
 * CSV format (columns in order, comma-separated, with header row):
 *   route, lin, lout, bs,
 *   prefill_gpu_ms, prefill_pim_ms, prefill_e2e_ms, prefill_energy_nj,
 *   decode_gpu_ms,  decode_pim_ms,  decode_e2e_ms,  decode_energy_nj
 *
 * Nearest-neighbor: minimize |lin - query.lin| + |lout - query.lout|
 *                             + |bs - query.bs| (L1 distance in feature space).
 *
 * A route with "pim" in its name is treated as uses_pim = true.
 */

#ifndef RAMULATOR_FRONTEND_IMPL_SERVING_ONLINE_COST_TABLE_H
#define RAMULATOR_FRONTEND_IMPL_SERVING_ONLINE_COST_TABLE_H

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "frontend/impl/serving_online/types.h"

namespace Ramulator::ServingOnline {

// One row in the cost CSV.
struct CostPoint {
  std::string route;
  int    lin, lout, bs;
  double prefill_gpu_ms,  prefill_pim_ms,  prefill_e2e_ms,  prefill_energy_nj;
  double decode_gpu_ms,   decode_pim_ms,   decode_e2e_ms,   decode_energy_nj;
};

class CostTable {
 public:
  /**
   * Load cost points from a CSV file.
   * Can be called multiple times; each call appends to existing data.
   * @throws std::runtime_error on file open or format errors.
   */
  void load(const std::string& path);

  /**
   * Find the nearest cost point for (route, context_tokens, generated_tokens, bs)
   * and return a CostEstimate.
   *
   * @return nullopt if no points exist for the requested route.
   */
  std::optional<CostEstimate> estimate(const std::string& route,
                                       int context_tokens,
                                       int generated_tokens,
                                       int bs = 1) const;

  // Returns the list of routes that have at least one cost point.
  std::vector<std::string> routes() const;

  // Returns the number of loaded cost points across all routes.
  size_t total_points() const;

 private:
  // m_points[route] = vector of cost points (all points for that route).
  std::unordered_map<std::string, std::vector<CostPoint>> m_points;
};

}  // namespace Ramulator::ServingOnline

#endif  // RAMULATOR_FRONTEND_IMPL_SERVING_ONLINE_COST_TABLE_H
