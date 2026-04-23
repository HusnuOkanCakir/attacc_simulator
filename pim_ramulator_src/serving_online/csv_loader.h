/**
 * serving_online/csv_loader.h
 *
 * Loads an Azure LLM inference trace CSV into a sorted vector of SourceRequests.
 *
 * Required CSV columns:
 *   TIMESTAMP       — ISO datetime string (e.g. "2023-01-01 12:00:00.123")
 *   ContextTokens   — number of prompt tokens (integer)
 *   GeneratedTokens — number of generated tokens (integer)
 *
 * The first row's timestamp becomes t = 0 ms. All subsequent timestamps are
 * converted to elapsed milliseconds. Rows with ContextTokens < 1 or
 * GeneratedTokens < 1 are silently dropped.
 */

#ifndef RAMULATOR_FRONTEND_IMPL_SERVING_ONLINE_CSV_LOADER_H
#define RAMULATOR_FRONTEND_IMPL_SERVING_ONLINE_CSV_LOADER_H

#include <string>
#include <vector>

#include "frontend/impl/serving_online/types.h"

namespace Ramulator::ServingOnline {

/**
 * Load an Azure LLM inference CSV.
 *
 * @param path           Path to the CSV file.
 * @param limit          Maximum number of rows to load (-1 = all).
 * @param arrival_scale  Multiply all arrival timestamps by this factor.
 *                       Values < 1.0 compress time (higher offered load).
 * @return               Sorted vector of SourceRequests (by arrival_ms).
 * @throws               std::runtime_error on file open or format errors.
 */
std::vector<SourceRequest> load_azure_csv(const std::string& path,
                                          int    limit         = -1,
                                          double arrival_scale = 1.0);

}  // namespace Ramulator::ServingOnline

#endif  // RAMULATOR_FRONTEND_IMPL_SERVING_ONLINE_CSV_LOADER_H
