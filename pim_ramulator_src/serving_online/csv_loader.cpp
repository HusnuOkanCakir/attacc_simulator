/**
 * serving_online/csv_loader.cpp
 *
 * Parses an Azure LLM inference CSV and returns SourceRequests sorted by
 * arrival time. See csv_loader.h for the expected column format.
 *
 * Timestamp parsing:
 *   Format: "YYYY-MM-DD HH:MM:SS" with an optional fractional-seconds suffix
 *   (.SSS or .SSSSSS). We compute elapsed milliseconds from the first row.
 *
 * No heavy dependencies: just the C++ standard library.
 */

#include "frontend/impl/serving_online/csv_loader.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Ramulator::ServingOnline {

// ─── Internal helpers ───────────────────────────────────────────────────────

namespace {

// Split a string by a single-character delimiter.
std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> parts;
  std::stringstream ss(s);
  std::string part;
  while (std::getline(ss, part, delim)) {
    parts.push_back(part);
  }
  return parts;
}

// Trim leading/trailing whitespace and quotes from a string.
std::string trim(const std::string& s) {
  size_t start = s.find_first_not_of(" \t\r\n\"");
  if (start == std::string::npos) return "";
  size_t end = s.find_last_not_of(" \t\r\n\"");
  return s.substr(start, end - start + 1);
}

/**
 * Parse an ISO datetime string to milliseconds since the Unix epoch.
 *
 * Accepted formats:
 *   "2023-01-01 12:34:56"
 *   "2023-01-01 12:34:56.123"
 *   "2023-01-01 12:34:56.123456"
 *   "2023-01-01T12:34:56"   (T separator also accepted)
 *
 * Returns milliseconds as a double. Throws on parse failure.
 */
double parse_timestamp_ms(const std::string& ts) {
  // Replace T separator with space if present.
  std::string s = ts;
  for (char& c : s) {
    if (c == 'T') c = ' ';
  }

  // Separate date-time from fractional seconds.
  double frac_ms = 0.0;
  size_t dot = s.find('.');
  if (dot != std::string::npos) {
    std::string frac_str = s.substr(dot + 1);
    // Pad or truncate to exactly 3 digits (milliseconds).
    while (frac_str.size() < 3) frac_str += '0';
    if (frac_str.size() > 3) frac_str = frac_str.substr(0, 3);
    frac_ms = std::stod(frac_str);
    s = s.substr(0, dot);
  }

  // Parse "YYYY-MM-DD HH:MM:SS".
  int year, month, day, hour, minute, second;
  char dash1, dash2, space, colon1, colon2;
  std::istringstream iss(s);
  if (!(iss >> year >> dash1 >> month >> dash2 >> day
             >> space >> hour >> colon1 >> minute >> colon2 >> second)) {
    throw std::runtime_error("csv_loader: cannot parse timestamp: " + ts);
  }

  // Convert to seconds since epoch using mktime (local time → UTC skew ignored;
  // only relative differences matter for this simulation).
  std::tm t{};
  t.tm_year = year - 1900;
  t.tm_mon  = month - 1;
  t.tm_mday = day;
  t.tm_hour = hour;
  t.tm_min  = minute;
  t.tm_sec  = second;
  t.tm_isdst = -1;

  time_t epoch_s = mktime(&t);
  if (epoch_s == -1) {
    throw std::runtime_error("csv_loader: mktime failed for timestamp: " + ts);
  }

  return static_cast<double>(epoch_s) * 1000.0 + frac_ms;
}

}  // namespace

// ─── Public API ─────────────────────────────────────────────────────────────

std::vector<SourceRequest> load_azure_csv(const std::string& path,
                                          int    limit,
                                          double arrival_scale) {
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error("csv_loader: cannot open file: " + path);
  }

  // --- Parse header ---
  std::string header_line;
  if (!std::getline(f, header_line)) {
    throw std::runtime_error("csv_loader: empty file: " + path);
  }
  auto header_cols = split(header_line, ',');
  for (auto& c : header_cols) c = trim(c);

  // Find required column indices.
  int idx_ts    = -1;
  int idx_ctx   = -1;
  int idx_gen   = -1;
  for (int i = 0; i < static_cast<int>(header_cols.size()); ++i) {
    const auto& col = header_cols[i];
    if (col == "TIMESTAMP")      idx_ts  = i;
    if (col == "ContextTokens")  idx_ctx = i;
    if (col == "GeneratedTokens") idx_gen = i;
  }
  if (idx_ts < 0 || idx_ctx < 0 || idx_gen < 0) {
    throw std::runtime_error(
        "csv_loader: missing required columns (TIMESTAMP, ContextTokens, GeneratedTokens) in: "
        + path);
  }

  // --- Read data rows ---
  struct Row {
    double ts_ms;
    int    context_tokens;
    int    generated_tokens;
  };
  std::vector<Row> rows;

  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    auto cols = split(line, ',');
    if (static_cast<int>(cols.size()) <= std::max({idx_ts, idx_ctx, idx_gen})) continue;

    // Parse and validate tokens.
    int ctx, gen;
    try {
      ctx = std::stoi(trim(cols[idx_ctx]));
      gen = std::stoi(trim(cols[idx_gen]));
    } catch (...) {
      continue;  // skip malformed rows silently
    }
    if (ctx < 1 || gen < 1) continue;

    double ts_ms;
    try {
      ts_ms = parse_timestamp_ms(trim(cols[idx_ts]));
    } catch (...) {
      continue;
    }

    rows.push_back({ts_ms, ctx, gen});
  }

  if (rows.empty()) {
    return {};
  }

  // Sort by timestamp (CSV is usually sorted, but we sort defensively).
  std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
    return a.ts_ms < b.ts_ms;
  });

  // Apply limit.
  if (limit > 0 && static_cast<int>(rows.size()) > limit) {
    rows.resize(limit);
  }

  // Normalize: subtract first timestamp so t=0 is the start.
  double t0 = rows[0].ts_ms;

  std::vector<SourceRequest> out;
  out.reserve(rows.size());
  for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
    double arrival = (rows[i].ts_ms - t0) * arrival_scale;
    out.push_back({i, arrival, rows[i].context_tokens, rows[i].generated_tokens});
  }
  return out;
}

}  // namespace Ramulator::ServingOnline
