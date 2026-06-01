#ifndef RAMULATOR_FRONTEND_IMPL_SERVING_INPUTS_INTERNAL_H
#define RAMULATOR_FRONTEND_IMPL_SERVING_INPUTS_INTERNAL_H

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "frontend/impl/serving/types.h"

namespace Ramulator::Serving::detail {

inline std::string trim(const std::string& s) {
  size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) {
    ++b;
  }
  if (b == s.size()) {
    return "";
  }
  size_t e = s.size() - 1;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e]))) {
    --e;
  }
  return s.substr(b, e - b + 1);
}

inline std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  bool in_quotes = false;
  for (char ch : line) {
    if (ch == '"') {
      in_quotes = !in_quotes;
      continue;
    }
    if (ch == ',' && !in_quotes) {
      out.push_back(trim(cur));
      cur.clear();
    } else {
      cur.push_back(ch);
    }
  }
  out.push_back(trim(cur));
  return out;
}

inline std::string lower(std::string v) {
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return v;
}

inline int to_int(const std::string& s, int default_v = 0) {
  try {
    return std::stoi(trim(s));
  } catch (...) {
    return default_v;
  }
}

inline double to_double(const std::string& s, double default_v = 0.0) {
  try {
    return std::stod(trim(s));
  } catch (...) {
    return default_v;
  }
}

inline bool parse_timestamp_ms(const std::string& s, double& out_ms) {
  std::string v = trim(s);
  if (v.empty()) {
    return false;
  }

  {
    char* end = nullptr;
    const double as_num = std::strtod(v.c_str(), &end);
    if (end != nullptr && *end == '\0') {
      out_ms = as_num;
      return true;
    }
  }

  std::string base = v;
  std::string frac;
  const size_t dot = v.find('.');
  if (dot != std::string::npos) {
    base = v.substr(0, dot);
    frac = v.substr(dot + 1);
  }

  std::tm tm = {};
  {
    std::istringstream ss(base);
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    if (ss.fail()) {
      std::istringstream ss2(base);
      ss2 >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
      if (ss2.fail()) {
        return false;
      }
    }
  }

  const std::time_t tt = std::mktime(&tm);
  if (tt < 0) {
    return false;
  }

  double frac_ms = 0.0;
  if (!frac.empty()) {
    std::string digits;
    for (char ch : frac) {
      if (std::isdigit(static_cast<unsigned char>(ch))) {
        digits.push_back(ch);
      } else {
        break;
      }
    }
    if (!digits.empty()) {
      while (digits.size() < 3) {
        digits.push_back('0');
      }
      if (digits.size() > 3) {
        digits.resize(3);
      }
      frac_ms = static_cast<double>(to_int(digits, 0));
    }
  }

  out_ms = static_cast<double>(tt) * 1000.0 + frac_ms;
  return true;
}

inline int find_col(const std::unordered_map<std::string, int>& idx, const std::vector<std::string>& keys) {
  for (const auto& key : keys) {
    auto it = idx.find(lower(key));
    if (it != idx.end()) {
      return it->second;
    }
  }
  return -1;
}

inline int bucket_value(int value, int bucket_size) {
  if (bucket_size <= 1) {
    return value;
  }
  const double ratio = static_cast<double>(value) / static_cast<double>(bucket_size);
  const int rounded = static_cast<int>(std::round(ratio)) * bucket_size;
  return std::max(1, rounded);
}

inline Clk_t ms_to_cycles(double cycles_per_ms, double ms) {
  if (!(ms > 0.0)) {
    return 0;
  }
  const double raw = ms * cycles_per_ms;
  const auto cyc = static_cast<Clk_t>(std::ceil(raw - 1e-12));
  return std::max<Clk_t>(1, cyc);
}

}  // namespace Ramulator::Serving::detail

#endif  // RAMULATOR_FRONTEND_IMPL_SERVING_INPUTS_INTERNAL_H
