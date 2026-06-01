#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "base/exception.h"
#include "frontend/impl/serving/trace_loader.h"

namespace Ramulator::Serving {

namespace fs = std::filesystem;

namespace {

std::string trim(const std::string& s) {
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

}  // namespace

int parse_trace_op(const std::string& op, bool allow_load_store) {
  if (allow_load_store) {
    if (op == "LD") {
      return Request::Type::Read;
    }
    if (op == "ST") {
      return Request::Type::Write;
    }
  }

  if (op == "PIM_MAC_AB") {
    return Request::Type::PIM_MAC_AB;
  }
  if (op == "PIM_MAC_SB") {
    return Request::Type::PIM_MAC_SB;
  }
  if (op == "PIM_MAC_PB") {
    return Request::Type::PIM_MAC_PB;
  }
  if (op == "PIM_WR_GB") {
    return Request::Type::PIM_WR_GB;
  }
  if (op == "PIM_MV_SB") {
    return Request::Type::PIM_MV_SB;
  }
  if (op == "PIM_MV_GB") {
    return Request::Type::PIM_MV_GB;
  }
  if (op == "PIM_SFM") {
    return Request::Type::PIM_SFM;
  }
  if (op == "PIM_SET_MODEL") {
    return Request::Type::PIM_SET_MODEL;
  }
  if (op == "PIM_SET_HEAD") {
    return Request::Type::PIM_SET_HEAD;
  }
  if (op == "PIM_BARRIER") {
    return Request::Type::PIM_BARRIER;
  }

  return -1;
}

std::vector<TraceCommand> load_trace_commands(const std::string& file_path, bool allow_load_store) {
  fs::path trace_path(file_path);
  if (!fs::exists(trace_path)) {
    throw ConfigurationError("Trace {} does not exist!", file_path);
  }

  std::ifstream trace_file(trace_path);
  if (!trace_file.is_open()) {
    throw ConfigurationError("Trace {} cannot be opened!", file_path);
  }

  std::vector<TraceCommand> commands;
  std::string line;
  while (std::getline(trace_file, line)) {
    const std::string cleaned = trim(line);
    if (cleaned.empty()) {
      continue;
    }

    std::stringstream ss(cleaned);
    std::string op;
    std::string addr_tok;
    ss >> op >> addr_tok;
    if (op.empty() || addr_tok.empty()) {
      throw ConfigurationError("Trace {} format invalid!", file_path);
    }

    const int req_type = parse_trace_op(op, allow_load_store);
    if (req_type < 0) {
      throw ConfigurationError("Trace {} contains unsupported op {}", file_path, op);
    }

    Addr_t addr = 0;
    try {
      if (addr_tok.rfind("0x", 0) == 0 || addr_tok.rfind("0X", 0) == 0) {
        addr = static_cast<Addr_t>(std::stoll(addr_tok.substr(2), nullptr, 16));
      } else {
        addr = static_cast<Addr_t>(std::stoll(addr_tok));
      }
    } catch (...) {
      throw ConfigurationError("Trace {} format invalid!", file_path);
    }

    commands.push_back({req_type, addr});
  }

  return commands;
}

}  // namespace Ramulator::Serving
