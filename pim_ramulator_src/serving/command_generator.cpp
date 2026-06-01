#include <algorithm>
#include <tuple>
#include <vector>

#include "frontend/impl/serving/command_generator_internal.h"

namespace Ramulator::Serving {

namespace {

void finalize_compact_layout(CompactKvObjectLayout& object_layout,
                             std::vector<KvLogicalRowKey>& rows) {
  std::sort(rows.begin(), rows.end());
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());

  object_layout.logical_rows = rows;
  object_layout.page_index_by_row.clear();
  object_layout.page_index_by_row.reserve(object_layout.logical_rows.size());
  for (uint32_t page_index = 0; page_index < object_layout.logical_rows.size(); ++page_index) {
    object_layout.page_index_by_row.emplace(object_layout.logical_rows[page_index], page_index);
  }
}

}  // namespace

// Logical command helpers
LogicalCommand make_passthrough_command(int type_id, uint64_t passthrough_addr) {
  LogicalCommand cmd;
  cmd.type_id = type_id;
  cmd.target = CommandTargetKind::Passthrough;
  cmd.passthrough_addr = passthrough_addr;
  return cmd;
}

LogicalCommand make_kv_command(int type_id,
                               CommandTargetKind target,
                               const LogicalKvAddress& kv_address) {
  LogicalCommand cmd;
  cmd.type_id = type_id;
  cmd.target = target;
  cmd.kv_address = kv_address;
  return cmd;
}

bool expects_callback(int type_id) {
  return type_id != Request::Type::PIM_BARRIER;
}

MemoryRegionKind region_for_target(CommandTargetKind target) {
  switch (target) {
    case CommandTargetKind::KvKey:
      return MemoryRegionKind::KvKey;
    case CommandTargetKind::KvValue:
      return MemoryRegionKind::KvValue;
    case CommandTargetKind::Passthrough:
      break;
  }
  return MemoryRegionKind::Invalid;
}

// Flat slice addr -> logical KV coords
LogicalKvAddress make_kv_address(int head_itr,
                                 int local_channel,
                                 uint64_t slice_relative_addr,
                                 const LpddrGeometry& gs) {
  LogicalKvAddress out;
  out.layer_idx = 0;
  out.head_itr = static_cast<uint16_t>(std::max(0, head_itr));
  out.local_channel = static_cast<uint16_t>(std::max(0, local_channel));

  uint64_t rem = slice_relative_addr;
  out.rank = static_cast<uint8_t>(rem / gs.rank);
  rem %= gs.rank;
  out.bank_group = static_cast<uint8_t>(rem / gs.bg);
  rem %= gs.bg;
  out.bank = static_cast<uint8_t>(rem / gs.ba);
  rem %= gs.ba;
  out.row_idx = static_cast<uint16_t>(rem / gs.row);
  rem %= gs.row;
  out.col_idx = static_cast<uint16_t>(rem);
  return out;
}

// Accessed logical rows -> compact page order
void collect_compact_layout_from_commands(const std::vector<LogicalCommand>& commands,
                                          RepresentativeKvLayout& layout) {
  std::vector<KvLogicalRowKey> key_rows;
  std::vector<KvLogicalRowKey> value_rows;

  for (const auto& cmd : commands) {
    if (!cmd.kv_address.has_value()) {
      continue;
    }
    const auto row = logical_row_key_for_kv_address(*cmd.kv_address);
    switch (region_for_target(cmd.target)) {
      case MemoryRegionKind::KvKey:
        key_rows.push_back(row);
        break;
      case MemoryRegionKind::KvValue:
        value_rows.push_back(row);
        break;
      case MemoryRegionKind::Invalid:
      case MemoryRegionKind::Score:
      case MemoryRegionKind::Context:
      case MemoryRegionKind::Scratch:
        break;
    }
  }

  finalize_compact_layout(layout.key, key_rows);
  finalize_compact_layout(layout.value, value_rows);
}

}  // namespace Ramulator::Serving
