#ifndef RAMULATOR_FRONTEND_IMPL_SERVING_COMMAND_GENERATOR_INTERNAL_H
#define RAMULATOR_FRONTEND_IMPL_SERVING_COMMAND_GENERATOR_INTERNAL_H

#include <cstdint>
#include <vector>

#include "frontend/impl/serving/command_generator.h"

namespace Ramulator::Serving {

// Fixed LPDDR5-bank geometry
constexpr int kPrefetchBytes = 32;
constexpr int kNPseudoChannels = 1;
constexpr int kNRank = 1;
constexpr int kNBankGroup = 4;
constexpr int kNBank = 4;
constexpr int kNRow = 1 << 13;
constexpr int kNCol = 1 << 7;

struct LpddrGeometry {
  uint64_t col = kPrefetchBytes;
  uint64_t row = static_cast<uint64_t>(kNCol) * col;
  uint64_t ba = static_cast<uint64_t>(kNRow) * row;
  uint64_t bg = static_cast<uint64_t>(kNBank) * ba;
  uint64_t rank = static_cast<uint64_t>(kNBankGroup) * bg;
  uint64_t pch = static_cast<uint64_t>(kNRank) * rank;
  uint64_t ch = static_cast<uint64_t>(kNPseudoChannels) * pch;
};

using CommandList = std::vector<LogicalCommand>;
using CommandGroups = std::vector<CommandList>;

LogicalCommand make_passthrough_command(int type_id, uint64_t passthrough_addr);
LogicalCommand make_kv_command(int type_id,
                               CommandTargetKind target,
                               const LogicalKvAddress& kv_address);
bool expects_callback(int type_id);
MemoryRegionKind region_for_target(CommandTargetKind target);

LogicalKvAddress make_kv_address(int head_itr,
                                 int local_channel,
                                 uint64_t slice_relative_addr,
                                 const LpddrGeometry& gs);

void collect_compact_layout_from_commands(const std::vector<LogicalCommand>& commands,
                                          RepresentativeKvLayout& layout);

}  // namespace Ramulator::Serving

#endif  // RAMULATOR_FRONTEND_IMPL_SERVING_COMMAND_GENERATOR_INTERNAL_H
