#ifndef RAMULATOR_FRONTEND_IMPL_SERVING_LAYOUT_PLANNER_H
#define RAMULATOR_FRONTEND_IMPL_SERVING_LAYOUT_PLANNER_H

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "base/base.h"

namespace Ramulator::Serving {

enum class MemoryRegionKind : uint8_t {
  Invalid = 0,
  KvKey = 1,
  KvValue = 2,
  Score = 3,
  Context = 4,
  Scratch = 5,
};

struct GeneratorGeometry {
  std::string backend = "lpddr5_bank";
  int dhead = 128;
  int heads_per_hbm = 1;
  int dtype_bytes = 2;
  int num_layers = 1;
  int channel_count = 16;
};

struct RequestMemoryFootprint {
  uint64_t kv_key_bytes = 0;
  uint64_t kv_value_bytes = 0;

  uint64_t total_bytes() const { return kv_key_bytes + kv_value_bytes; }
};

struct TaskMemoryFootprint {
  uint64_t score_bytes = 0;
  uint64_t context_bytes = 0;
  uint64_t scratch_bytes = 0;

  uint64_t total_bytes() const { return score_bytes + context_bytes + scratch_bytes; }
};

struct LogicalKvAddress {
  uint16_t layer_idx = 0;
  uint16_t head_itr = 0;
  uint16_t local_channel = 0;
  uint8_t rank = 0;
  uint8_t bank_group = 0;
  uint8_t bank = 0;
  uint16_t row_idx = 0;
  uint16_t col_idx = 0;
};

struct KvLogicalRowKey {
  uint16_t layer_idx = 0;
  uint16_t head_itr = 0;
  uint16_t local_channel = 0;
  uint8_t rank = 0;
  uint8_t bank_group = 0;
  uint8_t bank = 0;
  uint16_t row_idx = 0;

  bool operator==(const KvLogicalRowKey& other) const {
    return std::tie(layer_idx, head_itr, local_channel, rank, bank_group, bank, row_idx) ==
           std::tie(other.layer_idx,
                    other.head_itr,
                    other.local_channel,
                    other.rank,
                    other.bank_group,
                    other.bank,
                    other.row_idx);
  }

  bool operator<(const KvLogicalRowKey& other) const {
    return std::tie(layer_idx, head_itr, local_channel, rank, bank_group, bank, row_idx) <
           std::tie(other.layer_idx,
                    other.head_itr,
                    other.local_channel,
                    other.rank,
                    other.bank_group,
                    other.bank,
                    other.row_idx);
  }
};

struct KvLogicalRowKeyHash {
  size_t operator()(const KvLogicalRowKey& key) const;
};

struct TransientLogicalPageKey {
  uint16_t request_slot = 0;
  uint16_t local_channel = 0;
  uint8_t rank = 0;
  uint8_t bank_group = 0;
  uint8_t bank = 0;
  uint32_t page_slot = 0;
};

struct CompactKvObjectLayout {
  std::vector<KvLogicalRowKey> logical_rows;
  std::unordered_map<KvLogicalRowKey, uint32_t, KvLogicalRowKeyHash> page_index_by_row;

  uint32_t logical_page_count() const { return static_cast<uint32_t>(logical_rows.size()); }
};

struct CompactTransientObjectLayout {
  std::vector<TransientLogicalPageKey> logical_pages;

  uint32_t logical_page_count() const { return static_cast<uint32_t>(logical_pages.size()); }
};

struct RepresentativeKvLayout {
  GeneratorGeometry geometry;
  int context_tokens = 0;
  uint64_t page_size_bytes = 4096;
  CompactKvObjectLayout key;
  CompactKvObjectLayout value;
};

struct TransientTaskLayouts {
  CompactTransientObjectLayout score;
  CompactTransientObjectLayout context;
  CompactTransientObjectLayout scratch;
};

struct VirtualAddressFields {
  MemoryRegionKind region = MemoryRegionKind::Invalid;
  uint32_t request_id = 0;
  uint8_t object_id = 0;
  uint32_t page_index = 0;
  uint16_t page_offset = 0;
};

uint64_t align_up(uint64_t value, uint64_t alignment);

const char* region_name(MemoryRegionKind region);

RequestMemoryFootprint plan_request_memory(const GeneratorGeometry& geometry, int context_tokens);

TaskMemoryFootprint plan_decode_task_memory(const GeneratorGeometry& geometry,
                                            int decode_context_tokens,
                                            int batch_size);

KvLogicalRowKey logical_row_key_for_kv_address(const LogicalKvAddress& address);

std::optional<uint64_t> logical_offset_for_kv_address(const CompactKvObjectLayout& layout,
                                                      const LogicalKvAddress& address,
                                                      uint64_t page_size_bytes);

std::optional<KvLogicalRowKey> logical_row_for_page_index(const CompactKvObjectLayout& layout,
                                                          uint32_t page_index);

CompactTransientObjectLayout build_transient_object_layout(const GeneratorGeometry& geometry,
                                                           uint64_t bytes,
                                                           uint16_t request_slot,
                                                           uint64_t page_size_bytes);

Addr_t encode_virtual_address(MemoryRegionKind region,
                              uint32_t request_id,
                              uint8_t object_id,
                              uint64_t byte_offset);

std::optional<VirtualAddressFields> decode_virtual_address(Addr_t addr);

}  // namespace Ramulator::Serving

#endif  // RAMULATOR_FRONTEND_IMPL_SERVING_LAYOUT_PLANNER_H
