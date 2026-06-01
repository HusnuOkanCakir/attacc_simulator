#include <algorithm>

#include "frontend/impl/serving/layout_planner.h"

namespace Ramulator::Serving {

namespace {

constexpr uint64_t kVirtualOffsetBits = 12;
constexpr uint64_t kVirtualPageBits = 20;
constexpr uint64_t kVirtualObjectBits = 8;
constexpr uint64_t kVirtualRequestBits = 16;
constexpr uint64_t kVirtualRegionBits = 8;

constexpr uint64_t kVirtualOffsetMask = (1ULL << kVirtualOffsetBits) - 1;
constexpr uint64_t kVirtualPageMask = (1ULL << kVirtualPageBits) - 1;
constexpr uint64_t kVirtualObjectMask = (1ULL << kVirtualObjectBits) - 1;
constexpr uint64_t kVirtualRequestMask = (1ULL << kVirtualRequestBits) - 1;
constexpr uint64_t kVirtualRegionShift = kVirtualOffsetBits + kVirtualPageBits +
                                         kVirtualObjectBits + kVirtualRequestBits;
constexpr uint64_t kVirtualRequestShift = kVirtualOffsetBits + kVirtualPageBits +
                                          kVirtualObjectBits;
constexpr uint64_t kVirtualObjectShift = kVirtualOffsetBits + kVirtualPageBits;
constexpr uint64_t kVirtualPageShift = kVirtualOffsetBits;

int active_transient_channel_count(const GeneratorGeometry& geometry) {
  return std::max(1, std::min(std::max(1, geometry.channel_count), std::max(1, geometry.heads_per_hbm)));
}

}  // namespace

size_t KvLogicalRowKeyHash::operator()(const KvLogicalRowKey& key) const {
  size_t seed = 0;
  auto mix = [&](uint64_t value) {
    seed ^= static_cast<size_t>(value) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
  };
  mix(key.layer_idx);
  mix(key.head_itr);
  mix(key.local_channel);
  mix(key.rank);
  mix(key.bank_group);
  mix(key.bank);
  mix(key.row_idx);
  return seed;
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const uint64_t rem = value % alignment;
  return rem == 0 ? value : (value + alignment - rem);
}

const char* region_name(MemoryRegionKind region) {
  switch (region) {
    case MemoryRegionKind::KvKey:
      return "kv_key";
    case MemoryRegionKind::KvValue:
      return "kv_value";
    case MemoryRegionKind::Score:
      return "score";
    case MemoryRegionKind::Context:
      return "context";
    case MemoryRegionKind::Scratch:
      return "scratch";
    case MemoryRegionKind::Invalid:
      break;
  }
  return "invalid";
}

RequestMemoryFootprint plan_request_memory(const GeneratorGeometry& geometry, int context_tokens) {
  const uint64_t tokens = static_cast<uint64_t>(std::max(0, context_tokens));
  const uint64_t heads = static_cast<uint64_t>(std::max(1, geometry.heads_per_hbm));
  const uint64_t dhead = static_cast<uint64_t>(std::max(1, geometry.dhead));
  const uint64_t dtype_bytes = static_cast<uint64_t>(std::max(1, geometry.dtype_bytes));
  const uint64_t layers = static_cast<uint64_t>(std::max(1, geometry.num_layers));

  RequestMemoryFootprint out;
  out.kv_key_bytes = layers * tokens * heads * dhead * dtype_bytes;
  out.kv_value_bytes = out.kv_key_bytes;
  return out;
}

TaskMemoryFootprint plan_decode_task_memory(const GeneratorGeometry& geometry,
                                            int decode_context_tokens,
                                            int batch_size) {
  const uint64_t tokens = static_cast<uint64_t>(std::max(1, decode_context_tokens));
  const uint64_t heads = static_cast<uint64_t>(std::max(1, geometry.heads_per_hbm));
  const uint64_t dhead = static_cast<uint64_t>(std::max(1, geometry.dhead));
  const uint64_t dtype_bytes = static_cast<uint64_t>(std::max(1, geometry.dtype_bytes));
  const uint64_t bs = static_cast<uint64_t>(std::max(1, batch_size));

  TaskMemoryFootprint out;
  out.score_bytes = bs * tokens * heads * dtype_bytes;
  out.context_bytes = bs * heads * dhead * dtype_bytes;
  out.scratch_bytes = std::max(out.score_bytes, out.context_bytes);
  return out;
}

KvLogicalRowKey logical_row_key_for_kv_address(const LogicalKvAddress& address) {
  KvLogicalRowKey row;
  row.layer_idx = address.layer_idx;
  row.head_itr = address.head_itr;
  row.local_channel = address.local_channel;
  row.rank = address.rank;
  row.bank_group = address.bank_group;
  row.bank = address.bank;
  row.row_idx = address.row_idx;
  return row;
}

Addr_t encode_virtual_address(MemoryRegionKind region,
                              uint32_t request_id,
                              uint8_t object_id,
                              uint64_t byte_offset) {
  const uint64_t page_index = byte_offset >> kVirtualOffsetBits;
  const uint64_t page_offset = byte_offset & kVirtualOffsetMask;
  return (static_cast<uint64_t>(region) << kVirtualRegionShift) |
         ((static_cast<uint64_t>(request_id) & kVirtualRequestMask) << kVirtualRequestShift) |
         ((static_cast<uint64_t>(object_id) & kVirtualObjectMask) << kVirtualObjectShift) |
         ((page_index & kVirtualPageMask) << kVirtualPageShift) |
         page_offset;
}

std::optional<VirtualAddressFields> decode_virtual_address(Addr_t addr) {
  const uint64_t region_raw = (static_cast<uint64_t>(addr) >> kVirtualRegionShift) &
                              ((1ULL << kVirtualRegionBits) - 1);
  const auto region = static_cast<MemoryRegionKind>(region_raw);
  if (region == MemoryRegionKind::Invalid) {
    return std::nullopt;
  }

  VirtualAddressFields out;
  out.region = region;
  out.request_id = static_cast<uint32_t>((static_cast<uint64_t>(addr) >> kVirtualRequestShift) &
                                         kVirtualRequestMask);
  out.object_id = static_cast<uint8_t>((static_cast<uint64_t>(addr) >> kVirtualObjectShift) &
                                       kVirtualObjectMask);
  out.page_index = static_cast<uint32_t>((static_cast<uint64_t>(addr) >> kVirtualPageShift) &
                                         kVirtualPageMask);
  out.page_offset = static_cast<uint16_t>(static_cast<uint64_t>(addr) & kVirtualOffsetMask);
  return out;
}

std::optional<uint64_t> logical_offset_for_kv_address(const CompactKvObjectLayout& layout,
                                                      const LogicalKvAddress& address,
                                                      uint64_t page_size_bytes) {
  if (page_size_bytes != 4096) {
    return std::nullopt;
  }
  const auto row = logical_row_key_for_kv_address(address);
  const auto it = layout.page_index_by_row.find(row);
  if (it == layout.page_index_by_row.end()) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(it->second) * page_size_bytes + address.col_idx;
}

std::optional<KvLogicalRowKey> logical_row_for_page_index(const CompactKvObjectLayout& layout,
                                                          uint32_t page_index) {
  if (page_index >= layout.logical_rows.size()) {
    return std::nullopt;
  }
  return layout.logical_rows[page_index];
}

CompactTransientObjectLayout build_transient_object_layout(const GeneratorGeometry& geometry,
                                                           uint64_t bytes,
                                                           uint16_t request_slot,
                                                           uint64_t page_size_bytes) {
  CompactTransientObjectLayout layout;
  if (page_size_bytes == 0) {
    return layout;
  }

  const uint64_t pages = align_up(bytes, page_size_bytes) / page_size_bytes;
  if (pages == 0) {
    return layout;
  }

  const int active_channels = active_transient_channel_count(geometry);
  const int rank_count = 1;
  const int bank_group_count = 4;
  const int bank_count = 4;
  const uint32_t bucket_count = static_cast<uint32_t>(active_channels * rank_count *
                                                      bank_group_count * bank_count);

  std::vector<uint32_t> next_page_slot(bucket_count, 0);
  layout.logical_pages.reserve(pages);

  auto fill_key = [&](uint32_t bucket_idx, uint32_t page_slot) {
    TransientLogicalPageKey key;
    key.request_slot = request_slot;
    key.local_channel = static_cast<uint16_t>(bucket_idx / (rank_count * bank_group_count * bank_count));

    uint32_t remaining = bucket_idx % (rank_count * bank_group_count * bank_count);
    key.rank = static_cast<uint8_t>(remaining / (bank_group_count * bank_count));
    remaining %= (bank_group_count * bank_count);
    key.bank_group = static_cast<uint8_t>(remaining / bank_count);
    key.bank = static_cast<uint8_t>(remaining % bank_count);
    key.page_slot = page_slot;
    return key;
  };

  for (uint64_t page_idx = 0; page_idx < pages; ++page_idx) {
    const uint32_t bucket_idx =
        static_cast<uint32_t>((request_slot + page_idx) % static_cast<uint64_t>(bucket_count));
    layout.logical_pages.push_back(fill_key(bucket_idx, next_page_slot[bucket_idx]++));
  }

  return layout;
}

}  // namespace Ramulator::Serving
