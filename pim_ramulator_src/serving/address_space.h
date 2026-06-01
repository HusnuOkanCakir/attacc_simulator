#ifndef RAMULATOR_FRONTEND_IMPL_SERVING_ADDRESS_SPACE_H
#define RAMULATOR_FRONTEND_IMPL_SERVING_ADDRESS_SPACE_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/base.h"
#include "frontend/impl/serving/layout_planner.h"

namespace Ramulator::Serving {

struct ServingAllocatorConfig {
  Addr_t max_addr = 0;
  uint64_t page_size_bytes = 4096;
  uint64_t kv_pool_bytes = 0;
  uint64_t score_pool_bytes = 0;
  uint64_t context_pool_bytes = 0;
  uint64_t scratch_pool_bytes = 0;
  int placement_channel_count = 16;
  int placement_rank_count = 1;
  int placement_bank_group_count = 4;
  int placement_bank_count = 4;
};

struct ServingAllocatorStats {
  uint64_t current_total_bytes = 0;
  uint64_t peak_total_bytes = 0;
  std::unordered_map<MemoryRegionKind, uint64_t> current_region_bytes;
  std::unordered_map<MemoryRegionKind, uint64_t> peak_region_bytes;
};

struct AllocationOutcome {
  bool success = false;
  std::string reason;
};

struct PlacementBucketKey {
  uint16_t local_channel = 0;
  uint8_t rank = 0;
  uint8_t bank_group = 0;
  uint8_t bank = 0;

  bool operator==(const PlacementBucketKey& other) const {
    return local_channel == other.local_channel && rank == other.rank &&
           bank_group == other.bank_group && bank == other.bank;
  }
};

struct PlacementBucketKeyHash {
  size_t operator()(const PlacementBucketKey& key) const;
};

struct PlacementDebugRow {
  MemoryRegionKind region = MemoryRegionKind::Invalid;
  uint32_t request_id = 0;
  uint8_t object_id = 0;
  uint32_t logical_page_index = 0;
  uint16_t logical_request_slot = 0;
  uint16_t logical_local_channel = 0;
  uint8_t logical_rank = 0;
  uint8_t logical_bank_group = 0;
  uint8_t logical_bank = 0;
  uint32_t logical_page_slot = 0;
  uint16_t logical_layer = 0;
  uint16_t logical_head_itr = 0;
  uint16_t logical_row = 0;
  uint64_t physical_page = 0;
  uint64_t physical_addr = 0;
  uint16_t physical_channel = 0;
  uint8_t physical_rank = 0;
  uint8_t physical_bank_group = 0;
  uint8_t physical_bank = 0;
  uint16_t physical_row = 0;
  bool bucket_match = false;
};

struct RegionPressureRow {
  MemoryRegionKind region = MemoryRegionKind::Invalid;
  uint64_t total_pages = 0;
  uint64_t free_pages = 0;
  uint64_t used_pages = 0;
  uint64_t min_free_pages = 0;
  uint64_t peak_used_pages = 0;
};

struct BucketPressureRow {
  MemoryRegionKind region = MemoryRegionKind::Invalid;
  uint16_t local_channel = 0;
  uint8_t rank = 0;
  uint8_t bank_group = 0;
  uint8_t bank = 0;
  uint64_t total_pages = 0;
  uint64_t free_pages = 0;
  uint64_t used_pages = 0;
  uint64_t min_free_pages = 0;
  uint64_t peak_used_pages = 0;
};

class ServingAddressSpace {
 public:
  struct RegionBudgetSnapshot {
    uint64_t free_pages = 0;
    std::unordered_map<PlacementBucketKey, uint64_t, PlacementBucketKeyHash> bucket_free_pages;
  };

  struct BudgetSnapshot {
    std::unordered_map<MemoryRegionKind, RegionBudgetSnapshot> region_budgets;
  };

  explicit ServingAddressSpace(const ServingAllocatorConfig& config);

  const ServingAllocatorConfig& config() const { return m_config; }
  const ServingAllocatorStats& stats() const { return m_stats; }

  BudgetSnapshot snapshot_budget() const;
  AllocationOutcome forecast_kv_object(BudgetSnapshot& budget,
                                       MemoryRegionKind region,
                                       uint32_t request_id,
                                       uint8_t object_id,
                                       uint64_t bytes,
                                       const CompactKvObjectLayout& layout) const;
  AllocationOutcome forecast_transient_object(BudgetSnapshot& budget,
                                              MemoryRegionKind region,
                                              uint64_t bytes,
                                              const CompactTransientObjectLayout& layout) const;
  AllocationOutcome reserve_object(MemoryRegionKind region,
                                   uint32_t request_id,
                                   uint8_t object_id,
                                   uint64_t bytes);
  AllocationOutcome reserve_kv_object(MemoryRegionKind region,
                                      uint32_t request_id,
                                      uint8_t object_id,
                                      uint64_t bytes,
                                      const CompactKvObjectLayout& layout);
  AllocationOutcome reserve_transient_object(MemoryRegionKind region,
                                             uint32_t request_id,
                                             uint8_t object_id,
                                             uint64_t bytes,
                                             const CompactTransientObjectLayout& layout);
  void release_object(MemoryRegionKind region, uint32_t request_id, uint8_t object_id);
  bool translate_virtual(Addr_t vaddr, Addr_t& paddr) const;
  std::vector<PlacementDebugRow> collect_placement_rows() const;
  std::vector<RegionPressureRow> collect_region_pressure_rows() const;
  std::vector<BucketPressureRow> collect_bucket_pressure_rows() const;
  std::string region_pressure_debug_string() const;
  std::string hot_bucket_pressure_debug_string(size_t max_buckets) const;

 private:
  struct AllocationKey {
    MemoryRegionKind region = MemoryRegionKind::Invalid;
    uint32_t request_id = 0;
    uint8_t object_id = 0;

    bool operator==(const AllocationKey& other) const {
      return region == other.region && request_id == other.request_id &&
             object_id == other.object_id;
    }
  };

  struct AllocationKeyHash {
    size_t operator()(const AllocationKey& key) const;
  };

  struct RegionPool {
    struct BucketPool {
      PlacementBucketKey key;
      uint64_t base_ppn = 0;
      uint64_t page_count = 0;
      uint64_t free_page_count = 0;
      uint64_t min_free_page_count = 0;
      std::map<uint64_t, uint64_t> free_ranges;
    };

    uint64_t base_ppn = 0;
    uint64_t page_count = 0;
    uint64_t free_page_count = 0;
    uint64_t min_free_page_count = 0;
    std::map<uint64_t, uint64_t> free_ranges;
    bool uses_bucket_placement = false;
    std::vector<BucketPool> buckets;
    std::unordered_map<PlacementBucketKey, size_t, PlacementBucketKeyHash> bucket_index_by_key;
  };

  struct PlacementLogicalPageSpec {
    PlacementBucketKey bucket_key;
    PlacementDebugRow debug_row;
  };

  struct ObjectAllocation {
    MemoryRegionKind region = MemoryRegionKind::Invalid;
    uint32_t request_id = 0;
    uint8_t object_id = 0;
    uint64_t bytes = 0;
    uint64_t page_count = 0;
    uint64_t base_ppn = 0;
    std::vector<uint64_t> physical_page_by_logical_page;
    std::vector<uint64_t> reserved_physical_pages;
  };

  RegionPool& pool_for_region(MemoryRegionKind region);
  const RegionPool* pool_for_region(MemoryRegionKind region) const;
  AllocationOutcome reserve_placed_object(MemoryRegionKind region,
                                          uint32_t request_id,
                                          uint8_t object_id,
                                          uint64_t bytes,
                                          const std::vector<PlacementLogicalPageSpec>& logical_pages);
  AllocationOutcome forecast_placed_object(BudgetSnapshot& budget,
                                           MemoryRegionKind region,
                                           uint64_t bytes,
                                           const std::vector<PlacementLogicalPageSpec>& logical_pages) const;
  void record_reservation(MemoryRegionKind region, uint64_t bytes, bool add);
  void add_free_range(RegionPool& pool, uint64_t start_ppn, uint64_t page_count);

  ServingAllocatorConfig m_config;
  ServingAllocatorStats m_stats;
  std::unordered_map<MemoryRegionKind, RegionPool> m_pools;
  std::unordered_map<AllocationKey, ObjectAllocation, AllocationKeyHash> m_allocations;
  std::vector<PlacementDebugRow> m_debug_rows;
};

}  // namespace Ramulator::Serving

#endif  // RAMULATOR_FRONTEND_IMPL_SERVING_ADDRESS_SPACE_H
