#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "base/exception.h"
#include "frontend/impl/serving/allocator.h"

namespace Ramulator::Serving {

namespace {

constexpr uint64_t kLpddr5RowBytes = 4096;
constexpr uint64_t kLpddr5RowsPerBank = 1ULL << 13;

uint64_t object_pages(uint64_t bytes, uint64_t page_size_bytes) {
  return align_up(bytes, page_size_bytes) / page_size_bytes;
}

PlacementBucketKey bucket_key_for_row(const KvLogicalRowKey& row) {
  PlacementBucketKey key;
  key.local_channel = row.local_channel;
  key.rank = row.rank;
  key.bank_group = row.bank_group;
  key.bank = row.bank;
  return key;
}

PlacementBucketKey bucket_key_for_page(const TransientLogicalPageKey& page) {
  PlacementBucketKey key;
  key.local_channel = page.local_channel;
  key.rank = page.rank;
  key.bank_group = page.bank_group;
  key.bank = page.bank;
  return key;
}

PlacementBucketKey bucket_key_for_physical_page(const ServingAllocatorConfig& config,
                                                uint64_t physical_page) {
  const uint64_t bank_stride = kLpddr5RowsPerBank;
  const uint64_t bank_group_stride =
      static_cast<uint64_t>(config.placement_bank_count) * bank_stride;
  const uint64_t rank_stride =
      static_cast<uint64_t>(config.placement_bank_group_count) * bank_group_stride;
  const uint64_t channel_stride =
      static_cast<uint64_t>(config.placement_rank_count) * rank_stride;

  PlacementBucketKey key;
  uint64_t remaining = physical_page;
  key.local_channel = static_cast<uint16_t>(remaining / channel_stride);
  remaining %= channel_stride;
  key.rank = static_cast<uint8_t>(remaining / rank_stride);
  remaining %= rank_stride;
  key.bank_group = static_cast<uint8_t>(remaining / bank_group_stride);
  remaining %= bank_group_stride;
  key.bank = static_cast<uint8_t>(remaining / bank_stride);
  return key;
}

uint64_t bucket_base_ppn(const ServingAllocatorConfig& config, const PlacementBucketKey& key) {
  const uint64_t bank_stride = kLpddr5RowsPerBank;
  const uint64_t bank_group_stride =
      static_cast<uint64_t>(config.placement_bank_count) * bank_stride;
  const uint64_t rank_stride =
      static_cast<uint64_t>(config.placement_bank_group_count) * bank_group_stride;
  const uint64_t channel_stride =
      static_cast<uint64_t>(config.placement_rank_count) * rank_stride;
  return static_cast<uint64_t>(key.local_channel) * channel_stride +
         static_cast<uint64_t>(key.rank) * rank_stride +
         static_cast<uint64_t>(key.bank_group) * bank_group_stride +
         static_cast<uint64_t>(key.bank) * bank_stride;
}

std::vector<PlacementBucketKey> ordered_bucket_keys(const ServingAllocatorConfig& config) {
  std::vector<PlacementBucketKey> keys;
  for (int channel = 0; channel < std::max(1, config.placement_channel_count); ++channel) {
    for (int rank = 0; rank < std::max(1, config.placement_rank_count); ++rank) {
      for (int bank_group = 0; bank_group < std::max(1, config.placement_bank_group_count);
           ++bank_group) {
        for (int bank = 0; bank < std::max(1, config.placement_bank_count); ++bank) {
          keys.push_back(PlacementBucketKey{static_cast<uint16_t>(channel),
                                            static_cast<uint8_t>(rank),
                                            static_cast<uint8_t>(bank_group),
                                            static_cast<uint8_t>(bank)});
        }
      }
    }
  }
  return keys;
}

uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

uint64_t kv_bank_slot_count(const ServingAllocatorConfig& config) {
  const uint64_t bank_group_count =
      static_cast<uint64_t>(std::max(1, config.placement_bank_group_count));
  const uint64_t bank_count = static_cast<uint64_t>(std::max(1, config.placement_bank_count));
  return bank_group_count * bank_count;
}

uint64_t kv_bank_slot_index(const ServingAllocatorConfig& config, const PlacementBucketKey& key) {
  const uint64_t bank_count = static_cast<uint64_t>(std::max(1, config.placement_bank_count));
  return static_cast<uint64_t>(key.bank_group) * bank_count + static_cast<uint64_t>(key.bank);
}

PlacementBucketKey kv_bucket_key_from_bank_slot(const PlacementBucketKey& logical_key,
                                                const ServingAllocatorConfig& config,
                                                uint64_t bank_slot) {
  const uint64_t bank_count = static_cast<uint64_t>(std::max(1, config.placement_bank_count));
  PlacementBucketKey key = logical_key;
  key.bank_group = static_cast<uint8_t>(bank_slot / bank_count);
  key.bank = static_cast<uint8_t>(bank_slot % bank_count);
  return key;
}

uint64_t pressure_tiebreak_seed(MemoryRegionKind region,
                                uint32_t request_id,
                                uint8_t object_id,
                                const KvLogicalRowKey& logical_row) {
  uint64_t seed = splitmix64(static_cast<uint64_t>(request_id) + 1);
  seed ^= splitmix64((static_cast<uint64_t>(region) << 8) | static_cast<uint64_t>(object_id));
  seed ^= splitmix64((static_cast<uint64_t>(logical_row.layer_idx) << 48) |
                     (static_cast<uint64_t>(logical_row.head_itr) << 32) |
                     (static_cast<uint64_t>(logical_row.local_channel) << 16) |
                     static_cast<uint64_t>(logical_row.row_idx));
  return seed;
}

size_t bank_slot_distance(uint64_t from_slot, uint64_t to_slot, uint64_t total_slots) {
  if (total_slots == 0) {
    return 0;
  }
  return static_cast<size_t>((to_slot + total_slots - from_slot) % total_slots);
}

std::optional<PlacementBucketKey> choose_pressure_aware_kv_bucket_key(
    const ServingAllocatorConfig& config,
    const KvLogicalRowKey& logical_row,
    MemoryRegionKind region,
    uint32_t request_id,
    uint8_t object_id,
    const std::unordered_map<PlacementBucketKey, uint64_t, PlacementBucketKeyHash>& free_pages) {
  const PlacementBucketKey logical_key = bucket_key_for_row(logical_row);
  const uint64_t total_bank_slots = kv_bank_slot_count(config);
  if (total_bank_slots == 0) {
    return std::nullopt;
  }

  const uint64_t logical_slot = kv_bank_slot_index(config, logical_key);
  const uint64_t preferred_slot =
      (logical_slot + (pressure_tiebreak_seed(region, request_id, object_id, logical_row) %
                       total_bank_slots)) %
      total_bank_slots;

  std::optional<PlacementBucketKey> best_key;
  uint64_t best_free_pages = 0;
  size_t best_distance = std::numeric_limits<size_t>::max();
  uint64_t best_slot = std::numeric_limits<uint64_t>::max();
  for (uint64_t bank_slot = 0; bank_slot < total_bank_slots; ++bank_slot) {
    const auto candidate = kv_bucket_key_from_bank_slot(logical_key, config, bank_slot);
    const auto it = free_pages.find(candidate);
    if (it == free_pages.end() || it->second == 0) {
      continue;
    }
    const size_t distance = bank_slot_distance(preferred_slot, bank_slot, total_bank_slots);
    if (!best_key.has_value() || it->second > best_free_pages ||
        (it->second == best_free_pages &&
         (distance < best_distance || (distance == best_distance && bank_slot < best_slot)))) {
      best_key = candidate;
      best_free_pages = it->second;
      best_distance = distance;
      best_slot = bank_slot;
    }
  }

  return best_key;
}

std::vector<uint64_t> distribute_bucket_pages(uint64_t total_pages,
                                              const std::vector<uint64_t>& capacities,
                                              size_t start_index) {
  std::vector<uint64_t> assigned(capacities.size(), 0);
  if (total_pages == 0) {
    return assigned;
  }

  std::vector<size_t> eligible;
  uint64_t total_capacity = 0;
  for (size_t idx = 0; idx < capacities.size(); ++idx) {
    if (capacities[idx] == 0) {
      continue;
    }
    eligible.push_back(idx);
    total_capacity += capacities[idx];
  }

  if (eligible.empty() || total_pages > total_capacity) {
    throw ConfigurationError("Serving allocator: placed region exceeds physical LPDDR5 capacity");
  }

  const uint64_t base = total_pages / eligible.size();
  uint64_t assigned_total = 0;
  for (const auto idx : eligible) {
    assigned[idx] = std::min(base, capacities[idx]);
    assigned_total += assigned[idx];
  }

  uint64_t remaining_pages = total_pages - assigned_total;
  size_t cursor = start_index % eligible.size();
  while (remaining_pages > 0) {
    bool placed = false;
    for (size_t attempts = 0; attempts < eligible.size() && remaining_pages > 0; ++attempts) {
      const auto idx = eligible[(cursor + attempts) % eligible.size()];
      if (assigned[idx] >= capacities[idx]) {
        continue;
      }
      ++assigned[idx];
      --remaining_pages;
      cursor = (cursor + attempts + 1) % eligible.size();
      placed = true;
      break;
    }
    if (!placed) {
      throw ConfigurationError("Serving allocator: bucket distribution exceeds row capacity");
    }
  }

  return assigned;
}

void add_free_range_map(std::map<uint64_t, uint64_t>& free_ranges,
                        uint64_t start_ppn,
                        uint64_t page_count) {
  if (page_count == 0) {
    return;
  }
  uint64_t start = start_ppn;
  uint64_t count = page_count;

  auto it = free_ranges.lower_bound(start);
  if (it != free_ranges.begin()) {
    auto prev = std::prev(it);
    if (prev->first + prev->second == start) {
      start = prev->first;
      count += prev->second;
      free_ranges.erase(prev);
    }
  }
  if (it != free_ranges.end() && start + count == it->first) {
    count += it->second;
    free_ranges.erase(it);
  }
  free_ranges[start] = count;
}

template <typename BucketLike>
std::optional<uint64_t> allocate_page(BucketLike& bucket) {
  if (bucket.free_ranges.empty()) {
    return std::nullopt;
  }
  auto it = bucket.free_ranges.begin();
  const uint64_t page = it->first;
  const uint64_t remaining = it->second - 1;
  bucket.free_ranges.erase(it);
  if (remaining > 0) {
    bucket.free_ranges.emplace(page + 1, remaining);
  }
  if (bucket.free_page_count == 0) {
    throw ConfigurationError("Serving allocator: free page counter underflow");
  }
  --bucket.free_page_count;
  bucket.min_free_page_count = std::min(bucket.min_free_page_count, bucket.free_page_count);
  return page;
}

}  // namespace

ServingAddressSpace::ServingAddressSpace(const ServingAllocatorConfig& config) : m_config(config) {
  if (m_config.page_size_bytes == 0) {
    throw ConfigurationError("Serving allocator page size must be > 0");
  }
  if (m_config.page_size_bytes != kLpddr5RowBytes &&
      (m_config.kv_pool_bytes > 0 || m_config.score_pool_bytes > 0 ||
       m_config.context_pool_bytes > 0 || m_config.scratch_pool_bytes > 0)) {
    throw ConfigurationError("Serving allocator: placed regions require 4KB pages");
  }

  const uint64_t max_ppn = m_config.max_addr / m_config.page_size_bytes;
  const auto bucket_keys = ordered_bucket_keys(m_config);
  std::vector<uint64_t> remaining_capacities;
  remaining_capacities.reserve(bucket_keys.size());
  for (const auto& key : bucket_keys) {
    const uint64_t base_ppn = bucket_base_ppn(m_config, key);
    const uint64_t available_rows =
        (base_ppn >= max_ppn) ? 0 : std::min<uint64_t>(kLpddr5RowsPerBank, max_ppn - base_ppn);
    remaining_capacities.push_back(available_rows);
  }
  std::vector<uint64_t> bucket_offsets(bucket_keys.size(), 0);
  size_t distribution_cursor = 0;

  auto build_bucketed_pool = [&](uint64_t total_pages) {
    RegionPool pool;
    pool.page_count = total_pages;
    pool.free_page_count = total_pages;
    pool.min_free_page_count = total_pages;
    pool.uses_bucket_placement = true;

    const auto distribution =
        distribute_bucket_pages(total_pages, remaining_capacities, distribution_cursor);
    for (size_t idx = 0; idx < bucket_keys.size(); ++idx) {
      RegionPool::BucketPool bucket;
      bucket.key = bucket_keys[idx];
      bucket.base_ppn = bucket_base_ppn(m_config, bucket_keys[idx]) + bucket_offsets[idx];
      bucket.page_count = distribution[idx];
      bucket.free_page_count = distribution[idx];
      bucket.min_free_page_count = distribution[idx];
      if (bucket.page_count > 0) {
        bucket.free_ranges.emplace(bucket.base_ppn, bucket.page_count);
      }
      bucket_offsets[idx] += bucket.page_count;
      remaining_capacities[idx] -= bucket.page_count;
      pool.bucket_index_by_key.emplace(bucket.key, pool.buckets.size());
      pool.buckets.push_back(std::move(bucket));
    }
    const auto eligible_count =
        std::count_if(remaining_capacities.begin(), remaining_capacities.end(),
                      [](uint64_t capacity) { return capacity > 0; });
    if (eligible_count > 0) {
      distribution_cursor = (distribution_cursor + (total_pages % eligible_count)) % eligible_count;
    }

    return pool;
  };

  const uint64_t key_pages = (m_config.kv_pool_bytes / 2) / m_config.page_size_bytes;
  const uint64_t value_pages =
      (m_config.kv_pool_bytes - (m_config.kv_pool_bytes / 2)) / m_config.page_size_bytes;
  const uint64_t score_pages = m_config.score_pool_bytes / m_config.page_size_bytes;
  const uint64_t context_pages = m_config.context_pool_bytes / m_config.page_size_bytes;
  const uint64_t scratch_pages = m_config.scratch_pool_bytes / m_config.page_size_bytes;

  m_pools.emplace(MemoryRegionKind::KvKey, build_bucketed_pool(key_pages));
  m_pools.emplace(MemoryRegionKind::KvValue, build_bucketed_pool(value_pages));
  m_pools.emplace(MemoryRegionKind::Score, build_bucketed_pool(score_pages));
  m_pools.emplace(MemoryRegionKind::Context, build_bucketed_pool(context_pages));
  m_pools.emplace(MemoryRegionKind::Scratch, build_bucketed_pool(scratch_pages));
}

ServingAddressSpace::RegionPool& ServingAddressSpace::pool_for_region(MemoryRegionKind region) {
  auto it = m_pools.find(region);
  if (it == m_pools.end()) {
    throw ConfigurationError("Serving allocator: unknown region {}", region_name(region));
  }
  return it->second;
}

const ServingAddressSpace::RegionPool* ServingAddressSpace::pool_for_region(
    MemoryRegionKind region) const {
  auto it = m_pools.find(region);
  return (it == m_pools.end()) ? nullptr : &it->second;
}

void ServingAddressSpace::record_reservation(MemoryRegionKind region, uint64_t bytes, bool add) {
  auto& current_region = m_stats.current_region_bytes[region];
  if (add) {
    current_region += bytes;
    m_stats.current_total_bytes += bytes;
    m_stats.peak_region_bytes[region] =
        std::max(m_stats.peak_region_bytes[region], current_region);
    m_stats.peak_total_bytes = std::max(m_stats.peak_total_bytes, m_stats.current_total_bytes);
  } else {
    current_region = (current_region > bytes) ? (current_region - bytes) : 0;
    m_stats.current_total_bytes =
        (m_stats.current_total_bytes > bytes) ? (m_stats.current_total_bytes - bytes) : 0;
  }
}

void ServingAddressSpace::add_free_range(RegionPool& pool,
                                         uint64_t start_ppn,
                                         uint64_t page_count) {
  add_free_range_map(pool.free_ranges, start_ppn, page_count);
}

AllocationOutcome ServingAddressSpace::reserve_object(MemoryRegionKind region,
                                                      uint32_t request_id,
                                                      uint8_t object_id,
                                                      uint64_t bytes) {
  AllocationKey key{region, request_id, object_id};
  if (auto it = m_allocations.find(key); it != m_allocations.end()) {
    return {true, ""};
  }

  RegionPool& pool = pool_for_region(region);
  if (pool.uses_bucket_placement) {
    throw ConfigurationError("Serving allocator: reserve_object requires a contiguous region");
  }

  const uint64_t needed_pages = object_pages(bytes, m_config.page_size_bytes);
  if (needed_pages == 0) {
    return {true, ""};
  }

  for (auto it = pool.free_ranges.begin(); it != pool.free_ranges.end(); ++it) {
    if (it->second < needed_pages) {
      continue;
    }

    const uint64_t base_ppn = it->first;
    const uint64_t remaining = it->second - needed_pages;
    pool.free_ranges.erase(it);
    if (remaining > 0) {
      pool.free_ranges.emplace(base_ppn + needed_pages, remaining);
    }
    if (pool.free_page_count < needed_pages) {
      throw ConfigurationError("Serving allocator: contiguous free page counter underflow");
    }
    pool.free_page_count -= needed_pages;
    pool.min_free_page_count = std::min(pool.min_free_page_count, pool.free_page_count);

    m_allocations.emplace(
        key, ObjectAllocation{region, request_id, object_id, bytes, needed_pages, base_ppn, {}, {}});
    record_reservation(region, bytes, true);
    return {true, ""};
  }

  return {false, "pim_capacity_exceeded"};
}

AllocationOutcome ServingAddressSpace::reserve_placed_object(
    MemoryRegionKind region,
    uint32_t request_id,
    uint8_t object_id,
    uint64_t bytes,
    const std::vector<PlacementLogicalPageSpec>& logical_pages) {
  AllocationKey key{region, request_id, object_id};
  if (auto it = m_allocations.find(key); it != m_allocations.end()) {
    return {true, ""};
  }

  RegionPool& pool = pool_for_region(region);
  if (!pool.uses_bucket_placement) {
    throw ConfigurationError("Serving allocator: reserve_placed_object requires a placed region");
  }

  const uint64_t needed_pages = object_pages(bytes, m_config.page_size_bytes);
  if (logical_pages.size() > needed_pages) {
    return {false, "pim_layout_exceeds_capacity"};
  }
  if (needed_pages == 0) {
    return {true, ""};
  }

  std::vector<uint64_t> physical_page_by_logical_page;
  physical_page_by_logical_page.reserve(logical_pages.size());
  std::vector<uint64_t> reserved_physical_pages;
  reserved_physical_pages.reserve(needed_pages);
  std::vector<std::pair<size_t, uint64_t>> rollback_pages;
  rollback_pages.reserve(needed_pages);
  const uint64_t original_region_min_free = pool.min_free_page_count;
  std::unordered_map<size_t, uint64_t> original_bucket_min_free;

  std::vector<size_t> used_bucket_indices;
  std::unordered_set<size_t> seen_bucket_indices;

  for (const auto& page : logical_pages) {
    const auto bucket_it = pool.bucket_index_by_key.find(page.bucket_key);
    if (bucket_it == pool.bucket_index_by_key.end()) {
      return {false, "pim_capacity_exceeded"};
    }
    const size_t bucket_idx = bucket_it->second;
    original_bucket_min_free.try_emplace(bucket_idx, pool.buckets[bucket_idx].min_free_page_count);
    auto physical_page = allocate_page(pool.buckets[bucket_idx]);
    if (!physical_page.has_value()) {
      for (const auto& [rollback_bucket_idx, rollback_page] : rollback_pages) {
        add_free_range_map(pool.buckets[rollback_bucket_idx].free_ranges, rollback_page, 1);
        ++pool.buckets[rollback_bucket_idx].free_page_count;
        ++pool.free_page_count;
      }
      for (const auto& [rollback_bucket_idx, min_free] : original_bucket_min_free) {
        pool.buckets[rollback_bucket_idx].min_free_page_count = min_free;
      }
      pool.min_free_page_count = original_region_min_free;
      return {false, "pim_capacity_exceeded"};
    }
    const auto actual_bucket = bucket_key_for_physical_page(m_config, *physical_page);
    if (!(actual_bucket == page.bucket_key)) {
      for (const auto& [rollback_bucket_idx, rollback_page] : rollback_pages) {
        add_free_range_map(pool.buckets[rollback_bucket_idx].free_ranges, rollback_page, 1);
        ++pool.buckets[rollback_bucket_idx].free_page_count;
        ++pool.free_page_count;
      }
      for (const auto& [rollback_bucket_idx, min_free] : original_bucket_min_free) {
        pool.buckets[rollback_bucket_idx].min_free_page_count = min_free;
      }
      pool.min_free_page_count = original_region_min_free;
      throw ConfigurationError("Serving allocator: placed object bucket mismatch during allocation");
    }
    physical_page_by_logical_page.push_back(*physical_page);
    reserved_physical_pages.push_back(*physical_page);
    rollback_pages.emplace_back(bucket_idx, *physical_page);
    if (pool.free_page_count == 0) {
      throw ConfigurationError("Serving allocator: region free page counter underflow");
    }
    --pool.free_page_count;
    pool.min_free_page_count = std::min(pool.min_free_page_count, pool.free_page_count);
    if (seen_bucket_indices.insert(bucket_idx).second) {
      used_bucket_indices.push_back(bucket_idx);
    }
  }

  if (used_bucket_indices.empty()) {
    return {false, "pim_capacity_exceeded"};
  }

  while (reserved_physical_pages.size() < needed_pages) {
    bool progress = false;
    for (const size_t bucket_idx : used_bucket_indices) {
      if (reserved_physical_pages.size() >= needed_pages) {
        break;
      }
      original_bucket_min_free.try_emplace(bucket_idx, pool.buckets[bucket_idx].min_free_page_count);
      auto physical_page = allocate_page(pool.buckets[bucket_idx]);
      if (!physical_page.has_value()) {
        continue;
      }
      reserved_physical_pages.push_back(*physical_page);
      rollback_pages.emplace_back(bucket_idx, *physical_page);
      if (pool.free_page_count == 0) {
        throw ConfigurationError("Serving allocator: region free page counter underflow");
      }
      --pool.free_page_count;
      pool.min_free_page_count = std::min(pool.min_free_page_count, pool.free_page_count);
      progress = true;
    }
    if (!progress) {
      for (const auto& [rollback_bucket_idx, rollback_page] : rollback_pages) {
        add_free_range_map(pool.buckets[rollback_bucket_idx].free_ranges, rollback_page, 1);
        ++pool.buckets[rollback_bucket_idx].free_page_count;
        ++pool.free_page_count;
      }
      for (const auto& [rollback_bucket_idx, min_free] : original_bucket_min_free) {
        pool.buckets[rollback_bucket_idx].min_free_page_count = min_free;
      }
      pool.min_free_page_count = original_region_min_free;
      return {false, "pim_capacity_exceeded"};
    }
  }

  for (size_t page_idx = 0; page_idx < logical_pages.size(); ++page_idx) {
    PlacementDebugRow row = logical_pages[page_idx].debug_row;
    row.region = region;
    row.request_id = request_id;
    row.object_id = object_id;
    row.logical_page_index = static_cast<uint32_t>(page_idx);
    row.physical_page = physical_page_by_logical_page[page_idx];
    row.physical_addr = row.physical_page * m_config.page_size_bytes;

    const auto actual_bucket = bucket_key_for_physical_page(m_config, row.physical_page);
    row.physical_channel = actual_bucket.local_channel;
    row.physical_rank = actual_bucket.rank;
    row.physical_bank_group = actual_bucket.bank_group;
    row.physical_bank = actual_bucket.bank;
    row.physical_row = static_cast<uint16_t>(row.physical_page % kLpddr5RowsPerBank);
    row.bucket_match = (actual_bucket == logical_pages[page_idx].bucket_key);
    m_debug_rows.push_back(std::move(row));
  }

  m_allocations.emplace(key,
                        ObjectAllocation{region,
                                         request_id,
                                         object_id,
                                         bytes,
                                         needed_pages,
                                         0,
                                         std::move(physical_page_by_logical_page),
                                         std::move(reserved_physical_pages)});
  record_reservation(region, bytes, true);
  return {true, ""};
}

AllocationOutcome ServingAddressSpace::forecast_placed_object(
    BudgetSnapshot& budget,
    MemoryRegionKind region,
    uint64_t bytes,
    const std::vector<PlacementLogicalPageSpec>& logical_pages) const {
  const uint64_t needed_pages = object_pages(bytes, m_config.page_size_bytes);
  if (logical_pages.size() > needed_pages) {
    return {false, "pim_layout_exceeds_capacity"};
  }
  if (needed_pages == 0) {
    return {true, ""};
  }

  auto budget_it = budget.region_budgets.find(region);
  if (budget_it == budget.region_budgets.end()) {
    return {false, "pim_capacity_exceeded"};
  }
  auto& region_budget = budget_it->second;
  if (region_budget.free_pages < needed_pages) {
    return {false, "pim_capacity_exceeded"};
  }

  std::unordered_map<PlacementBucketKey, uint64_t, PlacementBucketKeyHash> bucket_demand;
  std::vector<PlacementBucketKey> used_bucket_keys;
  for (const auto& page : logical_pages) {
    auto [demand_it, inserted] = bucket_demand.emplace(page.bucket_key, 0);
    ++demand_it->second;
    if (inserted) {
      used_bucket_keys.push_back(page.bucket_key);
    }
  }
  if (used_bucket_keys.empty()) {
    return {false, "pim_capacity_exceeded"};
  }

  uint64_t filler_capacity = 0;
  for (const auto& bucket_key : used_bucket_keys) {
    auto bucket_it = region_budget.bucket_free_pages.find(bucket_key);
    if (bucket_it == region_budget.bucket_free_pages.end()) {
      return {false, "pim_capacity_exceeded"};
    }
    const auto demand = bucket_demand.at(bucket_key);
    if (bucket_it->second < demand) {
      return {false, "pim_capacity_exceeded"};
    }
    filler_capacity += (bucket_it->second - demand);
  }

  uint64_t filler_pages = needed_pages - logical_pages.size();
  if (filler_pages > filler_capacity) {
    return {false, "pim_capacity_exceeded"};
  }

  for (const auto& bucket_key : used_bucket_keys) {
    auto& free_pages = region_budget.bucket_free_pages[bucket_key];
    free_pages -= bucket_demand.at(bucket_key);
    region_budget.free_pages -= bucket_demand.at(bucket_key);
  }

  while (filler_pages > 0) {
    bool progress = false;
    for (const auto& bucket_key : used_bucket_keys) {
      if (filler_pages == 0) {
        break;
      }
      auto& free_pages = region_budget.bucket_free_pages[bucket_key];
      if (free_pages == 0) {
        continue;
      }
      --free_pages;
      --region_budget.free_pages;
      --filler_pages;
      progress = true;
    }
    if (!progress) {
      return {false, "pim_capacity_exceeded"};
    }
  }

  return {true, ""};
}

ServingAddressSpace::BudgetSnapshot ServingAddressSpace::snapshot_budget() const {
  BudgetSnapshot budget;
  for (const auto& [region, pool] : m_pools) {
    RegionBudgetSnapshot region_budget;
    region_budget.free_pages = pool.free_page_count;
    for (const auto& bucket : pool.buckets) {
      region_budget.bucket_free_pages.emplace(bucket.key, bucket.free_page_count);
    }
    budget.region_budgets.emplace(region, std::move(region_budget));
  }
  return budget;
}

AllocationOutcome ServingAddressSpace::forecast_kv_object(BudgetSnapshot& budget,
                                                          MemoryRegionKind region,
                                                          uint32_t request_id,
                                                          uint8_t object_id,
                                                          uint64_t bytes,
                                                          const CompactKvObjectLayout& layout) const {
  auto budget_it = budget.region_budgets.find(region);
  if (budget_it == budget.region_budgets.end()) {
    return {false, "pim_capacity_exceeded"};
  }

  auto free_pages = budget_it->second.bucket_free_pages;
  std::vector<PlacementLogicalPageSpec> logical_pages;
  logical_pages.reserve(layout.logical_rows.size());
  for (const auto& logical_row : layout.logical_rows) {
    auto bucket_key = choose_pressure_aware_kv_bucket_key(
        m_config, logical_row, region, request_id, object_id, free_pages);
    if (!bucket_key.has_value()) {
      return {false, "pim_capacity_exceeded"};
    }
    --free_pages[*bucket_key];

    PlacementLogicalPageSpec page;
    page.bucket_key = *bucket_key;
    logical_pages.push_back(std::move(page));
  }
  return forecast_placed_object(budget, region, bytes, logical_pages);
}

AllocationOutcome ServingAddressSpace::forecast_transient_object(
    BudgetSnapshot& budget,
    MemoryRegionKind region,
    uint64_t bytes,
    const CompactTransientObjectLayout& layout) const {
  std::vector<PlacementLogicalPageSpec> logical_pages;
  logical_pages.reserve(layout.logical_pages.size());
  for (const auto& logical_page : layout.logical_pages) {
    PlacementLogicalPageSpec page;
    page.bucket_key = bucket_key_for_page(logical_page);
    logical_pages.push_back(std::move(page));
  }
  return forecast_placed_object(budget, region, bytes, logical_pages);
}

AllocationOutcome ServingAddressSpace::reserve_kv_object(MemoryRegionKind region,
                                                         uint32_t request_id,
                                                         uint8_t object_id,
                                                         uint64_t bytes,
                                                         const CompactKvObjectLayout& layout) {
  const RegionPool& pool = pool_for_region(region);
  std::unordered_map<PlacementBucketKey, uint64_t, PlacementBucketKeyHash> free_pages;
  free_pages.reserve(pool.buckets.size());
  for (const auto& bucket : pool.buckets) {
    free_pages.emplace(bucket.key, bucket.free_page_count);
  }

  std::vector<PlacementLogicalPageSpec> logical_pages;
  logical_pages.reserve(layout.logical_rows.size());
  for (const auto& logical_row : layout.logical_rows) {
    auto bucket_key = choose_pressure_aware_kv_bucket_key(
        m_config, logical_row, region, request_id, object_id, free_pages);
    if (!bucket_key.has_value()) {
      return {false, "pim_capacity_exceeded"};
    }
    --free_pages[*bucket_key];

    PlacementLogicalPageSpec page;
    page.bucket_key = *bucket_key;
    page.debug_row.logical_local_channel = logical_row.local_channel;
    page.debug_row.logical_rank = logical_row.rank;
    page.debug_row.logical_bank_group = logical_row.bank_group;
    page.debug_row.logical_bank = logical_row.bank;
    page.debug_row.logical_layer = logical_row.layer_idx;
    page.debug_row.logical_head_itr = logical_row.head_itr;
    page.debug_row.logical_row = logical_row.row_idx;
    logical_pages.push_back(std::move(page));
  }
  return reserve_placed_object(region, request_id, object_id, bytes, logical_pages);
}

AllocationOutcome ServingAddressSpace::reserve_transient_object(
    MemoryRegionKind region,
    uint32_t request_id,
    uint8_t object_id,
    uint64_t bytes,
    const CompactTransientObjectLayout& layout) {
  std::vector<PlacementLogicalPageSpec> logical_pages;
  logical_pages.reserve(layout.logical_pages.size());
  for (const auto& logical_page : layout.logical_pages) {
    PlacementLogicalPageSpec page;
    page.bucket_key = bucket_key_for_page(logical_page);
    page.debug_row.logical_request_slot = logical_page.request_slot;
    page.debug_row.logical_local_channel = logical_page.local_channel;
    page.debug_row.logical_rank = logical_page.rank;
    page.debug_row.logical_bank_group = logical_page.bank_group;
    page.debug_row.logical_bank = logical_page.bank;
    page.debug_row.logical_page_slot = logical_page.page_slot;
    logical_pages.push_back(std::move(page));
  }
  return reserve_placed_object(region, request_id, object_id, bytes, logical_pages);
}

void ServingAddressSpace::release_object(MemoryRegionKind region,
                                         uint32_t request_id,
                                         uint8_t object_id) {
  AllocationKey key{region, request_id, object_id};
  auto it = m_allocations.find(key);
  if (it == m_allocations.end()) {
    return;
  }

  RegionPool& pool = pool_for_region(region);
  if (pool.uses_bucket_placement) {
    for (uint64_t physical_page : it->second.reserved_physical_pages) {
      const auto bucket_key = bucket_key_for_physical_page(m_config, physical_page);
      const auto bucket_it = pool.bucket_index_by_key.find(bucket_key);
      if (bucket_it == pool.bucket_index_by_key.end()) {
        throw ConfigurationError("Serving allocator: placed page {} has unknown bucket",
                                 physical_page);
      }
      auto& bucket = pool.buckets[bucket_it->second];
      if (physical_page < bucket.base_ppn || physical_page >= bucket.base_ppn + bucket.page_count) {
        throw ConfigurationError("Serving allocator: placed page {} is outside bucket ranges",
                                 physical_page);
      }
      add_free_range_map(bucket.free_ranges, physical_page, 1);
      ++bucket.free_page_count;
      ++pool.free_page_count;
    }
  } else {
    add_free_range(pool, it->second.base_ppn, it->second.page_count);
    pool.free_page_count += it->second.page_count;
  }

  record_reservation(region, it->second.bytes, false);
  m_allocations.erase(it);
}

}  // namespace Ramulator::Serving
