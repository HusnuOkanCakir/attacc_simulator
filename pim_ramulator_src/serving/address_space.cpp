#include <algorithm>
#include <sstream>
#include <tuple>

#include "frontend/impl/serving/address_space.h"

namespace Ramulator::Serving {

namespace {

constexpr uint64_t kLpddr5RowsPerBank = 1ULL << 13;

struct PhysicalKvLocation {
  uint16_t channel = 0;
  uint8_t rank = 0;
  uint8_t bank_group = 0;
  uint8_t bank = 0;
  uint16_t row = 0;
};

PhysicalKvLocation decode_physical_kv_page(const ServingAllocatorConfig& config, uint64_t physical_page) {
  const uint64_t bank_stride = kLpddr5RowsPerBank;
  const uint64_t bank_group_stride = static_cast<uint64_t>(config.placement_bank_count) * bank_stride;
  const uint64_t rank_stride = static_cast<uint64_t>(config.placement_bank_group_count) * bank_group_stride;
  const uint64_t channel_stride = static_cast<uint64_t>(config.placement_rank_count) * rank_stride;

  PhysicalKvLocation out;
  uint64_t remaining = physical_page;
  out.channel = static_cast<uint16_t>(remaining / channel_stride);
  remaining %= channel_stride;
  out.rank = static_cast<uint8_t>(remaining / rank_stride);
  remaining %= rank_stride;
  out.bank_group = static_cast<uint8_t>(remaining / bank_group_stride);
  remaining %= bank_group_stride;
  out.bank = static_cast<uint8_t>(remaining / bank_stride);
  out.row = static_cast<uint16_t>(remaining % bank_stride);
  return out;
}

}  // namespace

size_t PlacementBucketKeyHash::operator()(const PlacementBucketKey& key) const {
  return (static_cast<size_t>(key.local_channel) << 12) ^
         (static_cast<size_t>(key.rank) << 8) ^
         (static_cast<size_t>(key.bank_group) << 4) ^
         static_cast<size_t>(key.bank);
}

size_t ServingAddressSpace::AllocationKeyHash::operator()(const AllocationKey& key) const {
  return (static_cast<size_t>(key.request_id) << 8) ^
         (static_cast<size_t>(key.object_id) << 1) ^
         static_cast<size_t>(key.region);
}

bool ServingAddressSpace::translate_virtual(Addr_t vaddr, Addr_t& paddr) const {
  const auto fields = decode_virtual_address(vaddr);
  if (!fields.has_value()) {
    paddr = vaddr;
    return true;
  }

  const AllocationKey key{fields->region, fields->request_id, fields->object_id};
  auto it = m_allocations.find(key);
  if (it == m_allocations.end()) {
    return false;
  }

  uint64_t physical_page = 0;
  if (!it->second.physical_page_by_logical_page.empty()) {
    if (fields->page_index >= it->second.physical_page_by_logical_page.size()) {
      return false;
    }
    physical_page = it->second.physical_page_by_logical_page[fields->page_index];
  } else {
    if (fields->page_index >= it->second.page_count) {
      return false;
    }
    physical_page = it->second.base_ppn + fields->page_index;
  }

  paddr = physical_page * m_config.page_size_bytes + fields->page_offset;
  return true;
}

std::vector<PlacementDebugRow> ServingAddressSpace::collect_placement_rows() const {
  std::vector<PlacementDebugRow> rows = m_debug_rows;

  std::sort(rows.begin(), rows.end(), [](const PlacementDebugRow& a, const PlacementDebugRow& b) {
    return std::tie(a.request_id, a.region, a.object_id, a.logical_page_index) <
           std::tie(b.request_id, b.region, b.object_id, b.logical_page_index);
  });
  return rows;
}

std::vector<RegionPressureRow> ServingAddressSpace::collect_region_pressure_rows() const {
  std::vector<RegionPressureRow> rows;
  rows.reserve(m_pools.size());
  for (const auto& [region, pool] : m_pools) {
    rows.push_back(RegionPressureRow{
        region,
        pool.page_count,
        pool.free_page_count,
        pool.page_count - pool.free_page_count,
        pool.min_free_page_count,
        pool.page_count - pool.min_free_page_count,
    });
  }
  std::sort(rows.begin(), rows.end(), [](const RegionPressureRow& a, const RegionPressureRow& b) {
    return a.region < b.region;
  });
  return rows;
}

std::vector<BucketPressureRow> ServingAddressSpace::collect_bucket_pressure_rows() const {
  std::vector<BucketPressureRow> rows;
  for (const auto& [region, pool] : m_pools) {
    if (!pool.uses_bucket_placement) {
      continue;
    }
    for (const auto& bucket : pool.buckets) {
      if (bucket.page_count == 0) {
        continue;
      }
      rows.push_back(BucketPressureRow{
          region,
          bucket.key.local_channel,
          bucket.key.rank,
          bucket.key.bank_group,
          bucket.key.bank,
          bucket.page_count,
          bucket.free_page_count,
          bucket.page_count - bucket.free_page_count,
          bucket.min_free_page_count,
          bucket.page_count - bucket.min_free_page_count,
      });
    }
  }
  std::sort(rows.begin(), rows.end(), [](const BucketPressureRow& a, const BucketPressureRow& b) {
    return std::tie(a.region, a.local_channel, a.rank, a.bank_group, a.bank) <
           std::tie(b.region, b.local_channel, b.rank, b.bank_group, b.bank);
  });
  return rows;
}

std::string ServingAddressSpace::region_pressure_debug_string() const {
  std::ostringstream oss;
  const auto rows = collect_region_pressure_rows();
  for (size_t idx = 0; idx < rows.size(); ++idx) {
    if (idx > 0) {
      oss << '|';
    }
    oss << region_name(rows[idx].region) << ':' << rows[idx].free_pages << '/'
        << rows[idx].total_pages;
  }
  return oss.str();
}

std::string ServingAddressSpace::hot_bucket_pressure_debug_string(size_t max_buckets) const {
  auto rows = collect_bucket_pressure_rows();
  std::sort(rows.begin(), rows.end(), [](const BucketPressureRow& a, const BucketPressureRow& b) {
    const double a_used_ratio =
        (a.total_pages == 0) ? 0.0 : static_cast<double>(a.used_pages) / a.total_pages;
    const double b_used_ratio =
        (b.total_pages == 0) ? 0.0 : static_cast<double>(b.used_pages) / b.total_pages;
    if (a_used_ratio != b_used_ratio) {
      return a_used_ratio > b_used_ratio;
    }
    if (a.used_pages != b.used_pages) {
      return a.used_pages > b.used_pages;
    }
    return std::tie(a.region, a.local_channel, a.rank, a.bank_group, a.bank) <
           std::tie(b.region, b.local_channel, b.rank, b.bank_group, b.bank);
  });

  std::ostringstream oss;
  const size_t limit = std::min(max_buckets, rows.size());
  for (size_t idx = 0; idx < limit; ++idx) {
    if (idx > 0) {
      oss << '|';
    }
    const auto& row = rows[idx];
    oss << region_name(row.region) << "@c" << row.local_channel << "r"
        << static_cast<int>(row.rank) << "bg" << static_cast<int>(row.bank_group) << 'b'
        << static_cast<int>(row.bank) << ':' << row.free_pages << '/' << row.total_pages;
  }
  return oss.str();
}

}  // namespace Ramulator::Serving
