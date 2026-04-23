/**
 * serving_online/pim_allocator.cpp
 *
 * KV cache allocator for PIM DRAM. See pim_allocator.h for design details.
 *
 * VA bit constants (must match the layout documented in pim_allocator.h):
 *   bit  63    : marker
 *   bits 62–60 : region  (3 bits)
 *   bits 59–40 : request_id (20 bits)
 *   bits 39–12 : page_index (28 bits)
 *   bits 11–0  : page_offset (12 bits)
 */

#include "frontend/impl/serving_online/pim_allocator.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace Ramulator::ServingOnline {

// ─── VA bit field constants ─────────────────────────────────────────────────

static constexpr uint64_t VA_MARKER       = (1ULL << 63);
static constexpr int      VA_OFFSET_BITS  = 12;   // bits 11–0 : page offset
static constexpr int      VA_PAGE_BITS    = 28;   // bits 39–12: page index
static constexpr int      VA_REQUEST_BITS = 20;   // bits 59–40: request_id
static constexpr int      VA_REGION_BITS  = 3;    // bits 62–60: region

static constexpr uint64_t VA_OFFSET_MASK  = (1ULL << VA_OFFSET_BITS) - 1;
static constexpr uint64_t VA_PAGE_MASK    = (1ULL << VA_PAGE_BITS)   - 1;
static constexpr uint64_t VA_REQUEST_MASK = (1ULL << VA_REQUEST_BITS)- 1;
static constexpr uint64_t VA_REGION_MASK  = (1ULL << VA_REGION_BITS) - 1;

// Bit positions (shift amount to reach each field).
static constexpr int VA_OFFSET_SHIFT  = 0;
static constexpr int VA_PAGE_SHIFT    = VA_OFFSET_BITS;                            // 12
static constexpr int VA_REQUEST_SHIFT = VA_PAGE_SHIFT   + VA_PAGE_BITS;           // 40
static constexpr int VA_REGION_SHIFT  = VA_REQUEST_SHIFT + VA_REQUEST_BITS;       // 60

// ─── Constructor ─────────────────────────────────────────────────────────────

PimAllocator::PimAllocator(const Config& cfg)
    : m_page_size(cfg.page_size_bytes) {
  if (cfg.kv_pool_bytes == 0) {
    // No PIM memory — pools stay empty. can_fit() will always return false.
    return;
  }

  if (m_page_size == 0 || (m_page_size & (m_page_size - 1)) != 0) {
    throw std::invalid_argument("pim_allocator: page_size_bytes must be a power of two");
  }

  // Split the KV pool evenly: half for Key, half for Value.
  const uint64_t half = cfg.kv_pool_bytes / 2;
  const uint64_t key_pages = half / m_page_size;
  const uint64_t val_pages = half / m_page_size;

  // Key region: starts at PPN 0.
  auto& key_pool      = m_pools[static_cast<int>(KvRegion::Key)];
  key_pool.base_ppn   = 0;
  key_pool.total_pages = key_pages;

  // Value region: starts immediately after Key region.
  auto& val_pool      = m_pools[static_cast<int>(KvRegion::Value)];
  val_pool.base_ppn   = key_pages;
  val_pool.total_pages = val_pages;

  // Initialize Key free pages with channel-interleaved ordering.
  //
  // Goal: consecutive virtual pages (logical indices 0, 1, 2, …) should be
  // stored on consecutive LPDDR5 channels so that reading a full KV cache row
  // issues accesses to all channels in parallel.
  //
  // Mapping:  ppn(i) = base_ppn + ch * pages_per_ch + slot
  //             where  ch   = i % num_channels
  //                    slot = i / num_channels
  //
  // After building the list we reverse it so that back() = channel-0 page 0,
  // which is the first page allocated (pop_back semantics).
  {
    const int num_ch = std::max(1, cfg.num_channels);
    const uint64_t base_pages_per_ch =
        (num_ch > 0) ? (key_pages / static_cast<uint64_t>(num_ch)) : 0;
    const uint64_t extra_pages =
        (num_ch > 0) ? (key_pages % static_cast<uint64_t>(num_ch)) : 0;

    std::vector<uint64_t> ch_bases(static_cast<size_t>(num_ch), key_pool.base_ppn);
    uint64_t running_base = key_pool.base_ppn;
    for (int ch = 0; ch < num_ch; ++ch) {
      ch_bases[static_cast<size_t>(ch)] = running_base;
      const uint64_t ch_pages = base_pages_per_ch + (static_cast<uint64_t>(ch) < extra_pages ? 1 : 0);
      running_base += ch_pages;
    }

    key_pool.free_pages.reserve(key_pages);
    for (uint64_t slot = 0; key_pool.free_pages.size() < key_pages; ++slot) {
      for (int ch = 0; ch < num_ch; ++ch) {
        const uint64_t ch_pages =
            base_pages_per_ch + (static_cast<uint64_t>(ch) < extra_pages ? 1 : 0);
        if (slot >= ch_pages) {
          continue;
        }
        key_pool.free_pages.push_back(ch_bases[static_cast<size_t>(ch)] + slot);
      }
    }
    std::reverse(key_pool.free_pages.begin(), key_pool.free_pages.end());
  }

  // Initialize Value free pages sequentially: PPNs [val_base, val_base + val_pages).
  {
    val_pool.free_pages.resize(val_pages);
    for (uint64_t i = 0; i < val_pages; ++i) {
      val_pool.free_pages[i] = val_pool.base_ppn + i;
    }
    std::reverse(val_pool.free_pages.begin(), val_pool.free_pages.end());
  }
}

// ─── Capacity queries ────────────────────────────────────────────────────────

bool PimAllocator::can_fit(KvRegion r, uint64_t bytes) const {
  const auto& pool = m_pools[static_cast<int>(r)];
  uint64_t pages_needed = (bytes + m_page_size - 1) / m_page_size;
  return pool.free_pages.size() >= pages_needed;
}

uint64_t PimAllocator::free_bytes(KvRegion r) const {
  return m_pools[static_cast<int>(r)].free_pages.size() * m_page_size;
}

uint64_t PimAllocator::total_bytes(KvRegion r) const {
  return m_pools[static_cast<int>(r)].total_pages * m_page_size;
}

// ─── allocate() ─────────────────────────────────────────────────────────────

bool PimAllocator::allocate(KvRegion r, int request_id, uint64_t bytes) {
  uint64_t key = pack_key(r, request_id);

  // Idempotent: if already allocated, succeed without doing anything.
  if (m_allocs.count(key)) return true;

  auto& pool = m_pools[static_cast<int>(r)];
  uint64_t pages_needed = (bytes + m_page_size - 1) / m_page_size;

  // Check capacity before modifying anything (atomic succeed-or-fail).
  if (pool.free_pages.size() < pages_needed) return false;

  // Pop pages from the free stack.
  std::vector<uint64_t> pages(pages_needed);
  for (uint64_t i = 0; i < pages_needed; ++i) {
    pages[i] = pool.free_pages.back();
    pool.free_pages.pop_back();
  }

  m_allocs.emplace(key, std::move(pages));
  return true;
}

// ─── allocate_append() ──────────────────────────────────────────────────────

bool PimAllocator::allocate_append(KvRegion r, int request_id,
                                   uint64_t extra_bytes) {
  if (extra_bytes == 0) return true;

  uint64_t key = pack_key(r, request_id);
  auto it = m_allocs.find(key);
  if (it == m_allocs.end()) {
    // No prior allocation — fall back to a regular allocate().
    return allocate(r, request_id, extra_bytes);
  }

  auto& pool = m_pools[static_cast<int>(r)];
  uint64_t pages_needed = (extra_bytes + m_page_size - 1) / m_page_size;

  // Atomic capacity check.
  if (pool.free_pages.size() < pages_needed) return false;

  // Pop and append to the existing allocation.
  auto& pages = it->second;
  pages.reserve(pages.size() + pages_needed);
  for (uint64_t i = 0; i < pages_needed; ++i) {
    pages.push_back(pool.free_pages.back());
    pool.free_pages.pop_back();
  }
  return true;
}

// ─── free_tail() ────────────────────────────────────────────────────────────

uint64_t PimAllocator::free_tail(KvRegion r, int request_id,
                                  uint64_t bytes_to_free) {
  if (bytes_to_free == 0) return 0;

  uint64_t key = pack_key(r, request_id);
  auto it = m_allocs.find(key);
  if (it == m_allocs.end()) return 0;

  auto& pages = it->second;
  auto& pool  = m_pools[static_cast<int>(r)];

  uint64_t pages_to_free = (bytes_to_free + m_page_size - 1) / m_page_size;
  if (pages_to_free > pages.size()) pages_to_free = pages.size();

  for (uint64_t i = 0; i < pages_to_free; ++i) {
    pool.free_pages.push_back(pages.back());
    pages.pop_back();
  }
  if (pages.empty()) m_allocs.erase(it);
  return pages_to_free * m_page_size;
}

// ─── free() ─────────────────────────────────────────────────────────────────

void PimAllocator::free(KvRegion r, int request_id) {
  uint64_t key = pack_key(r, request_id);
  auto it = m_allocs.find(key);
  if (it == m_allocs.end()) return;  // idempotent: no-op if not allocated

  auto& pool = m_pools[static_cast<int>(r)];
  // Push all pages back onto the free stack.
  for (uint64_t ppn : it->second) {
    pool.free_pages.push_back(ppn);
  }
  m_allocs.erase(it);
}

// ─── encode_va() ─────────────────────────────────────────────────────────────

uint64_t PimAllocator::encode_va(KvRegion r, int request_id,
                                  uint64_t byte_offset) const {
  uint64_t page_index  = byte_offset >> VA_OFFSET_BITS;           // byte_offset / page_size
  uint64_t page_offset = byte_offset &  VA_OFFSET_MASK;           // byte_offset % page_size

  return VA_MARKER
       | (static_cast<uint64_t>(r)          & VA_REGION_MASK)   << VA_REGION_SHIFT
       | (static_cast<uint64_t>(request_id) & VA_REQUEST_MASK)  << VA_REQUEST_SHIFT
       | (page_index                         & VA_PAGE_MASK)     << VA_PAGE_SHIFT
       | (page_offset                        & VA_OFFSET_MASK)   << VA_OFFSET_SHIFT;
}

// ─── translate_va() ─────────────────────────────────────────────────────────

bool PimAllocator::translate_va(uint64_t va, uint64_t& pa) const {
  // Bit 63 must be set — if not, this is a raw physical address.
  if (!(va & VA_MARKER)) return false;

  // Decode VA fields.
  uint64_t region_bits = (va >> VA_REGION_SHIFT)  & VA_REGION_MASK;
  uint64_t request_id  = (va >> VA_REQUEST_SHIFT) & VA_REQUEST_MASK;
  uint64_t page_index  = (va >> VA_PAGE_SHIFT)    & VA_PAGE_MASK;
  uint64_t page_offset = (va >> VA_OFFSET_SHIFT)  & VA_OFFSET_MASK;

  KvRegion r = static_cast<KvRegion>(region_bits);
  uint64_t key = pack_key(r, static_cast<int>(request_id));

  auto it = m_allocs.find(key);
  if (it == m_allocs.end()) return false;   // no allocation for this request

  const auto& pages = it->second;
  if (page_index >= pages.size()) return false;  // out-of-bounds page

  pa = pages[page_index] * m_page_size + page_offset;
  return true;
}

}  // namespace Ramulator::ServingOnline
