#include "frontend/impl/serving/pim_allocator.h"

#include <algorithm>
#include <stdexcept>

namespace Ramulator {
namespace Serving {

// ============================================================
//  Virtual address bit layout constants
// ============================================================

static constexpr int      VA_OFFSET_BITS  = 12;   // bits 11:0  — byte offset within 4 KB page
static constexpr int      VA_PAGE_BITS    = 28;   // bits 39:12 — logical page index
static constexpr int      VA_REQUEST_BITS = 20;   // bits 59:40 — request_id
static constexpr int      VA_REGION_BITS  = 3;    // bits 62:60 — PimRegion enum value
static constexpr uint64_t VA_MARKER       = (1ULL << 63);
static constexpr uint64_t VA_OFFSET_MASK  = (1ULL << VA_OFFSET_BITS)  - 1;
static constexpr uint64_t VA_PAGE_MASK    = (1ULL << VA_PAGE_BITS)    - 1;
static constexpr uint64_t VA_REQUEST_MASK = (1ULL << VA_REQUEST_BITS) - 1;
static constexpr uint64_t VA_REGION_MASK  = (1ULL << VA_REGION_BITS)  - 1;


// ============================================================
//  Constructor: build one RegionPool per region
// ============================================================

PimAllocator::PimAllocator(const Config& cfg) : m_cfg(cfg) {

  // ---- Compute page counts and base page numbers ----
  //
  // Physical layout (flat, all regions laid out sequentially):
  //
  //   [0          ... kv/2)          KvKey
  //   [kv/2       ... kv)            KvValue
  //   [kv         ... kv+score)      Score
  //   [kv+score   ... +context)      Context
  //   [+context   ... +scratch)      Scratch
  //
  const uint64_t ps = m_cfg.page_size_bytes;
  const uint64_t kv_half_pages = (m_cfg.kv_pool_bytes / 2) / ps;
  const uint64_t score_pages   = m_cfg.score_pool_bytes   / ps;
  const uint64_t context_pages = m_cfg.context_pool_bytes / ps;
  const uint64_t scratch_pages = m_cfg.scratch_pool_bytes / ps;

  const uint64_t kv_key_base     = 0;
  const uint64_t kv_value_base   = kv_half_pages;
  const uint64_t score_base      = 2 * kv_half_pages;
  const uint64_t context_base    = score_base  + score_pages;
  const uint64_t scratch_base    = context_base + context_pages;

  // ---- Build KV pools with channel-interleaved free list ----
  //
  // LPDDR5 has num_channels independent channels. For the KV cache we
  // want consecutive logical pages to land on different channels so that
  // a single attention head's decode access hits all channels in parallel.
  //
  // We initialise the free_pages stack such that popping pages in order
  // produces the pattern:
  //   logical page 0 → channel 0  (ppn = base + 0 * pages_per_ch + 0)
  //   logical page 1 → channel 1  (ppn = base + 1 * pages_per_ch + 0)
  //   ...
  //   logical page N → channel (N % num_ch)  (ppn = base + ch*ppc + slot)
  //
  auto build_kv_pool = [&](uint64_t base_ppn, uint64_t total_pages) -> RegionPool {
    RegionPool pool;
    pool.base_ppn    = base_ppn;
    pool.total_pages = total_pages;

    const int num_ch = std::max(1, m_cfg.num_channels);
    const uint64_t pages_per_channel = total_pages / static_cast<uint64_t>(num_ch);

    // Push in interleave order: i=0 → ch0, i=1 → ch1, i=2 → ch2, ...
    pool.free_pages.reserve(total_pages);
    for (uint64_t i = 0; i < total_pages; ++i) {
      const uint64_t ch   = i % static_cast<uint64_t>(num_ch);
      const uint64_t slot = i / static_cast<uint64_t>(num_ch);
      if (slot < pages_per_channel) {
        pool.free_pages.push_back(base_ppn + ch * pages_per_channel + slot);
      }
      // Pages beyond pages_per_channel * num_ch are left unused to keep the
      // channel mapping symmetric. This wastes at most (num_channels - 1) pages.
    }

    // Reverse so that back() == logical page 0 (channel 0, first slot).
    std::reverse(pool.free_pages.begin(), pool.free_pages.end());
    return pool;
  };

  // ---- Build transient pools with simple sequential free list ----
  //
  // Score / Context / Scratch are short-lived (one decode task) and their
  // access pattern does not require cross-channel interleaving.
  //
  auto build_seq_pool = [&](uint64_t base_ppn, uint64_t total_pages) -> RegionPool {
    RegionPool pool;
    pool.base_ppn    = base_ppn;
    pool.total_pages = total_pages;
    pool.free_pages.reserve(total_pages);
    for (uint64_t i = 0; i < total_pages; ++i) {
      pool.free_pages.push_back(base_ppn + i);
    }
    // Reverse so that back() == base_ppn (first page allocated first).
    std::reverse(pool.free_pages.begin(), pool.free_pages.end());
    return pool;
  };

  m_pools[static_cast<uint8_t>(PimRegion::KvKey)]    = build_kv_pool(kv_key_base,   kv_half_pages);
  m_pools[static_cast<uint8_t>(PimRegion::KvValue)]  = build_kv_pool(kv_value_base, kv_half_pages);
  m_pools[static_cast<uint8_t>(PimRegion::Score)]    = build_seq_pool(score_base,   score_pages);
  m_pools[static_cast<uint8_t>(PimRegion::Context)]  = build_seq_pool(context_base, context_pages);
  m_pools[static_cast<uint8_t>(PimRegion::Scratch)]  = build_seq_pool(scratch_base, scratch_pages);
}


// ============================================================
//  Capacity interface
// ============================================================

bool PimAllocator::can_fit(PimRegion region, uint64_t bytes) const {
  if (bytes == 0) return true;
  auto it = m_pools.find(static_cast<uint8_t>(region));
  if (it == m_pools.end()) return false;
  return it->second.free_pages.size() >= pages_needed(bytes);
}

// Allocate pages for (region, request_id).
// We check capacity first, then pop pages — no rollback needed on failure.
bool PimAllocator::allocate(PimRegion region, uint32_t request_id, uint64_t bytes) {
  const uint64_t key = pack_key(region, request_id);

  // Guard against double-allocation of the same pair.
  if (m_allocations.count(key)) return false;

  // Zero-byte allocation: insert an empty record so free() is well-defined.
  if (bytes == 0) {
    m_allocations[key] = Allocation{};
    return true;
  }

  auto it = m_pools.find(static_cast<uint8_t>(region));
  if (it == m_pools.end()) return false;
  RegionPool& pool = it->second;

  const uint64_t needed = pages_needed(bytes);
  if (pool.free_pages.size() < needed) return false;  // OOM — no mutation yet

  // Pop pages from the free stack into the allocation record.
  Allocation alloc;
  alloc.pages.reserve(needed);
  for (uint64_t i = 0; i < needed; ++i) {
    alloc.pages.push_back(pool.free_pages.back());
    pool.free_pages.pop_back();
  }
  m_allocations[key] = std::move(alloc);
  return true;
}

// Return all pages of (region, request_id) to the free pool.
void PimAllocator::free(PimRegion region, uint32_t request_id) {
  const uint64_t key = pack_key(region, request_id);
  auto it = m_allocations.find(key);
  if (it == m_allocations.end()) return;  // idempotent — nothing to do

  auto pool_it = m_pools.find(static_cast<uint8_t>(region));
  if (pool_it != m_pools.end()) {
    for (uint64_t ppn : it->second.pages) {
      pool_it->second.free_pages.push_back(ppn);
    }
  }
  m_allocations.erase(it);
}


// ============================================================
//  Translation interface
// ============================================================

// Decode the virtual address and return the corresponding physical byte address.
bool PimAllocator::translate_va(uint64_t va, uint64_t& pa) const {
  // Bit 63 = 0 means this is a raw physical address, not our VA.
  if (!(va & VA_MARKER)) return false;

  const uint8_t  region_raw  = static_cast<uint8_t>((va >> 60) & VA_REGION_MASK);
  const uint32_t request_id  = static_cast<uint32_t>((va >> (VA_OFFSET_BITS + VA_PAGE_BITS)) & VA_REQUEST_MASK);
  const uint64_t page_index  = (va >> VA_OFFSET_BITS) & VA_PAGE_MASK;
  const uint64_t page_offset = va & VA_OFFSET_MASK;

  if (region_raw > 4) return false;  // invalid region value

  const uint64_t key = pack_key(static_cast<PimRegion>(region_raw), request_id);
  auto it = m_allocations.find(key);
  if (it == m_allocations.end()) return false;  // not allocated

  const auto& pages = it->second.pages;
  if (page_index >= pages.size()) return false;  // page index out of bounds

  pa = pages[page_index] * m_cfg.page_size_bytes + page_offset;
  return true;
}

// Build a virtual address for a byte at 'byte_offset' within (region, request_id).
uint64_t PimAllocator::encode_va(PimRegion region, uint32_t request_id, uint64_t byte_offset) const {
  const uint64_t page_index  = byte_offset / m_cfg.page_size_bytes;
  const uint64_t page_offset = byte_offset % m_cfg.page_size_bytes;
  return VA_MARKER
       | (static_cast<uint64_t>(region)     << 60)
       | (static_cast<uint64_t>(request_id) << (VA_OFFSET_BITS + VA_PAGE_BITS))
       | (page_index                        << VA_OFFSET_BITS)
       | page_offset;
}


// ============================================================
//  Stats
// ============================================================

uint64_t PimAllocator::free_bytes(PimRegion region) const {
  auto it = m_pools.find(static_cast<uint8_t>(region));
  if (it == m_pools.end()) return 0;
  return it->second.free_pages.size() * m_cfg.page_size_bytes;
}

uint64_t PimAllocator::total_bytes(PimRegion region) const {
  auto it = m_pools.find(static_cast<uint8_t>(region));
  if (it == m_pools.end()) return 0;
  return it->second.total_pages * m_cfg.page_size_bytes;
}


// ============================================================
//  Private helpers
// ============================================================

// Pack (region, request_id) into one 64-bit key. Region fits in 3 bits;
// request_id occupies the lower 32 bits. No collisions are possible.
uint64_t PimAllocator::pack_key(PimRegion region, uint32_t request_id) {
  return (static_cast<uint64_t>(region) << 32) | static_cast<uint64_t>(request_id);
}

uint64_t PimAllocator::pages_needed(uint64_t bytes) const {
  return (bytes + m_cfg.page_size_bytes - 1) / m_cfg.page_size_bytes;
}

}  // namespace Serving
}  // namespace Ramulator
