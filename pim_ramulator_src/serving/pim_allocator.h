#ifndef RAMULATOR_FRONTEND_IMPL_SERVING_PIM_ALLOCATOR_H
#define RAMULATOR_FRONTEND_IMPL_SERVING_PIM_ALLOCATOR_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================
//  PimAllocator — LPDDR5 PIM memory allocator for LLM serving
//
//  Manages five named memory regions on the PIM-side LPDDR5:
//    KvKey / KvValue  — persistent per request (KV-cache)
//    Score / Context / Scratch — transient per decode task
//
//  Two responsibilities:
//  1. Capacity gate: can_fit() / allocate() / free() let the
//     scheduler check and reserve memory before running a request.
//  2. VA→PA translation: encode_va() stamps a virtual address onto
//     each PIM command; translate_va() resolves it to a physical
//     DRAM address at command-issue time.
//
//  Virtual address format (64-bit):
//    bit 63     : marker = 1  (0 means raw physical address — pass through)
//    bits 62-60 : region  (3 bits, PimRegion enum values 0-4)
//    bits 59-40 : request_id  (20 bits)
//    bits 39-12 : page_index  (28 bits, index into this allocation's pages)
//    bits 11-0  : page_offset (12 bits, byte offset within 4 KB page)
// ============================================================

namespace Ramulator {
namespace Serving {

// ---- Region IDs ----

enum class PimRegion : uint8_t {
  KvKey    = 0,  // Key cache, persistent for request lifetime
  KvValue  = 1,  // Value cache, persistent for request lifetime
  Score    = 2,  // Attention scores, transient per decode task
  Context  = 3,  // Context output,  transient per decode task
  Scratch  = 4,  // Scratch buffer,  transient per decode task
};

// ---- Internal pool and allocation state ----

// One physical page pool per region. free_pages acts as a stack
// (back() = next page to hand out). For KV regions the pages are
// pre-ordered so consecutive pops stripe across LPDDR5 channels.
struct RegionPool {
  uint64_t base_ppn    = 0;  // first physical page number of this region
  uint64_t total_pages = 0;  // total page capacity
  std::vector<uint64_t> free_pages;  // available pages (stack, back = next)
};

// All physical pages allocated to one (region, request_id) pair.
// pages[logical_index] == physical page number.
struct Allocation {
  std::vector<uint64_t> pages;
};

// ---- Allocator class ----

class PimAllocator {
 public:
  // Configuration — mirrors the YAML serving config fields.
  struct Config {
    uint64_t page_size_bytes   = 4096;  // must be a power of two
    uint64_t kv_pool_bytes     = 0;     // split 50/50 between KvKey and KvValue
    uint64_t score_pool_bytes  = 0;
    uint64_t context_pool_bytes = 0;
    uint64_t scratch_pool_bytes = 0;
    int      num_channels      = 8;     // LPDDR5 channel count for KV interleaving
  };

  explicit PimAllocator(const Config& cfg);

  // ---- Capacity interface ----

  // Returns true if 'bytes' can be allocated from 'region' right now (no mutation).
  bool can_fit(PimRegion region, uint64_t bytes) const;

  // Reserves pages for (region, request_id). Returns true on success, false on OOM.
  // Atomically succeeds or fails — no partial state on failure.
  bool allocate(PimRegion region, uint32_t request_id, uint64_t bytes);

  // Returns all pages of (region, request_id) to the free pool.
  // Idempotent: safe to call even if the pair was never allocated.
  void free(PimRegion region, uint32_t request_id);

  // ---- Translation interface ----

  // If 'va' has the marker bit set (bit 63 = 1), decode it and fill 'pa'.
  // Returns true  → va was a PIM virtual address; pa is now valid.
  // Returns false → va is a raw physical address; caller should pass it through.
  bool translate_va(uint64_t va, uint64_t& pa) const;

  // Encode a virtual address for a byte at 'byte_offset' within the allocation
  // (region, request_id). The result carries the marker bit and all VA fields.
  uint64_t encode_va(PimRegion region, uint32_t request_id, uint64_t byte_offset) const;

  // ---- Stats ----

  uint64_t free_bytes(PimRegion region) const;
  uint64_t total_bytes(PimRegion region) const;

 private:
  // Pack (region, request_id) into a single 64-bit key for the allocation map.
  static uint64_t pack_key(PimRegion region, uint32_t request_id);

  // Number of pages required to hold 'bytes', rounded up to page boundary.
  uint64_t pages_needed(uint64_t bytes) const;

  Config m_cfg;

  // Pool state: key = PimRegion cast to uint8_t.
  std::unordered_map<uint8_t, RegionPool> m_pools;

  // Active allocations: key = pack_key(region, request_id).
  std::unordered_map<uint64_t, Allocation> m_allocations;
};

}  // namespace Serving
}  // namespace Ramulator

#endif  // RAMULATOR_FRONTEND_IMPL_SERVING_PIM_ALLOCATOR_H
