/**
 * serving_online/pim_allocator.h
 *
 * KV cache allocator for PIM DRAM.
 *
 * Manages two memory regions:
 *   Key   — stores K tensors for all layers/heads of a request
 *   Value — stores V tensors for all layers/heads of a request
 *
 * Each request gets a contiguous virtual address (VA) space. The VA encodes
 * the region, request id, page index, and byte offset in its bit fields.
 * At command issue time, the translation layer converts VA → physical address.
 *
 * VA bit layout (64-bit):
 *   bit  63    : marker = 1  (distinguishes PIM VA from a raw physical address)
 *   bits 62–60 : region  (0 = Key, 1 = Value)
 *   bits 59–40 : request_id  (20 bits, supports up to ~1M concurrent requests)
 *   bits 39–12 : page_index  (28 bits)
 *   bits 11–0  : page_offset (12 bits = 4096-byte pages)
 *
 * Free pages are maintained per region as a stack (LIFO). For the Key region,
 * pages are initialized in LPDDR5 channel-interleaved order so that consecutive
 * virtual pages map to consecutive pages across channels — this maximises
 * memory-level parallelism when reading KV cache lines.
 */

#ifndef RAMULATOR_FRONTEND_IMPL_SERVING_ONLINE_PIM_ALLOCATOR_H
#define RAMULATOR_FRONTEND_IMPL_SERVING_ONLINE_PIM_ALLOCATOR_H

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace Ramulator::ServingOnline {

// The two KV regions this allocator manages.
enum class KvRegion : uint8_t {
  Key   = 0,  // K tensors
  Value = 1,  // V tensors
};

class PimAllocator {
 public:
  struct Config {
    // Total bytes for Key + Value combined. Split 50/50 between the two regions.
    uint64_t kv_pool_bytes   = 0;
    // Page size in bytes. Must be a power of two. VA bit layout assumes 4096.
    uint64_t page_size_bytes = 4096;
    // Number of LPDDR5 channels for Key page interleaving.
    // Consecutive Key pages will be placed on consecutive channels mod num_channels.
    int num_channels = 8;
  };

  explicit PimAllocator(const Config& cfg);

  // ── Non-mutating capacity check ──────────────────────────────────────────

  // Returns true if at least `bytes` of free space remain in region `r`.
  bool can_fit(KvRegion r, uint64_t bytes) const;

  // Returns the number of currently free bytes in region `r`.
  uint64_t free_bytes(KvRegion r) const;

  // Returns the total capacity in bytes for region `r`.
  uint64_t total_bytes(KvRegion r) const;

  // ── Allocation / free ────────────────────────────────────────────────────

  /**
   * Allocate `bytes` in region `r` for `request_id`.
   *
   * On success: pages are popped from the free stack, stored in the allocation
   * table, and the function returns true.
   *
   * On failure (OOM): no pages are modified and the function returns false.
   * The caller should hold the request and retry after memory is freed.
   *
   * If `request_id` already has an allocation in `r`, this is a no-op and
   * returns true (idempotent re-allocation guard).
   */
  bool allocate(KvRegion r, int request_id, uint64_t bytes);

  /**
   * Append `extra_bytes` of pages to `request_id`'s existing allocation in `r`.
   *
   * Used by the `max_utilization` scheduler policy to grow KV on demand as
   * decoding produces more tokens. Rounds up to whole pages. If the request
   * has no allocation yet, behaves like `allocate()`. Atomic: checks capacity
   * first, no partial appends on OOM (returns false without modification).
   */
  bool allocate_append(KvRegion r, int request_id, uint64_t extra_bytes);

  /**
   * Free all pages allocated for `request_id` in region `r`.
   *
   * Pages are pushed back onto the free stack. Idempotent: safe to call even
   * if the request has no allocation in this region.
   */
  void free(KvRegion r, int request_id);

  /**
   * Free pages from the *back* of the allocation — the most recently allocated
   * pages, which under chronological allocation order hold the most recent
   * decode tokens. Used by tail-trim preemption: rewinds a victim's KV cache
   * by the last N tokens without touching prefill or earlier decode steps.
   *
   * Frees `ceil(bytes_to_free / page_size)` pages, capped at the number of
   * pages currently owned by the request. Returns the number of bytes actually
   * freed (page-aligned, may be slightly larger than requested, or smaller if
   * the request has fewer pages than requested). Idempotent: returns 0 if the
   * request has no allocation in this region.
   */
  uint64_t free_tail(KvRegion r, int request_id, uint64_t bytes_to_free);

  // ── VA ↔ PA translation ──────────────────────────────────────────────────

  /**
   * Translate a virtual address `va` to physical address `pa`.
   *
   * Returns true on success. Returns false if:
   *   - bit 63 is 0 (not a PIM VA — caller should pass the address through unchanged)
   *   - the allocation for (region, request_id) does not exist
   *   - page_index is out of bounds for the allocation
   */
  bool translate_va(uint64_t va, uint64_t& pa) const;

  /**
   * Encode a virtual address for a byte at `byte_offset` within the allocation
   * of `request_id` in region `r`.
   *
   * The allocation must already exist (allocate() must have been called first).
   * byte_offset is the linear offset in bytes from the start of the allocation.
   */
  uint64_t encode_va(KvRegion r, int request_id, uint64_t byte_offset) const;

 private:
  // One pool of pages for one region (Key or Value).
  struct RegionPool {
    uint64_t base_ppn    = 0;   // first physical page number of this region
    uint64_t total_pages = 0;   // total pages in region (= capacity / page_size)
    // Free page stack. back() = next page to allocate. Pages are pushed
    // back here when freed, enabling re-use.
    std::vector<uint64_t> free_pages;
  };

  // Pack (region, request_id) into a single 64-bit key for the allocation map.
  static uint64_t pack_key(KvRegion r, int request_id) {
    return (static_cast<uint64_t>(r) << 32) | static_cast<uint32_t>(request_id);
  }

  std::array<RegionPool, 2>                          m_pools;   // [Key, Value]
  std::unordered_map<uint64_t, std::vector<uint64_t>> m_allocs; // key→pages
  uint64_t m_page_size;
};

}  // namespace Ramulator::ServingOnline

#endif  // RAMULATOR_FRONTEND_IMPL_SERVING_ONLINE_PIM_ALLOCATOR_H
