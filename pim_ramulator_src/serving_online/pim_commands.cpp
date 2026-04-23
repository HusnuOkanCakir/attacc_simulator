/**
 * serving_online/pim_commands.cpp
 *
 * Generates LPDDR5 PIM commands for one pi0 attention decode step.
 *
 * KV memory layout (linear byte offset within an allocation):
 *
 *   For a request with L context tokens, the K tensor for all layers is:
 *
 *     K[layer][head][token][d_head]   stored as FP16 (dtype_bytes = 2)
 *
 *   byte_offset(K, layer, head) = (layer * num_heads * L * d_head
 *                                + head  * L * d_head) * dtype_bytes
 *
 *   V has the same layout, but lives in the KvRegion::Value allocation.
 *
 * Command granularity:
 *   One Read per (layer, head) K row and one per V row, pointed at the base
 *   of that row. This gives 640 commands per decode step for pi0 (40L × 8H),
 *   independent of context length, vs. 10k+ at per-cache-line granularity.
 *
 *   The Reads are purely for contention modelling: absolute decode latency
 *   is already accounted for by the cost-table estimate that advances
 *   m_pim_free_ms at start_task(). Using Read (instead of PIM_MAC_AB) avoids
 *   the PIM setup sequence (SET_MODEL/SET_HEAD/WR_GB/BARRIER) that the
 *   serving1_5 pipeline emits, which is why we can aggressively coarsen.
 */

#include "frontend/impl/serving_online/pim_commands.h"

#include <cstdint>

namespace Ramulator::ServingOnline {

PimCommandGen::PimCommandGen(const Params& p,
                              PimAllocator* alloc,
                              int read_type_id,
                              int pim_type_id)
    : m_p(p), m_alloc(alloc),
      m_read_type_id(read_type_id), m_pim_type_id(pim_type_id) {}

// ─── decode_step_count() ─────────────────────────────────────────────────────

int PimCommandGen::decode_step_count(int /*context_tokens*/) const {
  // One PIM_MAC_AB per (layer, head) K row + one per V row.
  // Each MAC_AB runs an all-bank MAC over the full context, so we do not need
  // per-cache-line requests — the controller models the internal timing.
  return m_p.num_layers * m_p.num_heads * 2;
}

// ─── decode_step() ───────────────────────────────────────────────────────────

std::vector<Request> PimCommandGen::decode_step(
    int request_id,
    int context_tokens,
    std::function<void(Request&)> on_complete) const {

  std::vector<Request> cmds;
  cmds.reserve(decode_step_count(context_tokens));

  const uint64_t bytes_per_token = static_cast<uint64_t>(m_p.d_head * m_p.dtype_bytes);
  const uint64_t bytes_per_head  = static_cast<uint64_t>(context_tokens) * bytes_per_token;
  const uint64_t bytes_per_layer = static_cast<uint64_t>(m_p.num_heads) * bytes_per_head;

  for (int layer = 0; layer < m_p.num_layers; ++layer) {
    for (int head = 0; head < m_p.num_heads; ++head) {
      const uint64_t base = static_cast<uint64_t>(layer) * bytes_per_layer
                          + static_cast<uint64_t>(head)  * bytes_per_head;

      // One Read that stands in for the K row access for this (layer, head).
      // Using Read (not PIM_MAC_AB) avoids the PIM setup sequence
      // (SET_MODEL/SET_HEAD/WR_GB/BARRIER) while still exercising the bank
      // scheduler for contention modelling. Absolute decode latency comes
      // from the cost table.
      uint64_t k_va = m_alloc->encode_va(KvRegion::Key, request_id, base);
      cmds.emplace_back(static_cast<Addr_t>(k_va), m_read_type_id,
                        request_id, on_complete);

      // One Read for the V row for this (layer, head).
      uint64_t v_va = m_alloc->encode_va(KvRegion::Value, request_id, base);
      cmds.emplace_back(static_cast<Addr_t>(v_va), m_read_type_id,
                        request_id, on_complete);
    }
  }

  return cmds;
}

}  // namespace Ramulator::ServingOnline
