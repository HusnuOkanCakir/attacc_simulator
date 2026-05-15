/**
 * serving_online/pim_commands.cpp
 *
 * Realistic LPDDR5-PIM attention decode-step command generator.
 *
 * Per (layer, kv_head), at cache-line block granularity. Iteration is over
 * num_kv_heads, not num_heads: K and V live once per KV head and are shared
 * across query heads in MQA/GQA. For MHA (num_kv_heads == num_heads) this
 * reduces to the original formula.
 *
 *   K-side score matmul:
 *     for each block in blocks(L):
 *       PIM_WR_GB(K[layer,kv_head,block])  // write query into global buffer
 *       PIM_MAC_AB(K[layer,kv_head,block]) // all-bank MAC against K row
 *     PIM_MV_SB(control)                   // collect per-bank partial scores
 *     PIM_SFM(control)                     // softmax across the score vector
 *
 *   V-side context matmul:
 *     for each block in blocks(L):
 *       PIM_MV_GB(V[layer,kv_head,block])  // broadcast prob vector
 *       PIM_MAC_AB(V[layer,kv_head,block]) // all-bank MAC against V row
 *     PIM_MV_SB(control)                   // collect partial contexts
 *     PIM_BARRIER(control)                 // serialize before next (layer, kv_head)
 *
 * Total commands per decode step: num_layers * num_kv_heads * (4 * blocks + 4)
 * with blocks = ceil(context_tokens / (kLineBytes / dtype_bytes)).
 *
 * Ported from serving1_5/pim_commands.cpp (only namespace renamed).
 */

#include "frontend/impl/serving_online/pim_commands.h"

#include <algorithm>
#include <cstdint>

namespace Ramulator::ServingOnline {

PimCommandGen::PimCommandGen(const Params& p,
                              PimAllocator* alloc,
                              int read_type_id,
                              int pim_type_id)
    : m_p(p), m_alloc(alloc),
      m_read_type_id(read_type_id), m_pim_type_id(pim_type_id) {}

// ─── control_addr() ──────────────────────────────────────────────────────────
//
// Synthesize a non-KV address for PIM control commands (MV_SB, SFM, BARRIER).
// Bit 63 is unset, so PimAllocator-aware translation leaves it untouched.

uint64_t PimCommandGen::control_addr(int request_id, int layer, int head,
                                     int phase_tag) const {
  return (static_cast<uint64_t>(phase_tag & 0xF) << 28) |
         (static_cast<uint64_t>(request_id & 0xFFFFF) << 8) |
         (static_cast<uint64_t>(layer & 0x1F) << 3) |
         static_cast<uint64_t>(head & 0x7);
}

// ─── decode_step_count() ─────────────────────────────────────────────────────

int PimCommandGen::decode_step_count(int context_tokens) const {
  if (m_p.mode == Mode::Simple) {
    // One PIM_MAC_AB for K and one for V per (layer, kv_head). Context-length
    // independent — diagnostic only.
    return m_p.num_layers * m_p.num_kv_heads * 2;
  }
  const int tokens_per_block = kLineBytes / m_p.dtype_bytes;
  const int blocks = (context_tokens + tokens_per_block - 1) / tokens_per_block;
  // Per (layer, kv_head): 2*blocks K-side + 2*blocks V-side + 4 control commands.
  return m_p.num_layers * m_p.num_kv_heads * (4 * blocks + 4);
}

// ─── decode_step() ───────────────────────────────────────────────────────────

std::vector<Request> PimCommandGen::decode_step(
    int request_id,
    int context_tokens,
    std::function<void(Request&)> on_complete) const {

  std::vector<Request> commands;
  commands.reserve(decode_step_count(context_tokens));

  const uint64_t bytes_per_token =
      static_cast<uint64_t>(m_p.d_head) * static_cast<uint64_t>(m_p.dtype_bytes);
  const uint64_t bytes_per_head =
      static_cast<uint64_t>(context_tokens) * bytes_per_token;
  // KV region is laid out as [layer][kv_head][ctx_token][d_head] — each
  // layer occupies num_kv_heads × bytes_per_head, NOT num_heads × bytes_per_head.
  // For MHA (num_kv_heads == num_heads) this is unchanged.
  const uint64_t bytes_per_layer =
      static_cast<uint64_t>(m_p.num_kv_heads) * bytes_per_head;

  // Simple mode: emit two PIM_MAC_AB per (layer, kv_head) — one for K, one
  // for V — at the kv_head's base address. No control commands, no per-block
  // stream. Diagnostic only; absolute decode latency still comes from the cost table.
  if (m_p.mode == Mode::Simple) {
    for (int layer = 0; layer < m_p.num_layers; ++layer) {
      for (int kv_head = 0; kv_head < m_p.num_kv_heads; ++kv_head) {
        const uint64_t key_base =
            static_cast<uint64_t>(layer)   * bytes_per_layer +
            static_cast<uint64_t>(kv_head) * bytes_per_head;
        const uint64_t key_va = m_alloc->encode_va(
            KvRegion::Key,   request_id, key_base);
        const uint64_t val_va = m_alloc->encode_va(
            KvRegion::Value, request_id, key_base);
        commands.emplace_back(static_cast<Addr_t>(key_va),
                              Request::Type::PIM_MAC_AB,
                              request_id, on_complete);
        commands.emplace_back(static_cast<Addr_t>(val_va),
                              Request::Type::PIM_MAC_AB,
                              request_id, on_complete);
      }
    }
    return commands;
  }

  const int tokens_per_block = kLineBytes / m_p.dtype_bytes;
  const int blocks = (context_tokens + tokens_per_block - 1) / tokens_per_block;

  for (int layer = 0; layer < m_p.num_layers; ++layer) {
    for (int kv_head = 0; kv_head < m_p.num_kv_heads; ++kv_head) {
      const uint64_t key_base =
          static_cast<uint64_t>(layer)   * bytes_per_layer +
          static_cast<uint64_t>(kv_head) * bytes_per_head;
      const uint64_t value_base = key_base;  // K and V live in separate VA regions

      // K-side: write query and MAC against each cache-line block of the K row.
      for (int block = 0; block < blocks; ++block) {
        const uint64_t block_offset =
            static_cast<uint64_t>(block * tokens_per_block) * bytes_per_token;
        const uint64_t key_va = m_alloc->encode_va(
            KvRegion::Key, request_id, key_base + block_offset);
        commands.emplace_back(static_cast<Addr_t>(key_va),
                              Request::Type::PIM_WR_GB,
                              request_id, on_complete);
        commands.emplace_back(static_cast<Addr_t>(key_va),
                              Request::Type::PIM_MAC_AB,
                              request_id, on_complete);
      }
      commands.emplace_back(static_cast<Addr_t>(control_addr(request_id, layer, kv_head, 1)),
                            Request::Type::PIM_MV_SB, request_id, on_complete);
      commands.emplace_back(static_cast<Addr_t>(control_addr(request_id, layer, kv_head, 2)),
                            Request::Type::PIM_SFM,   request_id, on_complete);

      // V-side: broadcast prob vector and MAC against each cache-line block of V.
      for (int block = 0; block < blocks; ++block) {
        const uint64_t block_offset =
            static_cast<uint64_t>(block * tokens_per_block) * bytes_per_token;
        const uint64_t value_va = m_alloc->encode_va(
            KvRegion::Value, request_id, value_base + block_offset);
        commands.emplace_back(static_cast<Addr_t>(value_va),
                              Request::Type::PIM_MV_GB,
                              request_id, on_complete);
        commands.emplace_back(static_cast<Addr_t>(value_va),
                              Request::Type::PIM_MAC_AB,
                              request_id, on_complete);
      }
      commands.emplace_back(static_cast<Addr_t>(control_addr(request_id, layer, kv_head, 3)),
                            Request::Type::PIM_MV_SB,  request_id, on_complete);
      // BARRIER is fire-and-forget: empty callback so it does not decrement
      // the outstanding-counter (matches serving1_5).
      commands.emplace_back(static_cast<Addr_t>(control_addr(request_id, layer, kv_head, 4)),
                            Request::Type::PIM_BARRIER, request_id,
                            std::function<void(Request&)>{});
    }
  }

  return commands;
}

}  // namespace Ramulator::ServingOnline
