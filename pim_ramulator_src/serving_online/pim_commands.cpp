/**
 * serving_online/pim_commands.cpp
 *
 * LPDDR5-PIM attention decode-step command generator.
 *
 * Four emission modes (selected via PimCommandGen::Params::mode):
 *
 * 1) Realistic — per-request, per (layer, kv_head), per block. Total cmds
 *    for a bs=B decode step ≈ B × num_layers × num_kv_heads × (4·blocks + 4).
 *
 * 2) Simple — diagnostic. Two PIM_MAC_AB per (layer, kv_head, request),
 *    context-independent. Total ≈ B × num_layers × num_kv_heads × 2.
 *
 * 3) AttAccQBroadcast — ATTACC-faithful. Per (layer, kv_head), one shared
 *    Q broadcast + B per-request K MACs + shared softmax + one shared V
 *    broadcast + B per-request V MACs + shared controls. Total ≈
 *    num_layers × num_kv_heads × (2·B·blocks + 6). Linear in B but with a
 *    smaller coefficient than Realistic (amortizes Q/V broadcast and
 *    control across the batch).
 *
 * 4) BankStripe — experimental. Per (layer, kv_head), one Q broadcast +
 *    one shared PIM_MAC_AB per K-block + shared softmax + one V broadcast
 *    + one shared PIM_MAC_AB per V-block + shared controls. The shared
 *    PIM_MAC_AB targets request 0's VA but, with the channel-interleaved
 *    allocator, the DRAM model parallelizes across channels naturally.
 *    APPROXIMATION: models the ideal case where bank-level parallelism
 *    perfectly amortizes the per-request K/V reads into one MAC_AB window.
 *    Total ≈ num_layers × num_kv_heads × (2·blocks + 6), independent of B.
 *
 * Common structure per (layer, kv_head):
 *
 *   K-side score matmul:
 *     [shared] PIM_WR_GB(Q)
 *     for each block in blocks(L):
 *       PIM_MAC_AB(K[block])  // per-request (Realistic/AttAcc) or shared (Stripe)
 *     PIM_MV_SB(control)
 *     PIM_SFM(control)
 *
 *   V-side context matmul:
 *     [shared] PIM_MV_GB(V_probs)
 *     for each block in blocks(L):
 *       PIM_MAC_AB(V[block])
 *     PIM_MV_SB(control)
 *     PIM_BARRIER(control)
 *
 * Realistic deviates: it emits PIM_WR_GB / PIM_MV_GB *per block* rather than
 * once-per-(layer, kv_head). All other modes hoist Q/V broadcasts out of
 * the block loop. This is the historical port from serving1_5.
 *
 * For batched (B > 1) modes the max context across the batch sets the
 * block count, matching real bank-PIM hardware where all banks step in
 * lockstep. Shorter requests pay a small upper-bound penalty in blocks.
 *
 * Ported from serving1_5/pim_commands.cpp (namespace renamed); batching
 * modes added 2026-05.
 */

#include "frontend/impl/serving_online/pim_commands.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>

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

// ─── decode_step_count_batched() ─────────────────────────────────────────────

int PimCommandGen::decode_step_count_batched(
    const std::vector<int>& context_tokens) const {
  if (context_tokens.empty()) return 0;
  const int B = static_cast<int>(context_tokens.size());
  int max_ctx = 0;
  for (int c : context_tokens) max_ctx = std::max(max_ctx, c);

  if (m_p.mode == Mode::Simple) {
    return B * m_p.num_layers * m_p.num_kv_heads * 2;
  }

  const int tokens_per_block = kLineBytes / m_p.dtype_bytes;
  const int max_blocks =
      (max_ctx + tokens_per_block - 1) / tokens_per_block;

  switch (m_p.mode) {
    case Mode::Realistic: {
      // Per-request (concat of decode_step()), with each request's own block
      // count. Caller passes per-request ctx so we honor it exactly here.
      int total = 0;
      for (int c : context_tokens) {
        const int blocks = (c + tokens_per_block - 1) / tokens_per_block;
        total += m_p.num_layers * m_p.num_kv_heads * (4 * blocks + 4);
      }
      return total;
    }
    case Mode::AttAccQBroadcast: {
      // Per (layer, kv_head): 1 PIM_WR_GB + Σ_i blocks_i PIM_MAC_AB (K) +
      // 1 PIM_MV_SB + 1 PIM_SFM + 1 PIM_MV_GB + Σ_i blocks_i PIM_MAC_AB (V) +
      // 1 PIM_MV_SB + 1 PIM_BARRIER = 6 + 2·Σ_i blocks_i.
      // For homogeneous ctx (all requests same length), this reduces to
      // 6 + 2·B·max_blocks.
      int sum_blocks = 0;
      for (int c : context_tokens) {
        sum_blocks += (c + tokens_per_block - 1) / tokens_per_block;
      }
      return m_p.num_layers * m_p.num_kv_heads * (2 * sum_blocks + 6);
    }
    case Mode::BankStripe:
      // Per (layer, kv_head): 1 PIM_WR_GB + blocks PIM_MAC_AB (K, shared) +
      // 1 PIM_MV_SB + 1 PIM_SFM + 1 PIM_MV_GB + blocks PIM_MAC_AB (V, shared) +
      // 1 PIM_MV_SB + 1 PIM_BARRIER = 2·blocks + 6. Independent of B.
      return m_p.num_layers * m_p.num_kv_heads * (2 * max_blocks + 6);
    case Mode::Simple:
      // Already handled above; fall through to unreachable.
      break;
  }
  return 0;
}

// ─── decode_step_batched() ───────────────────────────────────────────────────

std::vector<Request> PimCommandGen::decode_step_batched(
    const std::vector<int>& request_ids,
    const std::vector<int>& context_tokens,
    std::function<void(Request&)> on_complete) const {

  assert(request_ids.size() == context_tokens.size() && !request_ids.empty());

  // For Realistic / Simple modes, behavior is identical to concatenating
  // decode_step() per request — keep the legacy semantics.
  if (m_p.mode == Mode::Realistic || m_p.mode == Mode::Simple) {
    std::vector<Request> commands;
    commands.reserve(decode_step_count_batched(context_tokens));
    for (size_t i = 0; i < request_ids.size(); ++i) {
      auto sub = decode_step(request_ids[i], context_tokens[i], on_complete);
      commands.insert(commands.end(),
                      std::make_move_iterator(sub.begin()),
                      std::make_move_iterator(sub.end()));
    }
    return commands;
  }

  // ── Shared-command modes (AttAccQBroadcast, BankStripe) ────────────────────

  // Pick the longest-context request in the batch as the anchor. Its
  // KV allocation is large enough to cover every block offset we'll
  // encode below — important because shared commands (Q broadcast,
  // V broadcast, BankStripe MAC_AB) all use the anchor's VA. If we
  // picked a shorter request as anchor, encoded offsets would land
  // beyond its allocated region and the translation layer would fail.
  //
  // For real bank-PIM hardware all banks step in lockstep at the max
  // ctx; shorter requests pay an upper-bound penalty in blocks.
  size_t anchor_idx = 0;
  for (size_t i = 1; i < context_tokens.size(); ++i) {
    if (context_tokens[i] > context_tokens[anchor_idx]) anchor_idx = i;
  }
  const int anchor_req = request_ids[anchor_idx];
  const int max_ctx = context_tokens[anchor_idx];

  const uint64_t bytes_per_token =
      static_cast<uint64_t>(m_p.d_head) * static_cast<uint64_t>(m_p.dtype_bytes);
  const uint64_t bytes_per_head =
      static_cast<uint64_t>(max_ctx) * bytes_per_token;
  // KV region: [layer][kv_head][ctx_token][d_head]; one layer occupies
  // num_kv_heads × bytes_per_head. By using max_ctx (= anchor's ctx),
  // every (layer, kv_head, block) offset lands inside anchor's allocation.
  const uint64_t bytes_per_layer =
      static_cast<uint64_t>(m_p.num_kv_heads) * bytes_per_head;

  const int tokens_per_block = kLineBytes / m_p.dtype_bytes;
  const int blocks = (max_ctx + tokens_per_block - 1) / tokens_per_block;

  // Shared control commands carry request_id=0 by convention so the
  // cache-key fingerprint is deterministic regardless of which physical
  // requests are batched together.
  constexpr int kSharedControlReqId = 0;

  std::vector<Request> commands;
  commands.reserve(decode_step_count_batched(context_tokens));

  for (int layer = 0; layer < m_p.num_layers; ++layer) {
    for (int kv_head = 0; kv_head < m_p.num_kv_heads; ++kv_head) {
      const uint64_t kv_head_base =
          static_cast<uint64_t>(layer)   * bytes_per_layer +
          static_cast<uint64_t>(kv_head) * bytes_per_head;

      // 1× shared Q broadcast (load Q into the global buffer for all B
      // requests). Address: anchor request's K[layer, kv_head] base.
      {
        const uint64_t q_va =
            m_alloc->encode_va(KvRegion::Key, anchor_req, kv_head_base);
        commands.emplace_back(static_cast<Addr_t>(q_va),
                              Request::Type::PIM_WR_GB,
                              anchor_req, on_complete);
      }

      // K-side MAC: per-request in AttAccQBroadcast (one MAC per request per
      // block; uses each request's OWN ctx for VA sizing so we never address
      // beyond its allocation). Shared single MAC in BankStripe (uses anchor's
      // ctx, which is = max_ctx, so VA always lands inside anchor's region).
      for (int block = 0; block < blocks; ++block) {
        const uint64_t block_offset =
            static_cast<uint64_t>(block * tokens_per_block) * bytes_per_token;
        if (m_p.mode == Mode::AttAccQBroadcast) {
          for (size_t i = 0; i < request_ids.size(); ++i) {
            // Per-request sizing — uses request_i's own ctx, not max_ctx.
            // Requests with shorter ctx have fewer blocks; we emit ALL `blocks`
            // (max_ctx-sized) commands but skip blocks past this request's
            // own range. This matches real bank-PIM lockstep behavior where
            // shorter requests stall on their last valid block.
            const int ctx_i = context_tokens[i];
            const int blocks_i = (ctx_i + tokens_per_block - 1) / tokens_per_block;
            if (block >= blocks_i) continue;
            const uint64_t bytes_per_head_i =
                static_cast<uint64_t>(ctx_i) * bytes_per_token;
            const uint64_t bytes_per_layer_i =
                static_cast<uint64_t>(m_p.num_kv_heads) * bytes_per_head_i;
            const uint64_t kv_head_base_i =
                static_cast<uint64_t>(layer)   * bytes_per_layer_i +
                static_cast<uint64_t>(kv_head) * bytes_per_head_i;
            const uint64_t k_va = m_alloc->encode_va(
                KvRegion::Key, request_ids[i], kv_head_base_i + block_offset);
            commands.emplace_back(static_cast<Addr_t>(k_va),
                                  Request::Type::PIM_MAC_AB,
                                  request_ids[i], on_complete);
          }
        } else {  // BankStripe — one shared MAC_AB at anchor's address.
          const uint64_t k_va = m_alloc->encode_va(
              KvRegion::Key, anchor_req, kv_head_base + block_offset);
          commands.emplace_back(static_cast<Addr_t>(k_va),
                                Request::Type::PIM_MAC_AB,
                                anchor_req, on_complete);
        }
      }

      // 1× shared softmax control sequence.
      commands.emplace_back(
          static_cast<Addr_t>(control_addr(kSharedControlReqId, layer, kv_head, 1)),
          Request::Type::PIM_MV_SB, anchor_req, on_complete);
      commands.emplace_back(
          static_cast<Addr_t>(control_addr(kSharedControlReqId, layer, kv_head, 2)),
          Request::Type::PIM_SFM, anchor_req, on_complete);

      // 1× shared V probs broadcast.
      {
        const uint64_t v_broadcast_va =
            m_alloc->encode_va(KvRegion::Value, anchor_req, kv_head_base);
        commands.emplace_back(static_cast<Addr_t>(v_broadcast_va),
                              Request::Type::PIM_MV_GB,
                              anchor_req, on_complete);
      }

      // V-side MAC: same per-request vs shared dispatch as K-side, with
      // per-request VA sizing to avoid out-of-region addresses for the
      // AttAcc per-request path.
      for (int block = 0; block < blocks; ++block) {
        const uint64_t block_offset =
            static_cast<uint64_t>(block * tokens_per_block) * bytes_per_token;
        if (m_p.mode == Mode::AttAccQBroadcast) {
          for (size_t i = 0; i < request_ids.size(); ++i) {
            const int ctx_i = context_tokens[i];
            const int blocks_i = (ctx_i + tokens_per_block - 1) / tokens_per_block;
            if (block >= blocks_i) continue;
            const uint64_t bytes_per_head_i =
                static_cast<uint64_t>(ctx_i) * bytes_per_token;
            const uint64_t bytes_per_layer_i =
                static_cast<uint64_t>(m_p.num_kv_heads) * bytes_per_head_i;
            const uint64_t kv_head_base_i =
                static_cast<uint64_t>(layer)   * bytes_per_layer_i +
                static_cast<uint64_t>(kv_head) * bytes_per_head_i;
            const uint64_t v_va = m_alloc->encode_va(
                KvRegion::Value, request_ids[i], kv_head_base_i + block_offset);
            commands.emplace_back(static_cast<Addr_t>(v_va),
                                  Request::Type::PIM_MAC_AB,
                                  request_ids[i], on_complete);
          }
        } else {  // BankStripe.
          const uint64_t v_va = m_alloc->encode_va(
              KvRegion::Value, anchor_req, kv_head_base + block_offset);
          commands.emplace_back(static_cast<Addr_t>(v_va),
                                Request::Type::PIM_MAC_AB,
                                anchor_req, on_complete);
        }
      }

      // 1× shared final MV_SB + BARRIER.
      commands.emplace_back(
          static_cast<Addr_t>(control_addr(kSharedControlReqId, layer, kv_head, 3)),
          Request::Type::PIM_MV_SB, anchor_req, on_complete);
      // BARRIER is fire-and-forget: empty callback so it does not decrement
      // the outstanding-counter.
      commands.emplace_back(
          static_cast<Addr_t>(control_addr(kSharedControlReqId, layer, kv_head, 4)),
          Request::Type::PIM_BARRIER, anchor_req,
          std::function<void(Request&)>{});
    }
  }

  return commands;
}

}  // namespace Ramulator::ServingOnline
