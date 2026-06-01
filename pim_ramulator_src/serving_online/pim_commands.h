/**
 * serving_online/pim_commands.h
 *
 * Runtime LPDDR5-PIM command generator for attention decode steps.
 *
 * For each decode token, attention performs (per layer, per head):
 *
 *   Score matmul:   Q[1xD] x K^T[LxD] -> score[1xL]
 *   Softmax:        score[1xL] -> prob[1xL]
 *   Context matmul: prob[1xL] x V[LxD] -> ctx[1xD]
 *
 * Realistic command sequence per (layer, head), at cache-line block
 * granularity (block = kLineBytes / dtype_bytes tokens):
 *
 *   K-side:  for each block: PIM_WR_GB(K) -> PIM_MAC_AB(K)
 *            PIM_MV_SB (collect partial scores)
 *            PIM_SFM   (softmax)
 *   V-side:  for each block: PIM_MV_GB(V) -> PIM_MAC_AB(V)
 *            PIM_MV_SB (collect partial contexts)
 *            PIM_BARRIER
 *
 * K/V data commands address PIM DRAM via PimAllocator::encode_va(); control
 * commands use a synthesized control_addr() that lies outside the KV VA
 * space (bit 63 unset) so the translation layer passes it through. Absolute
 * decode latency still comes from the cost table; the realistic command
 * stream drives the LPDDR5-PIM bank-state machine for contention modelling.
 */

#ifndef RAMULATOR_FRONTEND_IMPL_SERVING_ONLINE_PIM_COMMANDS_H
#define RAMULATOR_FRONTEND_IMPL_SERVING_ONLINE_PIM_COMMANDS_H

#include <cstdint>
#include <functional>
#include <vector>

#include "base/request.h"
#include "frontend/impl/serving_online/pim_allocator.h"

namespace Ramulator::ServingOnline {

class PimCommandGen {
 public:
  // Mode selects per-decode-step command stream complexity. See
  // ServingOnlineConfig::pim_command_mode for the semantics. Diagnostic
  // toggle for bisecting wall-clock cost (DRAM-tick vs frontend overhead).
  enum class Mode {
    Realistic,         // full per-block sequence, emitted PER REQUEST (default).
                       // Total cmds for a bs=B step = B × N_L × N_KVH × (4·blocks+4).
    Simple,            // N_L × N_KVH × 2 PIM_MAC_AB per request only.
    AttAccQBroadcast,  // AttAcc-faithful: 1 shared Q broadcast + B per-request
                       // K MACs + shared softmax + 1 shared V broadcast + B
                       // per-request V MACs, per (layer, kv_head). K/V data
                       // is still per-request (unique cache), but control +
                       // Q/V broadcasts are amortized across the batch.
                       // Total cmds ≈ N_L × N_KVH × (2·blocks·B + 6).
    BankStripe,        // EXPERIMENTAL: one shared PIM_MAC_AB per block hits
                       // all batched requests in parallel (relies on the
                       // channel-interleaved allocator to disperse requests
                       // across banks/channels). Total cmds ≈ N_L × N_KVH ×
                       // (4·blocks + 4), independent of B up to num_channels.
                       // Does not change command count when bs=1.
  };

  struct Params {
    int  num_layers   = 18;    // number of attention layers (pi0 = 18)
    int  num_heads    = 8;     // Q-projection heads per layer (pi0 = 8)
    int  num_kv_heads = 1;     // K/V-projection heads per layer; MHA when == num_heads, MQA when 1
    int  d_head       = 256;   // key/value head dimension in elements (pi0 = 256)
    int  dtype_bytes  = 2;     // bytes per element: 2 = FP16
    Mode mode         = Mode::Realistic;
  };

  /**
   * @param p            Model parameters (pi0 defaults).
   * @param alloc        Allocator that owns the KV VA space. Must outlive this object.
   * @param read_type_id Request::Type::Read (0) -- used for cache-line reads.
   * @param pim_type_id  Request::Type::PIM_MAC_AB (4) -- used for bank-mode MACs.
   */
  PimCommandGen(const Params& p,
                PimAllocator* alloc,
                int read_type_id = Request::Type::Read,
                int pim_type_id  = Request::Type::PIM_MAC_AB);

  /**
   * Generate all memory requests for one attention decode step (single request).
   *
   * For each layer and KV head, emits one Read for the full K row and one
   * for the full V row. Total = num_layers * num_kv_heads * 2 requests in
   * Simple mode; Realistic mode emits per-block K/V command sequences.
   *
   * Each returned Request has its callback set to `on_complete` so the caller
   * can count outstanding requests and detect when the step is finished.
   *
   * For modes that share commands across batched requests
   * (AttAccQBroadcast, BankStripe), use `decode_step_batched()` instead; this
   * single-request signature emits a bs=1 stream regardless of mode.
   *
   * @param request_id    Request identifier (must have KV allocated in PimAllocator).
   * @param context_tokens  Number of KV tokens the row covers (for VA encoding).
   * @param on_complete   Called once per request when it departs the memory system.
   * @return              Vector of Requests ready to be sent to m_memory_system->send().
   */
  std::vector<Request> decode_step(int request_id,
                                   int context_tokens,
                                   std::function<void(Request&)> on_complete) const;

  /**
   * Generate all memory requests for one BATCHED attention decode step.
   *
   * Behavior depends on `Mode`:
   *   - Realistic / Simple: equivalent to concatenating decode_step() outputs
   *     for each (request_id, ctx) pair. Cmd count scales linearly with bs.
   *   - AttAccQBroadcast: emits 1 shared Q broadcast + B per-request K MACs +
   *     1 shared softmax + 1 shared V broadcast + B per-request V MACs per
   *     (layer, kv_head). Q/V/control commands are amortized across the batch.
   *   - BankStripe: emits 1 shared PIM_MAC_AB per block per (layer, kv_head)
   *     that targets request 0's VA (channel-interleaved layout disperses
   *     other requests' data onto neighbouring channels naturally).
   *
   * `context_tokens` is per request. The cmd stream uses
   * max_over_batch(context_tokens) to size the block count, matching real
   * bank-PIM hardware where all banks step in lockstep.
   *
   * Pre-condition: request_ids.size() == context_tokens.size() and >= 1.
   */
  std::vector<Request> decode_step_batched(
      const std::vector<int>& request_ids,
      const std::vector<int>& context_tokens,
      std::function<void(Request&)> on_complete) const;

  // Returns the number of Requests that decode_step() will produce for the
  // given parameters (without actually generating them). Useful for sizing
  // the pending-callback counter before issuing.
  int decode_step_count(int context_tokens) const;

  // Batched analogue of decode_step_count(). Uses the max context across the
  // batch (matching decode_step_batched()'s block-size choice) and the
  // current Mode to compute the exact cmd count.
  int decode_step_count_batched(const std::vector<int>& context_tokens) const;

 private:
  // Synthesize a non-KV "control" address for PIM_MV_SB / PIM_SFM /
  // PIM_BARRIER requests. Bit 63 is unset so the translation layer leaves
  // it untouched. Layout: phase_tag<<28 | request_id<<8 | layer<<3 | head.
  uint64_t control_addr(int request_id, int layer, int head, int phase_tag) const;

  static constexpr int kLineBytes = 64;  // LPDDR5 burst size / cache-line bytes

  Params        m_p;
  PimAllocator* m_alloc;
  int           m_read_type_id;
  int           m_pim_type_id;
};

}  // namespace Ramulator::ServingOnline

#endif  // RAMULATOR_FRONTEND_IMPL_SERVING_ONLINE_PIM_COMMANDS_H
