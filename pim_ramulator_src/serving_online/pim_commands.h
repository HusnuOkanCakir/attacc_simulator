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
    Realistic,   // full per-block sequence (default)
    Simple,      // num_layers × num_heads × 2 PIM_MAC_AB only
  };

  struct Params {
    int  num_layers  = 18;    // number of attention layers (pi0 = 18)
    int  num_heads   = 8;     // attention heads per layer (pi0 = 8)
    int  d_head      = 256;   // key/value head dimension in elements (pi0 = 256)
    int  dtype_bytes = 2;     // bytes per element: 2 = FP16
    Mode mode        = Mode::Realistic;
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
   * Generate all memory requests for one attention decode step.
   *
   * For each layer and head, emits one Read for the full K row and one for
   * the full V row. Total = num_layers * num_heads * 2 requests.
   *
   * Each returned Request has its callback set to `on_complete` so the caller
   * can count outstanding requests and detect when the step is finished.
   *
   * @param request_id    Request identifier (must have KV allocated in PimAllocator).
   * @param context_tokens  Number of KV tokens the row covers (for VA encoding).
   * @param on_complete   Called once per request when it departs the memory system.
   * @return              Vector of Requests ready to be sent to m_memory_system->send().
   */
  std::vector<Request> decode_step(int request_id,
                                   int context_tokens,
                                   std::function<void(Request&)> on_complete) const;

  // Returns the number of Requests that decode_step() will produce for the
  // given parameters (without actually generating them). Useful for sizing
  // the pending-callback counter before issuing.
  int decode_step_count(int context_tokens) const;

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
