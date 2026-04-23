/**
 * serving_online/pim_commands.h
 *
 * Runtime LPDDR5 PIM command generator for pi0 attention decode steps.
 *
 * For each decode token, pi0 attention performs (per layer, per head):
 *
 *   Score matmul:   Q[1x256] x K^T[Lx256] -> score[1xL]   (read K cache)
 *   Softmax:        score[1xL] -> prob[1xL]                (PIM softmax)
 *   Context matmul: prob[1xL] x V[Lx256]  -> ctx[1x256]   (read V cache)
 *
 * This class generates coarse Request tiles for the above operations. Virtual
 * addresses are produced via PimAllocator::encode_va() and are translated to
 * physical addresses at issue time by the translation layer.
 *
 * Command granularity:
 *   One Read per (layer, head) K row and one per V row. Absolute decode
 *   latency comes from the cost table; these Reads only exercise the bank
 *   scheduler for contention modelling. The callback on each Request
 *   decrements a caller-supplied counter so the frontend knows when all
 *   commands for a decode step have completed.
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
  struct Params {
    int num_layers  = 18;    // number of attention layers (pi0 = 18)
    int num_heads   = 8;     // attention heads per layer (pi0 = 8)
    int d_head      = 256;   // key/value head dimension in elements (pi0 = 256)
    int dtype_bytes = 2;     // bytes per element: 2 = FP16
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
  Params        m_p;
  PimAllocator* m_alloc;
  int           m_read_type_id;
  int           m_pim_type_id;
};

}  // namespace Ramulator::ServingOnline

#endif  // RAMULATOR_FRONTEND_IMPL_SERVING_ONLINE_PIM_COMMANDS_H
