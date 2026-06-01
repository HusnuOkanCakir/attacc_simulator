#ifndef RAMULATOR_FRONTEND_IMPL_SERVING_SERVING_ALLOCATOR_FACADE_H
#define RAMULATOR_FRONTEND_IMPL_SERVING_SERVING_ALLOCATOR_FACADE_H

#include <memory>
#include <utility>
#include <vector>

#include "frontend/impl/serving/address_space.h"

namespace Ramulator::Serving {

class ServingAllocatorFacade {
 public:
  explicit ServingAllocatorFacade(std::shared_ptr<ServingAddressSpace> state,
                                  GeneratorGeometry geometry)
      : m_state(std::move(state)), m_geometry(std::move(geometry)) {}

  std::shared_ptr<ServingAddressSpace> state() const { return m_state; }

  RequestMemoryFootprint request_footprint(int context_tokens) const;
  TaskMemoryFootprint task_footprint(int decode_context_tokens, int batch_size) const;

  AllocationOutcome can_fit_request_kv(uint32_t request_id,
                                       int capacity_context_tokens,
                                       const RepresentativeKvLayout& layout) const;
  AllocationOutcome can_fit_decode_task(const std::vector<int>& request_ids,
                                        int decode_context_tokens,
                                        int batch_size) const;

  AllocationOutcome reserve_request_kv(uint32_t request_id,
                                       int capacity_context_tokens,
                                       const RepresentativeKvLayout& layout);
  void release_request_kv(uint32_t request_id);

  AllocationOutcome reserve_decode_task(const std::vector<int>& request_ids,
                                        int decode_context_tokens,
                                        int batch_size);
  void release_decode_task(const std::vector<int>& request_ids);

 private:
  std::shared_ptr<ServingAddressSpace> m_state;
  GeneratorGeometry m_geometry;
};

}  // namespace Ramulator::Serving

#endif  // RAMULATOR_FRONTEND_IMPL_SERVING_SERVING_ALLOCATOR_FACADE_H
