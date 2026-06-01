#ifndef RAMULATOR_TRANSLATION_IMPL_ATTACC_SERVING_TRANSLATION_H
#define RAMULATOR_TRANSLATION_IMPL_ATTACC_SERVING_TRANSLATION_H

#include <memory>

#include "frontend/impl/serving/allocator.h"
#include "frontend/impl/serving/pim_allocator.h"

namespace Ramulator {

class IAttAccServingTranslationControl {
 public:
  virtual ~IAttAccServingTranslationControl() = default;

  // Legacy path — used by template_replay and old runtime_generated bridge.
  virtual void attach_state(std::shared_ptr<Serving::ServingAddressSpace> state) = 0;

  // New path — attach the clean PimAllocator for VA→PA translation.
  // When attached, addresses with bit 63 = 1 are resolved through PimAllocator.
  // Addresses with bit 63 = 0 are passed through as raw physical addresses.
  virtual void attach_pim_allocator(std::shared_ptr<Serving::PimAllocator> allocator) = 0;
};

}  // namespace Ramulator

#endif  // RAMULATOR_TRANSLATION_IMPL_ATTACC_SERVING_TRANSLATION_H
