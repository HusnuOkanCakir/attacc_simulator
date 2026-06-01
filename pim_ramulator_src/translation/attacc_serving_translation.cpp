#include <memory>

#include "base/exception.h"
#include "translation/translation.h"
#include "translation/impl/attacc_serving_translation.h"

namespace Ramulator {

class AttAccServingTranslation : public ITranslation,
                                 public Implementation,
                                 public IAttAccServingTranslationControl {
  RAMULATOR_REGISTER_IMPLEMENTATION(ITranslation,
                                    AttAccServingTranslation,
                                    "AttAccServingTranslation",
                                    "Deterministic VA->PA translation for realtime AttAcc serving.");

 public:
  void init() override {
    m_max_paddr = param<Addr_t>("max_addr").desc("Max physical address.").required();
    m_page_size_bytes =
        (param<Addr_t>("pagesize_KB").desc("Page size in KB.").default_val(4) << 10);
    m_logger = Logging::create_logger("AttAccServingTranslation");
  }

  void attach_state(std::shared_ptr<Serving::ServingAddressSpace> state) override {
    m_state = std::move(state);
  }

  // Attach the new PimAllocator. When set, addresses with bit 63 = 1 are
  // resolved through it; bit 63 = 0 addresses fall through to the legacy path.
  void attach_pim_allocator(std::shared_ptr<Serving::PimAllocator> allocator) override {
    m_pim_allocator = std::move(allocator);
  }

  bool translate(Request& req) override {
    const uint64_t addr = static_cast<uint64_t>(req.addr);

    // ---- New path: PimAllocator VA (bit 63 = 1) ----
    if (m_pim_allocator && (addr >> 63)) {
      uint64_t pa = 0;
      if (!m_pim_allocator->translate_va(addr, pa)) {
        m_logger->warn("PimAllocator: failed to translate VA {:#x} for source {}", addr, req.source_id);
        return false;
      }
      if (static_cast<Addr_t>(pa) >= m_max_paddr) {
        m_logger->warn("PimAllocator: translated address {:#x} exceeds max_paddr {:#x}", pa, m_max_paddr);
        return false;
      }
      req.addr = static_cast<Addr_t>(pa);
      return true;
    }

    // ---- Legacy path: ServingAddressSpace VA (used by template_replay) ----
    if (!m_state) {
      // No allocator attached at all — pass address through unchanged.
      return true;
    }

    // Passthrough commands that are already physical addresses stay unchanged.
    if (!Serving::decode_virtual_address(req.addr).has_value()) {
      return true;
    }

    Addr_t translated = 0;
    if (!m_state->translate_virtual(req.addr, translated)) {
      m_logger->warn("ServingAddressSpace: failed to translate VA {} for source {}", req.addr, req.source_id);
      return false;
    }
    if (translated >= m_max_paddr) {
      m_logger->warn("ServingAddressSpace: translated address {} exceeds max_paddr {}", translated, m_max_paddr);
      return false;
    }
    req.addr = translated;
    return true;
  }

  Addr_t get_max_addr() override { return m_max_paddr; }

 private:
  std::shared_ptr<Serving::PimAllocator>        m_pim_allocator;  // new clean path
  std::shared_ptr<Serving::ServingAddressSpace> m_state;          // legacy path
  Addr_t m_max_paddr       = 0;
  Addr_t m_page_size_bytes = 4096;
};

}  // namespace Ramulator
