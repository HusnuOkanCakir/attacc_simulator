#include <algorithm>
#include <memory>
#include <utility>

#include "base/exception.h"
#include "frontend/impl/serving/command_generator_internal.h"

namespace Ramulator::Serving {

// Address encoders
VirtualAddressEncoder::VirtualAddressEncoder(std::shared_ptr<const RepresentativeKvLayout> layout)
    : m_layout(std::move(layout)) {}

Addr_t VirtualAddressEncoder::encode(const LogicalCommand& cmd, int request_id) const {
  if (cmd.target == CommandTargetKind::Passthrough) {
    return static_cast<Addr_t>(cmd.passthrough_addr);
  }
  if (!m_layout || !cmd.kv_address.has_value()) {
    throw ConfigurationError("VirtualAddressEncoder: missing layout or logical KV address");
  }

  const auto region = region_for_target(cmd.target);
  const auto& object_layout = (region == MemoryRegionKind::KvKey) ? m_layout->key : m_layout->value;
  const auto logical_offset =
      logical_offset_for_kv_address(object_layout, *cmd.kv_address, m_layout->page_size_bytes);
  if (!logical_offset.has_value()) {
    throw ConfigurationError("VirtualAddressEncoder: no logical KV mapping for access");
  }

  const uint8_t object_id = (region == MemoryRegionKind::KvKey) ? 0 : 1;
  return encode_virtual_address(region,
                                static_cast<uint32_t>(std::max(0, request_id)),
                                object_id,
                                *logical_offset);
}

// Command stream
LogicalTemplateCommandStream::LogicalTemplateCommandStream(
    const std::vector<LogicalCommand>* logical_trace,
    std::shared_ptr<const HybridAddressEncoder> encoder,
    int request_id)
    : m_logical_trace(logical_trace), m_encoder(std::move(encoder)), m_request_id(request_id) {}

bool LogicalTemplateCommandStream::next(GeneratedCommand& out) {
  if (done()) {
    return false;
  }
  const auto& cmd = (*m_logical_trace)[m_next_idx++];
  out.type_id = cmd.type_id;
  out.addr = m_encoder->encode(cmd, m_request_id);
  out.expects_callback = expects_callback(cmd.type_id);
  return true;
}

bool LogicalTemplateCommandStream::done() const {
  return m_logical_trace == nullptr || m_next_idx >= m_logical_trace->size();
}

}  // namespace Ramulator::Serving
