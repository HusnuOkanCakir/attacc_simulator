#ifndef RAMULATOR_FRONTEND_IMPL_SERVING_COMMAND_GENERATOR_H
#define RAMULATOR_FRONTEND_IMPL_SERVING_COMMAND_GENERATOR_H

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "base/request.h"
#include "frontend/impl/serving/layout_planner.h"

namespace Ramulator::Serving {

enum class CommandTargetKind : uint8_t {
  Passthrough = 0,
  KvKey = 1,
  KvValue = 2,
};

struct LogicalCommand {
  int type_id = Request::Type::Read;
  CommandTargetKind target = CommandTargetKind::Passthrough;
  std::optional<LogicalKvAddress> kv_address;
  uint64_t passthrough_addr = 0;
};

struct GeneratedCommand {
  int type_id = Request::Type::Read;
  Addr_t addr = 0;
  bool expects_callback = false;
};

class HybridAddressEncoder {
 public:
  virtual ~HybridAddressEncoder() = default;
  virtual Addr_t encode(const LogicalCommand& cmd, int request_id) const = 0;
};

class VirtualAddressEncoder final : public HybridAddressEncoder {
 public:
  explicit VirtualAddressEncoder(std::shared_ptr<const RepresentativeKvLayout> layout);

  Addr_t encode(const LogicalCommand& cmd, int request_id) const override;

 private:
  std::shared_ptr<const RepresentativeKvLayout> m_layout;
};

class HybridCommandStream {
 public:
  virtual ~HybridCommandStream() = default;

  virtual bool next(GeneratedCommand& out) = 0;
  virtual bool done() const = 0;
};

class LogicalTemplateCommandStream final : public HybridCommandStream {
 public:
  LogicalTemplateCommandStream(const std::vector<LogicalCommand>* logical_trace,
                               std::shared_ptr<const HybridAddressEncoder> encoder,
                               int request_id);

  bool next(GeneratedCommand& out) override;
  bool done() const override;

 private:
  const std::vector<LogicalCommand>* m_logical_trace = nullptr;
  std::shared_ptr<const HybridAddressEncoder> m_encoder;
  int m_request_id = -1;
  size_t m_next_idx = 0;
};

struct GeneratedHybridArtifacts {
  std::vector<LogicalCommand> commands;
  std::shared_ptr<const RepresentativeKvLayout> kv_layout;
};

GeneratedHybridArtifacts build_lpddr5_bank_generated_artifacts(const GeneratorGeometry& geometry,
                                                               int context_tokens,
                                                               uint64_t page_size_bytes);

std::vector<LogicalCommand> build_lpddr5_bank_logical_trace(const GeneratorGeometry& geometry,
                                                            int context_tokens);

}  // namespace Ramulator::Serving

#endif  // RAMULATOR_FRONTEND_IMPL_SERVING_COMMAND_GENERATOR_H
