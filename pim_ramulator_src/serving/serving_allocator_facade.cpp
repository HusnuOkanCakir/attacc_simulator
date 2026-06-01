#include <limits>
#include <utility>

#include "frontend/impl/serving/serving_allocator_facade.h"

namespace Ramulator::Serving {

namespace {

constexpr uint8_t kKvKeyObjectId = 0;
constexpr uint8_t kKvValueObjectId = 1;
constexpr uint8_t kScoreObjectId = 2;
constexpr uint8_t kContextObjectId = 3;
constexpr uint8_t kScratchObjectId = 4;

}  // namespace

RequestMemoryFootprint ServingAllocatorFacade::request_footprint(int context_tokens) const {
  return plan_request_memory(m_geometry, context_tokens);
}

TaskMemoryFootprint ServingAllocatorFacade::task_footprint(int decode_context_tokens,
                                                           int batch_size) const {
  return plan_decode_task_memory(m_geometry, decode_context_tokens, batch_size);
}

AllocationOutcome ServingAllocatorFacade::can_fit_request_kv(
    uint32_t request_id,
    int capacity_context_tokens,
    const RepresentativeKvLayout& layout) const {
  if (!m_state) {
    return {true, ""};
  }

  const auto fp = request_footprint(capacity_context_tokens);
  auto budget = m_state->snapshot_budget();
  auto out = m_state->forecast_kv_object(
      budget, MemoryRegionKind::KvKey, request_id, kKvKeyObjectId, fp.kv_key_bytes, layout.key);
  if (!out.success) {
    return out;
  }
  return m_state->forecast_kv_object(budget,
                                     MemoryRegionKind::KvValue,
                                     request_id,
                                     kKvValueObjectId,
                                     fp.kv_value_bytes,
                                     layout.value);
}

AllocationOutcome ServingAllocatorFacade::can_fit_decode_task(const std::vector<int>& request_ids,
                                                              int decode_context_tokens,
                                                              int batch_size) const {
  if (!m_state || request_ids.empty()) {
    return {true, ""};
  }

  auto budget = m_state->snapshot_budget();
  const auto fp = task_footprint(decode_context_tokens, batch_size);
  const uint64_t nreq = static_cast<uint64_t>(request_ids.size());
  const uint64_t score_per_req = align_up(fp.score_bytes, nreq) / nreq;
  const uint64_t context_per_req = align_up(fp.context_bytes, nreq) / nreq;
  const uint64_t scratch_per_req = align_up(fp.scratch_bytes, nreq) / nreq;

  const std::vector<std::pair<MemoryRegionKind, uint8_t>> regions = {
      {MemoryRegionKind::Score, kScoreObjectId},
      {MemoryRegionKind::Context, kContextObjectId},
      {MemoryRegionKind::Scratch, kScratchObjectId},
  };

  for (size_t request_slot = 0; request_slot < request_ids.size(); ++request_slot) {
    for (const auto& [region, object_id] : regions) {
      (void)object_id;
      const uint64_t bytes = (region == MemoryRegionKind::Score)
                                 ? score_per_req
                                 : (region == MemoryRegionKind::Context ? context_per_req
                                                                       : scratch_per_req);
      const auto layout =
          build_transient_object_layout(m_geometry, bytes, static_cast<uint16_t>(request_slot),
                                        m_state->config().page_size_bytes);
      auto out = m_state->forecast_transient_object(budget, region, bytes, layout);
      if (!out.success) {
        return out;
      }
    }
  }

  return {true, ""};
}

AllocationOutcome ServingAllocatorFacade::reserve_request_kv(
    uint32_t request_id,
    int capacity_context_tokens,
    const RepresentativeKvLayout& layout) {
  const auto fp = request_footprint(capacity_context_tokens);
  auto out = m_state->reserve_kv_object(
      MemoryRegionKind::KvKey, request_id, kKvKeyObjectId, fp.kv_key_bytes, layout.key);
  if (!out.success) {
    return out;
  }

  out = m_state->reserve_kv_object(MemoryRegionKind::KvValue,
                                   request_id,
                                   kKvValueObjectId,
                                   fp.kv_value_bytes,
                                   layout.value);
  if (!out.success) {
    m_state->release_object(MemoryRegionKind::KvKey, request_id, kKvKeyObjectId);
  }
  return out;
}

void ServingAllocatorFacade::release_request_kv(uint32_t request_id) {
  m_state->release_object(MemoryRegionKind::KvKey, request_id, kKvKeyObjectId);
  m_state->release_object(MemoryRegionKind::KvValue, request_id, kKvValueObjectId);
}

AllocationOutcome ServingAllocatorFacade::reserve_decode_task(const std::vector<int>& request_ids,
                                                              int decode_context_tokens,
                                                              int batch_size) {
  if (request_ids.empty()) {
    return {true, ""};
  }

  const auto fp = task_footprint(decode_context_tokens, batch_size);
  const uint64_t nreq = static_cast<uint64_t>(request_ids.size());
  const uint64_t score_per_req = align_up(fp.score_bytes, nreq) / nreq;
  const uint64_t context_per_req = align_up(fp.context_bytes, nreq) / nreq;
  const uint64_t scratch_per_req = align_up(fp.scratch_bytes, nreq) / nreq;

  const std::vector<std::pair<MemoryRegionKind, uint8_t>> regions = {
      {MemoryRegionKind::Score, kScoreObjectId},
      {MemoryRegionKind::Context, kContextObjectId},
      {MemoryRegionKind::Scratch, kScratchObjectId},
  };

  for (size_t request_slot = 0; request_slot < request_ids.size(); ++request_slot) {
    const auto request_id = static_cast<uint32_t>(request_ids[request_slot]);
    std::vector<std::pair<MemoryRegionKind, uint8_t>> reserved_for_request;
    for (const auto& [region, object_id] : regions) {
      const uint64_t bytes = (region == MemoryRegionKind::Score)
                                 ? score_per_req
                                 : (region == MemoryRegionKind::Context ? context_per_req
                                                                       : scratch_per_req);
      const auto layout =
          build_transient_object_layout(m_geometry, bytes, static_cast<uint16_t>(request_slot),
                                        m_state->config().page_size_bytes);
      auto out = m_state->reserve_transient_object(region, request_id, object_id, bytes, layout);
      if (!out.success) {
        for (const auto& [rollback_region, rollback_object_id] : reserved_for_request) {
          m_state->release_object(rollback_region, request_id, rollback_object_id);
        }
        for (size_t prior_slot = 0; prior_slot < request_slot; ++prior_slot) {
          const auto prior_request_id = static_cast<uint32_t>(request_ids[prior_slot]);
          m_state->release_object(MemoryRegionKind::Score, prior_request_id, kScoreObjectId);
          m_state->release_object(MemoryRegionKind::Context, prior_request_id, kContextObjectId);
          m_state->release_object(MemoryRegionKind::Scratch, prior_request_id, kScratchObjectId);
        }
        return out;
      }
      reserved_for_request.emplace_back(region, object_id);
    }
  }

  return {true, ""};
}

void ServingAllocatorFacade::release_decode_task(const std::vector<int>& request_ids) {
  for (int request_id : request_ids) {
    const auto request_id_u32 = static_cast<uint32_t>(request_id);
    m_state->release_object(MemoryRegionKind::Score, request_id_u32, kScoreObjectId);
    m_state->release_object(MemoryRegionKind::Context, request_id_u32, kContextObjectId);
    m_state->release_object(MemoryRegionKind::Scratch, request_id_u32, kScratchObjectId);
  }
}

}  // namespace Ramulator::Serving
