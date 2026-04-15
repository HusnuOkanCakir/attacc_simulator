# Realtime Serving Memory Allocation Policy

This note describes the **current implemented memory-allocation policy** for the
realtime serving frontend in:

- `ramulator2/src/frontend/impl/serving/`

It describes what the code does today, not the longer-term target design.

## Scope

The allocator is only active for:

- `hybrid_command_source = runtime_generated`

It is not used for:

- `template_replay`
- pure `gpu_only` execution

Relevant entry points:

- `ramulator2/src/frontend/impl/serving/realtime_serving_frontend.cpp`
- `ramulator2/src/frontend/impl/serving/runtime.cpp`
- `ramulator2/src/frontend/impl/serving/allocator.cpp`
- `ramulator2/src/frontend/impl/serving/layout_planner.cpp`
- `ramulator2/src/translation/impl/attacc_serving_translation.cpp`

## High-Level Model

The current allocator is **not** a general OS-style allocator. It is a:

1. deterministic region-partitioned PIM-side capacity manager
2. virtual-to-physical translation state shared with the serving runtime
3. parity-preserving bridge for KV placement

So the current policy does two different jobs:

- track whether there is enough PIM-visible memory to run a hybrid request
- translate structured virtual addresses used by `runtime_generated`

## When The Allocator Is Enabled

Static validation in `ramulator2/src/frontend/impl/serving/inputs.cpp` requires:

- `hybrid_command_source` must be `template_replay` or `runtime_generated`
- if it is `runtime_generated`:
  - `generator_backend` must be `lpddr5_bank`
  - `translation_max_addr` must be set
  - `kv_pool_bytes` must be set

In `ramulator2/src/frontend/impl/serving/runtime.cpp`, the runtime creates:

- `ServingAddressSpace`
- `ServingAllocatorFacade`

only when `use_runtime_generated_hybrid()` is true.

In `ramulator2/src/frontend/impl/serving/realtime_serving_frontend.cpp`, the
frontend also creates the translation child only in `runtime_generated` mode and
attaches the shared allocator state to it.

## What Memory Is Modeled

The allocator models only the **Ramulator-visible PIM side**. It does not model
GPU memory.

Logical memory regions are defined in
`ramulator2/src/frontend/impl/serving/layout_planner.h`:

- `KvKey`
- `KvValue`
- `Score`
- `Context`
- `Scratch`

These regions are used both for:

- capacity tracking
- structured virtual address encoding

## Footprint Formulas

The footprint formulas are implemented in
`ramulator2/src/frontend/impl/serving/layout_planner.cpp`.

### Request-level persistent footprint

`plan_request_memory()` computes:

- `kv_key_bytes = num_layers * context_tokens * heads_per_hbm * dhead * dtype_bytes`
- `kv_value_bytes = kv_key_bytes`

This is the persistent memory reserved when a hybrid request is admitted.

### Decode-task transient footprint

`plan_decode_task_memory()` computes:

- `score_bytes = batch_size * decode_context_tokens * heads_per_hbm * dtype_bytes`
- `context_bytes = batch_size * heads_per_hbm * dhead * dtype_bytes`
- `scratch_bytes = max(score_bytes, context_bytes)`

This is the transient memory reserved for a hybrid decode task.

## Pool Structure

The region pools are built in
`ramulator2/src/frontend/impl/serving/allocator.cpp`.

The configured pools are:

- `kv_pool_bytes`
- `score_pool_bytes`
- `context_pool_bytes`
- `scratch_pool_bytes`

KV is split internally into:

- half for `KvKey`
- half for `KvValue`

The allocator then creates one contiguous page-number range per region. The pool
layout is deterministic:

1. `KvKey`
2. `KvValue`
3. `Score`
4. `Context`
5. `Scratch`

If the total configured capacity exceeds `translation_max_addr`, initialization
fails.

## Page Size And Allocation Style

The current allocator uses:

- fixed 4 KB pages by default
- deterministic first-fit allocation over free ranges

There is currently:

- no huge page support
- no fragmentation modeling beyond simple free ranges
- no migration
- no compaction
- no eviction

Allocation works by scanning the region’s free-range map and taking the first
range large enough to satisfy the object.

If no range is large enough, allocation returns:

- `pim_capacity_exceeded`

## Object Model

Objects are keyed by:

- region
- request id
- object id

Current object ids are effectively:

- `0`: KV key
- `1`: KV value
- `2`: score
- `3`: context
- `4`: scratch

This object model is implemented in
`ramulator2/src/frontend/impl/serving/allocator.cpp`.

## Request-Lifetime Policy

The runtime applies the allocator in two stages.

### 1. Persistent request allocation

When a request is admitted on the hybrid route, the runtime calls
`reserve_hybrid_request_memory()` in
`ramulator2/src/frontend/impl/serving/runtime.cpp`.

That function:

1. gets the generated artifacts for the mapped `lin`
2. reserves KV key/value through `reserve_request_kv()`
3. records `pim_kv_bytes`
4. updates allocator peak stats

If allocation fails:

- the runtime records the allocator block reason
- it tries to fall back to `gpu_only`
- if GPU fallback is viable, the request is admitted on GPU
- otherwise the request is dropped

So current request-level policy under hybrid KV pressure is:

- prefer hybrid
- fall back to GPU if possible
- otherwise drop

### 2. Transient decode-task allocation

When a hybrid decode task is about to start, the runtime calls
`reserve_hybrid_task_memory()` in
`ramulator2/src/frontend/impl/serving/runtime.cpp`.

That function:

1. computes `batch_lin` as the maximum active decode context across the task
2. reserves score/context/scratch through `reserve_decode_task()`
3. records per-request transient peak bytes
4. updates allocator peak stats

If transient allocation fails:

- the task is not started
- `allocator_hold_count` is incremented
- the runtime records the fallback reason
- it may attempt decode rebind to GPU
- otherwise the request remains waiting and retries later

So current task-level policy under hybrid transient pressure is:

- hold/retry
- optionally rebind to GPU
- do not force immediate drop

## Release Policy

Release points are also explicit in `runtime.cpp`.

### Request KV release

KV is released when the request finishes:

- after prefill if there is no decode work left
- after the final decode token completes

### Decode-task transient release

Score/context/scratch are released immediately when the active hybrid decode task
completes.

So current lifetime policy is:

- KV lives for the full request lifetime
- score/context/scratch live only for one decode task

## Virtual Address Format

The virtual address format is defined in
`ramulator2/src/frontend/impl/serving/layout_planner.cpp`.

A structured VA contains:

- region
- request id
- object id
- page index
- page offset

This is encoded by `encode_virtual_address()` and decoded by
`decode_virtual_address()`.

This means the runtime-generated serving path no longer sends only raw physical
addresses. It can send structured addresses that the serving translation resolves
at runtime.

## Translation Path

The serving translation lives in:

- `ramulator2/src/translation/impl/attacc_serving_translation.cpp`

Its behavior is:

1. if the request address is not a serving VA, leave it unchanged
2. if it is a serving VA:
   - decode the fields
   - ask `ServingAddressSpace::translate_virtual()`
   - reject the request if translation fails
   - reject the request if the translated address exceeds `max_addr`
   - otherwise replace the VA with the translated PA

So the translation policy is:

- structured VA only for serving-managed objects
- passthrough for non-VA addresses
- hard fail on missing allocation or out-of-range PA

## The Most Important Nuance: KV Placement Is Still Parity-Preserving

This is the key detail.

The allocator does capacity accounting using the new region pools, but the
translated physical pages for KV can still come from the generated
`RepresentativeKvLayout` page ordering.

That layout is built in:

- `ramulator2/src/frontend/impl/serving/command_generator.cpp`

What happens there:

1. logical KV commands are generated
2. the old legacy physical address for each KV command is reconstructed
3. the accessed physical pages are collected
4. those pages are compacted into a dense order
5. `VirtualAddressEncoder` maps a logical KV access to a dense byte offset
6. translation maps that dense page index back to the stored physical page list

This means:

- the runtime is using a real VA path
- but the physical page mapping for accessed KV pages still preserves legacy
  physical parity

So the current implementation is best understood as:

- new allocator for capacity and lifetime
- new VA format for runtime-generated hybrid commands
- old physical KV page ordering preserved for parity

## What Is Not Yet Modeled

The current policy does not yet implement:

- a new AttAcc-aware physical placement policy
- huge pages
- fragmented mixed-size allocation
- migration
- compaction
- eviction
- GPU-side memory accounting
- direct command addressing for score/context/scratch

Also, the generator still models representative-layer accesses, while KV capacity
uses full-model scale. So the active translated page set and the full capacity
model are not yet fully symmetric.

## Observability

The allocator reports both per-request and summary metrics.

Per-request fields include:

- `pim_kv_bytes`
- `pim_transient_peak_bytes`
- `allocator_fallback_reason`
- `allocator_wait_ms`

Summary fields include:

- `pim_memory_peak_bytes`
- `allocator_hold_count`
- `allocator_gpu_fallback_count`
- `allocator_drop_count`
- peak bytes for:
  - `kv`
  - `score`
  - `context`
  - `scratch`

These are emitted by:

- `ramulator2/src/frontend/impl/serving/runtime_reporting.cpp`

## Short Summary

The current memory-allocation policy in realtime serving is:

- deterministic region-based page pools
- first-fit contiguous allocation per object
- persistent KV reserved at hybrid admission
- transient score/context/scratch reserved at hybrid decode dispatch
- GPU fallback or drop on persistent KV OOM
- hold/retry or decode rebind on transient decode-task OOM
- shared serving VA->PA translation state
- legacy-parity-preserving KV physical page mapping for accessed pages

So it is already a real allocator and translation path, but it is still a
**parity-preserving bridge**, not yet the final request-aware physical placement
policy.
