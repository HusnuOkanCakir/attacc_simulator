# Realtime Serving Command Generation Pseudocode

This note summarizes the current `runtime_generated` command path in:

- `/home/okan/attacc_simulator/ramulator2/src/frontend/impl/serving/command_generator.cpp`
- `/home/okan/attacc_simulator/ramulator2/src/frontend/impl/serving/runtime.cpp`
- `/home/okan/attacc_simulator/ramulator2/src/frontend/impl/serving/pim_executor.cpp`
- `/home/okan/attacc_simulator/ramulator2/src/frontend/impl/serving/layout_planner.cpp`
- `/home/okan/attacc_simulator/ramulator2/src/translation/impl/attacc_serving_translation.cpp`

It is pseudocode, not a literal rewrite.

## 1. Runtime Entry

```text
function start_task(task):
  for each request in task.request_indices:
    if task.phase != Decode:
      continue

    if request.route != hybrid_route:
      continue

    if request.svc.uses_pim is false:
      continue

    if request.remaining_decode_tokens <= 0:
      continue

    artifacts = generated_artifacts_for_lin(request.svc.mapped_lin)

    job.command_stream =
      LogicalTemplateCommandStream(
        logical_trace = artifacts.commands,
        encoder = VirtualAddressEncoder(artifacts.kv_layout),
        request_id = request.request_id
      )
```

## 2. Artifact Cache

```text
function generated_artifacts_for_lin(lin):
  if lin exists in m_generated_artifacts:
    return cached artifacts

  artifacts =
    build_lpddr5_bank_generated_artifacts(
      geometry = generator_geometry(),
      context_tokens = lin,
      page_size_bytes = translation_page_size_bytes
    )

  cache artifacts by lin
  return artifacts
```

## 3. Main Builder

```text
function build_lpddr5_bank_generated_artifacts(geometry, context_tokens, page_size_bytes):
  L = max(1, context_tokens)
  dhead = max(1, geometry.dhead)
  heads_per_hbm = max(1, geometry.heads_per_hbm)
  channel_count = max(1, geometry.channel_count)
  dtype_bytes = max(1, geometry.dtype_bytes)
  n_mac = prefetch_bytes / dtype_bytes

  gs = LpddrGeometry(channel_count)
  partition_size = ceil(L * dhead / (pseudo_channels * ranks * bank_groups * banks))

  initialize empty command groups:
    score_wrgb
    score_mac
    score_mvsb
    sfm
    context_mvgb
    context_mac
    context_mvsb
    valid_channels

  num_itr = ceil(heads_per_hbm / channel_count)

  for itr in [0, num_itr):
    per_itr_channels = full channel_count or tail remainder
    attention(itr, per_itr_channels)

  barrier = one PIM_BARRIER per local channel

  total_cmd = interleave paired iterations
  if num_itr is odd:
    append final tail iteration

  layout = RepresentativeKvLayout(...)
  layout.partition_size_units = partition_size
  layout.value_base_addr = default_value_base_addr
  collect_dense_layout_from_commands(total_cmd, layout)

  return GeneratedHybridArtifacts(
    commands = total_cmd,
    kv_layout = layout
  )
```

## 4. Per-Iteration Builder

```text
function attention(itr, per_itr_channels):
  create empty groups for this iteration

  # score-side setup
  for ba_idx over banks:
    for col_idx over score write columns:
      for local_channel in per_itr_channels:
        emit PIM_WR_GB targeting KvKey

  # score-side MAC
  for n_idx over score_n:
    create one MAC group
    for k_idx over score_k:
      for local_channel in per_itr_channels:
        emit PIM_MAC_AB targeting KvKey

    if boundary reached:
      create one move group
      for bg_idx over bank groups:
        for rank over ranks:
          for local_channel in per_itr_channels:
            emit PIM_MV_SB targeting KvKey

  # softmax
  for local_channel in per_itr_channels:
    emit PIM_SFM as passthrough

  # context-side move from GB
  for rank over ranks:
    for bg_idx over bank groups:
      for col_idx over context columns:
        for local_channel in per_itr_channels:
          emit PIM_MV_GB targeting KvValue

  # context-side MAC and move
  for n_idx over context_n:
    create one MAC group
    for k_idx over context_k:
      for local_channel in per_itr_channels:
        emit PIM_MAC_AB targeting KvValue

    create one move group
    for ba_idx over banks:
      for rank over ranks:
        for local_channel in per_itr_channels:
          emit PIM_MV_SB targeting KvValue
```

## 5. Logical Address Construction

```text
function make_kv_address(head_itr, local_channel, slice_relative_addr, gs):
  out.layer_idx = 0
  out.head_itr = head_itr
  out.local_channel = local_channel

  rem = slice_relative_addr
  out.rank = rem / gs.rank
  rem = rem % gs.rank
  out.bank_group = rem / gs.bg
  rem = rem % gs.bg
  out.bank = rem / gs.ba
  rem = rem % gs.ba
  out.row_idx = rem / gs.row
  rem = rem % gs.row
  out.col_idx = rem

  return out
```

## 6. Dense KV Layout

```text
function collect_dense_layout_from_commands(commands, layout):
  key_pages = {}
  value_pages = {}

  for cmd in commands:
    if cmd has no kv_address:
      continue

    physical_addr = legacy_physical_address(layout, cmd.target, cmd.kv_address)

    if cmd.target is KvKey:
      register physical page in key_pages

    if cmd.target is KvValue:
      register physical page in value_pages

  layout.key = finalize_dense_layout(key_pages)
  layout.value = finalize_dense_layout(value_pages)
```

```text
function finalize_dense_layout(page_keys):
  order pages by:
    layer_idx
    head_itr
    local_channel
    physical_page

  assign dense_page 0, 1, 2, ...

  return:
    physical_pages
    dense_page_by_physical_page
    pages_per_layer
```

## 7. Logical Command Stream

```text
class LogicalTemplateCommandStream:
  next():
    if end of logical_trace:
      return false

    cmd = logical_trace[next_idx]
    next_idx += 1

    out.type_id = cmd.type_id
    out.addr = encoder.encode(cmd, request_id)
    out.expects_callback = (cmd.type_id != PIM_BARRIER)
    return true
```

## 8. Virtual Address Encoding

```text
function VirtualAddressEncoder.encode(cmd, request_id):
  if cmd.target is Passthrough:
    return cmd.passthrough_addr

  region = region_for_target(cmd.target)
  object_layout = layout.key or layout.value

  physical_addr = legacy_physical_address(layout, cmd.target, cmd.kv_address)
  dense_offset =
    dense_offset_for_physical_address(
      object_layout,
      physical_addr,
      cmd.kv_address.layer_idx,
      layout.page_size_bytes
    )

  object_id = 0 for KvKey, 1 for KvValue

  return encode_virtual_address(
    region,
    request_id,
    object_id,
    dense_offset
  )
```

## 9. Pump Into Ramulator

```text
function pump_pim_job(job):
  while true:
    generated = job.command_stream.next()
    if no command:
      break

    req = Request(
      addr = generated.addr,
      type = generated.type_id,
      source = request_id,
      callback = optional completion callback
    )

    if translation exists:
      translate(req)

    if memory_system.send(req) fails:
      stop pumping this cycle

    track outstanding callback-producing commands
```

## 10. Translation

```text
function translate(req):
  if req.addr is not a serving virtual address:
    return true

  fields = decode_virtual_address(req.addr)

  allocation = lookup by:
    region
    request_id
    object_id

  page_index -> physical_page
  req.addr = physical_page * page_size + page_offset

  return true on success
```

## 11. Command Sequence Summary

At a high level, the generated hybrid decode program is:

```text
score-side:
  PIM_WR_GB
  PIM_MAC_AB
  PIM_MV_SB

softmax:
  PIM_SFM

context-side:
  PIM_MV_GB
  PIM_MAC_AB
  PIM_MV_SB

ordering:
  PIM_BARRIER inserted between major pieces
  neighboring iterations may be interleaved
```

## 12. Short Summary

```text
lin
 -> generated_artifacts_for_lin(lin)
 -> build logical command groups
 -> assemble one total logical command stream
 -> build dense KV layout from accessed pages
 -> encode each logical command to VA or passthrough addr
 -> translate VA to PA
 -> send Request into Ramulator
```
