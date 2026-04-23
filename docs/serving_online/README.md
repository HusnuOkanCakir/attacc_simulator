# serving_online — Pi0 PIM+GPU LLM Serving Frontend

Clean, self-contained C++ IFrontEnd for pi0 attention serving in ramulator2.

## What it does

Reads an Azure LLM inference CSV (TIMESTAMP, ContextTokens, GeneratedTokens), admits
each request to either the `pim+gpu` or `gpu_only` route, then simulates execution:

- **GPU work** advances a wall-clock free-time estimate directly (no ramulator commands).
- **PIM decode** issues LPDDR5 Read requests per KV cache line to the ramulator memory
  system, which drives the actual LPDDR5-PIM timing model (MACAB, SFM, etc.).
- Outputs per-request timing CSV and YAML stats at the end.

## Files

| File | Purpose |
|------|---------|
| `types.h` | Request state machine, cost estimates, active tasks, config struct |
| `csv_loader.h/cpp` | Load Azure inference CSV; normalize arrival times |
| `cost_table.h/cpp` | Load pre-exported HistGBR cost CSVs; nearest-neighbor lookup |
| `pim_allocator.h/cpp` | KV cache page allocator; channel-interleaved free stack; VA encoding |
| `pim_commands.h/cpp` | Generate LPDDR5 Read requests for one pi0 attention decode step |
| `scheduler.h/cpp` | Admission (KV OOM hold/retry), prefill-priority FCFS decode selection |
| `frontend.cpp` | IFrontEnd: ticks scheduler, issues PIM commands, writes output |

Translation:
`ramulator2/src/translation/impl/serving_online_translation.h/cpp`

## YAML configuration

```yaml
Frontend:
  impl: ServingOnlineFrontend
  clock_ratio: 4
  requests_csv: path/to/azure_conv.csv
  cost_gpu_csv: path/to/cost_gpu.csv
  cost_hybrid_csv: path/to/cost_pim.csv
  kv_pool_bytes: 536870912        # 512 MB PIM DRAM for KV cache (0 = unlimited)
  requests_out_csv: out/serving_online_requests.csv
  # Optional overrides (defaults shown):
  gpu_route_name: gpu_only
  hybrid_route_name: pim+gpu
  route_policy: min_finish         # or latency_guarded_energy
  energy_latency_guard_ms: 5.0
  max_decode_batch_size: 8
  prompt_priority: true
  max_consecutive_decode_batches: 4
  arrival_time_scale: 1.0          # < 1 = compress arrivals (increase load)
  arrival_limit: -1                # -1 = load all requests
  generator_num_layers: 18
  generator_num_heads: 8
  generator_dhead: 256
  generator_dtype_bytes: 2
  generator_channel_count: 8
  translation_pagesize_KB: 4
  Translation:
    impl: ServingOnlineTranslation
    max_addr: 4294967296           # 4 GB physical DRAM
```

## Cost table CSV format

Generated offline with HistGBR models in Python, then exported:

```
route,lin,lout,bs,prefill_gpu_ms,prefill_pim_ms,prefill_e2e_ms,prefill_energy_nj,decode_gpu_ms,decode_pim_ms,decode_e2e_ms,decode_energy_nj
pim+gpu,256,1,1,0.8,1.42,1.42,120.0,0.3,0.7,0.7,8.5
gpu_only,256,1,1,2.51,0.0,2.51,200.0,0.9,0.0,0.9,15.0
```

`lin` = context tokens, `lout` = generated tokens, `bs` = batch size.
Lookup uses L1 nearest-neighbor: minimize |lin-q| + |lout-q| + |bs-q|.

## Route policies

- **min_finish**: pick the route with the earliest predicted finish time.
- **latency_guarded_energy**: among routes within `energy_latency_guard_ms` of the
  fastest, pick the lowest energy one.

## KV memory

When `kv_pool_bytes > 0`, PIM requests reserve Key and Value pages from a
channel-interleaved free-page stack. If both K and V fit, the request is admitted to
`pim+gpu`; otherwise it is held with reason `kv_oom` until memory is freed.

## Output CSV columns

`request_id, arrival_ms, context_tokens, generated_tokens, state, route, is_lost,
admission_attempts, admission_last_reject_reason, dropped_reason, prefill_start_ms,
prefill_end_ms, first_token_ms, completion_ms, ttft_ms, e2e_ms, tbt_mean_ms`
