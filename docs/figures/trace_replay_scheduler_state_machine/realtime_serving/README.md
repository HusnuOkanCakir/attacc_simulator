# Realtime Serving Scheduler Figures

This folder contains Mermaid source and rendered figures for the current
`ramulator2/src/frontend/impl/serving` pipeline.

These figures are intentionally separate from the replay-scheduler figures in
the parent folder because the realtime frontend has extra states and a real
PIM execution data path:

- persistent running states: `RunningPrefill`, `RunningDecode`
- callback-backed PIM completion
- `template_replay` and `runtime_generated` hybrid command sources
- allocator and translation in the `runtime_generated` path

Files:

- `1_realtime_request_state_machine.*`
  Request lifecycle including running states and optional decode rebind.
- `2_realtime_tick_loop.*`
  Top-level `tick()` order in `RealtimeServingRuntime`.
- `3_realtime_hybrid_decode_pipeline.*`
  Phase split and hybrid decode data path from serving runtime to LPDDR5-PIM.
- `4_realtime_command_generation_flow.*`
  Detailed `runtime_generated` command-generation flow from `lin` to `Request`.
- `5_realtime_command_generation_flow_high_level.*`
  Higher-level view of the `runtime_generated` command path.
- `6_realtime_command_generation_builder_detail.*`
  Zoomed-in view of `build_lpddr5_bank_generated_artifacts()`.

Primary code paths represented here:

- `/home/okan/attacc_simulator/ramulator2/src/frontend/impl/serving/realtime_serving_frontend.cpp`
- `/home/okan/attacc_simulator/ramulator2/src/frontend/impl/serving/runtime.cpp`
- `/home/okan/attacc_simulator/ramulator2/src/frontend/impl/serving/runtime_forecast.cpp`
- `/home/okan/attacc_simulator/ramulator2/src/frontend/impl/serving/pim_executor.cpp`
- `/home/okan/attacc_simulator/ramulator2/src/frontend/impl/serving/command_generator.cpp`
- `/home/okan/attacc_simulator/ramulator2/src/frontend/impl/serving/layout_planner.cpp`
- `/home/okan/attacc_simulator/ramulator2/src/frontend/impl/serving/allocator.cpp`
- `/home/okan/attacc_simulator/ramulator2/src/translation/impl/attacc_serving_translation.cpp`
- `/home/okan/attacc_simulator/ramulator2/src/memory_system/impl/PIM_DRAM_system.cpp`
- `/home/okan/attacc_simulator/ramulator2/src/dram_controller/impl/scheduler/pim_scheduler.cpp`
- `/home/okan/attacc_simulator/ramulator2/src/dram/impl/LPDDR5-PIM.cpp`

Export all figures with:

```bash
bash tools/export_realtime_serving_scheduler_figures.sh
```
