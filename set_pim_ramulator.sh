#!/usr/bin/env bash
set -euo pipefail

## initialize ramulator2
cd ramulator2
git reset --hard b7c70275f04126c647edb989270cc429776955d1
cd ..

# copy files
cp pim_ramulator_src/attacc_bank.yaml ramulator2/
cp pim_ramulator_src/attacc_bg.yaml ramulator2/
cp pim_ramulator_src/attacc_buffer.yaml ramulator2/
cp pim_ramulator_src/HBM3_base.yaml ramulator2/
cp pim_ramulator_src/hbm3_linear_mappers.cpp ramulator2/src/addr_mapper/impl/
cp pim_ramulator_src/hbm3_pim_linear_mappers.cpp ramulator2/src/addr_mapper/impl/
cp pim_ramulator_src/HBM3-PIM.cpp ramulator2/src/dram/impl/
cp pim_ramulator_src/hbm3_controller.cpp ramulator2/src/dram_controller/impl/
cp pim_ramulator_src/hbm3_trace_recorder.cpp ramulator2/src/dram_controller/impl/plugin/
cp pim_ramulator_src/all_bank_refresh_hbm3.cpp ramulator2/src/dram_controller/impl/refresh/
cp pim_ramulator_src/no_refresh.cpp ramulator2/src/dram_controller/impl/refresh/
cp pim_ramulator_src/pim_scheduler.cpp ramulator2/src/dram_controller/impl/scheduler/
cp pim_ramulator_src/pim_loadstore_trace.cpp ramulator2/src/frontend/impl/memory_trace/
cp pim_ramulator_src/PIM_DRAM_system.cpp ramulator2/src/memory_system/impl/
cp -r pim_ramulator_src/trace_gen ramulator2/
cp -r pim_ramulator_src/patches ramulator2/


# Apply patches
cd ramulator2;

for f in ./patches/*.patch
do
    # --forward: skip reversed/already-applied patches
    # --batch: never prompt interactively
    patch --forward --batch -p1 < "$f"
done

# LPDDR5-PIM DRAM/model/controller updates (apply after legacy patches)
required_lpddr_files=(
  ../pim_ramulator_src/LPDDR5-PIM.cpp
  ../pim_ramulator_src/LPDDR5.cpp
  ../pim_ramulator_src/hbm3_pim_controller.cpp
  ../pim_ramulator_src/CMakeLists.txt
  ../pim_ramulator_src/frontend_CMakeLists.txt
  ../pim_ramulator_src/serving/realtime_serving_frontend.cpp
  ../pim_ramulator_src/serving/types.h
  ../pim_ramulator_src/serving/trace_loader.h
  ../pim_ramulator_src/serving/trace_loader.cpp
  ../pim_ramulator_src/serving/inputs.h
  ../pim_ramulator_src/serving/inputs.cpp
  ../pim_ramulator_src/serving/inputs_internal.h
  ../pim_ramulator_src/serving/cost_inputs.cpp
  ../pim_ramulator_src/serving/predictor.h
  ../pim_ramulator_src/serving/predictor.cpp
  ../pim_ramulator_src/serving/pim_executor.h
  ../pim_ramulator_src/serving/pim_executor.cpp
  ../pim_ramulator_src/serving/runtime.h
  ../pim_ramulator_src/serving/runtime.cpp
  ../pim_ramulator_src/serving/runtime_forecast.cpp
  ../pim_ramulator_src/serving/runtime_debug.cpp
  ../pim_ramulator_src/serving/runtime_reporting.cpp
  ../pim_ramulator_src/serving/address_space.cpp
  ../pim_ramulator_src/serving/address_space.h
  ../pim_ramulator_src/serving/allocator.cpp
  ../pim_ramulator_src/serving/allocator.h
  ../pim_ramulator_src/serving/command_generator.cpp
  ../pim_ramulator_src/serving/command_generator.h
  ../pim_ramulator_src/serving/command_generator_builder.cpp
  ../pim_ramulator_src/serving/command_generator_internal.h
  ../pim_ramulator_src/serving/command_generator_stream.cpp
  ../pim_ramulator_src/serving/layout_planner.cpp
  ../pim_ramulator_src/serving/layout_planner.h
  ../pim_ramulator_src/serving/serving_allocator_facade.cpp
  ../pim_ramulator_src/serving/serving_allocator_facade.h
  ../pim_ramulator_src/translation/attacc_serving_translation.cpp
  ../pim_ramulator_src/translation/attacc_serving_translation.h
  ../pim_ramulator_src/alternating_serving/types.h
  ../pim_ramulator_src/alternating_serving/inputs.h
  ../pim_ramulator_src/alternating_serving/inputs.cpp
  ../pim_ramulator_src/alternating_serving/runtime.h
  ../pim_ramulator_src/alternating_serving/runtime_support.cpp
  ../pim_ramulator_src/alternating_serving/runtime.cpp
  ../pim_ramulator_src/alternating_serving/runtime_pim.cpp
  ../pim_ramulator_src/alternating_serving/reporting.cpp
  ../pim_ramulator_src/alternating_serving/alternating_serving_frontend.cpp
  ../pim_ramulator_src/trace_gen/gen_trace_attacc_lpddr5_bank.py
  ../pim_ramulator_src/trace_gen/gen_trace_attacc_lpddr5_bg.py
  ../pim_ramulator_src/trace_gen/gen_trace_attacc_lpddr5_buffer.py
)

for f in "${required_lpddr_files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing required LPDDR-PIM file: $f"
    echo "Copy LPDDR-PIM sources into pim_ramulator_src/ and rerun."
    exit 1
  fi
done

cp ../pim_ramulator_src/LPDDR5-PIM.cpp src/dram/impl/
cp ../pim_ramulator_src/LPDDR5.cpp src/dram/impl/
cp ../pim_ramulator_src/hbm3_pim_controller.cpp src/dram_controller/impl/
cp ../pim_ramulator_src/CMakeLists.txt src/dram/CMakeLists.txt
cp ../pim_ramulator_src/frontend_CMakeLists.txt src/frontend/CMakeLists.txt
mkdir -p src/frontend/impl/serving
rsync -a --delete ../pim_ramulator_src/serving/ src/frontend/impl/serving/
mkdir -p src/translation/impl
cp ../pim_ramulator_src/translation/attacc_serving_translation.cpp src/translation/impl/
cp ../pim_ramulator_src/translation/attacc_serving_translation.h src/translation/impl/
if [[ -f ../pim_ramulator_src/translation_CMakeLists.txt ]]; then
  cp ../pim_ramulator_src/translation_CMakeLists.txt src/translation/CMakeLists.txt
fi
mkdir -p src/frontend/impl/alternating_serving
cp ../pim_ramulator_src/alternating_serving/types.h src/frontend/impl/alternating_serving/
cp ../pim_ramulator_src/alternating_serving/inputs.h src/frontend/impl/alternating_serving/
cp ../pim_ramulator_src/alternating_serving/inputs.cpp src/frontend/impl/alternating_serving/
cp ../pim_ramulator_src/alternating_serving/runtime.h src/frontend/impl/alternating_serving/
cp ../pim_ramulator_src/alternating_serving/runtime_support.cpp src/frontend/impl/alternating_serving/
cp ../pim_ramulator_src/alternating_serving/runtime.cpp src/frontend/impl/alternating_serving/
cp ../pim_ramulator_src/alternating_serving/runtime_pim.cpp src/frontend/impl/alternating_serving/
cp ../pim_ramulator_src/alternating_serving/reporting.cpp src/frontend/impl/alternating_serving/
cp ../pim_ramulator_src/alternating_serving/alternating_serving_frontend.cpp src/frontend/impl/alternating_serving/

# serving_online frontend — online LLM serving with paged KV cache management
mkdir -p src/frontend/impl/serving_online
rsync -a --delete ../pim_ramulator_src/serving_online/ src/frontend/impl/serving_online/

# LPDDR-native trace generators
cp ../pim_ramulator_src/trace_gen/gen_trace_attacc_lpddr5_bank.py trace_gen/
cp ../pim_ramulator_src/trace_gen/gen_trace_attacc_lpddr5_bg.py trace_gen/
cp ../pim_ramulator_src/trace_gen/gen_trace_attacc_lpddr5_buffer.py trace_gen/
