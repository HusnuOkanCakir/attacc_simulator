import pandas as pd
import subprocess
import math
import os
import re
import json
from pathlib import Path
from src.config import *
from src.model import *
from src.type import *


class Ramulator:
    # Parsed from ramulator2/src/dram/impl/HBM3-PIM.cpp org_presets.
    HBM3_ORG_PRESET_COUNTS = {
        "HBM3_2Gb_1R": {"n_pch": 2, "n_rank": 1, "n_bg": 4, "n_bank": 4},
        "HBM3_4Gb_1R": {"n_pch": 2, "n_rank": 1, "n_bg": 4, "n_bank": 4},
        "HBM3_8Gb_1R": {"n_pch": 2, "n_rank": 1, "n_bg": 4, "n_bank": 4},
        "HBM3_4Gb_2R": {"n_pch": 2, "n_rank": 2, "n_bg": 4, "n_bank": 4},
        "HBM3_8Gb_2R": {"n_pch": 2, "n_rank": 2, "n_bg": 4, "n_bank": 4},
        "HBM3_16Gb_2R": {"n_pch": 2, "n_rank": 2, "n_bg": 4, "n_bank": 4},
        "HBM3_6Gb_3R": {"n_pch": 2, "n_rank": 3, "n_bg": 4, "n_bank": 4},
        "HBM3_12Gb_3R": {"n_pch": 2, "n_rank": 3, "n_bg": 4, "n_bank": 4},
        "HBM3_24Gb_3R": {"n_pch": 2, "n_rank": 3, "n_bg": 4, "n_bank": 4},
        "HBM3_8Gb_4R": {"n_pch": 2, "n_rank": 4, "n_bg": 4, "n_bank": 4},
        "HBM3_16Gb_4R": {"n_pch": 2, "n_rank": 4, "n_bg": 4, "n_bank": 4},
        "HBM3_32Gb_4R": {"n_pch": 2, "n_rank": 4, "n_bg": 4, "n_bank": 4},
    }

    # Parsed from ramulator2/src/dram/impl/LPDDR5-PIM.cpp org_presets.
    LPDDR5_ORG_PRESET_COUNTS = {
        "LPDDR5_2Gb_x16": {"n_pch": 1, "n_rank": 1, "n_bg": 4, "n_bank": 4},
        "LPDDR5_4Gb_x16": {"n_pch": 1, "n_rank": 1, "n_bg": 4, "n_bank": 4},
        "LPDDR5_8Gb_x16": {"n_pch": 1, "n_rank": 1, "n_bg": 4, "n_bank": 4},
        "LPDDR5_16Gb_x16": {"n_pch": 1, "n_rank": 1, "n_bg": 4, "n_bank": 4},
        "LPDDR5_32Gb_x16": {"n_pch": 1, "n_rank": 1, "n_bg": 4, "n_bank": 4},
    }

    LOG_COLUMNS = [
        'L', 'nhead', 'dhead', 'dbyte', 'pim_type', 'power_constraint',
        'dram_impl', 'timing_preset',
        'cycle', 'mac', 'softmax', 'mvgb', 'mvsb', 'wrgb'
    ]

    def __init__(self,
                 modelinfos,
                 ramulator_dir,
                 output_log='',
                 fast_mode=False,
                 num_hbm=5,
                 pim_config=None):
        self.df = pd.DataFrame()
        self.ramulator_dir = ramulator_dir
        self.output_log = output_log
        self.modelinfos = modelinfos
        if os.path.exists(output_log):
            self.df = pd.read_csv(output_log)
        self.num_hbm = num_hbm
        self.nhead = modelinfos['num_heads']
        self.dhead = modelinfos['dhead']
        self.num_layers = modelinfos.get('ndec', 1)
        dtype = modelinfos.get('dtype')
        self.dtype_bytes = 2 if dtype in (DataType.W16A16, DataType.W16A8) else 1
        self.fast_mode = fast_mode
        self.pim_config = pim_config or {}

        # Defaults are HBM3-compatible for backward compatibility.
        self.dram_impl = self.pim_config.get("DRAM_IMPL", "HBM3-PIM")
        self.dram_org_preset = self.pim_config.get("DRAM_ORG_PRESET", "HBM3_8Gb_2R")
        self.dram_timing_preset = self.pim_config.get("DRAM_TIMING_PRESET", "HBM3_5.2Gbps")
        self.controller_impl = self.pim_config.get("CONTROLLER_IMPL", "HBM3-PIM")
        self.refresh_impl = self.pim_config.get("REFRESH_MANAGER_IMPL", "AllBankHBM3")
        self.addr_mapper_impl = self.pim_config.get("ADDR_MAPPER_IMPL", "HBM3-PIM")
        self.trace_recorder_impl = self.pim_config.get("TRACE_RECORDER_IMPL", "HBM3TraceRecorder")
        self.trace_gen_prefix = self.pim_config.get("TRACE_GEN_PREFIX", "gen_trace_attacc_")
        self.channel_count = int(self.pim_config.get("CHANNEL_COUNT", 16))
        # Energy-only HBM-stack multiplier (see make_pim_config). Defaults
        # to num_hbm for backward compatibility with the hbm3 target.
        self.energy_num_hbm = int(self.pim_config.get("ENERGY_NUM_HBM", num_hbm))
        self.tCK = float(self.pim_config.get("TCK_NS", self._derive_tck_ns(self.dram_timing_preset)))
        self._ensure_log_schema()

    def _resolve_ramulator_binary(self):
        direct = os.path.join(self.ramulator_dir, "ramulator2")
        build = os.path.join(self.ramulator_dir, "build", "ramulator2")
        if os.path.exists(build):
            return build
        return direct

    def _ensure_log_schema(self):
        if self.df.empty:
            return
        for col in self.LOG_COLUMNS:
            if col not in self.df.columns:
                if col == 'dram_impl':
                    self.df[col] = self.dram_impl
                elif col == 'timing_preset':
                    self.df[col] = self.dram_timing_preset
                else:
                    self.df[col] = 0
        self.df = self.df[self.LOG_COLUMNS]

    def _derive_tck_ns(self, timing_preset: str) -> float:
        if timing_preset.startswith("LPDDR5_"):
            if timing_preset == "LPDDR5_6400":
                return 1.25
            m = re.search(r"LPDDR5_(\d+)", timing_preset)
            if m:
                rate = int(m.group(1))
                return 8000.0 / rate
            return 1.25

        # HBM3 presets look like HBM3_5.2Gbps(_NPC)
        m = re.search(r"HBM3_([0-9]+(?:\.[0-9]+)?)Gbps", timing_preset)
        if m:
            rate_mtps = float(m.group(1)) * 1000.0
            # HBM3 model in this repo uses QDR pin rate (same as parse_ramulator_output.py)
            return 1e6 / (rate_mtps / 4.0) / 1000.0
        return 0.769231

    def _get_topology_counts(self):
        if self.dram_impl.startswith("HBM3"):
            topo = self.HBM3_ORG_PRESET_COUNTS.get(self.dram_org_preset)
            if topo:
                return topo
            # Backward-compatible fallback.
            return {"n_pch": 2, "n_rank": 2, "n_bg": 4, "n_bank": 4}

        if self.dram_impl.startswith("LPDDR5"):
            topo = self.LPDDR5_ORG_PRESET_COUNTS.get(self.dram_org_preset)
            if topo:
                return topo
            return {"n_pch": 1, "n_rank": 1, "n_bg": 4, "n_bank": 4}

        # Conservative fallback for unknown backends.
        return {"n_pch": 1, "n_rank": 1, "n_bg": 1, "n_bank": 1}

    def _mem_acc_scale(self, pim_type: PIMType) -> int:
        topo = self._get_topology_counts()
        n_pch = topo["n_pch"]
        n_rank = topo["n_rank"]
        n_bg = topo["n_bg"]
        n_bank = topo["n_bank"]

        if pim_type == PIMType.BA:
            # All-bank MAC fanout: pCH x rank x BG x bank.
            return n_pch * n_rank * n_bg * n_bank
        if pim_type == PIMType.BG:
            # Same-bank MAC fanout: pCH x rank x BG.
            return n_pch * n_rank * n_bg
        # Per-bank MAC fanout: pCH (HBM pseudochannels). LPDDR5 has pCH=1.
        return n_pch

    def _build_traffic(self, pim_type: PIMType, mac: int, mvgb: int, mvsb: int,
                       wrgb: int, num_ops_group: int):
        si_io = wrgb * 32  # 256-bit granularity
        tsv_io = (wrgb + mvsb + mvgb) * 32
        giomux_io = (wrgb + mvsb + mvgb) * 32
        bgmux_io = (wrgb + mvsb + mvgb) * 32
        mem_acc = mac * 32 * self._mem_acc_scale(pim_type)

        traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
        # Traffic feeds the energy model only (timing comes from raw
        # ramulator cycles), so scale by the energy-only HBM count.
        traffic = [i * self.energy_num_hbm for i in traffic]
        traffic = [i * num_ops_group for i in traffic]
        return traffic

    def _cycles_to_seconds(self, cycle: int, num_ops_group: int):
        return self.tCK * cycle / 1_000_000_000 * num_ops_group

    def make_yaml_file(self, yaml_file, file_name, power_constraint):
        trace_path = os.path.join(self.ramulator_dir, file_name + ".trace")
        timing_preset = self.dram_timing_preset
        if self.dram_impl.startswith("HBM3") and power_constraint is False:
            if timing_preset == "HBM3_5.2Gbps":
                timing_preset = "HBM3_5.2Gbps_NPC"
            elif timing_preset == "HBM3_4.8Gbps":
                timing_preset = "HBM3_4.8Gbps_NPC"
            elif timing_preset == "HBM3_5.6Gbps":
                timing_preset = "HBM3_5.6Gbps_NPC"
            elif timing_preset == "HBM3_6.0Gbps":
                timing_preset = "HBM3_6.0Gbps_NPC"
            elif timing_preset == "HBM3_6.4Gbps":
                timing_preset = "HBM3_6.4Gbps_NPC"

        # Keep local tCK in sync with the emitted YAML timing preset.
        self.tCK = self._derive_tck_ns(timing_preset)

        line = ""
        line += "Frontend:\n"
        line += "  impl: PIMLoadStoreTrace\n"
        line += "  path: {}\n".format(trace_path)
        line += "  clock_ratio: 1\n"
        line += "\n"
        line += "  Translation:\n"
        line += "    impl: NoTranslation\n"
        line += "    max_addr: 2147483648\n"
        line += "              \n"
        line += "\n"
        line += "MemorySystem:\n"
        line += "  impl: PIMDRAM\n"
        line += "  clock_ratio: 1\n"
        line += "  DRAM:\n"
        line += "    impl: {}\n".format(self.dram_impl)
        line += "    org:\n"
        line += "      preset: {}\n".format(self.dram_org_preset)
        line += "      channel: {}\n".format(self.channel_count)
        line += "    timing:\n"
        line += "      preset: {}\n".format(timing_preset)
        line += "\n"
        line += "  Controller:\n"
        line += "    impl: {}\n".format(self.controller_impl)
        line += "    Scheduler:\n"
        line += "      impl: PIM\n"
        line += "    RefreshManager:\n"
        line += "      impl: {}\n".format(self.refresh_impl)
        line += "      #impl: No\n"
        line += "    plugins:\n"
        line += "    - ControllerPlugin:\n"
        line += "        impl: {}\n".format(self.trace_recorder_impl)
        line += "        path: ./log/{}/cmd.log\n".format(file_name)
        line += "\n"
        line += "  AddrMapper:\n"
        line += "    impl: {}\n".format(self.addr_mapper_impl)
        with open(yaml_file, 'w') as f:
            f.write(line)

    def _online_frontend_impl_name(self, online_frontend: str) -> str:
        if online_frontend == "realtime":
            return "RealtimeServingFrontend"
        if online_frontend == "alternating":
            return "AlternatingServingFrontend"
        if online_frontend == "serving1_5":
            return "Serving1_5Frontend"
        raise ValueError(
            f"Unsupported online frontend {online_frontend}; expected realtime, alternating, or serving1_5"
        )

    def make_online_yaml_file(self,
                              yaml_file,
                              requests_csv,
                              requests_out_csv,
                              kv_placement_csv,
                              allocator_pressure_csv,
                              template_dir,
                              hybrid_command_source,
                              generator_backend,
                              generator_dhead,
                              generator_heads_per_hbm,
                              generator_dtype_bytes,
                              generator_num_layers,
                              generator_num_heads,
                              generator_channel_count,
                              translation_max_addr,
                              translation_pagesize_kb,
                              kv_pool_bytes,
                              score_pool_bytes,
                              context_pool_bytes,
                              scratch_pool_bytes,
                              cost_gpu_csv,
                              cost_hybrid_csv,
                              route_policy,
                              pim_wait_threshold_ms,
                              unsupported_policy,
                              arrival_time_scale=1.0,
                              lin_bucket=1,
                              lout_bucket=1,
                              batch_size=1,
                              gpu_route_name="gpu_only",
                              hybrid_route_name="lpddr5_pim_bank",
                              enable_prefill_batching=False,
                              max_prefill_batch_size=1,
                              enable_decode_batching=False,
                              max_decode_batch_size=1,
                              prefill_guard_ms=-1.0,
                              max_consecutive_decode_batches=0,
                              decode_batch_cap_with_prefill=0,
                              local_scheduling_policy="prefill_priority_fcfs_decode",
                              fcfs_decode_prediction_mode="incremental",
                              enable_decode_rebind=False,
                              decode_rebind_margin_ms=0.0,
                              enable_predictive_admission=False,
                              enable_slo_guarded_admission=False,
                              slo_tbt_ms=-1.0,
                              slo_e2e_ms=-1.0,
                              slo_ttft_ms=-1.0,
                              admission_shadow_max_steps=10000,
                              admission_retry_interval_ms=0.0,
                              admission_max_harmed_requests=0,
                              admission_max_total_harm_ms=0.0,
                              admission_max_single_harm_ms=0.0,
                              admission_max_own_miss_ms=0.0,
                              admission_check_own_slo=False,
                              admission_max_bypass_count=0,
                              admission_max_wait_ms=0.0,
                              admission_aging_harm_ms_per_ms=0.0,
                              admission_aging_harmed_requests_per_ms=0.0,
                              share_gpu_prefill_across_routes=False,
                              gpu_queue_alpha=0.0,
                              pim_queue_alpha=0.0,
                              active_request_alpha=0.0,
                              decode_token_alpha=0.0,
                              prompt_priority=True,
                              prefer_gpu_on_tie=True,
                              energy_latency_guard_ms=5.0,
                              route_load_balance_ms_per_active_request=0.0,
                              capture_debug_events=False,
                              debug_log_path="log/online_serving/frontend_debug.log",
                              online_frontend="realtime"):
        self._validate_online_serving_config(route_policy=route_policy,
                                             batch_size=batch_size,
                                             local_scheduling_policy=local_scheduling_policy,
                                             fcfs_decode_prediction_mode=fcfs_decode_prediction_mode,
                                             enable_predictive_admission=enable_predictive_admission,
                                             enable_slo_guarded_admission=enable_slo_guarded_admission,
                                             enable_prefill_batching=enable_prefill_batching,
                                             slo_tbt_ms=slo_tbt_ms,
                                             slo_e2e_ms=slo_e2e_ms,
                                             online_frontend=online_frontend)
        frontend_impl = self._online_frontend_impl_name(online_frontend)
        line = ""
        line += "Frontend:\n"
        line += f"  impl: {frontend_impl}\n"
        line += "  requests_csv: {}\n".format(requests_csv)
        line += "  requests_out_csv: {}\n".format(requests_out_csv)
        line += "  kv_placement_csv: {}\n".format(kv_placement_csv)
        line += "  allocator_pressure_csv: {}\n".format(allocator_pressure_csv)
        line += "  hybrid_command_source: {}\n".format(hybrid_command_source)
        line += "  template_dir: {}\n".format(template_dir)
        line += "  generator_backend: {}\n".format(generator_backend)
        line += "  generator_dhead: {}\n".format(int(generator_dhead))
        line += "  generator_heads_per_hbm: {}\n".format(int(generator_heads_per_hbm))
        line += "  generator_dtype_bytes: {}\n".format(int(generator_dtype_bytes))
        line += "  generator_num_layers: {}\n".format(int(generator_num_layers))
        line += "  generator_num_heads: {}\n".format(int(generator_num_heads))
        line += "  generator_channel_count: {}\n".format(int(generator_channel_count))
        line += "  translation_max_addr: {}\n".format(int(translation_max_addr))
        line += "  translation_pagesize_KB: {}\n".format(int(translation_pagesize_kb))
        line += "  kv_pool_bytes: {}\n".format(int(kv_pool_bytes))
        line += "  score_pool_bytes: {}\n".format(int(score_pool_bytes))
        line += "  context_pool_bytes: {}\n".format(int(context_pool_bytes))
        line += "  scratch_pool_bytes: {}\n".format(int(scratch_pool_bytes))
        line += "  cost_gpu_csv: {}\n".format(cost_gpu_csv)
        line += "  cost_hybrid_csv: {}\n".format(cost_hybrid_csv)
        line += "  gpu_route_name: {}\n".format(gpu_route_name)
        line += "  hybrid_route_name: {}\n".format(hybrid_route_name)
        line += "  route_policy: {}\n".format(route_policy)
        line += "  pim_wait_threshold_ms: {}\n".format(float(pim_wait_threshold_ms))
        line += "  unsupported_policy: {}\n".format(unsupported_policy)
        line += "  arrival_time_scale: {}\n".format(float(arrival_time_scale))
        line += "  lin_bucket: {}\n".format(int(lin_bucket))
        line += "  lout_bucket: {}\n".format(int(lout_bucket))
        line += "  batch_size: {}\n".format(int(batch_size))
        line += "  enable_prefill_batching: {}\n".format(str(bool(enable_prefill_batching)).lower())
        line += "  max_prefill_batch_size: {}\n".format(int(max_prefill_batch_size))
        line += "  enable_decode_batching: {}\n".format(str(bool(enable_decode_batching)).lower())
        line += "  max_decode_batch_size: {}\n".format(int(max_decode_batch_size))
        line += "  prefill_guard_ms: {}\n".format(float(prefill_guard_ms))
        line += "  max_consecutive_decode_batches: {}\n".format(int(max_consecutive_decode_batches))
        line += "  decode_batch_cap_with_prefill: {}\n".format(int(decode_batch_cap_with_prefill))
        line += "  local_scheduling_policy: {}\n".format(local_scheduling_policy)
        line += "  fcfs_decode_prediction_mode: {}\n".format(fcfs_decode_prediction_mode)
        line += "  enable_decode_rebind: {}\n".format(str(bool(enable_decode_rebind)).lower())
        line += "  decode_rebind_margin_ms: {}\n".format(float(decode_rebind_margin_ms))
        line += "  enable_predictive_admission: {}\n".format(str(bool(enable_predictive_admission)).lower())
        line += "  enable_slo_guarded_admission: {}\n".format(str(bool(enable_slo_guarded_admission)).lower())
        line += "  slo_tbt_ms: {}\n".format(float(slo_tbt_ms))
        line += "  slo_e2e_ms: {}\n".format(float(slo_e2e_ms))
        line += "  slo_ttft_ms: {}\n".format(float(slo_ttft_ms))
        line += "  admission_shadow_max_steps: {}\n".format(int(admission_shadow_max_steps))
        line += "  admission_retry_interval_ms: {}\n".format(float(admission_retry_interval_ms))
        line += "  admission_max_harmed_requests: {}\n".format(int(admission_max_harmed_requests))
        line += "  admission_max_total_harm_ms: {}\n".format(float(admission_max_total_harm_ms))
        line += "  admission_max_single_harm_ms: {}\n".format(float(admission_max_single_harm_ms))
        line += "  admission_max_own_miss_ms: {}\n".format(float(admission_max_own_miss_ms))
        line += "  admission_check_own_slo: {}\n".format(str(bool(admission_check_own_slo)).lower())
        line += "  admission_max_bypass_count: {}\n".format(int(admission_max_bypass_count))
        line += "  admission_max_wait_ms: {}\n".format(float(admission_max_wait_ms))
        line += "  admission_aging_harm_ms_per_ms: {}\n".format(float(admission_aging_harm_ms_per_ms))
        line += "  admission_aging_harmed_requests_per_ms: {}\n".format(float(admission_aging_harmed_requests_per_ms))
        line += "  share_gpu_prefill_across_routes: {}\n".format(str(bool(share_gpu_prefill_across_routes)).lower())
        line += "  gpu_queue_alpha: {}\n".format(float(gpu_queue_alpha))
        line += "  pim_queue_alpha: {}\n".format(float(pim_queue_alpha))
        line += "  active_request_alpha: {}\n".format(float(active_request_alpha))
        line += "  decode_token_alpha: {}\n".format(float(decode_token_alpha))
        line += "  prompt_priority: {}\n".format(str(bool(prompt_priority)).lower())
        line += "  prefer_gpu_on_tie: {}\n".format(str(bool(prefer_gpu_on_tie)).lower())
        line += "  energy_latency_guard_ms: {}\n".format(float(energy_latency_guard_ms))
        line += "  route_load_balance_ms_per_active_request: {}\n".format(
            float(route_load_balance_ms_per_active_request))
        line += "  capture_debug_events: {}\n".format(str(bool(capture_debug_events)).lower())
        line += "  debug_log_path: {}\n".format(debug_log_path)
        line += "  clock_ratio: 1\n"
        needs_translation = hybrid_command_source == "runtime_generated" or online_frontend == "serving1_5"
        if needs_translation:
            line += "\n"
            line += "  Translation:\n"
            if online_frontend == "serving1_5":
                line += "    impl: Serving1_5Translation\n"
            else:
                line += "    impl: AttAccServingTranslation\n"
            line += "    max_addr: {}\n".format(int(translation_max_addr))
            line += "    pagesize_KB: {}\n".format(int(translation_pagesize_kb))
        line += "\n"
        line += "MemorySystem:\n"
        line += "  impl: PIMDRAM\n"
        line += "  clock_ratio: 1\n"
        line += "  DRAM:\n"
        line += "    impl: {}\n".format(self.dram_impl)
        line += "    org:\n"
        line += "      preset: {}\n".format(self.dram_org_preset)
        line += "      channel: {}\n".format(self.channel_count)
        line += "    timing:\n"
        line += "      preset: {}\n".format(self.dram_timing_preset)
        line += "\n"
        line += "  Controller:\n"
        line += "    impl: {}\n".format(self.controller_impl)
        line += "    Scheduler:\n"
        line += "      impl: PIM\n"
        line += "    RefreshManager:\n"
        line += "      impl: {}\n".format(self.refresh_impl)
        line += "    plugins:\n"
        line += "    - ControllerPlugin:\n"
        line += "        impl: {}\n".format(self.trace_recorder_impl)
        line += "        path: ./log/online_serving/cmd.log\n"
        line += "\n"
        line += "  AddrMapper:\n"
        line += "    impl: {}\n".format(self.addr_mapper_impl)
        with open(yaml_file, 'w') as f:
            f.write(line)

    def _validate_online_serving_config(self,
                                        route_policy,
                                        batch_size,
                                        local_scheduling_policy="prefill_priority_fcfs_decode",
                                        fcfs_decode_prediction_mode="incremental",
                                        enable_predictive_admission=False,
                                        enable_slo_guarded_admission=False,
                                        enable_prefill_batching=False,
                                        slo_tbt_ms=-1.0,
                                        slo_e2e_ms=-1.0,
                                        online_frontend="realtime"):
        frontend_impl = self._online_frontend_impl_name(online_frontend)
        if online_frontend == "alternating":
            if route_policy != "min_finish":
                raise ValueError(
                    f"{frontend_impl} only supports route_policy=min_finish, got {route_policy}"
                )
            if int(batch_size) != 1:
                raise ValueError(
                    f"{frontend_impl} only supports batch_size=1, got {batch_size}"
                )
            return
        if online_frontend == "serving1_5":
            if route_policy not in ("min_finish", "latency_guarded_energy"):
                raise ValueError(
                    f"{frontend_impl} only supports route_policy=min_finish or latency_guarded_energy, got {route_policy}"
                )
            return

        if local_scheduling_policy == "prefill_priority_fcfs_decode":
            if enable_slo_guarded_admission and fcfs_decode_prediction_mode != "incremental":
                raise ValueError("slo_guarded_admission requires fcfs_decode_prediction_mode=incremental")
        else:
            if fcfs_decode_prediction_mode != "shadow":
                raise ValueError("fcfs_decode_prediction_mode only applies to prefill_priority_fcfs_decode")
            if enable_slo_guarded_admission:
                raise ValueError("slo_guarded_admission requires prefill_priority_fcfs_decode")

        if enable_predictive_admission and enable_slo_guarded_admission:
            raise ValueError("predictive admission and slo_guarded_admission are mutually exclusive")

        if route_policy == "slack_then_finish" and float(slo_e2e_ms) < 0.0:
            raise ValueError("slack_then_finish requires slo_e2e_ms")
        if route_policy == "latency_guarded_energy":
            if local_scheduling_policy != "prefill_priority_fcfs_decode":
                raise ValueError("latency_guarded_energy requires prefill_priority_fcfs_decode")
            if fcfs_decode_prediction_mode != "incremental":
                raise ValueError("latency_guarded_energy requires fcfs_decode_prediction_mode=incremental")
        if enable_slo_guarded_admission:
            if float(slo_e2e_ms) < 0.0:
                raise ValueError("slo_guarded_admission requires slo_e2e_ms")
            if float(slo_tbt_ms) < 0.0:
                raise ValueError("slo_guarded_admission requires slo_tbt_ms")

    def update_log_file(self, log):
        columns = self.LOG_COLUMNS
        if self.df.empty:
            if os.path.exists(self.output_log):
                df = pd.read_csv(self.output_log)
            else:
                df = pd.DataFrame(columns=columns)
        else:
            df = self.df

        # Backward-compatible schema upgrade for older cache logs.
        for col in columns:
            if col not in df.columns:
                if col == 'dram_impl':
                    df[col] = self.dram_impl
                elif col == 'timing_preset':
                    df[col] = self.dram_timing_preset
                else:
                    df[col] = 0
        df = df[columns]

        new_df = pd.DataFrame(columns=df.columns)
        new_df.loc[0] = log
        df = pd.concat([df, new_df]).drop_duplicates()
        self.df = df
        self.df.to_csv(self.output_log, index=False)

    #def run_ramulator(self):
    def run_ramulator(self, pim_type: PIMType, l, num_ops_per_hbm, dbyte,
                      yaml_file, file_name):
        pim_type_name = pim_type.name.lower(
        ) if not pim_type == PIMType.BA else "bank"
        trace_file = os.path.join(self.ramulator_dir, file_name + '.trace')

        trace_exc = os.path.join(
            self.ramulator_dir,
            "trace_gen/{}{}.py".format(self.trace_gen_prefix, pim_type_name))
        trace_args = "--dhead {} --nhead {} --seqlen {} --dbyte {} -o {}".format(
            self.dhead, num_ops_per_hbm, l, dbyte, trace_file)

        gen_trace_cmd = f"python {trace_exc} {trace_args}"

        # generate trace
        try:
            os.system(gen_trace_cmd)
        except Exception as e:
            print(f"Error: {e}")

        # run ramulator
        ramulator_file = self._resolve_ramulator_binary()
        run_ramulator_cmd = f"{ramulator_file} -f {yaml_file}"
        try:
            result = subprocess.run(run_ramulator_cmd,
                                    stdout=subprocess.PIPE,
                                    text=True,
                                    shell=True)
            output_lines = result.stdout.strip().split('\n')
            output_list = [line.strip() for line in output_lines]
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            assert 0

        # remove trace
        rm_trace_cmd = f"rm {trace_file}"
        try:
            os.system(rm_trace_cmd)
        except Exception as e:
            print(f"Error: {e}")

        # parsing output
        n_cmds = {"mac": 0, "sfm": 0, "mvgb": 0, "mvsb": 0, "wrgb": 0}
        cycle = 0
        for line in output_list:
            if "mac" in line:
                n_cmds["mac"] += int(line.split()[-1])
            elif "softmax_requests" in line:
                n_cmds["sfm"] += int(line.split()[-1])
            elif "move_to_gemv_buffer" in line:
                n_cmds["mvgb"] += int(line.split()[-1])
            elif "move_to_softmax_buffer" in line:
                n_cmds["mvsb"] += int(line.split()[-1])
            elif "write_to_gemv_buffer" in line:
                n_cmds["wrgb"] += int(line.split()[-1])
            elif "memory_system_cycles" in line:
                cycle += int(line.split()[-1])

        out = [
            cycle, n_cmds["mac"], n_cmds["sfm"], n_cmds["mvgb"], n_cmds["mvsb"],
            n_cmds["wrgb"]
        ]
        return out

    def _parse_ramulator_yaml_from_stdout(self, stdout_text: str):
        try:
            import yaml  # type: ignore
        except Exception:
            return {}

        start = stdout_text.find("Frontend:")
        if start < 0:
            return {}
        body = stdout_text[start:]
        try:
            parsed = yaml.safe_load(body)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
        return {}

    def run(self, pim_type: PIMType, layer: Layer, power_constraint=True):
        if os.path.exists(self.ramulator_dir):
            l = layer.n
            dhead = self.dhead
            dbyte = layer.dbyte
            num_ops_per_attacc = layer.numOp
            num_ops_per_hbm = math.ceil(num_ops_per_attacc / self.num_hbm)
            num_ops_group = 1
            if self.fast_mode:
                minimum_heads = 64
                num_ops_group = math.ceil(num_ops_per_hbm / minimum_heads)
                num_ops_per_hbm = minimum_heads

            file_name = "attacc_l{}_nattn{}_dhead{}_dbyte{}_pc{}".format(
                l, num_ops_per_hbm, dhead, layer.dbyte, int(power_constraint))
            yaml_file = os.path.join(self.ramulator_dir, file_name + '.yaml')
            self.make_yaml_file(yaml_file, file_name, power_constraint)

            result = self.run_ramulator(pim_type, l, num_ops_per_hbm,
                                        layer.dbyte, yaml_file, file_name)

            # remove trace
            rm_yaml_cmd = f"rm {yaml_file}"
            try:
                os.system(rm_yaml_cmd)
            except Exception as e:
                print(f"Error: {e}")

            # post processing
            cycle, mac, sfm, mvgb, mvsb, wrgb = result

            ## update log file

            log = [
                l, num_ops_per_hbm, dhead, dbyte, pim_type.name,
                power_constraint, self.dram_impl, self.dram_timing_preset
            ] + result
            self.update_log_file(log)

            ## si, tsv, giomux to bgmux, bgmux to column decoder, bank RD
            traffic = self._build_traffic(
                pim_type, mac, mvgb, mvsb, wrgb, num_ops_group)
            exec_time = self._cycles_to_seconds(cycle, num_ops_group)
            return exec_time, traffic

        else:
            assert 0, "Need to install ramulator"

    def output(self, pim_type: PIMType, layer: Layer, power_constraint=True):
        self._ensure_log_schema()
        if self.df.empty:
            self.run(pim_type, layer, power_constraint)

        num_ops_per_attacc = layer.numOp
        num_ops_per_hbm = math.ceil(num_ops_per_attacc / self.num_hbm)
        num_ops_group = 1
        if self.fast_mode:
            minimum_heads = 64
            num_ops_group = math.ceil(num_ops_per_hbm / minimum_heads)
            num_ops_per_hbm = minimum_heads

        l = layer.n
        dhead = layer.k
        dbyte = layer.dbyte
        row = self.df[(self.df['L'] == l) & (self.df['nhead'] == num_ops_per_hbm) & \
                      (self.df['dbyte'] == dbyte) & (self.df['dhead'] == dhead) & \
                      (self.df['power_constraint'] == power_constraint) &  \
                      (self.df['pim_type'] == pim_type.name) & \
                      (self.df['dram_impl'] == self.dram_impl) & \
                      (self.df['timing_preset'] == self.dram_timing_preset)]
        if row.empty:
            return self.run(pim_type, layer, power_constraint)

        else:
            cycle = int(row.iloc[0]['cycle'])
            mac = int(row.iloc[0]['mac'])
            mvgb = int(row.iloc[0]['mvgb'])
            mvsb = int(row.iloc[0]['mvsb'])
            wrgb = int(row.iloc[0]['wrgb'])

            ## si, tsv, giomux to bgmux, bgmux to column decoder, bank RD
            traffic = self._build_traffic(
                pim_type, mac, mvgb, mvsb, wrgb, num_ops_group)
            exec_time = self._cycles_to_seconds(cycle, num_ops_group)
            return exec_time, traffic

    def run_online_serving(self,
                           requests_csv,
                           template_dir,
                           cost_gpu_csv,
                           cost_hybrid_csv,
                           hybrid_command_source="template_replay",
                           generator_backend=None,
                           generator_dhead=None,
                           generator_heads_per_hbm=None,
                           generator_dtype_bytes=None,
                           generator_num_layers=None,
                           generator_num_heads=None,
                           generator_channel_count=None,
                           translation_max_addr=None,
                           translation_pagesize_kb=4,
                           kv_pool_bytes=None,
                           score_pool_bytes=None,
                           context_pool_bytes=None,
                           scratch_pool_bytes=None,
                           route_policy="min_finish",
                           pim_wait_threshold_ms=1.0,
                           unsupported_policy="clip",
                           output_summary_json="cluster_outputs/online_serving_summary.json",
                           output_requests_csv="cluster_outputs/online_serving_requests.csv",
                           output_kv_placement_csv="",
                           output_allocator_pressure_csv="",
                           yaml_file="ramulator2/online_serving.yaml",
                           arrival_time_scale=1.0,
                           lin_bucket=1,
                           lout_bucket=1,
                           batch_size=1,
                           gpu_route_name="gpu_only",
                           hybrid_route_name="lpddr5_pim_bank",
                           enable_prefill_batching=False,
                           max_prefill_batch_size=1,
                           enable_decode_batching=False,
                           max_decode_batch_size=1,
                           prefill_guard_ms=-1.0,
                           max_consecutive_decode_batches=0,
                           decode_batch_cap_with_prefill=0,
                           local_scheduling_policy="prefill_priority_fcfs_decode",
                           fcfs_decode_prediction_mode="incremental",
                           enable_decode_rebind=False,
                           decode_rebind_margin_ms=0.0,
                           enable_predictive_admission=False,
                           enable_slo_guarded_admission=False,
                           slo_tbt_ms=-1.0,
                           slo_e2e_ms=-1.0,
                           slo_ttft_ms=-1.0,
                           admission_shadow_max_steps=10000,
                           admission_retry_interval_ms=0.0,
                           admission_max_harmed_requests=0,
                           admission_max_total_harm_ms=0.0,
                           admission_max_single_harm_ms=0.0,
                           admission_max_own_miss_ms=0.0,
                           admission_check_own_slo=False,
                           admission_max_bypass_count=0,
                           admission_max_wait_ms=0.0,
                           admission_aging_harm_ms_per_ms=0.0,
                           admission_aging_harmed_requests_per_ms=0.0,
                           share_gpu_prefill_across_routes=False,
                           gpu_queue_alpha=0.0,
                           pim_queue_alpha=0.0,
                           active_request_alpha=0.0,
                           decode_token_alpha=0.0,
                           prompt_priority=True,
                           prefer_gpu_on_tie=True,
                           energy_latency_guard_ms=5.0,
                           route_load_balance_ms_per_active_request=0.0,
                           capture_debug_events=False,
                           debug_log_path="log/online_serving/frontend_debug.log",
                           online_frontend="realtime"):
        self._validate_online_serving_config(route_policy=route_policy,
                                             batch_size=batch_size,
                                             local_scheduling_policy=local_scheduling_policy,
                                             fcfs_decode_prediction_mode=fcfs_decode_prediction_mode,
                                             enable_predictive_admission=enable_predictive_admission,
                                             enable_slo_guarded_admission=enable_slo_guarded_admission,
                                             enable_prefill_batching=enable_prefill_batching,
                                             slo_tbt_ms=slo_tbt_ms,
                                             slo_e2e_ms=slo_e2e_ms,
                                             online_frontend=online_frontend)
        Path(os.path.dirname(output_summary_json) or ".").mkdir(
            parents=True, exist_ok=True)
        Path(os.path.dirname(output_requests_csv) or ".").mkdir(
            parents=True, exist_ok=True)
        if output_kv_placement_csv:
            Path(os.path.dirname(output_kv_placement_csv) or ".").mkdir(
                parents=True, exist_ok=True)
        if output_allocator_pressure_csv:
            Path(os.path.dirname(output_allocator_pressure_csv) or ".").mkdir(
                parents=True, exist_ok=True)

        generator_backend = generator_backend or "lpddr5_bank"
        generator_dhead = int(generator_dhead or self.dhead)
        generator_heads_per_hbm = int(generator_heads_per_hbm or math.ceil(self.nhead / self.num_hbm))
        generator_dtype_bytes = int(generator_dtype_bytes or self.dtype_bytes)
        generator_num_layers = int(generator_num_layers or self.num_layers)
        generator_num_heads = int(generator_num_heads or self.nhead)
        generator_channel_count = int(generator_channel_count or self.channel_count)
        translation_max_addr = int(translation_max_addr or (2 * 1024 * 1024 * 1024))
        if not kv_pool_bytes:
            kv_pool_bytes = int(translation_max_addr * 0.70)
        remaining_bytes = max(0, translation_max_addr - int(kv_pool_bytes))
        default_aux_pool = remaining_bytes // 3 if remaining_bytes > 0 else 0
        score_pool_bytes = int(score_pool_bytes if score_pool_bytes else default_aux_pool)
        context_pool_bytes = int(context_pool_bytes if context_pool_bytes else default_aux_pool)
        scratch_pool_bytes = int(
            scratch_pool_bytes if scratch_pool_bytes
            else max(0, remaining_bytes - score_pool_bytes - context_pool_bytes))

        self.make_online_yaml_file(
            yaml_file=yaml_file,
            requests_csv=requests_csv,
            requests_out_csv=output_requests_csv,
            kv_placement_csv=output_kv_placement_csv,
            allocator_pressure_csv=output_allocator_pressure_csv,
            template_dir=template_dir,
            hybrid_command_source=hybrid_command_source,
            generator_backend=generator_backend,
            generator_dhead=generator_dhead,
            generator_heads_per_hbm=generator_heads_per_hbm,
            generator_dtype_bytes=generator_dtype_bytes,
            generator_num_layers=generator_num_layers,
            generator_num_heads=generator_num_heads,
            generator_channel_count=generator_channel_count,
            translation_max_addr=translation_max_addr,
            translation_pagesize_kb=translation_pagesize_kb,
            kv_pool_bytes=kv_pool_bytes,
            score_pool_bytes=score_pool_bytes,
            context_pool_bytes=context_pool_bytes,
            scratch_pool_bytes=scratch_pool_bytes,
            cost_gpu_csv=cost_gpu_csv,
            cost_hybrid_csv=cost_hybrid_csv,
            route_policy=route_policy,
            pim_wait_threshold_ms=pim_wait_threshold_ms,
            unsupported_policy=unsupported_policy,
            arrival_time_scale=arrival_time_scale,
            lin_bucket=lin_bucket,
            lout_bucket=lout_bucket,
            batch_size=batch_size,
            gpu_route_name=gpu_route_name,
            hybrid_route_name=hybrid_route_name,
            enable_prefill_batching=enable_prefill_batching,
            max_prefill_batch_size=max_prefill_batch_size,
            enable_decode_batching=enable_decode_batching,
            max_decode_batch_size=max_decode_batch_size,
            prefill_guard_ms=prefill_guard_ms,
            max_consecutive_decode_batches=max_consecutive_decode_batches,
            decode_batch_cap_with_prefill=decode_batch_cap_with_prefill,
            local_scheduling_policy=local_scheduling_policy,
            fcfs_decode_prediction_mode=fcfs_decode_prediction_mode,
            enable_decode_rebind=enable_decode_rebind,
            decode_rebind_margin_ms=decode_rebind_margin_ms,
            enable_predictive_admission=enable_predictive_admission,
            enable_slo_guarded_admission=enable_slo_guarded_admission,
            slo_tbt_ms=slo_tbt_ms,
            slo_e2e_ms=slo_e2e_ms,
            slo_ttft_ms=slo_ttft_ms,
            admission_shadow_max_steps=admission_shadow_max_steps,
            admission_retry_interval_ms=admission_retry_interval_ms,
            admission_max_harmed_requests=admission_max_harmed_requests,
            admission_max_total_harm_ms=admission_max_total_harm_ms,
            admission_max_single_harm_ms=admission_max_single_harm_ms,
            admission_max_own_miss_ms=admission_max_own_miss_ms,
            admission_check_own_slo=admission_check_own_slo,
            admission_max_bypass_count=admission_max_bypass_count,
            admission_max_wait_ms=admission_max_wait_ms,
            admission_aging_harm_ms_per_ms=admission_aging_harm_ms_per_ms,
            admission_aging_harmed_requests_per_ms=admission_aging_harmed_requests_per_ms,
            share_gpu_prefill_across_routes=share_gpu_prefill_across_routes,
            gpu_queue_alpha=gpu_queue_alpha,
            pim_queue_alpha=pim_queue_alpha,
            active_request_alpha=active_request_alpha,
            decode_token_alpha=decode_token_alpha,
            prompt_priority=prompt_priority,
            prefer_gpu_on_tie=prefer_gpu_on_tie,
            energy_latency_guard_ms=energy_latency_guard_ms,
            route_load_balance_ms_per_active_request=route_load_balance_ms_per_active_request,
            capture_debug_events=capture_debug_events,
            debug_log_path=debug_log_path,
            online_frontend=online_frontend)

        ramulator_file = self._resolve_ramulator_binary()
        run_cmd = f"{ramulator_file} -f {yaml_file}"
        result = subprocess.run(run_cmd,
                                stdout=subprocess.PIPE,
                                text=True,
                                shell=True,
                                check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Ramulator online serving failed (code={result.returncode})")

        parsed = self._parse_ramulator_yaml_from_stdout(result.stdout)
        frontend_stats = parsed.get("Frontend", {}) if isinstance(parsed, dict) else {}
        memory_stats = parsed.get("MemorySystem", {}) if isinstance(parsed, dict) else {}
        summary = {
            "frontend": frontend_stats,
            "memory_system": memory_stats,
        }
        with open(output_summary_json, "w") as f:
            json.dump(summary, f, indent=2)

        return summary
