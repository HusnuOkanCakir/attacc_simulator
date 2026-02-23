import pandas as pd
import subprocess
import math
import os
import re
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
        if os.path.exists(output_log):
            self.df = pd.read_csv(output_log)
        self.num_hbm = num_hbm
        self.nhead = modelinfos['num_heads']
        self.dhead = modelinfos['dhead']
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
        self.tCK = float(self.pim_config.get("TCK_NS", self._derive_tck_ns(self.dram_timing_preset)))
        self._ensure_log_schema()

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
        traffic = [i * self.num_hbm for i in traffic]
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
        ramulator_file = os.path.join(self.ramulator_dir, "ramulator2")
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
