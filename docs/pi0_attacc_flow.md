# PI0 - AttAcc PIM Flow




**Command Summary**
```bash
PI0_DUMP_DIR=/mnt/galactica/ocakir/smolvla/lerobot/pi0_dump python /mnt/galactica/ocakir/smolvla/lerobot/prof/pi0_inference.py     --checkpoint_path lerobot/pi0_base     --device cuda     --warmup_iters 5     --profile_iters 20     --nvtx_layers     --nvtx_leaf_only     --fresh_batch_each_iter     --time_preprocessor     --batch_size 1


python tools/pi0_trace_info.py   --dump-dir /mnt/galactica/ocakir/smolvla/lerobot/pi0_dump   --print-trace-cmds   --emit-yaml   --yaml-prefix pi0

For LPDDR5:
python tools/pi0_trace_info.py \
  --dump-dir /mnt/galactica/ocakir/smolvla/lerobot/pi0_dump \
  --print-trace-cmds \
  --emit-yaml \
  --yaml-prefix pi0 \
  --yaml-target lpddr5-pim


python ramulator2/trace_gen/gen_trace_attacc_bank.py --dhead 256 --nhead 2 --seqlen 867 --maxl 867 --dbyte 4 -o ramulator2/trace_gen/pi0_bank.trace


./ramulator2/ramulator2 -f ramulator2/pi0_bank.yaml | tee ram_out.txt


python tools/parse_ramulator_output.py   --input ram_out.txt   --dump-dir /mnt/galactica/ocakir/smolvla/lerobot/pi0_dump   --heads-per-hbm 2   --total-heads 8 --config-yaml ramulator2/pi0_bank.yaml
```


---

**Goal**

Use PI0’s real attention shapes (sequence length `L`, heads, `dhead`, dtype) to generate AttAcc PIM traces, simulate them in Ramulator2, and interpret the performance.

---

**How We Pick the Attention Layer**

PI0 attention consists of:
1. QKV projections (GPU)
2. Core attention: **score matmul + softmax + context matmul**
3. Output projection (GPU)

AttAcc models **only the core attention** (score/softmax/context) in PIM.  
Therefore, we use PI0’s actual attention shapes to parameterize AttAcc:
- `L` (total sequence length of PI0 attention)
- `num_heads`
- `dhead` (head dimension)
- `dtype` → `dbyte` (2 for FP16/BF16, 4 for FP32)
- `heads_per_hbm = ceil(num_heads / num_hbm)`

If PI0’s Q/K/V shapes match the trace parameters, then the trace corresponds to **the same attention kernel shape**, but executed by AttAcc PIM instead of GPU.

---

**One-Flag PI0 Dump (Masks + Attention Shape)**

We added a unified dump flag so PI0 writes everything needed in one folder.

**Command**
```bash
PI0_DUMP_DIR=/path/to/pi0_dump \
python /path/to/pi0_inference.py ...
```

**Outputs (in PI0_DUMP_DIR)**
- `lang_masks.npy`  
  Purpose: real language token mask; used to compute language length.
- `img_token_mask.npy`  
  Purpose: real image token mask (token-level); used to compute image token count.
- `pi0_attn_info.json`  
  Purpose: attention shape metadata (Q/K/V shapes, `num_heads`, `dhead`, `seqlen`, `dtype`, `num_layers`, etc).

---

**Generate Trace Parameters**

The helper reads the dump folder and produces the correct trace commands and YAMLs.

**Command**
```bash
python tools/pi0_trace_info.py \
  --dump-dir /path/to/pi0_dump \
  --emit-yaml \
  --yaml-prefix pi0 \
  --print-trace-cmds
```

**What it does**
- Reads `pi0_attn_info.json`, `lang_masks.npy`, `img_token_mask.npy`
- Computes `L = img_tokens + lang_len + 1 (state) + chunk_size`
- Computes `heads_per_hbm`
- Emits trace generation commands
- Writes `ramulator2/pi0_{bank,bg,buffer}.yaml`

**Outputs**
- `ramulator2/trace_gen/pi0_bank.trace`
- `ramulator2/trace_gen/pi0_bg.trace`
- `ramulator2/trace_gen/pi0_buffer.trace`
- `ramulator2/pi0_bank.yaml`
- `ramulator2/pi0_bg.yaml`
- `ramulator2/pi0_buffer.yaml`

---

**Run Ramulator2 (PIM Simulation)**

Pick one PIM mode and simulate:

```bash
./ramulator2/ramulator2 -f ramulator2/pi0_bank.yaml | tee ram_out.txt
```

**Outputs**
- `ram_out.txt`  
  Purpose: raw ramulator output (command counts + total cycles).
- `ramulator2/log/pi0_bank/cmd.log`  
  Purpose: detailed command trace log from Ramulator2.

**Key counters in ramulator output**
- `total_num_pim_mac_all_bank_requests`
- `total_num_pim_write_to_gemv_buffer_requests`
- `total_num_pim_move_to_*`
- `memory_system_cycles`

These are for **one trace**, i.e., **one attention layer**, and **heads_per_hbm** only.

---

**Parse and Scale Results**

Convert cycles → time, estimate bytes moved, and scale to full model:

```bash
python tools/parse_ramulator_output.py \
  --input ram_out.txt \
  --dump-dir /path/to/pi0_dump \
  --heads-per-hbm 2 \
  --total-heads 8
```

**Outputs**
- Per-trace time in µs
- Per-trace bytes moved (prefetch size = 32B)
- Scaled totals using:
  - `num_layers` from `pi0_attn_info.json`
  - `total_heads` from `pi0_attn_info.json`
  - `heads_per_hbm` (user input)
  - `batch` (from JSON or user override)

**Scaling math**

The trace models **heads_per_hbm** only. Full attention scaling is:

```
scale = (total_heads / heads_per_hbm) * num_layers * batch
```

Example for PI0:
```
scale = (8 / 2) * 18 * 1 = 72
```

---

**Raw Command Outputs (Example)**

Ramulator run:
```bash
./ramulator2/ramulator2 -f ramulator2/pi0_bank.yaml | tee ram_out.txt
```

Parsed summary (bank, scaled):
```
cycles: 7016
time_us (per trace): 5.397
head_scale: 4.000
layers: 18, batch: 1
time_us_total: 388.578
mem_acc_total: 3.867 MB
```

---

**Summary of Key Files**

Inputs:
- `PI0_DUMP_DIR/lang_masks.npy`
- `PI0_DUMP_DIR/img_token_mask.npy`
- `PI0_DUMP_DIR/pi0_attn_info.json`
- `ramulator2/trace_gen/pi0_*.trace`
- `ramulator2/pi0_*.yaml`
- `ram_out.txt`

Outputs:
- `ram_out.txt` (raw ramulator stdout)
- Parsed metrics from `tools/parse_ramulator_output.py`

---

**Notes / Limitations**

- AttAcc trace models **only core attention** (score, softmax, context).
- QKV projections and output projection stay on GPU and are not in this PIM trace.
- If you want end-to-end PI0 timing, you must add GPU layers separately.

---

**Final PI0 Attention Results (L=867, dhead=256, 8 heads, 18 layers)**

| PIM Mode | Per-trace Time | Full Attention Time (all heads × layers) | Mem Access Total |
|---|---|---|---|
| Bank | 5.397 µs | 0.389 ms | 3.87 MB |
| BG | 17.893 µs | 1.288 ms | 15.47 MB |
| Buffer | 47.361 µs | 3.410 ms | 122.06 MB |

**Key Observations**
- Bank PIM is the fastest.
- BG is ~3.3× slower than bank.
- Buffer is ~8.8× slower than bank and has much higher memory access.

---

**GPU Comparison (Attention Core Only)**


Extract the core attention time from the NCU CSV:

```bash
python tools/parse_ncu_attn_core.py \
  --input /home/okan/lerobot/prof/ncu_roofline_nvtx_fury0_pi0_full_launch_bs1_blocks/ncu_roofline_nvtx_fury0_pi0_full_launch_bs1_blocks.csv \
  --iters 200 \
  --include-pattern "BLOCK.attn"
```

This prints:
- `duration_unit_in_csv`
- `core_time_us_total` / `core_time_ms_total`
- `core_time_us_per_iter_(N_iters)` / `core_time_ms_per_iter_(N_iters)`

**Compare**

Compare the GPU per‑iteration attention core time with the PIM **full attention time**:
- PIM (bank, full heads × layers): **0.389 ms**
- GPU attention core (per iteration): computed from NCU

This is an apples‑to‑apples comparison of **attention core only**.

---

**Trace Generator Globals**

These are the key globals at the top of `ramulator2/trace_gen/gen_trace_attacc_bank.py`
and how they affect trace generation.

**Model / data parameters**
- `dhead`: attention head dimension. Sets the inner MAC dimension.
- `max_L`: maximum sequence length used for address partition sizing.
- `data_size`: bytes per element (2 for FP16/BF16, 4 for FP32).
- `n_mac`: elements per MAC = `prefetch_size / data_size`.  
  Example: 32B prefetch and FP16 (2B) → `n_mac = 16`.

**HBM topology parameters**
These define the assumed HBM hierarchy for address mapping:
- `n_channel`: channels per HBM stack
- `n_pch`: pseudo‑channels per channel
- `n_rank`: ranks per pseudo‑channel
- `n_bank`: banks per bank group
- `n_bg`: bank groups per rank
- `n_row`: rows per bank
- `n_col`: columns per row
- `prefetch_size`: DRAM access granularity (bytes)


- `dhead`, `L`, and `n_mac` determine how many commands are generated.
- The HBM topology determines where commands land in address space.

---

**HBM3-PIM (HBM3_8Gb_2R, HBM3_5.2Gbps) vs LPDDR5-PIM (LPDDR5_2Gb_x16, LPDDR5_6400) (Bank Mode, PI0 L=867)**

(`nhead=2`, `dhead=256`, same PI0 dump, `L=867`):

| Metric (per trace) | HBM3-PIM Bank | LPDDR5-PIM Bank | LPDDR5 / HBM3 | |
|---|---:|---:|---:|---:|
| `memory_system_cycles` | 7016 | 15774 | 2.248x |  total Ramulator memory cycles to finish the trace.
| `PIM_MAC_AB` | 1760 | 6944 | 3.945x | all-bank MAC ops
| `PIM_MV_GB` | 112 | 224 | 2.000x | move to GEMV buffer
| `PIM_MV_SB` | 192 | 176 | 0.917x | move to softmax buffer
| `mem_acc` | 56,320 B (55.0 KB) | 222,208 B (217.0 KB) | 3.945x | estimated memory-access bytes from MAC commands. mem_acc = (mac_ab + mac_sb + mac_pb) * prefetch_bytes
| `tCK_ps` | 769.231 | 1250.000 | 1.625x | DRAM clock period (ps)
| `time_us` | 5.397 us | 19.718 us | 3.654x | per-trace time in microseconds. time_us = memory_system_cycles * tCK_ps / 1e6
| `time_us_total (scaled)` | 388.578 us | 1419.660 us | 3.653x | scaled to full model execution. time_us_total = time_us * (total_heads / heads_per_hbm) * layers * batch
| `mem_acc_total (scaled)` | 3.867 MB | 15.258 MB | 3.945x | scaled memory-access bytes with same scale factor. (8 / 2) * 18 * 1 = 72


- LPDDR5 bank trace is heavier than HBM3 bank trace for the same PI0 attention shape.
- Main reason is lower structural parallelism used in LPDDR trace generator:
- `ramulator2/trace_gen/gen_trace_attacc_lpddr5_bank.py` uses `n_pch=1`, `n_rank=1`.
- `ramulator2/trace_gen/gen_trace_attacc_bank.py` uses `n_pch=2`, `n_rank=2`.
- This increases MAC partition count in LPDDR trace generation, which directly raises `PIM_MAC_AB` and `mem_acc`.

---

**NVIDIA A100-PCIE-40GB GPU (fury0) vs LPDDR5-PIM (LPDDR5_2Gb_x16, LPDDR5_6400) Comparison (attention core only)**

NCU raw output csv file parse command:

```bash
python tools/parse_ncu_attn_core.py \
  --input /home/okan/lerobot/prof/ncu_nvtx_pi0_bs1_attn_core_times/ncu_nvtx_pi0_bs1_attn_core_times.csv \
  --include-pattern "BLOCK.attn.core_layer" \
  --iters 200
```

GPU times:
- `core_time_ms_total`: `502.305696`
- `core_time_ms_per_iter_(200_iters)`: `2.51152848`

- HBM3-PIM Bank: `388.578 us` = `0.388578 ms`
- LPDDR5-PIM Bank: `1419.660 us` = `1.419660 ms`

GPU (NVIDIA A100-PCIE-40GB)
- vs HBM3-PIM Bank: `2.51152848 / 0.388578 = 6.463x`
- vs LPDDR5-PIM Bank: `2.51152848 / 1.419660 = 1.769x`

- GPU attention core time per iteration is larger than both PIM bank-mode estimates.

---

**GPU vs LPDDR5-PIM Memory Access BW**

GPU:
- `dram_bytes_total`: `19736701440` (`~19.74 GB`)
- `dram_bw_effective_GBps`: `39.2227`
- `dram_bytes_per_iter_(200_iters)`: `98683507.2` (`~98.68 MB`)

LPDDR5-PIM:
- `mem_acc_bw_GBps_total`: `11.270`
- `mem_acc_total`: `15998976 (15.258 MB)`


- GPU/LPDDR5-PIM BW ratio: `39.2227 / 11.270 = 3.48x`

Note:
- GPU BW is from NCU DRAM read/write counters.
- PIM BW is trace derived from `mem_acc` and simulated time.

---

**End-to-end Inference Integration into Main Simulation (PI0 + LPDDR5-PIM)**

This flow is wired into `main.py`/`src` so PI0 can be simulated directly with either HBM3-PIM or LPDDR5-PIM backend selection.

Code integration points:
- `src/config.py`
  - Added `model_table['PI0'] = [18, 2048, 8, 256, 8, 1]` in `make_model_config(...)`.
  - Extended `make_pim_config(...)` with `yaml_target` switch:
    - `hbm3-pim` -> `DRAM_IMPL=HBM3-PIM`, `TRACE_GEN_PREFIX=gen_trace_attacc_`, HBM presets.
    - `lpddr5-pim` -> `DRAM_IMPL=LPDDR5-PIM`, `TRACE_GEN_PREFIX=gen_trace_attacc_lpddr5_`, LPDDR5 presets.
- `main.py`
  - Added CLI option `--yaml-target {hbm3-pim,lpddr5-pim}`.
  - Passes `yaml_target` into `make_pim_config(...)` for `dgx-attacc` runs.
  - `--model PI0` uses the PI0 entry from `make_model_config(...)`.
- `src/ramulator_wrapper.py`
  - Reads DRAM/backend fields from `pim_config` (`DRAM_IMPL`, `ORG_PRESET`, `TIMING_PRESET`, `ADDR_MAPPER_IMPL`, `TRACE_GEN_PREFIX`, etc.).
  - Generates YAML according to selected backend.
  - Chooses trace generator scripts by prefix:
    - HBM3: `gen_trace_attacc_{bank,bg,buffer}.py`
    - LPDDR5: `gen_trace_attacc_lpddr5_{bank,bg,buffer}.py`
  - Uses backend-specific tCK (`TCK_NS`) for cycle->time conversion in main simulation.

How to run from main simulation:

HBM3-PIM:
```bash
python main.py \
  --system dgx-attacc \
  --model PI0 \
  --pim bank \
  --yaml-target hbm3-pim \
  --lin 867 \
  --lout 50 \
  --batch 1
```

LPDDR5-PIM:
```bash
python main.py \
  --system dgx-attacc \
  --model PI0 \
  --pim bank \
  --yaml-target lpddr5-pim \
  --lin 867 \
  --lout 50 \
  --batch 1
```

Notes:
- `--lin` should match the PI0 attention sequence length used in trace studies (e.g., `867` from dump).
- Main simulation uses the analytical transformer pipeline plus ramulator-backed attention timing; it is not a direct replay of Nsight kernels.

---

**Main.py Comparison Results (A100a, ngpu=1, gmemcap=40GB, PI0, lin=867, lout=50, batch=1)**

Commands used:

```bash
python main.py \
  --system dgx \
  --model PI0 \
  --gpu A100a \
  --ngpu 1 \
  --gmemcap 40 \
  --word 2 \
  --lin 867 \
  --lout 50 \
  --batch 1

python main.py \
  --system dgx-attacc \
  --model PI0 \
  --gpu A100a \
  --ngpu 1 \
  --gmemcap 40 \
  --word 2 \
  --pim bank \
  --yaml-target lpddr5-pim \
  --lin 867 \
  --lout 50 \
  --batch 1

python main.py \
  --system dgx-attacc \
  --model PI0 \
  --gpu A100a \
  --ngpu 1 \
  --gmemcap 40 \
  --word 2 \
  --pim bank \
  --yaml-target hbm3-pim \
  --lin 867 \
  --lout 50 \
  --batch 1
```

Observed outputs:

| Mode | Throughput (tokens/s) | Latency (ms) |
|---|---:|---:|
| GPU-only (`dgx`) | 447.65 | 2.23 |
| GPU + LPDDR5-PIM Bank | 561.01 | 1.78 |
| GPU + HBM3-PIM Bank | 609.87 | 1.64 |

Relative to GPU-only:
- LPDDR5-PIM: `+25.3%` throughput, `-20.2%` latency.
- HBM3-PIM: `+36.2%` throughput, `-26.5%` latency.

HBM3-PIM vs LPDDR5-PIM:
- HBM3-PIM: `+8.7%` throughput, `-7.9%` latency.

---

**Output.csv Field-Level Observations (same 3 runs)**

From the `output.csv` rows:

| Metric | GPU-only (`dgx`) | LPDDR5-PIM Bank | HBM3-PIM Bank |
|---|---:|---:|---:|
| `g_time (ms)` | 2.2339 | 1.7825 | 1.6397 |
| `g_matmul` | 0.6263 | 0.1773 | 0.0344 |
| `g_fc` | 1.2080 | 1.2080 | 1.2080 |
| `g_qkv_time` | 0.1790 | 0.1790 | 0.1790 |
| `g_prj_time` | 0.0671 | 0.0671 | 0.0671 |
| `g_ff_time` | 0.9619 | 0.9619 | 0.9619 |
| `s_time` | 20.5804 | 20.5804 | 20.5804 |
| `g_energy (nJ)` | 104,104,140 | 123,087,218 | 105,731,829 |
| `g_attn_mem_energy` | 3,811,310 | 23,206,375 | 5,850,986 |

What this indicates:
- `s_*` values are identical across runs because summarization stage remains on GPU in this model.
- Main speedup comes from generation-side attention path (`g_matmul` drop), which lowers `g_time`.
- HBM3-PIM gives lower `g_time` than LPDDR5-PIM, consistent with Ramulator command/cycle results.
- FC/projection/feedforward parts are unchanged; PIM is not accelerating those in the current split.
- LPDDR5-PIM has noticeably higher attention memory energy (`g_attn_mem_energy`) and total `g_energy` than HBM3-PIM in this setup.

Derived speedups (using `s_time + g_time`):
- HBM3-PIM vs GPU-only: `1.027x` (`22.8143 / 22.2201`)
- LPDDR5-PIM vs GPU-only: `1.020x` (`22.8143 / 22.3629`)
- HBM3-PIM vs LPDDR5-PIM: `1.006x` (`22.3629 / 22.2201`)

About `cap`/`bw` in CSV:
- GPU-only row shows `cap=40`, `bw=1.0` (single GPU baseline).
- PIM rows show larger `cap` and `bw` scale because simulator reports combined GPU+PIM system capacity/bandwidth scale, not Nsight-measured runtime bandwidth.

---

**Similar observations with slightly changed attention in PIM**

Source files:
- `cluster_outputs/output_gpu.csv`
- `cluster_outputs/output_hbm3.csv`
- `cluster_outputs/output_lpddr5.csv`

Same run setup for all three:
- `Lin=867`, `Lout=50`, `bs=1`, `ngpu=1`, `model=PI0`, `gpu=A100a`

Key timing comparison:

| Metric | GPU-only (`dgx`) | LPDDR5-PIM Bank | HBM3-PIM Bank |
|---|---:|---:|---:|
| `g_time (ms)` | 2.2339 | 1.7816 | 1.6395 |
| `g_matmul` | 0.6263 | 0.1764 | 0.0343 |
| `g_fc` | 1.2080 | 1.2080 | 1.2080 |
| `g_qkv_time` | 0.1790 | 0.1790 | 0.1790 |
| `g_prj_time` | 0.0671 | 0.0671 | 0.0671 |
| `g_ff_time` | 0.9619 | 0.9619 | 0.9619 |
| `s_time (ms)` | 20.5804 | 12.7059 | 12.5674 |
| `s_matmul` | 5.9950 | 0.1712 | 0.0333 |
| `s_softmax` | 2.0515 | 0.0008 | 0.0002 |
| `s_fc` | 11.1169 | 11.1169 | 11.1169 |
| `s_time + g_time (ms)` | 22.8144 | 14.4876 | 14.2069 |
| `g_energy (nJ)` | 104,104,140 | 123,086,382 | 105,730,993 |
| `g_attn_mem_energy` | 3,811,310 | 23,206,361 | 5,850,972 |

Derived speedups (using `s_time + g_time`):
- HBM3-PIM vs GPU-only: `1.606x`
- LPDDR5-PIM vs GPU-only: `1.575x`
- HBM3-PIM vs LPDDR5-PIM: `1.020x`

Energy comparison:

| Metric | GPU-only (`dgx`) | LPDDR5-PIM Bank | HBM3-PIM Bank |
|---|---:|---:|---:|
| `g_energy (nJ)` | 104.10M | 123.09M | 105.73M |
| `g_attn_mem_energy (nJ)` | 3.81M | 23.21M | 5.85M |
| `g_attn_comp_energy (nJ)` | 0.522M | 0.084M | 0.084M |

Main interpretation:
- The new offload path is active in both `sum` and `gen`: attention (`score/softmax/context`) shifts to PIM, while FC/proj/FF remain on GPU.
- This is visible from large drops in `s_matmul/s_softmax` and `g_matmul`.
- HBM3-PIM is faster and more energy-efficient than LPDDR5-PIM in this current model setup.
- LPDDR5-PIM still improves latency vs GPU-only, but with significantly higher attention memory energy.
