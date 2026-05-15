import argparse
import csv
import os
import json
from src.system import *
from src.type import *
from src.config import *
from src.ramulator_wrapper import *

RAMULATOR = False


def write_csv(logfile, perfs):
    if logfile is not None:
        firstrow = False
        if not os.path.exists(logfile):
            firstrow = True

        f = open(logfile, 'a')
        wrt = csv.writer(f)
        if firstrow:
            col_name = [
                'model', 'dtype', 'xpu', 'cap', 'bw', 'sys_opb', 'hw', 'cores',
                'pipe_level', 'is parallel', 'power constraint', 'gqa_size',
                'Lin', 'Lout', 'bs', 'required_cap', 's_flops',
                'g_flops', 's_time', 's_matmul', 's_fc', 's_comm', 's_softmax',
                's_act', 's_lnorm', 'g_time (ms)', 'g_matmul', 'g_fc', 'g_comm',
                'g_etc', 'g_qkv_time', 'g_prj_time', 'g_ff_time', 'g2g_comm',
                'c2g_comm', 'g_softmax', 'g_act', 'g_lnorm', 's_energy (nJ)', 'g_energy (nJ)',
                'g_dram_energy', 'g_l2_energy', 'g_l1_energy', 'g_reg_energy',
                'g_alu_energy', 'g_fc_mem_energy', 'g_fc_comp_energy',
                'g_attn_mem_energy', 'g_attn_comp_energy', 'g_etc_mem_energy',
                'g_etc_comp_energy', 'g_comm_energy'
            ]
            wrt.writerow(col_name)

        for perf in perfs:
            tag, config, time, energy = perf
            info = tag + config + time + energy
            wrt.writerow(info)
        f.close()


def run(system: System,
        batch,
        lin,
        lout,
        power_constraint=False,
        pipe=0,
        parallel=False,
        output_file=None):
    print("---Run simple mode Batch {} Lin {} Lout {} pipe {} parall {}---".
          format(batch, lin, lout, pipe, parallel))
    assert system.model_set, "Need to SetModel"
    perfs = []
    system.simulate(batch,
                    lin,
                    lout,
                    perfs=perfs,
                    pipe=pipe,
                    parallel_ff=parallel,
                    power_constraint=power_constraint)
    if output_file is not None:
        write_csv(output_file, perfs)


def _pim_type_from_arg(pim: str):
    if pim == "bg":
        return PIMType.BG
    if pim == "buffer":
        return PIMType.BUFFER
    return PIMType.BA


def _validate_online_serving_args(args):
    if args.system != 'dgx-attacc':
        raise ValueError("--online-serving currently requires --system dgx-attacc")


def _print_run_banner(args):
    if args.online_serving:
        print("{}: ({} x {}), PIM:{}, target:{}, online-serving frontend:{}, requests_csv:{}, batch:{}".format(
            args.system, args.gpu, args.ngpu, args.pim, args.yaml_target,
            args.online_frontend, args.requests_csv, args.batch))
        return

    if args.system == 'dgx-attacc':
        print("{}: ({} x {}), PIM:{}, target:{}, [Lin, Lout, batch]: {}".format(
            args.system, args.gpu, args.ngpu, args.pim, args.yaml_target,
            [args.lin, args.lout, args.batch]))
    else:
        print("{}: ({} x {}), [Lin, Lout, batch]: {}".format(
            args.system, args.gpu, args.ngpu,
            [args.lin, args.lout, args.batch]))


def _run_online_serving(args, modelinfos):
    _validate_online_serving_args(args)

    pim_type = _pim_type_from_arg(args.pim)
    pim_config = make_pim_config(pim_type,
                                 InterfaceType.NVLINK3,
                                 power_constraint=args.powerlimit,
                                 yaml_target=args.yaml_target)
    ramulator = Ramulator(modelinfos,
                          "ramulator2",
                          "ramulator.out",
                          pim_config=pim_config)
    summary = ramulator.run_online_serving(
        requests_csv=args.requests_csv,
        template_dir=args.template_dir,
        hybrid_command_source=args.hybrid_command_source,
        generator_backend=args.generator_backend,
        generator_dhead=args.generator_dhead,
        generator_heads_per_hbm=args.generator_heads_per_hbm,
        generator_dtype_bytes=args.generator_dtype_bytes,
        generator_num_layers=args.generator_num_layers,
        generator_channel_count=args.generator_channel_count,
        translation_max_addr=args.translation_max_addr,
        translation_pagesize_kb=args.translation_pagesize_kb,
        kv_pool_bytes=args.kv_pool_bytes,
        score_pool_bytes=args.score_pool_bytes,
        context_pool_bytes=args.context_pool_bytes,
        scratch_pool_bytes=args.scratch_pool_bytes,
        cost_gpu_csv=args.cost_gpu_csv,
        cost_hybrid_csv=args.cost_hybrid_csv,
        route_policy=args.route_policy,
        pim_wait_threshold_ms=args.pim_wait_threshold_ms,
        unsupported_policy=args.unsupported_policy,
        output_summary_json=args.online_summary_json,
        output_requests_csv=args.online_requests_out,
        output_kv_placement_csv=args.online_kv_placement_out,
        output_allocator_pressure_csv=args.online_allocator_pressure_out,
        yaml_file=args.online_yaml,
        arrival_time_scale=args.arrival_time_scale,
        lin_bucket=args.lin_bucket,
        lout_bucket=args.lout_bucket,
        batch_size=args.batch,
        gpu_route_name=args.gpu_route_name,
        hybrid_route_name=args.hybrid_route_name,
        enable_prefill_batching=args.enable_prefill_batching,
        max_prefill_batch_size=args.max_prefill_batch_size,
        enable_decode_batching=args.enable_decode_batching,
        max_decode_batch_size=args.max_decode_batch_size,
        prefill_guard_ms=args.prefill_guard_ms,
        max_consecutive_decode_batches=args.max_consecutive_decode_batches,
        decode_batch_cap_with_prefill=args.decode_batch_cap_with_prefill,
        local_scheduling_policy=args.local_scheduling_policy,
        fcfs_decode_prediction_mode=args.fcfs_decode_prediction_mode,
        enable_decode_rebind=args.enable_decode_rebind,
        decode_rebind_margin_ms=args.decode_rebind_margin_ms,
        enable_predictive_admission=args.enable_predictive_admission,
        enable_slo_guarded_admission=args.enable_slo_guarded_admission,
        slo_tbt_ms=args.slo_tbt_ms,
        slo_e2e_ms=args.slo_e2e_ms,
        slo_ttft_ms=args.slo_ttft_ms,
        admission_shadow_max_steps=args.admission_shadow_max_steps,
        admission_retry_interval_ms=args.admission_retry_interval_ms,
        admission_max_harmed_requests=args.admission_max_harmed_requests,
        admission_max_total_harm_ms=args.admission_max_total_harm_ms,
        admission_max_single_harm_ms=args.admission_max_single_harm_ms,
        admission_max_own_miss_ms=args.admission_max_own_miss_ms,
        admission_check_own_slo=args.admission_check_own_slo,
        admission_max_bypass_count=args.admission_max_bypass_count,
        admission_max_wait_ms=args.admission_max_wait_ms,
        admission_aging_harm_ms_per_ms=args.admission_aging_harm_ms_per_ms,
        admission_aging_harmed_requests_per_ms=args.admission_aging_harmed_requests_per_ms,
        share_gpu_prefill_across_routes=args.share_gpu_prefill_across_routes,
        gpu_queue_alpha=args.gpu_queue_alpha,
        pim_queue_alpha=args.pim_queue_alpha,
        active_request_alpha=args.active_request_alpha,
        decode_token_alpha=args.decode_token_alpha,
        prefer_gpu_on_tie=args.prefer_gpu_on_tie,
        energy_latency_guard_ms=args.energy_latency_guard_ms,
        route_load_balance_ms_per_active_request=args.route_load_balance_ms_per_active_request,
        prompt_priority=(not args.no_prompt_priority),
        capture_debug_events=args.capture_debug_events,
        debug_log_path=args.debug_log_path,
        online_frontend=args.online_frontend)
    print("Online serving summary:")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Model configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## set system configuration
    parser.add_argument(
        "--system",
        type=str,
        default="dgx",
        help="dgx (each GPU has 80GB HBM), \
              dgx-cpu (In dgx, offloading the attention layer to cpu), \
              dgx-attacc (dgx + attacc)")
    parser.add_argument(
        "--gpu",
        type=str,
        default='A100a',
        help="GPU type (A100a and H100), A100a is A100 with HBM3")
    parser.add_argument("--ngpu",
                        type=int,
                        default=8,
                        help="number of GPUs in DGX system. default=8")
    parser.add_argument("--gmemcap",
                        type=int,
                        default=80,
                        help="memory capacity per GPU (GB). default=80")



    ## set attacc configuration
    parser.add_argument("--pim",
                        type=str,
                        default='bank',
                        help="pim mode. list: bank, bg, buffer")
    parser.add_argument("--powerlimit",
                        action='store_true',
                        help="power constraint for PIM ")
    parser.add_argument("--yaml-target",
                        type=str,
                        default="hbm3-pim",
                        choices=["hbm3-pim", "lpddr5-pim"],
                        help="Ramulator DRAM target profile for PIM runs")
    parser.add_argument("--ffopt",
                        action='store_true',
                        help="apply feedforward parallel optimization")
    parser.add_argument("--pipeopt",
                        action='store_true',
                        help="apply pipeline optimization ")

    ## set model and service environment
    parser.add_argument(
        "--model",
        type=str,
        default='GPT-175B',
        help="model list: GPT-175B, LLAMA-65B, MT-530B, OPT-66B, PI0")
    parser.add_argument("--word",
                        type=int,
                        default='2',
                        help="word size (precision): 1(INT8), 2(FP16)")
    parser.add_argument("--lin",
                        type=int,
                        default=2048,
                        help="input sequence length")
    parser.add_argument("--lout",
                        type=int,
                        default=128,
                        help="number of generated tokens")
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help=
        "batch size, default = 1"
    )
    parser.add_argument("--online-serving",
                        action="store_true",
                        help="Run online request scheduling mode inside Ramulator.")
    parser.add_argument("--online-frontend",
                        type=str,
                        default="realtime",
                        choices=["realtime", "alternating", "serving1_5"],
                        help="Online serving frontend implementation.")
    parser.add_argument("--requests-csv",
                        type=str,
                        default="cluster_outputs/replay_requests_sw.csv",
                        help="Input request CSV for online serving mode.")
    parser.add_argument("--template-dir",
                        type=str,
                        default="ramulator2/trace_gen/templates_lpddr5_bank",
                        help="Directory containing lin_*.trace PIM templates.")
    parser.add_argument("--hybrid-command-source",
                        type=str,
                        default="template_replay",
                        choices=["template_replay", "runtime_generated"],
                        help="Hybrid decode command source.")
    parser.add_argument("--generator-backend",
                        type=str,
                        default="lpddr5_bank",
                        help="Runtime command generator backend.")
    parser.add_argument("--generator-dhead",
                        type=int,
                        default=0,
                        help="Override runtime generator dhead; 0 uses the selected model.")
    parser.add_argument("--generator-heads-per-hbm",
                        type=int,
                        default=0,
                        help="Override runtime generator heads-per-HBM; 0 derives from model and HBM count.")
    parser.add_argument("--generator-dtype-bytes",
                        type=int,
                        default=0,
                        help="Override runtime generator data type size in bytes; 0 derives from model dtype.")
    parser.add_argument("--generator-num-layers",
                        type=int,
                        default=0,
                        help="Override runtime generator decoder layer count; 0 uses the selected model.")
    parser.add_argument("--generator-channel-count",
                        type=int,
                        default=0,
                        help="Override runtime generator channel count; 0 uses the DRAM target profile.")
    parser.add_argument("--translation-max-addr",
                        type=int,
                        default=0,
                        help="Max physical address for runtime-generated serving translation; 0 uses the wrapper default.")
    parser.add_argument("--translation-pagesize-kb",
                        type=int,
                        default=4,
                        help="Page size in KB for runtime-generated serving translation.")
    parser.add_argument("--kv-pool-bytes",
                        type=int,
                        default=0,
                        help="Explicit KV pool size in bytes for runtime-generated serving; 0 uses the wrapper default.")
    parser.add_argument("--score-pool-bytes",
                        type=int,
                        default=0,
                        help="Explicit score pool size in bytes for runtime-generated serving; 0 uses the wrapper default.")
    parser.add_argument("--context-pool-bytes",
                        type=int,
                        default=0,
                        help="Explicit context pool size in bytes for runtime-generated serving; 0 uses the wrapper default.")
    parser.add_argument("--scratch-pool-bytes",
                        type=int,
                        default=0,
                        help="Explicit scratch pool size in bytes for runtime-generated serving; 0 uses the wrapper default.")
    parser.add_argument("--online-yaml",
                        type=str,
                        default="ramulator2/online_serving.yaml",
                        help="Generated Ramulator YAML path in online mode.")
    parser.add_argument("--route-policy",
                        type=str,
                        default="min_finish",
                        choices=["min_finish", "slack_then_finish", "latency_guarded_energy"],
                        help="Route policy in online mode.")
    parser.add_argument("--pim-wait-threshold-ms",
                        type=float,
                        default=0.0,
                        help="Reject PIM route if predicted PIM wait exceeds this value.")
    parser.add_argument("--unsupported-policy",
                        type=str,
                        default="clip",
                        choices=["clip", "nearest", "drop"],
                        help="Unsupported-shape handling in cost lookups.")
    parser.add_argument("--arrival-time-scale",
                        type=float,
                        default=1.0,
                        help="Scale factor applied to request arrival timeline.")
    parser.add_argument("--lin-bucket",
                        type=int,
                        default=1,
                        help="Lin bucketing size for online cost lookup.")
    parser.add_argument("--lout-bucket",
                        type=int,
                        default=1,
                        help="Lout bucketing size for online cost lookup.")
    parser.add_argument("--online-summary-json",
                        type=str,
                        default="cluster_outputs/online_serving_summary.json",
                        help="Online summary JSON output path.")
    parser.add_argument("--online-requests-out",
                        type=str,
                        default="cluster_outputs/online_serving_requests.csv",
                        help="Per-request CSV output path for online mode.")
    parser.add_argument("--online-kv-placement-out",
                        type=str,
                        default="",
                        help="Optional KV placement CSV output path for runtime-generated online mode.")
    parser.add_argument("--online-allocator-pressure-out",
                        type=str,
                        default="",
                        help="Optional allocator pressure CSV output path for online mode.")
    parser.add_argument("--cost-gpu-csv",
                        type=str,
                        default="cluster_outputs/cost_tables/gpu_only.csv",
                        help="Cost table CSV for gpu_only route.")
    parser.add_argument("--cost-hybrid-csv",
                        type=str,
                        default="cluster_outputs/cost_tables/lpddr5_pim_bank.csv",
                        help="Cost table CSV for hybrid route.")
    parser.add_argument("--gpu-route-name",
                        type=str,
                        default="gpu_only",
                        help="Route name for the GPU-only profile.")
    parser.add_argument("--hybrid-route-name",
                        type=str,
                        default="lpddr5_pim_bank",
                        help="Route name for the hybrid profile.")
    parser.add_argument("--enable-prefill-batching",
                        action="store_true",
                        help="Enable batched prefill scheduling.")
    parser.add_argument("--max-prefill-batch-size",
                        type=int,
                        default=1,
                        help="Maximum number of requests in a prefill batch.")
    parser.add_argument("--enable-decode-batching",
                        action="store_true",
                        help="Enable batched decode scheduling.")
    parser.add_argument("--max-decode-batch-size",
                        type=int,
                        default=1,
                        help="Maximum number of requests in a decode batch.")
    parser.add_argument("--prefill-guard-ms",
                        type=float,
                        default=-1.0,
                        help="Force a prefill task when queued prefill wait reaches this threshold.")
    parser.add_argument("--max-consecutive-decode-batches",
                        type=int,
                        default=0,
                        help="Maximum decode batches allowed in a row before forcing prefill.")
    parser.add_argument("--decode-batch-cap-with-prefill",
                        type=int,
                        default=0,
                        help="Cap decode batch size while prefills are waiting.")
    parser.add_argument("--local-scheduling-policy",
                        type=str,
                        default="prefill_priority_fcfs_decode",
                        choices=["priority", "fcfs_strict", "prefill_priority_fcfs_decode"],
                        help="Local scheduling policy inside the realtime serving frontend.")
    parser.add_argument("--fcfs-decode-prediction-mode",
                        type=str,
                        default="incremental",
                        choices=["shadow", "legacy", "incremental", "heuristic_refresh"],
                        help="Forecast mode for FCFS decode scheduling.")
    parser.add_argument("--enable-decode-rebind",
                        action="store_true",
                        help="Allow one-shot route rebind after prefill completes.")
    parser.add_argument("--decode-rebind-margin-ms",
                        type=float,
                        default=0.0,
                        help="Minimum improvement required to rebind decode to gpu_only.")
    parser.add_argument("--enable-predictive-admission",
                        action="store_true",
                        help="Enable predictive admission queueing.")
    parser.add_argument("--enable-slo-guarded-admission",
                        action="store_true",
                        help="Enable SLO-guarded admission queueing.")
    parser.add_argument("--slo-tbt-ms",
                        type=float,
                        default=-1.0,
                        help="Optional TBT SLO threshold.")
    parser.add_argument("--slo-e2e-ms",
                        type=float,
                        default=-1.0,
                        help="Optional E2E SLO threshold.")
    parser.add_argument("--slo-ttft-ms",
                        type=float,
                        default=-1.0,
                        help="Optional TTFT SLO threshold.")
    parser.add_argument("--admission-shadow-max-steps",
                        type=int,
                        default=10000,
                        help="Maximum forecast steps when shadow-projecting admission.")
    parser.add_argument("--admission-retry-interval-ms",
                        type=float,
                        default=0.0,
                        help="Retry interval for held admission requests.")
    parser.add_argument("--admission-max-harmed-requests",
                        type=int,
                        default=0,
                        help="Maximum number of existing requests that may be harmed by an admission.")
    parser.add_argument("--admission-max-total-harm-ms",
                        type=float,
                        default=0.0,
                        help="Maximum total added deadline miss across harmed existing requests.")
    parser.add_argument("--admission-max-single-harm-ms",
                        type=float,
                        default=0.0,
                        help="Maximum added miss for any single existing request.")
    parser.add_argument("--admission-max-own-miss-ms",
                        type=float,
                        default=0.0,
                        help="Maximum allowed own miss when admission_check_own_slo is enabled.")
    parser.add_argument("--admission-check-own-slo",
                        action="store_true",
                        help="Check the candidate request's own SLOs during admission.")
    parser.add_argument("--admission-max-bypass-count",
                        type=int,
                        default=0,
                        help="Force best-effort admission after this many bypasses.")
    parser.add_argument("--admission-max-wait-ms",
                        type=float,
                        default=0.0,
                        help="Force best-effort admission after this queue wait.")
    parser.add_argument("--admission-aging-harm-ms-per-ms",
                        type=float,
                        default=0.0,
                        help="Loosen harm budgets linearly with queue wait.")
    parser.add_argument("--admission-aging-harmed-requests-per-ms",
                        type=float,
                        default=0.0,
                        help="Loosen harmed-request-count budget linearly with queue wait.")
    parser.add_argument("--share-gpu-prefill-across-routes",
                        action="store_true",
                        help="Reuse gpu_only prefill estimates for hybrid routes when available.")
    parser.add_argument("--gpu-queue-alpha",
                        type=float,
                        default=0.0,
                        help="Queue-pressure weight for pending GPU work.")
    parser.add_argument("--pim-queue-alpha",
                        type=float,
                        default=0.0,
                        help="Queue-pressure weight for pending PIM work.")
    parser.add_argument("--active-request-alpha",
                        type=float,
                        default=0.0,
                        help="Queue-pressure weight for active request count.")
    parser.add_argument("--decode-token-alpha",
                        type=float,
                        default=0.0,
                        help="Queue-pressure weight for pending decode tokens.")
    parser.add_argument("--no-prompt-priority",
                        action="store_true",
                        help="Disable prompt/prefill priority when comparing local task candidates.")
    parser.add_argument("--no-prefer-gpu-on-tie",
                        dest="prefer_gpu_on_tie",
                        action="store_false",
                        help="Disable GPU preference for equal-scoring route ties.")
    parser.add_argument("--energy-latency-guard-ms",
                        type=float,
                        default=5.0,
                        help="Latency guard window for latency_guarded_energy route policy.")
    parser.add_argument("--route-load-balance-ms-per-active-request",
                        type=float,
                        default=0.0,
                        help="Per-active-request penalty added to effective route finish time.")
    parser.add_argument("--capture-debug-events",
                        action="store_true",
                        help="Emit structured debug-event lines to the frontend debug log.")
    parser.add_argument("--debug-log-path",
                        type=str,
                        default="",
                        help="Override frontend debug log path in online-serving mode.")
    parser.set_defaults(prefer_gpu_on_tie=True)

    args = parser.parse_args()

    global RAMULATOR
    if RAMULATOR:
        print("The Ramulator {}".format(RAMULATOR))

    if args.gpu == 'H100':
        gpu_device = GPUType.H100
    elif args.gpu == 'A100a':
        gpu_device = GPUType.A100a
    elif args.gpu == 'A6000':
        gpu_device = GPUType.A6000
    else:
        raise ValueError(f"Unknown --gpu '{args.gpu}' (expected H100, A100a, or A6000)")

    num_gpu = args.ngpu
    gmem_cap = args.gmemcap * 1024 * 1024 * 1024
    dtype = DataType.W16A16 if args.word == 2 else DataType.W8A8
    modelinfos = make_model_config(args.model, dtype)
    _print_run_banner(args)

    if args.online_serving:
        _run_online_serving(args, modelinfos)
        return

    output_path = "output.csv"
    if os.path.exists(output_path):
        os.system("rm " + output_path)

    # set system
    xpu_config = make_xpu_config(gpu_device, num_gpu=num_gpu, mem_cap=gmem_cap)
    system = System(xpu_config['GPU'], modelinfos)
    if args.system in ['dgx-attacc']:
        pim_type = _pim_type_from_arg(args.pim)
        pim_config = make_pim_config(pim_type,
                                     InterfaceType.NVLINK3,
                                     power_constraint=args.powerlimit,
                                     yaml_target=args.yaml_target)
        system.set_accelerator(modelinfos, DeviceType.PIM, pim_config)

    elif args.system in ['dgx-cpu']:
        xpu_config = make_xpu_config(gpu_device)
        system.set_xpu(xpu_config['GPU'])
        system.set_accelerator(modelinfos, DeviceType.CPU, xpu_config['CPU'])

    run(system,
        args.batch,
        args.lin,
        args.lout,
        pipe=args.pipeopt,
        parallel=args.ffopt,
        output_file=output_path,
        power_constraint=args.powerlimit)


if __name__ == "__main__":
    main()
