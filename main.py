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
                'c2g_comm', 'g_softmax', 'g_act', 'g_lnorm', 'g_energy (nJ)',
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
    parser.add_argument("--requests-csv",
                        type=str,
                        default="cluster_outputs/replay_requests_sw.csv",
                        help="Input request CSV for online serving mode.")
    parser.add_argument("--template-dir",
                        type=str,
                        default="ramulator2/trace_gen/templates_lpddr5_bank",
                        help="Directory containing lin_*.trace PIM templates.")
    parser.add_argument("--online-yaml",
                        type=str,
                        default="ramulator2/online_serving.yaml",
                        help="Generated Ramulator YAML path in online mode.")
    parser.add_argument("--route-policy",
                        type=str,
                        default="min_finish",
                        choices=["min_finish", "slack_then_finish"],
                        help="Route policy in online mode.")
    parser.add_argument("--pim-wait-threshold-ms",
                        type=float,
                        default=1.0,
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
    parser.add_argument("--cost-gpu-csv",
                        type=str,
                        default="cluster_outputs/cost_tables/gpu_only.csv",
                        help="Cost table CSV for gpu_only route.")
    parser.add_argument("--cost-hybrid-csv",
                        type=str,
                        default="cluster_outputs/cost_tables/lpddr5_pim_bank.csv",
                        help="Cost table CSV for hybrid route.")

    args = parser.parse_args()

    global RAMULATOR
    if RAMULATOR:
        print("The Ramulator {}".format(RAMULATOR))

    if args.gpu == 'H100':
        gpu_device = GPUType.H100
    elif args.gpu == 'A100a':
        gpu_device = GPUType.A100a
    else:
        assert 0

    num_gpu = args.ngpu
    gmem_cap = args.gmemcap * 1024 * 1024 * 1024
    dtype = DataType.W16A16 if args.word == 2 else DataType.W8A8
    modelinfos = make_model_config(args.model, dtype)
    if args.system == 'dgx-attacc':
        print("{}: ({} x {}), PIM:{}, target:{}, [Lin, Lout, batch]: {}".format(
            args.system, args.gpu, args.ngpu, args.pim, args.yaml_target,
            [args.lin, args.lout, args.batch]))
    else:
        print("{}: ({} x {}), [Lin, Lout, batch]: {}".format(
            args.system, args.gpu, args.ngpu,
            [args.lin, args.lout, args.batch]))

    if args.online_serving:
        if args.system != 'dgx-attacc':
            raise ValueError("--online-serving currently requires --system dgx-attacc")
        if args.pim == "bg":
            pim_type = PIMType.BG
        elif args.pim == "buffer":
            pim_type = PIMType.BUFFER
        else:
            pim_type = PIMType.BA

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
            cost_gpu_csv=args.cost_gpu_csv,
            cost_hybrid_csv=args.cost_hybrid_csv,
            route_policy=args.route_policy,
            pim_wait_threshold_ms=args.pim_wait_threshold_ms,
            unsupported_policy=args.unsupported_policy,
            output_summary_json=args.online_summary_json,
            output_requests_csv=args.online_requests_out,
            yaml_file=args.online_yaml,
            arrival_time_scale=args.arrival_time_scale,
            lin_bucket=args.lin_bucket,
            lout_bucket=args.lout_bucket,
            batch_size=args.batch)
        print("Online serving summary:")
        print(json.dumps(summary, indent=2))
        return

    output_path = "output.csv"
    if os.path.exists(output_path):
        os.system("rm " + output_path)

    # set system
    xpu_config = make_xpu_config(gpu_device, num_gpu=num_gpu, mem_cap=gmem_cap)
    system = System(xpu_config['GPU'], modelinfos)
    if args.system in ['dgx-attacc']:
        if args.pim == "bg":
            pim_type = PIMType.BG
        elif args.pim == "buffer":
            pim_type = PIMType.BUFFER
        else:
            pim_type = PIMType.BA
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
