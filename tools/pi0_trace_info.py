#!/usr/bin/env python3
"""
Utility to estimate PI0 attention trace parameters (L, dhead, dtype, heads/HBM)
without requiring a full training/inference run.

This script does not import lerobot to avoid heavy dependencies. It parses
PI0 config/modeling files if available, with safe defaults otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple


DEFAULT_LEROBOT_ROOT = "/home/okan/lerobot"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_PATCH_SIZE = 14


@dataclass
class GemmaConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


def _read_text(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _extract_int(pattern: str, text: str) -> Optional[int]:
    m = re.search(pattern, text)
    if not m:
        return None
    return int(m.group(1))


def _extract_str(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text)
    if not m:
        return None
    return m.group(1)


def _extract_block(text: str, start_pat: str, end_pat: Optional[str]) -> Optional[str]:
    start = text.find(start_pat)
    if start == -1:
        return None
    if end_pat:
        end = text.find(end_pat, start + len(start_pat))
        if end == -1:
            end = len(text)
    else:
        end = len(text)
    return text[start:end]


def _parse_gemma_config(modeling_text: str, variant: str) -> Optional[GemmaConfig]:
    if variant == "gemma_300m":
        start_pat = 'if variant == "gemma_300m":'
        end_pat = 'elif variant == "gemma_2b":'
    elif variant == "gemma_2b":
        start_pat = 'elif variant == "gemma_2b":'
        end_pat = "else:"
    else:
        return None
    block = _extract_block(modeling_text, start_pat, end_pat)
    if not block:
        return None

    def grab(name: str) -> Optional[int]:
        val = _extract_str(rf"{name}\s*=\s*([0-9_]+)", block)
        if val is None:
            return None
        return int(val.replace("_", ""))

    fields = {
        "width": grab("width"),
        "depth": grab("depth"),
        "mlp_dim": grab("mlp_dim"),
        "num_heads": grab("num_heads"),
        "num_kv_heads": grab("num_kv_heads"),
        "head_dim": grab("head_dim"),
    }
    if any(v is None for v in fields.values()):
        return None
    return GemmaConfig(**fields)  # type: ignore[arg-type]


def _default_gemma_config(variant: str) -> Optional[GemmaConfig]:
    defaults = {
        "gemma_300m": GemmaConfig(
            width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256
        ),
        "gemma_2b": GemmaConfig(
            width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256
        ),
    }
    return defaults.get(variant)


def _parse_pi0_defaults(lerobot_root: str) -> Tuple[Optional[int], Optional[str], Optional[GemmaConfig]]:
    config_path = os.path.join(
        lerobot_root, "src", "lerobot", "policies", "pi0", "configuration_pi0.py"
    )
    modeling_path = os.path.join(
        lerobot_root, "src", "lerobot", "policies", "pi0", "modeling_pi0.py"
    )
    config_text = _read_text(config_path) or ""
    modeling_text = _read_text(modeling_path) or ""

    chunk_size = _extract_int(r"chunk_size:\s*int\s*=\s*(\d+)", config_text)
    dtype = _extract_str(r'dtype:\s*str\s*=\s*"([^"]+)"', config_text)

    # paligemma_variant default
    variant = _extract_str(r'paligemma_variant:\s*str\s*=\s*"([^"]+)"', config_text)
    gemma_cfg = None
    if variant and modeling_text:
        gemma_cfg = _parse_gemma_config(modeling_text, variant)
    return chunk_size, dtype, gemma_cfg


def _parse_numeric(val: str) -> float:
    v = val.strip().replace("'", "")
    if v == "" or v.lower() == "nan":
        return 0.0
    try:
        return float(v)
    except ValueError:
        return 0.0


def infer_dtype_from_ncu(csv_path: str, max_rows: int = 5000) -> Optional[str]:
    if not os.path.exists(csv_path):
        return None

    hfma_cols = []
    ffma_cols = []
    dfma_cols = []
    kernel_col = None

    hfma_seen = False
    ffma_seen = False
    dfma_seen = False
    kernel_hint = None

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return None

        for i, name in enumerate(header):
            lname = name.lower()
            if "kernel name" in lname:
                kernel_col = i
            if "op_hfma" in lname:
                hfma_cols.append(i)
            if "op_ffma" in lname:
                ffma_cols.append(i)
            if "op_dfma" in lname:
                dfma_cols.append(i)

        for row_idx, row in enumerate(reader):
            if row_idx >= max_rows:
                break

            if kernel_col is not None and kernel_col < len(row):
                kname = row[kernel_col].lower()
                if "hgemm" in kname or "hmma" in kname or "hfma" in kname:
                    kernel_hint = "fp16/bf16"
                elif "sgemm" in kname or "tf32" in kname:
                    kernel_hint = kernel_hint or "fp32/tf32"
                elif "dgemm" in kname or "dfma" in kname:
                    kernel_hint = kernel_hint or "fp64"

            for idx in hfma_cols:
                if idx < len(row) and _parse_numeric(row[idx]) > 0:
                    hfma_seen = True
                    break
            for idx in ffma_cols:
                if idx < len(row) and _parse_numeric(row[idx]) > 0:
                    ffma_seen = True
                    break
            for idx in dfma_cols:
                if idx < len(row) and _parse_numeric(row[idx]) > 0:
                    dfma_seen = True
                    break

            if hfma_seen and ffma_seen:
                # We already have a strong signal; continue a bit for stability.
                pass

    if hfma_seen:
        return "fp16/bf16 (heuristic: hfma)"
    if dfma_seen:
        return "fp64 (heuristic: dfma)"
    if ffma_seen:
        return "fp32/tf32 (heuristic: ffma)"
    return kernel_hint


def _write_yaml(
    yaml_path: str,
    trace_path: str,
    power_constraint: bool = True,
    yaml_target: str = "hbm3-pim",
    channel_count: int = 16,
) -> None:
    log_base = os.path.splitext(os.path.basename(yaml_path))[0]
    line = ""
    line += "Frontend:\n"
    line += "  impl: PIMLoadStoreTrace\n"
    line += f"  path: {trace_path}\n"
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
    if yaml_target == "hbm3-pim":
        line += "    impl: HBM3-PIM\n"
        line += "    org:\n"
        line += "      preset: HBM3_8Gb_2R\n"
        line += f"      channel: {channel_count}\n"
        line += "    timing:\n"
        if power_constraint:
            line += "      preset: HBM3_5.2Gbps\n"
        else:
            line += "      preset: HBM3_5.2Gbps_NPC\n"
    elif yaml_target == "lpddr5-pim":
        line += "    impl: LPDDR5-PIM\n"
        line += "    org:\n"
        line += "      preset: LPDDR5_2Gb_x16\n"
        line += f"      channel: {channel_count}\n"
        line += "    timing:\n"
        line += "      preset: LPDDR5_6400\n"
    else:
        raise ValueError(f"Unsupported yaml_target: {yaml_target}")
    line += "\n"
    line += "  Controller:\n"
    # Current PIM controller registration is named HBM3-PIM but is generic enough
    # to service PIM requests for other DRAM backends as well.
    line += "    impl: HBM3-PIM\n"
    line += "    Scheduler:\n"
    line += "      impl: PIM\n"
    line += "    RefreshManager:\n"
    if yaml_target == "hbm3-pim":
        line += "      impl: AllBankHBM3\n"
        line += "      #impl: No\n"
    else:
        line += "      impl: AllBank\n"
        line += "      #impl: No\n"
    line += "    plugins:\n"
    line += "    - ControllerPlugin:\n"
    if yaml_target == "hbm3-pim":
        line += "        impl: HBM3TraceRecorder\n"
    else:
        line += "        impl: TraceRecorder\n"
    line += f"        path: ./log/{log_base}/cmd.log\n"
    line += "\n"
    line += "  AddrMapper:\n"
    if yaml_target == "hbm3-pim":
        line += "    impl: HBM3-PIM\n"
    else:
        line += "    impl: ChRaBaRoCo\n"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(line)


def _load_array(path: str, key: Optional[str] = None):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".npy", ".npz"]:
        try:
            import numpy as np  # type: ignore
        except Exception as exc:
            raise SystemExit(f"numpy is required to load {path}: {exc}")
        if ext == ".npy":
            return np.load(path, allow_pickle=False)
        npz = np.load(path, allow_pickle=False)
        if key is None:
            keys = list(npz.keys())
            if not keys:
                raise SystemExit(f"No arrays in npz file: {path}")
            key = keys[0]
        return npz[key]
    if ext in [".pt", ".pth"]:
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise SystemExit(f"torch is required to load {path}: {exc}")
        obj = torch.load(path, map_location="cpu")
        if key is not None:
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                raise SystemExit(f"Key '{key}' not found in {path}")
        if hasattr(obj, "detach"):
            obj = obj.detach().cpu().numpy()
        return obj
    raise SystemExit(f"Unsupported mask file extension: {ext}")


def _infer_len_from_mask(mask_arr) -> int:
    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        raise SystemExit(f"numpy is required to process mask arrays: {exc}")
    arr = np.asarray(mask_arr)
    if arr.ndim == 0:
        return int(arr)
    if arr.ndim == 1:
        return int(np.sum(arr))
    # For 2D or higher: take the first element along leading dims, sum last dim.
    slicer = (0,) * (arr.ndim - 1)
    token_mask = arr[slicer]
    return int(np.sum(token_mask))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate PI0 attention trace parameters (L, dhead, dtype)."
    )
    parser.add_argument("--lerobot-root", type=str, default=DEFAULT_LEROBOT_ROOT)
    parser.add_argument("--paligemma-variant", type=str, default="gemma_2b")
    parser.add_argument("--num-heads", type=int, default=None, help="Override num_heads.")
    parser.add_argument("--head-dim", type=int, default=None, help="Override head_dim (dhead).")
    parser.add_argument("--seqlen", type=int, default=None, help="Override total L.")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--lang-len", type=int, default=None, help="Number of language tokens.")
    parser.add_argument("--lang-mask", type=str, default=None, help="Path to language mask (.npy/.npz/.pt).")
    parser.add_argument("--lang-mask-key", type=str, default=None, help="Key for .npz/.pt language mask.")
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--img-tokens", type=int, default=None, help="Override image token count.")
    parser.add_argument("--img-mask", type=str, default=None, help="Path to image token mask (.npy/.npz/.pt).")
    parser.add_argument("--img-mask-key", type=str, default=None, help="Key for .npz/.pt image mask.")
    parser.add_argument(
        "--dump-dir",
        type=str,
        default=None,
        help="PI0 dump directory containing lang_masks.npy, img_token_mask.npy, and pi0_attn_info.json.",
    )
    parser.add_argument("--img-extra-tokens", type=int, default=0)
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"],
    )
    parser.add_argument("--n-hbm", type=int, default=5)
    parser.add_argument("--ncu-csv", type=str, default=None, help="Optional NCU CSV to infer dtype.")
    parser.add_argument(
        "--trace-dir",
        type=str,
        default="ramulator2/trace_gen",
        help="Directory for generated trace files.",
    )
    parser.add_argument(
        "--emit-yaml",
        action="store_true",
        help="Write ramulator YAMLs alongside traces (bank/bg/buffer).",
    )
    parser.add_argument(
        "--yaml-dir",
        type=str,
        default="ramulator2",
        help="Directory to write YAML files when --emit-yaml is set.",
    )
    parser.add_argument(
        "--yaml-prefix",
        type=str,
        default="pi0",
        help="Prefix for YAML and trace files.",
    )
    parser.add_argument(
        "--yaml-target",
        type=str,
        choices=["hbm3-pim", "lpddr5-pim"],
        default="hbm3-pim",
        help="Select emitted YAML backend profile.",
    )
    parser.add_argument(
        "--yaml-channel-count",
        type=int,
        default=16,
        help="Channel count written into emitted YAML org settings.",
    )
    parser.add_argument("--print-trace-cmds", action="store_true")
    args = parser.parse_args()

    chunk_size_default, dtype_default, gemma_cfg = _parse_pi0_defaults(args.lerobot_root)
    if gemma_cfg is None:
        gemma_cfg = _default_gemma_config(args.paligemma_variant)
    if gemma_cfg is None:
        raise SystemExit(f"Unknown paligemma variant: {args.paligemma_variant}")

    dump_info = {}
    if args.dump_dir:
        info_path = os.path.join(args.dump_dir, "pi0_attn_info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    dump_info = json.load(f)
            except Exception:
                dump_info = {}
        if args.lang_mask is None:
            lm = os.path.join(args.dump_dir, "lang_masks.npy")
            if os.path.exists(lm):
                args.lang_mask = lm
        if args.img_mask is None:
            im = os.path.join(args.dump_dir, "img_token_mask.npy")
            if os.path.exists(im):
                args.img_mask = im

    if "num_heads" in dump_info and args.num_heads is None:
        args.num_heads = int(dump_info["num_heads"])
    if "head_dim" in dump_info and args.head_dim is None:
        args.head_dim = int(dump_info["head_dim"])
    if "dtype" in dump_info and args.dtype is None:
        # dtype in dump is torch.<dtype>
        dtype_str = str(dump_info["dtype"]).replace("torch.", "")
        args.dtype = dtype_str
    if "chunk_size" in dump_info and args.chunk_size is None:
        args.chunk_size = int(dump_info["chunk_size"])
    if "seqlen" in dump_info and args.seqlen is None:
        args.seqlen = int(dump_info["seqlen"])

    if args.num_heads is not None:
        gemma_cfg.num_heads = args.num_heads
    if args.head_dim is not None:
        gemma_cfg.head_dim = args.head_dim

    chunk_size = args.chunk_size if args.chunk_size is not None else (chunk_size_default or 50)
    dtype = args.dtype or dtype_default or "float32"

    inferred_dtype = None
    if args.ncu_csv:
        inferred_dtype = infer_dtype_from_ncu(args.ncu_csv)

    if args.img_mask:
        img_mask_arr = _load_array(args.img_mask, args.img_mask_key)
        num_img_tokens = _infer_len_from_mask(img_mask_arr)
        img_tokens_note = f"inferred from img_mask ({os.path.basename(args.img_mask)})"
    elif args.img_tokens is not None:
        num_img_tokens = args.img_tokens
        img_tokens_note = "user-specified"
    else:
        tokens_per_img = (args.image_size // args.patch_size) ** 2
        num_img_tokens = tokens_per_img * args.num_images + args.img_extra_tokens
        img_tokens_note = "estimated from image_size/patch_size"

    if args.lang_mask:
        lang_mask_arr = _load_array(args.lang_mask, args.lang_mask_key)
        lang_len = _infer_len_from_mask(lang_mask_arr)
        lang_note = f"inferred from lang_mask ({os.path.basename(args.lang_mask)})"
    elif args.lang_len is None:
        lang_len = 0
        lang_note = "lang_len not provided (assumed 0)"
    else:
        lang_len = args.lang_len
        lang_note = "user-specified"

    # L = image tokens + language tokens + 1 state token + chunk_size action tokens
    if args.seqlen is not None:
        L = args.seqlen
        l_note = "user-specified"
    else:
        L = num_img_tokens + lang_len + 1 + chunk_size
        l_note = "computed"

    # Map dtype to bytes
    dtype_norm = dtype.lower()
    if dtype_norm in ["float32", "fp32"]:
        dbyte = 4
    elif dtype_norm in ["bfloat16", "bf16", "float16", "fp16"]:
        dbyte = 2
    else:
        dbyte = 4

    heads_per_hbm = math.ceil(gemma_cfg.num_heads / args.n_hbm)

    print("PI0 attention trace estimate")
    print(f"  paligemma_variant: {args.paligemma_variant}")
    print(f"  num_heads: {gemma_cfg.num_heads}")
    print(f"  head_dim (dhead): {gemma_cfg.head_dim}")
    print(f"  chunk_size: {chunk_size}")
    print(f"  num_img_tokens: {num_img_tokens} ({img_tokens_note})")
    print(f"  lang_len: {lang_len} ({lang_note})")
    print(f"  L (total seq len): {L} ({l_note})")
    print(f"  dtype: {dtype} (dbyte={dbyte})")
    if inferred_dtype:
        print(f"  dtype_inferred_from_ncu: {inferred_dtype}")
    print(f"  heads_per_hbm: {heads_per_hbm} (n_hbm={args.n_hbm})")

    if args.print_trace_cmds:
        trace_dir = args.trace_dir
        base = f"--dhead {gemma_cfg.head_dim} --nhead {heads_per_hbm} --seqlen {L} --maxl {L} --dbyte {dbyte}"
        if args.yaml_target == "lpddr5-pim":
            bank_script = "ramulator2/trace_gen/gen_trace_attacc_lpddr5_bank.py"
            bg_script = "ramulator2/trace_gen/gen_trace_attacc_lpddr5_bg.py"
            buffer_script = "ramulator2/trace_gen/gen_trace_attacc_lpddr5_buffer.py"
        else:
            bank_script = "ramulator2/trace_gen/gen_trace_attacc_bank.py"
            bg_script = "ramulator2/trace_gen/gen_trace_attacc_bg.py"
            buffer_script = "ramulator2/trace_gen/gen_trace_attacc_buffer.py"
        print("\nSuggested trace generation commands:")
        print(
            f"  python {bank_script} {base} "
            f"-o {trace_dir}/{args.yaml_prefix}_bank.trace"
        )
        print(
            f"  python {bg_script} {base} "
            f"-o {trace_dir}/{args.yaml_prefix}_bg.trace"
        )
        print(
            f"  python {buffer_script} {base} "
            f"-o {trace_dir}/{args.yaml_prefix}_buffer.trace"
        )

        if args.emit_yaml:
            print(f"\nYAML target profile: {args.yaml_target}")
            print("\nSuggested ramulator commands:")
            print(f"  ./ramulator2/ramulator2 -f {args.yaml_dir}/{args.yaml_prefix}_bank.yaml")
            print(f"  ./ramulator2/ramulator2 -f {args.yaml_dir}/{args.yaml_prefix}_bg.yaml")
            print(f"  ./ramulator2/ramulator2 -f {args.yaml_dir}/{args.yaml_prefix}_buffer.yaml")
            if args.yaml_target == "lpddr5-pim":
                print(
                    "  note: controller impl remains HBM3-PIM (current generic PIM controller registration name)"
                )

    if args.emit_yaml:
        os.makedirs(args.yaml_dir, exist_ok=True)
        os.makedirs(args.trace_dir, exist_ok=True)
        for mode in ["bank", "bg", "buffer"]:
            yaml_path = os.path.join(args.yaml_dir, f"{args.yaml_prefix}_{mode}.yaml")
            trace_path = os.path.join(args.trace_dir, f"{args.yaml_prefix}_{mode}.trace")
            _write_yaml(
                yaml_path,
                trace_path,
                power_constraint=True,
                yaml_target=args.yaml_target,
                channel_count=args.yaml_channel_count,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
