#!/usr/bin/env python3
"""
Unit test for PIM command-count formulas across all four pim_command_mode
values. Computes the expected cmd count from the formula and (optionally)
runs the simulator with --debug-log to verify the actual emitted count.

Formulas per (layer, kv_head) at batch size B with `blocks = ceil(ctx/32)`
(for dtype_bytes=2):

  realistic         :  B · (4·blocks + 4)         # per-request, per-block hoist=no
  simple            :  B · 2                      # per-request, ctx-independent
  attacc_qbroadcast :  2·B·blocks + 6             # Q/V/control hoisted across batch
  bank_stripe       :  2·blocks + 6               # K/V MAC also hoisted (1 per block)

Total cmds = N_LAYERS · N_KV_HEADS · (above).

Usage:
  python tools/test_pim_commands_counts.py                       # formula-only check
  python tools/test_pim_commands_counts.py --run-simulator       # also run sim + parse
"""

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "tools" / "run_serving_online_policy_compare.py"
PY = "/home/okan/anaconda3/envs/lerobot/bin/python"


class Shape(NamedTuple):
    n_layers: int
    n_kv_heads: int
    ctx_tokens: int
    bs: int
    dtype_bytes: int = 2


def blocks(ctx_tokens: int, dtype_bytes: int = 2) -> int:
    tokens_per_block = 64 // dtype_bytes  # kLineBytes=64
    return (ctx_tokens + tokens_per_block - 1) // tokens_per_block


def expected_count(mode: str, s: Shape) -> int:
    """Cmd count assuming all `bs` requests in the batch have ctx == s.ctx_tokens.

    For heterogeneous-ctx batches use expected_count_hetero() with explicit
    per-request ctx list.
    """
    b = blocks(s.ctx_tokens, s.dtype_bytes)
    if mode == "realistic":
        per_layer_kv = s.bs * (4 * b + 4)
    elif mode == "simple":
        per_layer_kv = s.bs * 2
    elif mode == "attacc_qbroadcast":
        per_layer_kv = 2 * s.bs * b + 6
    elif mode == "bank_stripe":
        per_layer_kv = 2 * b + 6
    else:
        raise ValueError(f"unknown mode {mode}")
    return s.n_layers * s.n_kv_heads * per_layer_kv


def expected_count_hetero(mode: str, ctxs: list[int], n_layers: int,
                           n_kv_heads: int, dtype_bytes: int = 2) -> int:
    """Cmd count for heterogeneous-ctx batch. Each request_i has ctx ctxs[i].

    Reduces to expected_count() when all ctxs are equal.
    """
    if not ctxs:
        return 0
    per_req_blocks = [blocks(c, dtype_bytes) for c in ctxs]
    max_b = max(per_req_blocks)
    B = len(ctxs)
    if mode == "realistic":
        per_layer_kv = sum(4 * b + 4 for b in per_req_blocks)
    elif mode == "simple":
        per_layer_kv = B * 2
    elif mode == "attacc_qbroadcast":
        per_layer_kv = 2 * sum(per_req_blocks) + 6
    elif mode == "bank_stripe":
        per_layer_kv = 2 * max_b + 6
    else:
        raise ValueError(f"unknown mode {mode}")
    return n_layers * n_kv_heads * per_layer_kv


# ─── formula-only test ──────────────────────────────────────────────────────

def test_formulas() -> None:
    """Hard-coded reference values for the documented test case."""
    print("\n[test_formulas]")
    # OpenVLA (32L, 32 KV heads) at ctx=630, bs=8.
    # blocks(630) = ceil(630/32) = 20.
    # realistic         : 8 · (80 + 4) = 672 per (L,KVH); × 1024 = 688,128
    # simple            : 8 · 2        = 16 per (L,KVH); × 1024 = 16,384
    # attacc_qbroadcast : 2·8·20 + 6   = 326 per (L,KVH); × 1024 = 333,824
    # bank_stripe       : 2·20 + 6     = 46 per (L,KVH); × 1024 = 47,104
    s = Shape(n_layers=32, n_kv_heads=32, ctx_tokens=630, bs=8)
    expected = {
        "realistic":         688128,
        "simple":             16384,
        "attacc_qbroadcast": 333824,
        "bank_stripe":        47104,
    }
    ok = True
    for mode, want in expected.items():
        got = expected_count(mode, s)
        marker = "OK" if got == want else "FAIL"
        print(f"  {marker:>4}  OpenVLA bs=8 ctx=630 mode={mode:<20} "
              f"want={want:>7}  got={got:>7}")
        if got != want:
            ok = False

    # Pi0 (18L, 1 KV head MQA) at ctx=320, bs=4.
    # blocks(320) = ceil(320/32) = 10.
    # realistic         : 4 · (40 + 4) = 176 per (L,KVH); × 18 = 3168
    # simple            : 4 · 2        = 8 per (L,KVH); × 18 = 144
    # attacc_qbroadcast : 2·4·10 + 6   = 86 per (L,KVH); × 18 = 1548
    # bank_stripe       : 2·10 + 6     = 26 per (L,KVH); × 18 = 468
    s_pi0 = Shape(n_layers=18, n_kv_heads=1, ctx_tokens=320, bs=4)
    expected_pi0 = {
        "realistic":         3168,
        "simple":             144,
        "attacc_qbroadcast": 1548,
        "bank_stripe":        468,
    }
    for mode, want in expected_pi0.items():
        got = expected_count(mode, s_pi0)
        marker = "OK" if got == want else "FAIL"
        print(f"  {marker:>4}  Pi0 bs=4 ctx=320 mode={mode:<20} "
              f"want={want:>7}  got={got:>7}")
        if got != want:
            ok = False

    # bs=1 should give identical counts for AttAccQBroadcast and BankStripe
    # (no batching to amortize across). Verify.
    s1 = Shape(n_layers=4, n_kv_heads=2, ctx_tokens=32, bs=1)
    qb = expected_count("attacc_qbroadcast", s1)
    bs = expected_count("bank_stripe", s1)
    marker = "OK" if qb == bs else "FAIL"
    print(f"  {marker:>4}  bs=1 sanity   attacc_qbroadcast={qb}  bank_stripe={bs}")
    if qb != bs:
        ok = False

    if not ok:
        sys.exit(1)
    print("  [PASS] All formula values match reference table.")


# ─── simulator integration test ─────────────────────────────────────────────

CMD_GEN_RE = re.compile(
    r"CMD_GEN\s+bs=(?P<bs>\d+)\s+max_ctx=\s*(?P<max_ctx>\d+)\s+cmds=(?P<cmds>\d+)")


def test_simulator_mode(mode: str, model: str = "pi0",
                        n_requests: int = 10, kv_pool_gb: float = 4.0,
                        max_decode_batch_size: int = 2,
                        timeout_s: int = 600) -> bool:
    """Run the simulator briefly in `mode`, parse CMD_GEN lines, verify
    actual emitted cmds == expected formula for each (bs, max_ctx, mode)."""
    import tempfile

    if model == "openvla":
        n_layers, n_kv_heads = 32, 32
    elif model == "pi0":
        n_layers, n_kv_heads = 18, 1
    else:
        raise ValueError(f"unknown model {model}")

    # Test artifacts live inside the repo so the runner's relative-path
    # logic works. We clean up the tmpdir at the end.
    tmproot = REPO / "cluster_outputs" / "pim_command_tests"
    tmproot.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"pimtest_{mode}_",
                                     dir=str(tmproot)) as tmpdir:
        cache_path = Path(tmpdir) / f"{model}_{mode}.bin"
        run_dir   = Path(tmpdir) / "run"
        cmd = [
            PY, str(RUNNER),
            "--model", model,
            "--n-requests", str(n_requests),
            "--policies", "max_util_full",
            "--kv-pool-gb", str(kv_pool_gb),
            "--max-active", "32",
            "--max-decode-batch-size", str(max_decode_batch_size),
            "--ml",
            "--pim-command-mode", mode,
            "--pim-cache-file", str(cache_path),
            "--debug-log",
            "--run-dir", str(run_dir),
        ]
        result = subprocess.run(cmd, cwd=str(REPO),
                                capture_output=True, text=True, timeout=timeout_s)
        if result.returncode != 0:
            print(f"  [FAIL] mode={mode}: runner exited {result.returncode}")
            print(f"  stderr tail:\n{result.stderr[-1000:]}")
            return False

        # Parse debug log for CMD_GEN lines.
        debug_logs = list(run_dir.rglob("debug.log"))
        if not debug_logs:
            print(f"  [FAIL] mode={mode}: no debug.log in {run_dir}")
            return False

        # Mode-specific assertion: Simple and BankStripe count commands only
        # against max_ctx (ignoring per-request ctx distribution), so they
        # match the homogeneous-ctx formula exactly. Realistic and AttAccQBroadcast
        # emit per-request commands and may emit FEWER if some batched requests
        # have shorter ctx than max_ctx — so assert `got <= homogeneous_formula`.
        exact_modes = {"simple", "bank_stripe"}
        all_ok = True
        n_checked = 0
        for log_path in debug_logs:
            for line in log_path.read_text().splitlines():
                m = CMD_GEN_RE.search(line)
                if not m:
                    continue
                bs = int(m.group("bs"))
                max_ctx = int(m.group("max_ctx"))
                got = int(m.group("cmds"))
                s = Shape(n_layers=n_layers, n_kv_heads=n_kv_heads,
                          ctx_tokens=max_ctx, bs=bs)
                want = expected_count(mode, s)
                if mode in exact_modes:
                    if got != want:
                        print(f"  [FAIL] {log_path.name} {mode} bs={bs} "
                              f"max_ctx={max_ctx} got={got} want={want}")
                        all_ok = False
                else:
                    # Lower bound: bs=1 special case must equal homogeneous formula.
                    lower = expected_count(mode, Shape(n_layers, n_kv_heads, max_ctx, 1))
                    if not (lower <= got <= want):
                        print(f"  [FAIL] {log_path.name} {mode} bs={bs} "
                              f"max_ctx={max_ctx} got={got} not in [{lower},{want}]")
                        all_ok = False
                n_checked += 1
        if n_checked == 0:
            print(f"  [FAIL] mode={mode}: no CMD_GEN lines parsed from debug.log")
            return False
        if all_ok:
            print(f"  [PASS] mode={mode}: {n_checked} CMD_GEN lines, all consistent with formula.")
        return all_ok


def test_simulator() -> None:
    print("\n[test_simulator] running each mode at n=10, pi0, bs=2 (small to stay fast)")
    overall = True
    for mode in ["realistic", "simple", "attacc_qbroadcast", "bank_stripe"]:
        if not test_simulator_mode(mode):
            overall = False
    if not overall:
        sys.exit(1)


# ─── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-simulator", action="store_true",
                    help="Also run the simulator (~1 min) to check actual emitted "
                         "cmd counts via debug.log CMD_GEN lines.")
    args = ap.parse_args()

    test_formulas()
    if args.run_simulator:
        test_simulator()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
