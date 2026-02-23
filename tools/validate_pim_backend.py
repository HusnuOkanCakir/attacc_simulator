#!/usr/bin/env python3
"""
Static validation checks for HBM3-PIM and LPDDR5-PIM integration.

This is a lightweight "completeness gate" that checks:
1) PIM command/request coverage in DRAM implementation files
2) Frontend/controller support for PIM trace tokens/request types
3) Trace-generator emitted command tokens are parseable by frontend
4) Topology-aware mem-traffic scaling hooks exist in ramulator_wrapper.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable


REQUIRED_PIM_COMMANDS = [
    "MACAB",
    "MACSB",
    "MACPB",
    "WRGB",
    "MVSB",
    "MVGB",
    "SFM",
    "SETM",
    "SETH",
    "BARRIER",
]

REQUIRED_PIM_REQUESTS = [
    ("pim-mac-all-bank", "MACAB"),
    ("pim-mac-same-bank", "MACSB"),
    ("pim-mac-per-bank", "MACPB"),
    ("pim-write-to-gemv-buffer", "WRGB"),
    ("pim-move-to-softmax-buffer", "MVSB"),
    ("pim-move-to-gemv-buffer", "MVGB"),
    ("pim-softmax", "SFM"),
    ("pim-set-model", "SETM"),
    ("pim-set-head", "SETH"),
    ("pim-barrier", "BARRIER"),
]

FRONTEND_PIM_TOKENS = {
    "PIM_MAC_AB",
    "PIM_MAC_SB",
    "PIM_MAC_PB",
    "PIM_WR_GB",
    "PIM_MV_SB",
    "PIM_MV_GB",
    "PIM_SFM",
    "PIM_SET_MODEL",
    "PIM_SET_HEAD",
    "PIM_BARRIER",
}


@dataclass
class CheckResult:
    ok: bool
    name: str
    detail: str


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _exists(path: str) -> CheckResult:
    return CheckResult(
        ok=os.path.exists(path),
        name=f"exists:{path}",
        detail="found" if os.path.exists(path) else "missing",
    )


def _contains(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.MULTILINE) is not None


def _validate_impl(backend: str, impl_path: str) -> list[CheckResult]:
    text = _read(impl_path)
    out: list[CheckResult] = []

    for cmd in REQUIRED_PIM_COMMANDS:
        out.append(
            CheckResult(
                ok=f"\"{cmd}\"" in text,
                name=f"{backend}:command:{cmd}",
                detail="present in impl",
            )
        )

    for req_name, cmd in REQUIRED_PIM_REQUESTS:
        # Flexible spacing in LUT entries.
        pat = rf'\{{\s*"{re.escape(req_name)}"\s*,\s*"{re.escape(cmd)}"\s*\}}'
        out.append(
            CheckResult(
                ok=_contains(text, pat),
                name=f"{backend}:request-map:{req_name}->{cmd}",
                detail="present in request translations",
            )
        )

    for cmd in ("MACAB", "MACSB", "MACPB"):
        preq_pat = rf'm_preqs\[[^\n]+\]\[m_commands\["{cmd}"\]\]'
        out.append(
            CheckResult(
                ok=_contains(text, preq_pat),
                name=f"{backend}:preq:{cmd}",
                detail="preq handler exists",
            )
        )

    for cmd in ("ACTAB", "ACTSB", "ACTPB"):
        action_pat = rf'm_actions\[[^\n]+\]\[m_commands\["{cmd}"\]\]'
        out.append(
            CheckResult(
                ok=_contains(text, action_pat),
                name=f"{backend}:action:{cmd}",
                detail="action handler exists",
            )
        )

    # Timing-table signal that PIM timing arcs were added.
    for cmd in ("MACAB", "MACSB", "MACPB", "WRGB", "MVSB", "MVGB", "SFM"):
        out.append(
            CheckResult(
                ok=f'{{"{cmd}"' in text,
                name=f"{backend}:timing-ref:{cmd}",
                detail="timing/metadata references exist",
            )
        )

    return out


def _validate_frontend(frontend_path: str) -> list[CheckResult]:
    text = _read(frontend_path)
    out: list[CheckResult] = []
    for token in sorted(FRONTEND_PIM_TOKENS):
        out.append(
            CheckResult(
                ok=token in text,
                name=f"frontend-token:{token}",
                detail="trace token parser/dispatch support",
            )
        )
    return out


def _validate_controller(controller_path: str) -> list[CheckResult]:
    text = _read(controller_path)
    out: list[CheckResult] = []
    for req in (
        "PIM_MAC_AB",
        "PIM_MAC_SB",
        "PIM_MAC_PB",
        "PIM_WR_GB",
        "PIM_MV_SB",
        "PIM_MV_GB",
        "PIM_SFM",
        "PIM_SET_MODEL",
        "PIM_SET_HEAD",
        "PIM_BARRIER",
    ):
        out.append(
            CheckResult(
                ok=f"Request::Type::{req}" in text,
                name=f"controller-request:{req}",
                detail="request routed by controller",
            )
        )
    return out


def _extract_trace_tokens(trace_script_path: str) -> set[str]:
    text = _read(trace_script_path)
    return set(re.findall(r"PIM_[A-Z_]+", text))


def _validate_trace_scripts(trace_scripts: Iterable[str]) -> list[CheckResult]:
    out: list[CheckResult] = []
    for script in trace_scripts:
        if not os.path.exists(script):
            out.append(CheckResult(False, f"trace-script:{script}", "missing"))
            continue
        tokens = _extract_trace_tokens(script)
        unknown = sorted(tokens - FRONTEND_PIM_TOKENS)
        out.append(
            CheckResult(
                ok=len(unknown) == 0,
                name=f"trace-token-compat:{os.path.basename(script)}",
                detail="all emitted PIM_* tokens parseable by frontend"
                if not unknown
                else f"unknown tokens: {unknown}",
            )
        )
    return out


def _validate_wrapper_scaling(wrapper_path: str) -> list[CheckResult]:
    text = _read(wrapper_path)
    checks = []
    checks.append(
        CheckResult(
            ok="_mem_acc_scale" in text,
            name="wrapper:topology-scale-hook",
            detail="topology-aware mem_acc scaling helper exists",
        )
    )
    checks.append(
        CheckResult(
            ok="HBM3_ORG_PRESET_COUNTS" in text and "LPDDR5_ORG_PRESET_COUNTS" in text,
            name="wrapper:org-topology-tables",
            detail="org preset topology tables exist",
        )
    )
    checks.append(
        CheckResult(
            ok="_build_traffic" in text,
            name="wrapper:traffic-builder",
            detail="centralized traffic post-processing exists",
        )
    )
    return checks


def _print_results(results: list[CheckResult]) -> int:
    passed = 0
    for r in results:
        tag = "PASS" if r.ok else "FAIL"
        print(f"[{tag}] {r.name} -- {r.detail}")
        if r.ok:
            passed += 1
    total = len(results)
    print(f"\nSummary: {passed}/{total} checks passed")
    return 0 if passed == total else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate LPDDR5/HBM3 PIM integration completeness checks."
    )
    parser.add_argument(
        "--backend",
        choices=["all", "hbm3-pim", "lpddr5-pim"],
        default="all",
        help="Backend to validate.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root containing ramulator2/ and src/.",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    ramulator_root = os.path.join(repo_root, "ramulator2")

    frontend_path = os.path.join(
        ramulator_root, "src", "frontend", "impl", "memory_trace", "pim_loadstore_trace.cpp"
    )
    controller_path = os.path.join(
        ramulator_root, "src", "dram_controller", "impl", "hbm3_pim_controller.cpp"
    )
    wrapper_path = os.path.join(repo_root, "src", "ramulator_wrapper.py")

    backends = [args.backend] if args.backend != "all" else ["hbm3-pim", "lpddr5-pim"]
    results: list[CheckResult] = []

    for path in (frontend_path, controller_path, wrapper_path):
        results.append(_exists(path))

    backend_meta = {
        "hbm3-pim": {
            "impl": os.path.join(ramulator_root, "src", "dram", "impl", "HBM3-PIM.cpp"),
            "trace_scripts": [
                os.path.join(ramulator_root, "trace_gen", "gen_trace_attacc_bank.py"),
                os.path.join(ramulator_root, "trace_gen", "gen_trace_attacc_bg.py"),
                os.path.join(ramulator_root, "trace_gen", "gen_trace_attacc_buffer.py"),
            ],
        },
        "lpddr5-pim": {
            "impl": os.path.join(ramulator_root, "src", "dram", "impl", "LPDDR5-PIM.cpp"),
            "trace_scripts": [
                os.path.join(ramulator_root, "trace_gen", "gen_trace_attacc_lpddr5_bank.py"),
                os.path.join(ramulator_root, "trace_gen", "gen_trace_attacc_lpddr5_bg.py"),
                os.path.join(ramulator_root, "trace_gen", "gen_trace_attacc_lpddr5_buffer.py"),
            ],
        },
    }

    for backend in backends:
        impl_path = backend_meta[backend]["impl"]
        results.append(_exists(impl_path))
        if os.path.exists(impl_path):
            results.extend(_validate_impl(backend, impl_path))
        results.extend(_validate_trace_scripts(backend_meta[backend]["trace_scripts"]))

    if os.path.exists(frontend_path):
        results.extend(_validate_frontend(frontend_path))
    if os.path.exists(controller_path):
        results.extend(_validate_controller(controller_path))
    if os.path.exists(wrapper_path):
        results.extend(_validate_wrapper_scaling(wrapper_path))

    return _print_results(results)


if __name__ == "__main__":
    sys.exit(main())
