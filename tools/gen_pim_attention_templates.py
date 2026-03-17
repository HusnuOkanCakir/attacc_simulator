#!/usr/bin/env python3
"""Generate PIM attention command templates for online serving frontend.

This wraps the existing LPDDR5 AttAcc trace generator and emits one trace per Lin.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_SCRIPT = ROOT / "ramulator2" / "trace_gen" / "gen_trace_attacc_lpddr5_bank.py"


def _parse_int_list(spec: str) -> List[int]:
    vals: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            parts = [p.strip() for p in chunk.split(":")]
            if len(parts) != 3:
                raise ValueError(f"Invalid range spec: {chunk}")
            start, end, step = (int(parts[0]), int(parts[1]), int(parts[2]))
            if step <= 0:
                raise ValueError(f"step must be > 0 in {chunk}")
            cur = start
            while cur <= end:
                vals.append(cur)
                cur += step
        else:
            vals.append(int(chunk))
    vals = sorted(set(v for v in vals if v > 0))
    if not vals:
        raise ValueError("No valid Lin values parsed")
    return vals


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _run(cmd: Iterable[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(list(cmd), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate LPDDR5-PIM attention templates (score+softmax+context).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lin-values",
        type=str,
        default="128:1024:128,1152:4096:512,867",
        help="Comma/range list. Range format: start:end:step",
    )
    parser.add_argument("--dhead", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=2, help="Heads per HBM in trace generator")
    parser.add_argument("--dbyte", type=int, default=2)
    parser.add_argument("--python-exec", type=str, default=sys.executable)
    parser.add_argument("--trace-script", type=str, default=str(DEFAULT_TRACE_SCRIPT))
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "ramulator2" / "trace_gen" / "templates_lpddr5_bank"),
    )
    parser.add_argument("--force", action="store_true", help="Regenerate existing files")
    parser.add_argument("--manifest", type=str, default="manifest.json")

    args = parser.parse_args()

    lin_values = _parse_int_list(args.lin_values)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    script = Path(args.trace_script)
    if not script.exists():
        raise FileNotFoundError(f"Trace generator script not found: {script}")

    generated = []
    for lin in lin_values:
        out_file = out_dir / f"lin_{lin}.trace"
        if out_file.exists() and not args.force:
            print(f"[skip] {out_file}")
        else:
            cmd = [
                args.python_exec,
                str(script),
                "--dhead",
                str(args.dhead),
                "--nhead",
                str(args.nhead),
                "--seqlen",
                str(lin),
                "--maxl",
                str(lin),
                "--dbyte",
                str(args.dbyte),
                "-o",
                str(out_file),
            ]
            _run(cmd)

        if not out_file.exists():
            raise RuntimeError(f"Template file not generated: {out_file}")

        generated.append(
            {
                "lin": lin,
                "path": str(out_file),
                "commands": _count_lines(out_file),
            }
        )

    manifest_path = out_dir / args.manifest
    manifest = {
        "generator": str(script),
        "dhead": args.dhead,
        "nhead": args.nhead,
        "dbyte": args.dbyte,
        "count": len(generated),
        "templates": generated,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(generated)} templates to {out_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
