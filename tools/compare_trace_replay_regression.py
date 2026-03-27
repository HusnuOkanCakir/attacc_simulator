#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ABS_TOL = 1e-9


def _is_missing(v: Any) -> bool:
    try:
        return pd.isna(v)
    except Exception:
        return False


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _compare_json(a: Any, b: Any, path: str, diffs: List[str]) -> None:
    if _is_missing(a) and _is_missing(b):
        return
    if _is_number(a) and _is_number(b):
        af = float(a)
        bf = float(b)
        if math.isnan(af) and math.isnan(bf):
            return
        if not math.isclose(af, bf, rel_tol=0.0, abs_tol=ABS_TOL):
            diffs.append(f"{path}: {af} != {bf}")
        return
    if type(a) != type(b):
        diffs.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
        return
    if isinstance(a, dict):
        if set(a) != set(b):
            diffs.append(f"{path}: keys {sorted(a)} != {sorted(b)}")
            return
        for key in sorted(a):
            _compare_json(a[key], b[key], f"{path}.{key}" if path else key, diffs)
        return
    if isinstance(a, list):
        if len(a) != len(b):
            diffs.append(f"{path}: len {len(a)} != {len(b)}")
            return
        for i, (av, bv) in enumerate(zip(a, b)):
            _compare_json(av, bv, f"{path}[{i}]", diffs)
        return
    if a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")


def _compare_csv(a_path: Path, b_path: Path, diffs: List[str]) -> None:
    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)
    if list(a.columns) != list(b.columns):
        diffs.append(f"{a_path.name}: columns differ")
        return
    if len(a) != len(b):
        diffs.append(f"{a_path.name}: row count {len(a)} != {len(b)}")
        return
    for row_idx in range(len(a)):
        for col in a.columns:
            av = a.iloc[row_idx][col]
            bv = b.iloc[row_idx][col]
            path = f"{a_path.name}[{row_idx}].{col}"
            if _is_missing(av) and _is_missing(bv):
                continue
            if _is_number(av) and _is_number(bv):
                af = float(av)
                bf = float(bv)
                if math.isnan(af) and math.isnan(bf):
                    continue
                if not math.isclose(af, bf, rel_tol=0.0, abs_tol=ABS_TOL):
                    diffs.append(f"{path}: {af} != {bf}")
                continue
            if av != bv:
                diffs.append(f"{path}: {av!r} != {bv!r}")


def main() -> int:
    p = argparse.ArgumentParser(description="Compare before/after trace replay regression outputs.")
    p.add_argument('--before-dir', type=Path, default=Path('cluster_outputs/refactor_regression/before'))
    p.add_argument('--after-dir', type=Path, default=Path('cluster_outputs/refactor_regression/after'))
    p.add_argument('--report-json', type=Path, default=Path('cluster_outputs/refactor_regression/compare_report.json'))
    p.add_argument('--report-text', type=Path, default=Path('cluster_outputs/refactor_regression/compare_report.txt'))
    args = p.parse_args()

    diffs: Dict[str, List[str]] = {}
    before_files = sorted([p for p in args.before_dir.iterdir() if p.is_file()])
    after_map = {p.name: p for p in args.after_dir.iterdir() if p.is_file()}

    missing_after = [p.name for p in before_files if p.name not in after_map]
    extra_after = sorted(name for name in after_map if not (args.before_dir / name).exists())
    if missing_after:
        diffs['missing_after'] = missing_after
    if extra_after:
        diffs['extra_after'] = extra_after

    for before_path in before_files:
        after_path = after_map.get(before_path.name)
        if after_path is None:
            continue
        file_diffs: List[str] = []
        if before_path.suffix == '.json':
            with before_path.open() as f:
                a = json.load(f)
            with after_path.open() as f:
                b = json.load(f)
            _compare_json(a, b, '', file_diffs)
        elif before_path.suffix == '.csv':
            _compare_csv(before_path, after_path, file_diffs)
        if file_diffs:
            diffs[before_path.name] = file_diffs

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open('w') as f:
        json.dump(diffs, f, indent=2, sort_keys=True)

    lines = []
    if not diffs:
        lines.append('No differences found.')
    else:
        for name, items in diffs.items():
            lines.append(f'## {name}')
            for item in items:
                lines.append(f'- {item}')
            lines.append('')
    args.report_text.write_text('\n'.join(lines).rstrip() + '\n')

    print(f'Wrote JSON report: {args.report_json}')
    print(f'Wrote text report: {args.report_text}')
    if diffs:
        print(f'Found differences in {len(diffs)} entries')
        return 1
    print('No differences found')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
