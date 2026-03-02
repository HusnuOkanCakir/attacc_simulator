#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = ROOT / "main.py"
TMP_OUTPUT_CSV = ROOT / "output.csv"


# Route names here are the same strings we can use later in replay profiles.
ROUTE_PRESETS: Dict[str, Dict[str, str]] = {
    "gpu_only": {
        "system": "dgx",
    },
    "hbm3_pim_bank": {
        "system": "dgx-attacc",
        "pim": "bank",
        "yaml_target": "hbm3-pim",
    },
    "hbm3_pim_bg": {
        "system": "dgx-attacc",
        "pim": "bg",
        "yaml_target": "hbm3-pim",
    },
    "hbm3_pim_buffer": {
        "system": "dgx-attacc",
        "pim": "buffer",
        "yaml_target": "hbm3-pim",
    },
    "lpddr5_pim_bank": {
        "system": "dgx-attacc",
        "pim": "bank",
        "yaml_target": "lpddr5-pim",
    },
    "lpddr5_pim_bg": {
        "system": "dgx-attacc",
        "pim": "bg",
        "yaml_target": "lpddr5-pim",
    },
    "lpddr5_pim_buffer": {
        "system": "dgx-attacc",
        "pim": "buffer",
        "yaml_target": "lpddr5-pim",
    },
    # Short aliases
    "hbm3_bank": {
        "system": "dgx-attacc",
        "pim": "bank",
        "yaml_target": "hbm3-pim",
    },
    "lpddr5_bank": {
        "system": "dgx-attacc",
        "pim": "bank",
        "yaml_target": "lpddr5-pim",
    },
}


@dataclass(frozen=True)
class SweepPoint:
    lin: int
    lout: int
    batch: int


def _parse_int_set_spec(spec: str) -> List[int]:
    """Parse a comma-separated list of ints and/or ranges: '867,768:960:32'."""
    values: Set[int] = set()
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        if ":" not in part:
            values.add(int(part))
            continue
        toks = [t.strip() for t in part.split(":")]
        if len(toks) not in (2, 3):
            raise ValueError(f"Invalid range spec '{part}'. Use start:end[:step].")
        start = int(toks[0])
        end = int(toks[1])
        step = int(toks[2]) if len(toks) == 3 else 1
        if step == 0:
            raise ValueError(f"Invalid range spec '{part}': step must be non-zero.")
        # Inclusive range. Support descending inputs too.
        if start <= end and step < 0:
            step = -step
        if start > end and step > 0:
            step = -step
        cur = start
        if step > 0:
            while cur <= end:
                values.add(cur)
                cur += step
        else:
            while cur >= end:
                values.add(cur)
                cur += step
    out = sorted(v for v in values if v > 0)
    if not out:
        raise ValueError(f"No positive integers parsed from spec '{spec}'.")
    return out


def _read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"{path}: missing CSV header")
    return fieldnames, rows


def _load_existing_keys(path: Path) -> Set[Tuple[int, int, int]]:
    if not path.exists():
        return set()
    _, rows = _read_csv_rows(path)
    keys: Set[Tuple[int, int, int]] = set()
    for r in rows:
        try:
            keys.add((int(float(r["Lin"])), int(float(r["Lout"])), int(float(r["bs"]))))
        except Exception:
            continue
    return keys


def _append_row(path: Path, fieldnames: Sequence[str], row: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _sort_csv_in_place(path: Path) -> None:
    if not path.exists():
        return
    fieldnames, rows = _read_csv_rows(path)
    if not rows:
        return

    def _key(r: Dict[str, str]):
        def _num(k: str) -> float:
            try:
                return float(r.get(k, "nan"))
            except Exception:
                return float("inf")
        return (_num("Lin"), _num("Lout"), _num("bs"))

    rows = sorted(rows, key=_key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _route_to_cmd_args(route: str) -> Dict[str, str]:
    if route not in ROUTE_PRESETS:
        known = ", ".join(sorted(ROUTE_PRESETS.keys()))
        raise ValueError(f"Unknown route '{route}'. Known routes: {known}")
    return dict(ROUTE_PRESETS[route])


def _build_main_cmd(args: argparse.Namespace, route: str, point: SweepPoint) -> List[str]:
    route_cfg = _route_to_cmd_args(route)
    cmd = [
        args.python,
        str(MAIN_PY),
        "--model",
        args.model,
        "--gpu",
        args.gpu,
        "--ngpu",
        str(args.ngpu),
        "--gmemcap",
        str(args.gmemcap),
        "--word",
        str(args.word),
        "--batch",
        str(point.batch),
        "--lin",
        str(point.lin),
        "--lout",
        str(point.lout),
        "--system",
        route_cfg["system"],
    ]
    if "pim" in route_cfg:
        cmd += ["--pim", route_cfg["pim"]]
    if "yaml_target" in route_cfg:
        cmd += ["--yaml-target", route_cfg["yaml_target"]]
    if args.pipeopt:
        cmd.append("--pipeopt")
    if args.ffopt:
        cmd.append("--ffopt")
    if args.powerlimit:
        cmd.append("--powerlimit")
    return cmd


def _run_main_once(cmd: Sequence[str], cwd: Path, timeout_s: Optional[float]) -> None:
    # main.py overwrites ROOT/output.csv on each invocation.
    if TMP_OUTPUT_CSV.exists():
        TMP_OUTPUT_CSV.unlink()
    print("$ " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(list(cmd), cwd=str(cwd), check=True, timeout=timeout_s)
    if not TMP_OUTPUT_CSV.exists():
        raise RuntimeError(f"Expected main.py to write {TMP_OUTPUT_CSV}, but it does not exist.")


def _read_single_output_row(path: Path) -> Tuple[List[str], Dict[str, str]]:
    fieldnames, rows = _read_csv_rows(path)
    if not rows:
        raise RuntimeError(f"{path}: no data rows found")
    if len(rows) != 1:
        # main.py should write one row per invocation; if not, use the last row and warn.
        print(f"[warn] {path} has {len(rows)} rows; using the last row.")
    return fieldnames, rows[-1]


def _point_key(point: SweepPoint) -> Tuple[int, int, int]:
    return (point.lin, point.lout, point.batch)


def _default_outfile(route: str, out_dir: Path) -> Path:
    return out_dir / f"{route}.csv"


def _parse_route_out(item: str) -> Tuple[str, Optional[Path]]:
    if "=" in item:
        route, out = item.split("=", 1)
        route = route.strip()
        out = out.strip()
        if not route:
            raise ValueError(f"Invalid --route '{item}': empty route")
        return route, (Path(out) if out else None)
    return item.strip(), None


def main() -> int:
    p = argparse.ArgumentParser(
        description=("Sweep main.py over (Lin, Lout) points and build replay cost tables "
                     "(output.csv-style) for one or more routes."))
    p.add_argument("--route",
                   action="append",
                   required=True,
                   help=("Route preset or route=output.csv. Examples: gpu_only, "
                         "lpddr5_pim_bank=cluster_outputs/cost_tables/lpddr5_pi0.csv"))
    p.add_argument("--out-dir",
                   type=Path,
                   default=ROOT / "cost_tables",
                   help="Default output directory for per-route CSVs")

    # PI0-focused defaults. Use explicit sweeps to broaden.
    p.add_argument("--lin-values",
                   type=str,
                   default="867",
                   help=("Comma/range spec for Lin values. Examples: '867' or "
                         "'768:960:32,1024'"))
    p.add_argument("--lout-values",
                   type=str,
                   default="50",
                   help="Comma/range spec for Lout values. Example: '32,50,64'")
    p.add_argument("--batch-values",
                   type=str,
                   default="1",
                   help="Comma/range spec for batch sizes. Usually '1' for PI0.")

    p.add_argument("--model", type=str, default="PI0")
    p.add_argument("--gpu", type=str, default="A100a")
    p.add_argument("--ngpu", type=int, default=1)
    p.add_argument("--gmemcap", type=int, default=40, help="Per-GPU memory cap in GB")
    p.add_argument("--word", type=int, default=2, help="2=FP16, 1=INT8")

    p.add_argument("--pipeopt", action="store_true", help="Pass --pipeopt to main.py")
    p.add_argument("--ffopt", action="store_true", help="Pass --ffopt to main.py")
    p.add_argument("--powerlimit", action="store_true", help="Pass --powerlimit to main.py")

    p.add_argument("--python",
                   type=str,
                   default=sys.executable,
                   help="Python interpreter used to invoke main.py")
    p.add_argument("--timeout-s",
                   type=float,
                   default=None,
                   help="Optional timeout per main.py invocation")
    p.add_argument("--resume",
                   action="store_true",
                   help="Skip points already present in target route CSVs (Lin,Lout,bs)")
    p.add_argument("--sort",
                   action="store_true",
                   help="Sort route CSVs by (Lin,Lout,bs) after completion")
    p.add_argument("--dry-run",
                   action="store_true",
                   help="Print planned commands without running main.py")

    args = p.parse_args()

    if not MAIN_PY.exists():
        raise FileNotFoundError(f"Cannot find {MAIN_PY}")

    lin_values = _parse_int_set_spec(args.lin_values)
    lout_values = _parse_int_set_spec(args.lout_values)
    batch_values = _parse_int_set_spec(args.batch_values)

    points = [
        SweepPoint(lin=lin, lout=lout, batch=bs)
        for bs in batch_values
        for lin in lin_values
        for lout in lout_values
    ]
    points = sorted(points, key=lambda pnt: (pnt.batch, pnt.lin, pnt.lout))

    route_to_out: Dict[str, Path] = {}
    for item in args.route:
        route, out_path = _parse_route_out(item)
        if not route:
            raise ValueError("Empty route in --route")
        if route in route_to_out:
            raise ValueError(f"Duplicate route '{route}'")
        # Validate route preset early.
        _route_to_cmd_args(route)
        route_to_out[route] = out_path if out_path is not None else _default_outfile(route, args.out_dir)

    existing_by_route: Dict[str, Set[Tuple[int, int, int]]] = {}
    if args.resume:
        for route, out_path in route_to_out.items():
            existing_by_route[route] = _load_existing_keys(out_path)
    else:
        for route in route_to_out:
            existing_by_route[route] = set()

    total_runs = 0
    skipped = 0

    print("Cost table sweep plan")
    print(f"  routes: {list(route_to_out.keys())}")
    print(f"  Lin values: {lin_values}")
    print(f"  Lout values: {lout_values}")
    print(f"  batch values: {batch_values}")
    print(f"  total points per route: {len(points)}")
    print(f"  total potential invocations: {len(points) * len(route_to_out)}")

    for route, out_path in route_to_out.items():
        print(f"\n[route] {route} -> {out_path}")
        route_runs = 0
        route_skips = 0
        for point in points:
            key = _point_key(point)
            if key in existing_by_route[route]:
                route_skips += 1
                skipped += 1
                continue

            cmd = _build_main_cmd(args, route, point)
            if args.dry_run:
                print("$ " + " ".join(shlex.quote(c) for c in cmd))
                route_runs += 1
                total_runs += 1
                continue

            _run_main_once(cmd, cwd=ROOT, timeout_s=args.timeout_s)
            fieldnames, row = _read_single_output_row(TMP_OUTPUT_CSV)

            # Sanity-check returned point.
            try:
                row_lin = int(float(row["Lin"]))
                row_lout = int(float(row["Lout"]))
                row_bs = int(float(row["bs"]))
            except Exception as e:
                raise RuntimeError(f"main.py output row missing Lin/Lout/bs fields: {e}") from e
            if (row_lin, row_lout, row_bs) != key:
                raise RuntimeError(
                    f"main.py returned unexpected point {(row_lin, row_lout, row_bs)} != {key}")

            _append_row(out_path, fieldnames, row)
            existing_by_route[route].add(key)
            route_runs += 1
            total_runs += 1

        print(f"  completed new runs: {route_runs}")
        print(f"  skipped existing: {route_skips}")
        if args.sort and out_path.exists() and not args.dry_run:
            _sort_csv_in_place(out_path)
            print("  sorted output by (Lin,Lout,bs)")

    print("\nSweep done")
    print(f"  new invocations: {total_runs}")
    print(f"  skipped: {skipped}")
    if args.dry_run:
        print("  mode: dry-run (no commands executed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
