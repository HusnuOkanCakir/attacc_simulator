"""Fabricate dummy Ramulator-shaped data + run P0_single_detailed.py.

Goal: produce one reference-style figure so we can visually anchor
the thesis_plotting/style/ choices. The output lands at
thesis_plotting/figures/Fig6_IPC_detailed.pdf (and .png if matplotlib
can re-emit it — we just rename).

Usage:
    python thesis_plotting/examples_test/reproduce_P0_with_dummy.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DUMMY_DIR = HERE / "dummy_data"
SWEEP_DIR = DUMMY_DIR / "SWEEPT"
WORKPROF  = DUMMY_DIR / "workprof.csv"

SWEEP_DIR.mkdir(parents=True, exist_ok=True)


# ---- 1. Fabricate workload profile (trace -> RBMPKI) ----------------

TRACES = [
    ("traceA", 1.0),    # low
    ("traceB", 2.0),    # low
    ("traceC", 5.0),    # mid
    ("traceD", 15.0),   # high
    ("traceE", 20.0),   # high
]
pd.DataFrame(
    [{"trace": t, "RBMPKI": r} for t, r in TRACES]
).to_csv(WORKPROF, index=False)


# ---- 2. Fabricate compiled_results.csv ------------------------------

# Schema (subset that P0 actually reads after the parse branch):
# trace, mitigation, cd_threshold, cycles_recorded, energy
THRESHOLDS  = [2**14, 2**17, 2**20]     # 16K, 128K, 1M
MITIGATIONS = ["DUAL", "PROBABILISTIC-3", "PROBABILISTIC-12"]
BASE_INSTS  = 1_000_000_000             # matches load_config()

# Baseline (NONE) per trace. cd_threshold=0 satisfies the
# "cd_threshold > 0" filter (it'll be excluded from the main df).
rng = np.random.default_rng(seed=42)
rows = []
for trace, rbmpki in TRACES:
    base_cycles = BASE_INSTS    # IPC = 1.0
    base_energy = 50.0 + 10.0 * rbmpki
    rows.append({
        "trace": trace, "mitigation": "NONE",
        "cd_threshold": 0,
        "cycles_recorded": base_cycles,
        "energy": base_energy,
    })

# Mitigated cells. Three trends woven in so the plot tells a story:
#   - perf_deg shrinks as cd_threshold loosens (16K < 128K < 1M)
#   - PROBABILISTIC-12 is the cheapest mitigation
#   - DUAL is the most expensive
# Add small noise so geomean isn't degenerate.
MTG_BASE_DEG = {"DUAL": 0.60, "PROBABILISTIC-3": 0.75,
                "PROBABILISTIC-12": 0.85}
THRESH_BONUS = {2**14: 0.00, 2**17: 0.18, 2**20: 0.30}

for trace, rbmpki in TRACES:
    base_cycles = BASE_INSTS
    base_energy = 50.0 + 10.0 * rbmpki
    for thresh in THRESHOLDS:
        for mtg in MITIGATIONS:
            deg = MTG_BASE_DEG[mtg] + THRESH_BONUS[thresh]
            deg *= 1.0 + 0.02 * rng.standard_normal()
            deg = max(0.3, min(deg, 1.05))
            cyc = base_cycles / deg     # IPC_mtg = IPC_base * deg
            energy = base_energy * (1.0 + 0.1 + 0.02 * rng.standard_normal())
            rows.append({
                "trace": trace, "mitigation": mtg,
                "cd_threshold": int(thresh),
                "cycles_recorded": cyc,
                "energy": energy,
            })

pd.DataFrame(rows).to_csv(SWEEP_DIR / "compiled_results.csv", index=False)
print(f"[dummy] wrote {SWEEP_DIR / 'compiled_results.csv'}  ({len(rows)} rows)")
print(f"[dummy] wrote {WORKPROF}")


# ---- 3. Run P0_single_detailed.py with our stub on PYTHONPATH ------

P0 = Path("/home/okan/plotting_examples/P0_single_detailed.py")
if not P0.is_file():
    sys.exit(f"[error] reference script not found: {P0}")

env = os.environ.copy()
env["PYTHONPATH"] = str(HERE) + (
    os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

# The reference script writes to FIGDIR/Fig6_IPC_detailed.pdf
fig_dir = Path("/home/okan/attacc_simulator/thesis_plotting/figures")
out_pdf = fig_dir / "Fig6_IPC_detailed.pdf"
target_pdf = fig_dir / "reference_P0_dummy.pdf"
target_png = fig_dir / "reference_P0_dummy.png"

# Remove old output if any.
if out_pdf.exists():
    out_pdf.unlink()
if target_pdf.exists():
    target_pdf.unlink()
if target_png.exists():
    target_png.unlink()

print(f"[run] {P0.name}  (PYTHONPATH={HERE})")
cp = subprocess.run(
    [sys.executable, str(P0), "--expname", "SWEEPT"],
    env=env,
    cwd=str(HERE),
    capture_output=True, text=True,
)
print(cp.stdout)
if cp.returncode != 0:
    print(cp.stderr)
    sys.exit(f"[error] reference script exited with code {cp.returncode}")


# ---- 4. Rename the output to make it clear it's a reference --------

if out_pdf.exists():
    shutil.move(str(out_pdf), str(target_pdf))
    print(f"[ok] {target_pdf}")

# Also emit a PNG for inline preview.
if target_pdf.exists() and shutil.which("pdftoppm"):
    subprocess.run(
        ["pdftoppm", "-png", "-r", "180",
         str(target_pdf), str(target_png.with_suffix(""))],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # pdftoppm writes <prefix>-1.png; rename.
    candidate = Path(str(target_png.with_suffix("")) + "-1.png")
    if candidate.exists():
        candidate.rename(target_png)
        print(f"[ok] {target_png}")

print("[done] dummy P0 reference figure produced.")
