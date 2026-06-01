"""Stub config module for the reference plotting scripts."""

from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent

FIGDIR       = str(_HERE.parent / "figures")
RESULTS_PATH = str(_HERE / "dummy_data")
PLOT_FMT     = "pdf"

# (label, path) — the reference scripts read TRACE_PROFILE[1] as the
# CSV path; we point to our fabricated workload profile.
TRACE_PROFILE = ("dummy_workprof",
                 str(_HERE / "dummy_data" / "workprof.csv"))


def load_config():
    """Returns the slice the reference scripts actually read."""
    return {
        "Frontend": {
            "num_expected_insts": 1_000_000_000,  # → IPC ≈ 1 when cycles ≈ 1e9
        }
    }
