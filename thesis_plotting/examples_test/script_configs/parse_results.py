"""Stub parse_results module.

Forwards style helpers to thesis_plotting.style; provides minimal
placeholders for the data-loading + naming dictionaries that the
reference scripts import.
"""

import shutil
import subprocess
import sys
from pathlib import Path

# Make thesis_plotting importable when this stub is loaded.
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from thesis_plotting.style import (  # noqa: E402
    configure_plotting as _configure_plotting,
    set_spines as _set_spines,
    bold as _bold,
    bold_legend as _bold_legend,
)


# ---- Style forwards -------------------------------------------------

def configure_plotting():
    _configure_plotting()


def set_spines(ax, *args, **kwargs):
    _set_spines(ax, *args, **kwargs)


def bold(s):
    return _bold(s)


def bold_legend(legend):
    _bold_legend(legend)
    return legend


def crop_pdf(path: str):
    """Run `pdfcrop` if installed; otherwise no-op."""
    if shutil.which("pdfcrop"):
        try:
            subprocess.run(["pdfcrop", path, path],
                           check=True,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            pass


def horizontal_lines(ax, *args, **kwargs):
    """No-op stub — the reference's horizontal_lines just adds extra
    y-axis baselines; for the dummy reproduction we don't need it."""
    pass


# ---- Data / naming ---------------------------------------------------

# Reference uses three mitigation methods. Colors picked to match the
# warm/cool/green pattern seen in P0_single_detailed.py output.
MITIGATION_COLORS = {
    "NONE":              "#888888",
    "DUAL":              "#1f6e8c",   # deep teal
    "PROBABILISTIC-3":   "#c44e52",   # rust
    "PROBABILISTIC-12":  "#2ca02c",   # green
}

RENAME_DICT = {
    "DUAL":             "DUAL",
    "PROBABILISTIC-3":  "PROB-3",
    "PROBABILISTIC-12": "PROB-12",
}

ALERT_SEPARATOR = "---"


def fmt_threshold(n: int) -> str:
    """Format an integer threshold as a bold-math label, e.g.
    fmt_threshold(2**14) -> r'$\\mathbf{N_{CD} = 16K}$'."""
    if n >= 2**20:
        s = f"{n // (2**20)}M"
    elif n >= 2**10:
        s = f"{n // (2**10)}K"
    else:
        s = str(n)
    return r"$\mathbf{N_{CD} = " + s + r"}$"


class RamulatorExperimentCollection:
    """Stub — never actually called in the dummy run path because we
    pre-create the compiled_results.csv. We define it so the import
    succeeds even on import-time evaluation."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Stub RamulatorExperimentCollection invoked — the dummy "
            "reproduction expected compiled_results.csv to already exist."
        )

    def getExperimentList(self):
        return []
