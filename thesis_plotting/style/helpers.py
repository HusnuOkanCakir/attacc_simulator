"""Styling helpers ported from the reference scripts at
/home/okan/plotting_examples/. Each function is small and composable —
call from any plotter after configure_plotting().
"""

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt

from . import palette
from . import config as cfg


def set_spines(ax, color: str = "black", linewidth: float = 0.5):
    """Match the reference look: keep only left + bottom spines, thin."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(color)
        ax.spines[side].set_linewidth(linewidth)


def bold(s: str) -> str:
    r"""LaTeX bold-math wrapper. e.g. bold('Pi0') -> r'$\mathbf{Pi0}$'."""
    return r"$\mathbf{" + str(s) + r"}$"


def bold_legend(legend):
    """Apply bold weight to every text item in a legend."""
    if legend is None:
        return
    for txt in legend.get_texts():
        txt.set_fontweight("bold")


def category_shade(ax,
                   regions: Sequence[tuple],
                   alpha: float = 0.3) -> None:
    """Paint alpha=0.3 background bands across the x-axis.

    Each `regions` entry is `(xmin, xmax, category_name)`, where
    category_name is one of palette.COLOR_CATEGORY keys ('low', 'mid',
    'high') OR a literal color string.
    """
    for xmin, xmax, name in regions:
        color = palette.COLOR_CATEGORY.get(name, name)
        ax.axvspan(xmin, xmax, color=color, alpha=alpha, zorder=0)


def vertical_separators(ax, xs: Iterable[float],
                        color: str = palette.COLOR_SEPARATOR,
                        linewidth: float = 0.5):
    """Dashed-grey verticals between category groups."""
    for x in xs:
        ax.axvline(x, color=color, linewidth=linewidth, linestyle="--",
                   alpha=0.6, zorder=1)


def baseline_line(ax, y: float = 1.0,
                  color: str = palette.COLOR_BASELINE,
                  linewidth: float = 0.8,
                  linestyle: str = "--"):
    """Red dashed reference line at y (default y=1.0)."""
    ax.axhline(y, color=color, linewidth=linewidth, linestyle=linestyle,
               zorder=2)


def annotate_bar(ax, x: float, y: float, label: str,
                 color: str = "#444444",
                 fontsize: float = cfg.FONT_ANNOTATE,
                 weight: str = "normal",
                 dy: float = 0.0):
    """Compact bar-top annotation. dy is in y-axis units (caller-handled)."""
    ax.text(x, y + dy, label,
            ha="center", va="bottom",
            fontsize=fontsize, color=color, fontweight=weight)


def save_fig(fig, name: str, out_dir: Path,
             also_png: bool = True) -> dict:
    """Save figure as <out_dir>/<name>.pdf and (optionally) .png.

    If `pdfcrop` is on PATH, run it on the PDF to trim white margins.

    Returns a dict of saved paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"

    fig.savefig(pdf_path, dpi=cfg.DPI_PDF)
    saved = {"pdf": pdf_path}

    if also_png:
        fig.savefig(png_path, dpi=cfg.DPI_PNG)
        saved["png"] = png_path

    if shutil.which("pdfcrop"):
        try:
            subprocess.run(["pdfcrop", str(pdf_path), str(pdf_path)],
                           check=True,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            pass  # output PDF is still usable, just uncropped

    for kind, p in saved.items():
        print(f"  [{kind}] {p}")
    return saved


def add_panel_label(ax, label: str, loc: str = "upper left",
                    weight: str = "bold", offset=(0.02, 0.98)):
    """Add an '(a)' / '(b)' label in the corner of a subplot."""
    ax.text(offset[0], offset[1], label,
            transform=ax.transAxes,
            ha="left" if loc.endswith("left") else "right",
            va="top" if loc.startswith("upper") else "bottom",
            fontsize=cfg.FONT_TITLE, fontweight=weight)


def dual_legend(ax, primary_handles, primary_labels,
                secondary_handles, secondary_labels,
                primary_loc: str = "upper left",
                secondary_loc: str = "lower left"):
    """Pattern from P0/P3: stack two legends on the same axes.

    Returns (primary_legend, secondary_legend).
    """
    primary = ax.legend(primary_handles, primary_labels,
                        loc=primary_loc)
    bold_legend(primary)
    ax.add_artist(primary)
    secondary = ax.legend(secondary_handles, secondary_labels,
                          loc=secondary_loc)
    bold_legend(secondary)
    return primary, secondary
