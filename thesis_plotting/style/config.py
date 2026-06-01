"""Figure-sizing + rcParams setup. Call configure_plotting() once at the
top of every thesis_plotting script.

Sizes match the reference scripts (STD_FIGSIZE ~ 6.5 x 2.2 in for a
multi-panel boxplot row). WIDE_FIGSIZE is for sweep-compare style
multi-metric panels.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Reference figure sizes (inches).
STD_FIGSIZE   = (6.5, 2.2)
WIDE_FIGSIZE  = (11.0, 4.5)
SQUARE_FIGSIZE = (5.5, 5.0)
TALL_FIGSIZE  = (6.5, 4.5)

# Output resolution.
DPI_PDF = 300
DPI_PNG = 200

# Font sizes — small enough for two-column thesis layouts.
FONT_BASE      = 7
FONT_LABEL     = 7
FONT_LEGEND    = 6
FONT_TICK      = 6
FONT_TITLE     = 8
FONT_ANNOTATE  = 5


def configure_plotting():
    """Apply the thesis rcParams. Idempotent."""
    mpl.rcdefaults()
    plt.rcParams.update({
        # Fonts
        "font.family":       "serif",
        "font.serif":        ["DejaVu Serif", "STIXGeneral", "Times"],
        "font.size":         FONT_BASE,
        "mathtext.fontset":  "stix",        # matches "STIXGeneral" body
        "axes.labelsize":    FONT_LABEL,
        "axes.titlesize":    FONT_TITLE,
        "legend.fontsize":   FONT_LEGEND,
        "xtick.labelsize":   FONT_TICK,
        "ytick.labelsize":   FONT_TICK,

        # Lines + spines
        "axes.linewidth":    0.5,
        "lines.linewidth":   1.2,
        "patch.linewidth":   0.4,
        "axes.spines.top":   False,
        "axes.spines.right": False,

        # Grid: dashed, thin, low-contrast — reference style.
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "grid.linewidth":    0.3,
        "grid.linestyle":    "--",
        "grid.color":        "0.75",
        "grid.alpha":        0.8,

        # Ticks
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size":  2.5,
        "ytick.major.size":  2.5,
        "xtick.minor.size":  1.5,
        "ytick.minor.size":  1.5,
        "xtick.direction":   "out",
        "ytick.direction":   "out",

        # Legend frame: subtle box for contrast against background shading.
        "legend.frameon":    True,
        "legend.framealpha": 0.95,
        "legend.edgecolor":  "black",
        "legend.fancybox":   False,
        "legend.borderpad":  0.3,
        "legend.handletextpad": 0.5,

        # Layout: constrained is more robust than tight for multi-panel.
        "figure.constrained_layout.use": True,
        "figure.dpi":  120,
        "savefig.dpi": DPI_PDF,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        # PDF compatibility for thesis builds.
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
    })
