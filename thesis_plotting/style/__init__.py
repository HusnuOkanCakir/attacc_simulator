"""Shared style module for thesis figures.

Usage:

    from thesis_plotting.style import (
        configure_plotting, save_fig, set_spines, bold, bold_legend,
        COLOR_MODEL, COLOR_ROUTE, COLOR_CATEGORY, COLOR_HEAVY,
        STD_FIGSIZE, WIDE_FIGSIZE,
    )

Always call `configure_plotting()` before creating any figure.
"""

from .config import (
    configure_plotting,
    STD_FIGSIZE, WIDE_FIGSIZE, SQUARE_FIGSIZE, TALL_FIGSIZE,
    DPI_PDF, DPI_PNG,
    FONT_BASE, FONT_LABEL, FONT_LEGEND, FONT_TICK, FONT_TITLE, FONT_ANNOTATE,
)

from .palette import (
    COLOR_MODEL,
    COLOR_ROUTE,
    COLOR_PHASE,
    COLOR_CATEGORY,
    COLOR_HEAVY,
    COLOR_BASELINE,
    COLOR_SEPARATOR,
)

from .helpers import (
    set_spines,
    bold,
    bold_legend,
    category_shade,
    vertical_separators,
    baseline_line,
    annotate_bar,
    save_fig,
    add_panel_label,
    dual_legend,
)

__all__ = [
    "configure_plotting",
    "STD_FIGSIZE", "WIDE_FIGSIZE", "SQUARE_FIGSIZE", "TALL_FIGSIZE",
    "DPI_PDF", "DPI_PNG",
    "FONT_BASE", "FONT_LABEL", "FONT_LEGEND", "FONT_TICK", "FONT_TITLE",
    "FONT_ANNOTATE",
    "COLOR_MODEL", "COLOR_ROUTE", "COLOR_PHASE", "COLOR_CATEGORY",
    "COLOR_HEAVY", "COLOR_BASELINE", "COLOR_SEPARATOR",
    "set_spines", "bold", "bold_legend", "category_shade",
    "vertical_separators", "baseline_line", "annotate_bar",
    "save_fig", "add_panel_label", "dual_legend",
]
