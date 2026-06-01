"""Semantic color palettes for thesis figures.

All colors are picked to be:
- print-safe (no pure white-on-pastel)
- distinguishable at thesis figure scale (small panels)
- consistent with the reference scripts at /home/okan/plotting_examples/

The lightgreen / orange / lightcoral trio at alpha=0.3 is the reference
signature for "category background shading" — reused across all
boxplot and bar plots that want to convey low/mid/high regions.
"""

# Per-model dark colors for line plots, scatters, hue-mapping.
COLOR_MODEL = {
    "pi0":     "#1f6e8c",   # deep teal
    "openvla": "#c44e52",   # warm rust
}

# Per-route colors. gpu_only is the "lighter" baseline tone; hybrid /
# lpddr5_pim_bank use the saturated emphasis colors.
COLOR_ROUTE = {
    "gpu_only":          "#9ec1ce",   # light teal — the "baseline"
    "lpddr5_pim_bank":   "#1f6e8c",   # deep teal — the "winning" route
    "hybrid":            "#2ca02c",   # green — emphasized variant
}

# Phase colors (used by KV-pool timelines etc).
COLOR_PHASE = {
    "prefill": "#1f77b4",
    "decode":  "#2ca02c",
    "waiting": "#dddddd",
}

# Background category shading — used for low/mid/high regions of an axis.
# Alpha=0.3 when applied, matching the reference scripts.
COLOR_CATEGORY = {
    "low":  "lightgreen",
    "mid":  "orange",
    "high": "lightcoral",
}

# Saturated palette for grouped bars / 4-6 categorical comparisons.
COLOR_HEAVY = [
    "#1f6e8c",   # deep teal
    "#c44e52",   # rust
    "#2ca02c",   # green
    "#f4a261",   # amber
    "#8172b2",   # purple
    "#5a5a5a",   # neutral grey
]

# Reference / baseline line color (the red dashed y=1.0 signature).
COLOR_BASELINE = "#c0392b"

# Neutral grey for vertical separators between groups.
COLOR_SEPARATOR = "#888888"
