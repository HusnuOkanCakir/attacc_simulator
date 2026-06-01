# thesis_plotting/style — quick reference

All thesis figures should start with:

```python
from thesis_plotting.style import configure_plotting, save_fig, ...
configure_plotting()
```

## When to call which helper

| Helper | Use it when |
|---|---|
| `configure_plotting()` | **Always**, at the top of every script. Sets fonts, rcParams, grid style. Idempotent. |
| `set_spines(ax)` | After creating each subplot. Removes top + right spines, thins left + bottom to 0.5px. |
| `save_fig(fig, name, out_dir)` | At the end. Writes `<name>.pdf` AND `<name>.png`. Runs `pdfcrop` if available. |
| `bold("Pi0")` | Wrap axis labels / titles / annotations to get LaTeX bold math (`$\mathbf{Pi0}$`). |
| `bold_legend(legend)` | After `ax.legend(...)` to make legend text bold. |
| `category_shade(ax, regions)` | When you want lightgreen/orange/lightcoral background bands. `regions = [(xmin, xmax, "low"), (xmin, xmax, "mid"), (xmin, xmax, "high")]`. |
| `vertical_separators(ax, xs)` | Dashed-grey verticals between category groups (paired with `category_shade`). |
| `baseline_line(ax, y=1.0)` | Red dashed reference line — universal y=1.0 marker for normalized metrics. |
| `annotate_bar(ax, x, y, label)` | Compact text label above a bar. |
| `add_panel_label(ax, "(a)")` | (a)/(b)/(c) panel letters in the corner. |
| `dual_legend(ax, h1, l1, h2, l2)` | Two stacked legends on the same axes (e.g. mitigation + category). |

## Palettes

- `COLOR_MODEL` — `pi0` (deep teal), `openvla` (rust). For per-model lines/bars.
- `COLOR_ROUTE` — `gpu_only` (light teal), `lpddr5_pim_bank` (deep teal), `hybrid` (green). For route comparisons.
- `COLOR_PHASE` — `prefill` (blue), `decode` (green), `waiting` (grey). For KV-pool stacks.
- `COLOR_CATEGORY` — `low`/`mid`/`high` → lightgreen/orange/lightcoral. Applied at alpha=0.3 by `category_shade`.
- `COLOR_HEAVY` — saturated 6-color palette for grouped bars or sweep axes.
- `COLOR_BASELINE` — `#c0392b` (the baseline_line color).
- `COLOR_SEPARATOR` — `#888888` (between-group verticals).

## Figure sizes

- `STD_FIGSIZE = (6.5, 2.2)` — one-row multi-panel boxplots / bars, the reference default.
- `WIDE_FIGSIZE = (11.0, 4.5)` — multi-metric sweep_compare-style plots.
- `SQUARE_FIGSIZE = (5.5, 5.0)` — single-panel scatter/heatmap.
- `TALL_FIGSIZE = (6.5, 4.5)` — two-row stacked plots.

DPI: PDF=300 (paper), PNG=200 (preview).
