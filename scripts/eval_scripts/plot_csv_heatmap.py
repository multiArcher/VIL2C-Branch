"""Plot raw CSV evaluations on a delay-mean / delay-std grid.

Each CSV row is one equally weighted evaluation. The annotation's SD is
the sample SD between rows, not the delay-distribution std on the y-axis.
Use one map, distribution and checkpoint per input (or filter beforehand).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from study_plotting import aggregate_runs, plt, PercentFormatter


def resolve_column(data, explicit, aliases, role):
    if explicit is not None:
        if explicit not in data:
            raise ValueError(f"Missing {role} column {explicit!r}; available: {list(data)}")
        return explicit
    matches = [name for name in aliases if name in data]
    if len(matches) != 1:
        raise ValueError(f"Specify {role} column explicitly; candidates: {matches}; "
                         f"available: {list(data)}")
    return matches[0]


def plot_csv_heatmap(csv_path, output_prefix, *, mean_col=None, std_col=None,
                     value_col=None, title=None, percent=True):
    """Save PNG/PDF and aggregated CSV; return (summary, mean_grid).

    Recognizes common column names, or accepts explicit column mappings.
    percent=True expects a win-rate fraction in [0, 1]; use percent=False
    for rewards or other numeric metrics. Missing cells remain blank.
    Single observations have undefined sample SD, so no ± label is shown.
    """
    data = pd.read_csv(csv_path)
    for key in ("map", "delay_type", "distribution", "checkpoint", "model_id"):
        if key in data and data[key].nunique(dropna=False) > 1:
            raise ValueError(f"Multiple {key} values: filter to one before plotting")
    mean_col = resolve_column(data, mean_col, ("obs_mean", "delay_mean", "mean"), "mean")
    std_col = resolve_column(data, std_col, ("obs_std", "delay_std", "std"), "std")
    value_col = resolve_column(data, value_col, ("win_rate", "battle_won_mean", "test_battle_won_mean"), "value")
    if len({mean_col, std_col, value_col}) != 3:
        raise ValueError("Mean, std and value columns must be distinct")
    rows = data[[mean_col, std_col, value_col]].copy()
    rows.columns = ["delay_mean", "delay_std", "value"]
    rows = rows.apply(pd.to_numeric, errors="raise")
    if rows.empty or not np.isfinite(rows.to_numpy()).all():
        raise ValueError("Input must contain finite numeric means, stds and values")
    if (rows.delay_std < 0).any():
        raise ValueError("Delay std must be nonnegative")
    if percent and not rows.value.between(0, 1).all():
        raise ValueError("Percentage mode expects fractions in [0, 1]")

    # Same aggregation as aggregate_study.py: equal row weights, ddof=1 SD.
    summary = aggregate_runs(rows, ["delay_mean", "delay_std"], ["value"])
    grid = summary.pivot(index="delay_std", columns="delay_mean", values="value")
    grid = grid.sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(max(7, 1.35 * len(grid.columns)),
                                    max(5, 0.8 * len(grid.index))))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#000ccc")
    im = ax.imshow(np.ma.masked_invalid(grid.to_numpy()), origin="lower",
                   aspect="auto", cmap=cmap,
                   vmin=0 if percent else None, vmax=1 if percent else None)
    for row in summary.itertuples(index=False):
        x = grid.columns.get_loc(row.delay_mean)
        y = grid.index.get_loc(row.delay_std)
        label = f"{row.value:.1%}" if percent else f"{row.value:.2f}"
        if row.value_n > 1:
            sd = 100 * row.value_std if percent else row.value_std
            label += f"\n± {sd:.2f}"
        color = "black" if im.norm(row.value) > 0.55 else "white"
        ax.text(x, y, label, ha="center", va="center", color=color, fontsize=10)
    ax.set(xticks=range(len(grid.columns)), xticklabels=[f"{v:g}" for v in grid.columns],
           yticks=range(len(grid.index)), yticklabels=[f"{v:g}" for v in grid.index],
           xlabel="Raw delay mean", ylabel="Raw delay standard deviation",
           title=title or Path(csv_path).stem)
    colorbar = {"format": PercentFormatter(1)} if percent else {}
    fig.colorbar(im, ax=ax, label=value_col.replace("_", " "), **colorbar)
    unit = " (SD in percentage points)" if percent else ""
    fig.text(0.5, 0.01, "Cell: evaluation mean ± sample SD" + unit,
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        for suffix in (".png", ".pdf"):
            fig.savefig(str(prefix) + suffix, dpi=180, bbox_inches="tight")
        summary.to_csv(str(prefix) + "_summary.csv", index=False)
    finally:
        plt.close(fig)
    return summary, grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Output path prefix")
    parser.add_argument("--mean-col")
    parser.add_argument("--std-col")
    parser.add_argument("--value-col")
    parser.add_argument("--title")
    parser.add_argument("--numeric", action="store_true", help="Plot non-percentage values")
    args = parser.parse_args()
    summary, grid = plot_csv_heatmap(
        args.csv, args.output, mean_col=args.mean_col, std_col=args.std_col,
        value_col=args.value_col, title=args.title, percent=not args.numeric)
    print(f"Saved {args.output}.png/.pdf and summary CSV; "
          f"{len(summary)} cells, grid {grid.shape}")


if __name__ == "__main__":
    main()
