"""Continuous episode-time diagnostics across delay regime changes."""
from collections import defaultdict
import gzip
import json

import pandas as pd

from study_plotting import load, save, plt, percent_axis, error_limit, aggregate_runs, band


def main():
    root, summary = load(raw=True)
    if summary.empty or "condition_id" not in summary:
        return
    curves = []
    dynamic = summary[summary.condition_id.str.startswith(("periodic", "markov"))]
    for row in dynamic.itertuples():
        totals = defaultdict(lambda: dict(count=0, error=0., agreement=0.,
                                           episodes=0, high=0, missing=0, agents=0))
        directory = root / "runs" / row.model_id / row.condition_id
        for path in directory.glob("batch_*/decisions.jsonl.gz"):
            if not (path.parent / "result.json").exists():
                continue
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    decision = json.loads(line)
                    total = totals[decision["step"]]
                    total["agents"] += 1
                    total["missing"] += int(bool(decision["missing"]))
                    if decision["agent"] == 0:
                        total["episodes"] += 1
                        total["high"] += decision["regime"] == 1
                    if decision.get("eligible") and "generated_z_mse" in decision:
                        total["count"] += 1
                        total["error"] += decision["generated_z_mse"]
                        total["agreement"] += decision["agreement"]
        if not totals:
            continue
        data = pd.DataFrame.from_dict(totals, orient="index").sort_index()
        count = data["count"].replace(0, float("nan"))
        data["mse"] = data.error / count
        data["agreement"] = data.agreement / count
        data["missing_fraction"] = data.missing / data.agents
        data["high_fraction"] = data.high / data.episodes
        data = data.rename_axis("step").reset_index()
        data["map"] = row.map
        data["model_id"] = row.model_id
        data["condition_id"] = row.condition_id
        curves.append(data)

    if not curves:
        return
    runs = pd.concat(curves, ignore_index=True)
    runs.to_csv(root / "tables/dynamic_per_run.csv", index=False)
    has_latent = runs["count"].sum() > 0
    metrics = ["high_fraction", "missing_fraction"]
    if has_latent:
        metrics = ["mse", "agreement", *metrics]
    summary = aggregate_runs(runs, ["map", "condition_id", "step"], metrics)
    summary.to_csv(root / "tables/run_dynamic.csv", index=False)
    for (map_name, condition), data in summary.groupby(["map", "condition_id"]):
        panels = [("missing_fraction", "Missing current observation", True),
                  ("high_fraction", "Episodes in high-delay regime", True)]
        if has_latent:
            panels = [("mse", "Latent MSE", False),
                      ("agreement", "Action agreement", True), *panels]
        fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(10, 2.6 * len(panels)))
        if len(panels) == 1:
            axes = [axes]
        for ax, (metric, label, percent) in zip(axes, panels):
            band(ax, data, "step", metric)
            ax.set(ylabel=label)
            if metric == "mse":
                ax.set(ylim=(0, error_limit(data.mse + data.mse_std.fillna(0))))
            if percent:
                percent_axis(ax)
            ax.grid(alpha=0.2)
        axes[-1].set_xlabel("Environment step")
        fig.suptitle(f"{map_name}, {condition}: run mean ± SD")
        save(root, f"dynamic_{map_name}_{condition}", fig)


if __name__ == "__main__":
    main()
