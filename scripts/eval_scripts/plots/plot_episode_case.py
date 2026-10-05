"""Usage: plot_episode_case.py STUDY MODEL CONDITION EPISODE [AGENT]."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from summarize_study import read_rows
from study_plotting import load, save, plt


def main():
    root, _ = load()
    model, condition, episode = sys.argv[2], sys.argv[3], int(sys.argv[4])
    agent = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    rows = [r for p in (root / "runs" / model / condition).glob("batch_*/decisions.jsonl.gz")
            if (p.parent / "result.json").exists() for r in read_rows(p)
            if r["episode"] == episode and r["agent"] == agent]
    data = pd.DataFrame(rows).sort_values("step")
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    axes[0].step(data.step, data.age.fillna(-1), label="Latest observation age")
    action_keys = [key for key in ("reference_action", "candidate_action", "actual_action") if key in data]
    if action_keys:
        for key in action_keys:
            axes[1].step(data.step, data[key], label=key)
    else:
        axes[1].step(data.step, data.missing.astype(float), label="Missing observation")
    error_keys = [key for key in ("mask_z_mse", "generated_z_mse") if key in data]
    if error_keys:
        for key in error_keys:
            axes[2].plot(data.step, data[key], label=key)
    else:
        axes[2].step(data.step, data.regime.fillna(0), label="Delay regime")
    for ax in axes:
        ax.legend()
    axes[2].set_xlabel("Environment step")
    save(root, f"case_{model}_{condition}_{episode}_{agent}", fig)


if __name__ == "__main__":
    main()
