"""Copy to a ``*.local.py`` file, edit the run list, then execute that copy.

Example:
    python -u scripts/eval_scripts/my_study.local.py
    python scripts/eval_scripts/draw_all.py results/evaluate/<STUDY>
"""
import delay_study as study


RUNS = [
    ("seed1", "<sacred-run-directory-name>"),
]

study.STUDY = "vil2c_delay"
study.MODELS = [
    {
        "id": run_id,
        "config": f"results/sacred/{run}/1/config.json",
        "checkpoint": f"results/models/{run}/best_model",
        "tensorboard": f"results/tb_logs/{run}",
    }
    for run_id, run in RUNS
]
study.EPISODES = 64
study.PARALLEL = 4
study.SEED = 101
study.DEVICE = None
# The training config selects the grid: sc2 keeps the SMAC preset, mpe uses the MPE preset.

if __name__ == "__main__":
    study.main()
