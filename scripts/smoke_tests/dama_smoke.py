"""Small real-simulator checks, NOT a convergence experiment.

Runs the unchanged main/run entrypoints, then reloads each frozen checkpoint.
Use --smac-only for the three requested SMAC maps; default checks spread/tag.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smac-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    options = parser.parse_args()
    output = ROOT / "results/dama_smoke" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    scenarios = [("sc2", name, []) for name in ("5m_vs_6m", "8m_vs_9m", "MMM2")] if options.smac_only else [
        ("mpe", "simple_spread_v3", ["env_args.time_limit=10"]),
        ("mpe", "simple_tag_v3", ["env_args.time_limit=10", "common_reward=False", "dama_checkpoint_agents=[0,1,2]"]),
    ]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    report = []
    for environment, scenario, extra in scenarios:
        name = "dama_smoke_" + scenario.lower()
        previous = set((ROOT / "results/sacred").glob(name + "__*"))
        command = [sys.executable, "-u", "src/main.py", "--config=dama_ddpg",
                   "--env-config=" + environment, "with", "name=" + name,
                   "env_args.map_name=" + scenario, "seed=1", "batch_size_run=1",
                   "batch_size=1", "buffer_size=4", "dama_history_length=16", "hidden_dim=32",
                   "device=" + options.device, "use_cuda=" + str(options.device.startswith("cuda")),
                   "t_max=1", "test_nepisode=1", "test_interval=50000", "use_tensorboard=False",
                   "learner_log_interval=1", "log_interval=1"] + extra
        print(f"Training smoke: {scenario}", flush=True)
        with (output / (scenario + "_train.log")).open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=240)
        created = set((ROOT / "results/sacred").glob(name + "__*")) - previous
        if len(created) != 1:
            raise RuntimeError(f"Expected one new Sacred run, found {created}")
        run = next(iter(created)) / "1"
        status = json.loads((run / "run.json").read_text(encoding="utf-8"))["status"]
        if status != "COMPLETED":
            raise RuntimeError(f"Sacred run status: {status}")
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        for key in ("critic_loss", "pg_loss"):
            values = metrics[key]["values"]
            if not values or not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"Missing/nonfinite {key}")
        config_path = run / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        checkpoint = ROOT / "results/models" / config["unique_token"] / "best_model"
        print(f"Frozen delay smoke: {scenario}", flush=True)
        evaluation = output / (scenario + "_eval")
        command = [sys.executable, "-u", "scripts/eval_scripts/dama_obs_delay.py",
                   "--config", str(config_path), "--checkpoint", str(checkpoint),
                   "--fixed", "0", "2", "--episodes", "1", "--device", options.device,
                   "--output", str(evaluation)]
        with (output / (scenario + "_eval.log")).open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=240)
        report.append(dict(scenario=scenario, training_status=status,
                           config=str(config_path), checkpoint=str(checkpoint), evaluation=str(evaluation),
                           critic_loss=metrics["critic_loss"]["values"][-1],
                           actor_loss=metrics["pg_loss"]["values"][-1]))
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Passed: {scenario}", flush=True)
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
