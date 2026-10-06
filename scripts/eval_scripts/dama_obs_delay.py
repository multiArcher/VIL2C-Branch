"""Evaluate one frozen DAMA checkpoint using the shared arrival-time protocol.

No learner is instantiated. The existing single-episode runner keeps per-agent
rewards intact for tag; both teams' returns are reported separately.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from components.episode_buffer import EpisodeBatch
from components.evaluation_delay import EvaluationDelay
from components.transforms import OneHot
from run import parse_buffer_scheme
from runners.delayed_episode_runner import DelayedEpisodeRunner
from utils.maker import MACMaker, EnvMaker


class EvaluationLogger:
    def __init__(self):
        self.metrics = {}

    def log_stat(self, key, value, timestep):
        self.metrics[key] = float(value)

    def warning(self, message):
        print(message)


def delay_conditions(means, stds, fixed, cap):
    if cap < 0 or any(std < 0 for std in stds):
        raise ValueError("cap and standard deviations must be nonnegative")
    cases = [dict(id="fixed_0", kind="fixed", value=0, cap=cap)]
    for value in fixed:
        if not 0 <= value <= cap:
            raise ValueError("fixed delays must be between zero and cap")
        case = dict(id=f"fixed_{value}", kind="fixed", value=value, cap=cap)
        if case not in cases:
            cases.append(case)
    for mean in means:
        for std in stds:
            cases.append(dict(id=f"gaussian_{mean:g}_{std:g}", kind="gaussian",
                              mean=mean, std=std, cap=cap))
    return list({case["id"]: case for case in cases}.values())


def create_runner(config, checkpoint, device):
    config = json.loads(json.dumps(config))
    if config["mac"] != "dama_mac":
        raise ValueError("This evaluator expects mac=dama_mac")
    if config["env"] in ("sc2", "mpe"):
        config["env"] = "delayed_" + config["env"]
    if config["env"] not in ("delayed_sc2", "delayed_mpe"):
        raise ValueError("DAMA evaluation supports SMAC and MPE")
    config.update(device=device, use_cuda=device.startswith("cuda"),
                  batch_size_run=1, test_nepisode=1, render=False, runner="delayed_episode")
    # The per-condition EvaluationDelay supplies the exact packet distribution.
    config["env_args"].update(delay_mean=0, delay_std=0, max_delay=0)
    args = SimpleNamespace(**config)
    args.device = torch.device(device)
    logger = EvaluationLogger()
    runner = DelayedEpisodeRunner(args, logger)
    try:
        info = runner.get_env_info()
        args.n_agents, args.n_actions = info["n_agents"], info["n_actions"]
        args.state_shape = info["state_shape"]
        scheme = parse_buffer_scheme(info, args.common_reward, args)
        groups = {"agents": args.n_agents}
        preprocess = {"actions": ("actions_onehot", [OneHot(args.n_actions)])}
        template = EpisodeBatch(scheme, groups, 1, 1, preprocess=preprocess, device=device)
        mac = MACMaker.make(args.mac, template.scheme, groups, args).to(device)
        mac.load_models(checkpoint)
        mac.eval()
        runner.setup(scheme, groups, preprocess, mac)
        return runner, mac, logger
    except Exception:
        runner.close_env()
        raise


def evaluate_condition(runner, condition, episodes, seed, logger):
    rows = []
    agent_names = list(getattr(runner.env.env, "agents", []))
    with torch.no_grad():
        for episode in range(episodes):
            # Same initial environment seed in every condition; delay randomness
            # has an independent stream. Never reseed/retrain the policy.
            episode_seed = seed + episode
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            if runner.args.env == "delayed_sc2":
                # SMAC seed() is a getter. Its game seed is set at construction
                # and applied on launch, so construct a fresh game per episode
                # rather than silently failing to reseed a running game.
                runner.env.close()
                env_args = dict(runner.args.env_args, seed=episode_seed)
                runner.env = EnvMaker.make("delayed_sc2", **env_args,
                                           common_reward=runner.args.common_reward,
                                           reward_scalarisation=runner.args.reward_scalarisation)
            else:
                runner.env.seed(episode_seed)
            runner.env.delay_model = EvaluationDelay(condition, episode_seed + 1000003)
            logger.metrics.clear()
            batch = runner.run(test_mode=True)
            length = runner.t
            reward = batch["reward"][0, :length].sum(0).cpu().tolist()
            gen = batch["obs_gen_t"][0, :length, :, 0]
            times = torch.arange(length, device=gen.device).unsqueeze(-1)
            arrived = gen >= 0
            beyond = arrived & ((times - gen) > runner.mac.inputs.history_length)
            row = dict(condition=condition["id"], episode=episode, seed=episode_seed,
                       length=length, return_total=float(sum(reward)),
                       win=logger.metrics.get("running/test_battle_won_mean",
                                              logger.metrics.get("metric/test_battle_won_mean")),
                       unarrived_fraction=float((~arrived).float().mean()),
                       history_overflow_fraction=float(beyond.float().mean()))
            if not runner.args.common_reward:
                for name, value in zip(agent_names or [f"agent_{i}" for i in range(len(reward))], reward):
                    row[f"return_{name}"] = value
                predators = [v for name, v in zip(agent_names, reward) if name.startswith("adversary_")]
                prey = [v for name, v in zip(agent_names, reward) if name.startswith("agent_")]
                if predators:
                    row["return_predator_mean"] = float(np.mean(predators))
                if prey:
                    row["return_prey_mean"] = float(np.mean(prey))
            rows.append(row)
    return rows


def summarize(rows):
    summary = []
    for condition in dict.fromkeys(row["condition"] for row in rows):
        selected = [row for row in rows if row["condition"] == condition]
        result = dict(condition=condition, episodes=len(selected))
        for key in selected[0]:
            if key.startswith("return_") or key.endswith("_fraction") or key == "win":
                values = [row[key] for row in selected if row[key] is not None]
                if values:
                    result[key + "_mean"] = float(np.mean(values))
                    result[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(result)
    return summary


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Saved Sacred config.json")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Directory containing agent.th")
    parser.add_argument("--means", type=float, nargs="*", default=[])
    parser.add_argument("--stds", type=float, nargs="+", default=[0, 1, 2])
    parser.add_argument("--fixed", type=int, nargs="*", default=[0, 1, 2, 4, 8])
    parser.add_argument("--cap", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    options = parser.parse_args()
    if options.episodes < 1:
        parser.error("episodes must be positive")
    conditions = delay_conditions(options.means, options.stds, options.fixed, options.cap)
    config = json.loads(options.config.read_text(encoding="utf-8"))
    checkpoint = options.checkpoint.resolve()
    fingerprint = hashlib.sha256((checkpoint / "agent.th").read_bytes()).hexdigest()
    output = options.output or ROOT / "results/dama_delay_eval" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(config_path=str(options.config.resolve()), checkpoint=str(checkpoint),
                    checkpoint_sha256=fingerprint, config=config, conditions=conditions,
                    episodes_per_condition=options.episodes, seed=options.seed, device=options.device)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    torch.set_num_threads(1)
    runner, mac, logger = create_runner(config, checkpoint, options.device)
    frozen = {key: value.detach().clone() for key, value in mac.state_dict().items()}
    rows = []
    try:
        for condition in conditions:
            rows.extend(evaluate_condition(runner, condition, options.episodes, options.seed, logger))
            write_csv(output / "episodes.csv", rows)
            write_csv(output / "summary.csv", summarize(rows))
            print(f"{condition['id']}: {options.episodes} episodes complete", flush=True)
        if any(not torch.equal(value, frozen[key]) for key, value in mac.state_dict().items()):
            raise RuntimeError("Frozen policy weights changed during evaluation")
        if fingerprint != hashlib.sha256((checkpoint / "agent.th").read_bytes()).hexdigest():
            raise RuntimeError("Checkpoint file changed during evaluation")
    finally:
        runner.close_env()
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()
