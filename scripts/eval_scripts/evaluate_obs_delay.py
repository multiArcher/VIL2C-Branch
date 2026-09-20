"""Evaluate one frozen VIL2C policy over observation delays; never construct a learner."""

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
from datetime import datetime
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def resolve_checkpoint(path, selection="latest"):
    path = Path(path).resolve()
    if (path / "agent.th").is_file():
        return path
    if selection == "latest":
        candidates = [p for p in path.iterdir()
                      if p.is_dir() and p.name.isdigit() and (p / "agent.th").is_file()]
        if not candidates:
            raise ValueError(f"No numeric checkpoints with agent.th in {path}")
        return max(candidates, key=lambda p: int(p.name))
    if selection != "best_model" and not selection.isdigit():
        raise ValueError("--checkpoint must be latest, best_model, or an exact saved step")
    selected = path / selection
    if not (selected / "agent.th").is_file():
        raise FileNotFoundError(selected / "agent.th")
    return selected


def find_config(checkpoint, explicit=None):
    if explicit:
        return Path(explicit).resolve()
    run_dir = checkpoint.parent
    # Locate the matching Sacred run even when the whole results folder is moved.
    candidates = list((run_dir.parent.parent / "sacred" / run_dir.name).glob("*/config.json"))
    if len(candidates) != 1:
        raise ValueError("Cannot uniquely locate training config.json; supply --config PATH")
    return candidates[0]


def delay_pair(value):
    try:
        mean, std = map(float, value.split(":") if ":" in value else (value, "0"))
        if not all(math.isfinite(x) for x in (mean, std)) or std < 0:
            raise ValueError
        return mean, std
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use finite MEAN:STD, MIN:MAX, or fixed STEPS; second value must be nonnegative") from exc


def resolve_delay_grid(parser, opts):
    if opts.means is not None or opts.stds is not None:
        if opts.delay_type != "gaussian":
            parser.error("--means/--stds are only supported for gaussian delays")
        if opts.delays is not None:
            parser.error("Use either --delays or --means with --stds, not both")
        if opts.means is None or opts.stds is None:
            parser.error("--means and --stds must be supplied together")
        if not all(math.isfinite(x) for x in opts.means):
            parser.error("--means must contain finite values (negative means are allowed)")
        if not all(math.isfinite(x) and x >= 0 for x in opts.stds):
            parser.error("--stds must contain finite nonnegative values")
        return [(mean, std) for mean in dict.fromkeys(opts.means)
                for std in dict.fromkeys(opts.stds)]
    if opts.delays is not None:
        return opts.delays
    return [(0, 0), (0, 2), (0, 4)] if opts.delay_type == "uniform" else [(0, 0), (1, 0), (2, 0), (4, 0)]


def make_config(training, checkpoint, mean, std, seed, episodes, batch_size, device, discretization,
                delay_type="gaussian"):
    config = copy.deepcopy(training)
    if config.get("mac") != "vil2c_mac" or config.get("env") not in ("sc2", "mpe"):
        raise ValueError("This evaluator supports VIL2C on plain sc2 or mpe (no stacked delay wrapper)")
    if training.get("obs_delay_enabled", False) and training.get("obs_delay_apply_train", False):
        raise ValueError("Training config enables observation delay during training")
    if episodes < 1 or batch_size < 1 or episodes % batch_size:
        raise ValueError("--episodes must be positive and divisible by --batch-size")
    if delay_type not in ("gaussian", "fixed", "uniform"):
        raise ValueError(f"Unknown delay type: {delay_type}")
    if not all(math.isfinite(x) for x in (mean, std)) or std < 0:
        raise ValueError("Delay parameters must be finite; standard deviation / upper bound must be nonnegative")
    if delay_type != "gaussian" and mean < 0:
        raise ValueError("Fixed delay and uniform lower bound must be nonnegative")
    if delay_type == "fixed" and (int(mean) != mean or std != 0):
        raise ValueError("Fixed delay requires integer STEPS (or STEPS:0)")
    if delay_type == "uniform" and (int(mean) != mean or int(std) != std or mean > std):
        raise ValueError("Uniform delay requires integer MIN:MAX with 0 <= MIN <= MAX")
    config.update(
        evaluate=True, checkpoint_path=str(checkpoint), save_model=False,
        save_replay=False, save_evaluate_state=False, use_tensorboard=False, use_wandb=False,
        runner="parallel", batch_size_run=batch_size, test_nepisode=episodes,
        seed=seed, device=device, use_cuda=device.startswith("cuda"), test_greedy=True,
        obs_delay_enabled=bool(mean or std), obs_delay_apply_train=False,
        obs_delay_apply_test=True, obs_gaussian_delay_mean=mean,
        obs_gaussian_delay_std=std, obs_delay_discretization=discretization,
        comm_delay_type="gaussian", comm_gaussian_delay_mean=0.0,
        comm_gaussian_delay_std=0.0,
    )
    config.update(obs_delay_type=delay_type,
                  obs_fixed_delay=int(mean) if delay_type == "fixed" else 0,
                  obs_uniform_delay_min=int(mean) if delay_type == "uniform" else 0,
                  obs_uniform_delay_max=int(std) if delay_type == "uniform" else 0)
    if delay_type != "gaussian":
        config.update(obs_gaussian_delay_mean=0.0, obs_gaussian_delay_std=0.0)
    config["env_args"]["seed"] = seed
    return config


class MetricLogger:
    def __init__(self):
        self.metrics = {}

    def log_stat(self, key, value, timestep):
        self.metrics[key] = float(value)


def evaluate(config, checkpoint):
    import numpy as np
    import torch
    from components.episode_buffer import EpisodeBatch
    from components.transforms import OneHot
    from run import parse_buffer_scheme
    from utils.maker import MACMaker, RunnerMaker

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    args = SimpleNamespace(**copy.deepcopy(config))
    args.device = torch.device(args.device)
    logger = MetricLogger()
    runner = RunnerMaker.make(args.runner, args=args, logger=logger)
    try:
        info = runner.get_env_info()
        args.n_agents, args.n_actions = info["n_agents"], info["n_actions"]
        args.state_shape = info["state_shape"]
        scheme = parse_buffer_scheme(info, args.common_reward, args)
        groups = {"agents": args.n_agents}
        preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
        template = EpisodeBatch(scheme, groups, 1, 1, preprocess=preprocess)
        mac = MACMaker.make(args.mac, template.scheme, groups, args)
        # Only policy weights are needed. No critic, optimizer, or training loop.
        mac.load_models(str(checkpoint))
        mac.to(args.device)
        mac.eval()
        runner.setup(scheme, groups, preprocess, mac)
        runner.t_env = int(checkpoint.name) if checkpoint.name.isdigit() else 0
        runner.log_train_stats_t = runner.t_env  # prevent early partial test-stat flush
        with torch.no_grad():
            for index in range(args.test_nepisode // args.batch_size_run):
                runner.run(test_mode=True)
                print(f"  episodes {(index + 1) * args.batch_size_run}/{args.test_nepisode}", flush=True)
        required = [return_metric_prefix(config) + "_mean"]
        if args.env == "sc2":
            required.append("metric/test_battle_won_mean")
        if any(key not in logger.metrics for key in required):
            raise RuntimeError(f"Missing evaluation metrics: {logger.metrics}")
        return logger.metrics
    finally:
        try:
            runner.close_env()
        finally:
            for process in runner.ps:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            for connection in (*runner.parent_conns, *runner.worker_conns):
                connection.close()


def return_metric_prefix(config):
    return "metric/test_return" if config["common_reward"] else "metric/test_total_return"


def summary_row(config, checkpoint, metrics):
    prefix = return_metric_prefix(config)
    return dict(checkpoint=str(checkpoint), env=config["env"], map=config["env_args"]["map_name"],
                common_reward=config["common_reward"], reward_scalarisation=config["reward_scalarisation"],
                return_metric=prefix + "_mean",
                delay_type=config["obs_delay_type"], delay_parameters=delay_description(config),
                obs_mean=config["obs_gaussian_delay_mean"] if config["obs_delay_type"] == "gaussian" else None,
                obs_std=config["obs_gaussian_delay_std"] if config["obs_delay_type"] == "gaussian" else None,
                seed=config["seed"], episodes=config["test_nepisode"],
                win_rate=metrics.get("metric/test_battle_won_mean") if config["env"] == "sc2" else None,
                return_mean=metrics[prefix + "_mean"], return_std=metrics.get(prefix + "_std"),
                episode_length_mean=metrics.get("metric/test_ep_length_mean"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--checkpoint", default="latest", help="latest, best_model, or exact step")
    parser.add_argument("--config", type=Path, help="Saved training Sacred config.json (auto-detected)")
    parser.add_argument("--delay-type", choices=["gaussian", "fixed", "uniform"], default="gaussian")
    parser.add_argument("--delays", nargs="+", type=delay_pair,
                        help="Gaussian MEAN:STD; fixed STEPS; discrete uniform MIN:MAX (inclusive)")
    parser.add_argument("--means", nargs="+", type=float, help="Gaussian means; crossed with every --stds value")
    parser.add_argument("--stds", nargs="+", type=float, help="Gaussian standard deviations; crossed with every --means value")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--episodes", type=int, default=32, help="Episodes per delay per seed")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--discretization", choices=["round", "floor", "ceil"], default="round")
    parser.add_argument("--sc2-path", type=Path)
    parser.add_argument("--output", type=Path, help="New output directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print plan without launching environments")
    opts = parser.parse_args(argv)
    opts.delays = resolve_delay_grid(parser, opts)
    checkpoint = resolve_checkpoint(opts.model_dir, opts.checkpoint)
    config_path = find_config(checkpoint, opts.config)
    training = json.loads(config_path.read_text(encoding="utf-8"))
    configs = [make_config(training, checkpoint, mean, std, seed, opts.episodes,
                           opts.batch_size, opts.device, opts.discretization, opts.delay_type)
               for mean, std in dict.fromkeys(opts.delays) for seed in dict.fromkeys(opts.seeds)]
    print(f"Training config: {config_path}\nFrozen checkpoint: {checkpoint}\nEnvironment: {training['env']}\nMap: {training['env_args']['map_name']}")
    print(f"Evaluations: {len(configs)}; episodes each: {opts.episodes}; total episodes: {len(configs) * opts.episodes}; communication delay: 0")
    for config in configs:
        print(f"  obs={delay_description(config)} seed={config['seed']}")
    if opts.dry_run:
        return
    import torch
    if opts.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    torch.set_num_threads(1)
    if opts.sc2_path:
        os.environ["SC2PATH"] = str(opts.sc2_path.resolve())
    for key in ("NO_PROXY", "no_proxy"):
        os.environ[key] = os.environ.get(key, "") + ",127.0.0.1,localhost"
    output = opts.output or ROOT / "results" / "obs_delay_eval" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(checkpoint=str(checkpoint), training_config=str(config_path), evaluations=configs)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    fields = ["checkpoint", "env", "map", "common_reward", "reward_scalarisation", "return_metric", "delay_type", "delay_parameters", "obs_mean", "obs_std", "seed", "episodes", "win_rate", "return_mean", "return_std", "episode_length_mean"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        file.flush()
        for index, config in enumerate(configs):
            print(f"Evaluation {index + 1}/{len(configs)}", flush=True)
            metrics = evaluate(config, checkpoint)
            (output / f"metrics_{index:03d}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            row = summary_row(config, checkpoint, metrics)
            writer.writerow(row)
            file.flush()
            win_text = f"win_rate={row['win_rate']:.4f}, " if row["win_rate"] is not None else ""
            print(f"  {win_text}return_mean={row['return_mean']:.4f}", flush=True)
    print(f"Results: {output / 'summary.csv'}")


def delay_description(config):
    kind = config["obs_delay_type"]
    if kind == "fixed":
        return f"fixed({config['obs_fixed_delay']})"
    if kind == "uniform":
        return f"uniform[{config['obs_uniform_delay_min']},{config['obs_uniform_delay_max']}]"
    return f"gaussian(mean={config['obs_gaussian_delay_mean']},std={config['obs_gaussian_delay_std']})"


if __name__ == "__main__":
    main()
