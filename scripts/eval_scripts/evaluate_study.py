"""Evaluate batches while reusing compatible environment workers."""
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


class EvaluationLogger:
    def __init__(self):
        self.metrics = {}

    def log_stat(self, key, value, timestep):
        self.metrics[key] = float(value)


def resolve_device(config, requested):
    import torch
    if requested is None:
        requested = config.get("device")
    if requested is None:
        requested = "cuda" if config.get("use_cuda", False) else "cpu"
    requested = str(requested)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; evaluating on CPU.", file=sys.__stdout__, flush=True)
        requested = "cpu"
    return requested


def prepare_config(config, job):
    """Apply packet delay in the environment, not a second time inside the MAC."""
    env = config["env"]
    if env == "sc2":
        config["env"] = "delayed_sc2"
    elif env == "mpe":
        config["env"] = "delayed_mpe"
    elif env not in ("delayed_sc2", "delayed_mpe"):
        raise ValueError(f"Delay study supports sc2 and mpe, got env={env}")
    config.update(batch_size_run=job["parallel"], test_nepisode=job["parallel"],
                  runner="parallel", render=False, evaluate=True, test_greedy=True,
                  obs_delay_enabled=False, obs_delay_apply_train=False, obs_delay_apply_test=False)
    config["env_args"] = dict(config["env_args"])
    config["env_args"].update(seed=job["seed"], max_delay=job["condition"]["cap"])
    config["device"] = resolve_device(config, job.get("device"))
    config["use_cuda"] = config["device"].startswith("cuda")
    return config


def evaluate_batch(job_path, runner=None):
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    from components.episode_buffer import EpisodeBatch
    from components.transforms import OneHot
    from run import parse_buffer_scheme
    from runners.parallel_runner import ParallelRunner
    from utils.maker.mac_maker import MACMaker
    from study_diagnostics import ReturnDiagnostics

    job = json.loads(job_path.read_text(encoding="utf-8"))
    config = prepare_config(json.loads(Path(job["config"]).read_text(encoding="utf-8")), job)
    args = SimpleNamespace(**config)
    args.device = torch.device(config["device"])
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    torch.set_num_threads(1)
    (job_path.parent / "effective_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if runner is None:
        print(f"  Starting {job['parallel']} environment workers...", file=sys.__stdout__, flush=True)
        runner = ParallelRunner(args, EvaluationLogger())
        runner.evaluation_seed = job["seed"]
        runner.evaluation_batch_index = 0
    else:
        runner.args = args
        runner.logger = EvaluationLogger()
    for index, connection in enumerate(runner.parent_conns):
        connection.send(("set_evaluation_delay", dict(condition=job["condition"], seed=job["seed"] + index)))
    for connection in runner.parent_conns:
        connection.recv()
    info = runner.get_env_info()
    args.n_agents = info["n_agents"]
    args.n_actions = info["n_actions"]
    args.state_shape = info["state_shape"]
    args.env_info = info
    scheme = parse_buffer_scheme(info, args.common_reward, args)
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    template = EpisodeBatch(scheme, groups, 1, 1, preprocess=preprocess, device=args.device)
    mac = MACMaker.make(config["mac"], template.scheme, groups, args).to(args.device)
    mac.load_models(job["checkpoint"])
    mac.eval()
    print("  Model loaded; resetting environments and running episodes...",
          file=sys.__stdout__, flush=True)
    select = mac.select_actions
    last_progress = time.perf_counter()

    def timed_select(*values, **keywords):
        nonlocal last_progress
        if args.use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        actions = select(*values, **keywords)
        if args.use_cuda:
            torch.cuda.synchronize()
        mac.decision_ms = 1000 * (time.perf_counter() - start)
        now = time.perf_counter()
        if now - last_progress >= 10:
            print(f"  Episode step {runner.t}/{runner.episode_limit}",
                  file=sys.__stdout__, flush=True)
            last_progress = now
        return actions

    mac.select_actions = timed_select
    runner.setup(scheme, groups, preprocess, mac)
    runner.diagnostics = ReturnDiagnostics(job_path.parent, job["parallel"], job["episode_offset"])
    if args.use_cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        runner.run(test_mode=True)
    seconds = time.perf_counter() - start
    result = dict(seconds=seconds, episodes=job["parallel"],
                  environment_seed=runner.evaluation_seed,
                  environment_batch_index=runner.evaluation_batch_index,
                  worker_pids=[process.pid for process in runner.ps],
                  device_name=torch.cuda.get_device_name() if args.use_cuda else "CPU",
                  torch_version=str(torch.__version__),
                  diagnostic_inclusive_peak_bytes=torch.cuda.max_memory_allocated() if args.use_cuda else 0,
                  logger_metrics=runner.logger.metrics)
    for connection in runner.parent_conns:
        connection.send(("get_final_info", None))
    final_infos = [connection.recv() for connection in runner.parent_conns]
    episode_path = job_path.parent / "episodes.jsonl"
    rows = [json.loads(line) for line in episode_path.read_text(encoding="utf-8").splitlines()]
    for row, info in zip(rows, final_infos):
        row.update(dead_allies=info.get("dead_allies"), dead_enemies=info.get("dead_enemies"))
    episode_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    del mac.select_actions
    runner.mac = None
    runner.diagnostics = None
    runner.batch = None
    runner.evaluation_batch_index += 1
    path = job_path.parent / "result.tmp"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    path.replace(job_path.parent / "result.json")
    return runner


def environment_key(job):
    config = json.loads(Path(job["config"]).read_text(encoding="utf-8"))
    env = config["env"]
    if env == "sc2":
        env = "delayed_sc2"
    elif env == "mpe":
        env = "delayed_mpe"
    env_args = dict(config["env_args"])
    env_args.pop("seed", None)
    env_args["max_delay"] = job["condition"]["cap"]
    return json.dumps(dict(
        env=env, env_args=env_args, parallel=job["parallel"],
        common_reward=config["common_reward"],
        reward_scalarisation=config["reward_scalarisation"],
    ), sort_keys=True)


def close_runner(runner):
    runner.close_env()
    for process in runner.ps:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def main(job_path):
    request = json.loads(Path(job_path).read_text(encoding="utf-8"))
    paths = [Path(path) for path in request["jobs"]] if "jobs" in request else [Path(job_path)]
    groups = {}
    for path in paths:
        job = json.loads(path.read_text(encoding="utf-8"))
        groups.setdefault(environment_key(job), []).append(path)
    total = len(paths)
    completed = 0
    started = time.perf_counter()
    for paths in groups.values():
        runner = None
        try:
            for path in paths:
                job = json.loads(path.read_text(encoding="utf-8"))
                print(
                    f"[{completed + 1}/{total}] {path.parent.parent.parent.name} | "
                    f"{path.parent.parent.name} | episodes "
                    f"{job['episode_offset'] + 1}-{job['episode_offset'] + job['parallel']} | "
                    f"{'new environments' if runner is None else 'reuse environments'}",
                    flush=True,
                )
                with (path.parent / "run.log").open("w", encoding="utf-8") as log:
                    with redirect_stdout(log), redirect_stderr(log):
                        runner = evaluate_batch(path, runner)
                result = json.loads((path.parent / "result.json").read_text(encoding="utf-8"))
                rows = [json.loads(line) for line in
                        (path.parent / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
                config = json.loads((path.parent / "effective_config.json").read_text(encoding="utf-8"))
                if config.get("env") in ("mpe", "delayed_mpe"):
                    returns = [row["episode_return"] for row in rows]
                    mean = sum(returns) / len(returns)
                    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1) if len(returns) > 1 else 0
                    score = f"return={mean:.3f}±{variance ** 0.5:.3f}"
                else:
                    score = f"win={sum(row['won'] for row in rows) / len(rows):.1%}"
                completed += 1
                elapsed = time.perf_counter() - started
                remaining = elapsed / completed * (total - completed)
                print(
                    f"  Done {completed}/{total} ({completed / total:.1%}) | "
                    f"{score} | rollout={result['seconds']:.1f}s | "
                    f"elapsed={elapsed / 60:.1f}min | ETA~{remaining / 60:.1f}min",
                    flush=True,
                )
        finally:
            if runner is not None:
                close_runner(runner)


if __name__ == "__main__":
    main(sys.argv[1])
