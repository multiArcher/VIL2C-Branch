"""Delay-grid definitions and offline study figures; no simulator."""
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "eval_scripts"))

import delay_study
from components.evaluation_delay import EvaluationDelay
from envs.wrappers.delayed_wrapper import DelayedObservationWrapper


class ToyEnv:
    def __init__(self):
        self.episode_limit = 4
        self.t = 0

    def get_env_info(self):
        return {"n_agents": 2, "episode_limit": self.episode_limit}

    def reset(self, seed=None, options=None):
        self.t = 0
        return self.get_obs(), {}

    def step(self, actions):
        self.t += 1
        return self.get_obs(), 1.0, self.t >= self.episode_limit, False, {}

    def get_obs(self):
        return [np.full(3, self.t, np.float32), np.full(3, 10 + self.t, np.float32)]


def test_condition_grid_matches_epymarl():
    delay_study.MEANS = [-2, -1, 0, 1, 2]
    delay_study.STDS = [0, 0.5, 1, 1.5, 2]
    delay_study.CAP = 8
    cases, cells = delay_study.conditions()
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)) == 49
    assert ids[0] == "fixed_0"
    for name in ("fixed_1", "fixed_2", "fixed_4", "fixed_8",
                 "gaussian_0_1", "uniform_0_1",
                 "mixture_balanced", "mixture_rare_severe", "periodic_16", "markov_09"):
        assert name in ids
    assert len(cells["gaussian"]) == len(cells["uniform"]) == 25
    shared = {cell["condition_id"] for cell in cells["gaussian"] if cell["std"] == 0}
    assert shared == {cell["condition_id"] for cell in cells["uniform"] if cell["std"] == 0}
    uniform = next(case for case in cases if case["id"] == "uniform_0_1")
    assert np.isclose(uniform["high"] - uniform["low"], np.sqrt(12))


def test_mpe_grid_uses_its_own_delays():
    cases, cells = delay_study.conditions("mpe")
    by_id = {case["id"]: case for case in cases}
    assert len(by_id) == len(cases) == 70
    assert [by_id[f"fixed_{value}"]["value"] for value in (0, 1, 2, 4, 6, 8, 10, 12, 14, 16)] == [
        0, 1, 2, 4, 6, 8, 10, 12, 14, 16]
    assert all(case["cap"] == 16 for case in cases)
    assert len(cells["gaussian"]) == len(cells["uniform"]) == 35
    assert {cell["mean"] for cell in cells["gaussian"]} == {-2, 0, 2, 4, 6, 8, 10}
    assert {cell["std"] for cell in cells["gaussian"]} == {0, 2, 4, 6, 8}
    assert cells["gaussian"][0]["condition_id"] == cells["uniform"][0]["condition_id"] == "fixed_0"
    uniform = by_id["uniform_0_2"]
    assert np.isclose(uniform["high"] - uniform["low"], math.sqrt(12) * 2)
    assert by_id["mixture_balanced"]["means"] == [0, 4]
    assert by_id["mixture_balanced"]["stds"] == [4, 4]
    assert by_id["mixture_balanced"]["high_probability"] == 0.5
    assert by_id["mixture_rare_severe"]["means"] == [0, 8]
    assert by_id["mixture_rare_severe"]["high_probability"] == 0.1
    assert by_id["periodic_16"]["means"] == [0, 4]
    assert by_id["periodic_16"]["stds"] == [4, 4]
    assert by_id["markov_09"]["means"] == [0, 4]
    assert by_id["markov_09"]["stds"] == [2, 4]
    assert by_id["markov_09"]["stay_probability"] == 0.9


def test_evaluation_delay_matches_packet_arrival():
    fixed = EvaluationDelay({"kind": "fixed", "value": 3, "cap": 8}, seed=3)
    assert torch.equal(fixed._draw((2,), "cpu"), torch.tensor([3, 3]))

    zero = EvaluationDelay({"kind": "gaussian", "mean": -2, "std": 0, "cap": 8}, seed=1)
    assert torch.equal(zero._draw((4,), "cpu"), torch.zeros(4, dtype=torch.long))

    capped = EvaluationDelay({"kind": "fixed", "value": 100, "cap": 8}, seed=1)
    assert torch.equal(capped._draw((2,), "cpu"), torch.full((2,), 8))
    assert capped.clipped_count == 2

    point = EvaluationDelay({"kind": "uniform", "low": 2.0, "high": 2.0, "cap": 8}, seed=1)
    assert torch.equal(point._draw((3,), "cpu"), torch.full((3,), 2))

    periodic = EvaluationDelay(
        {"kind": "periodic", "means": [0, 2], "stds": [0, 0], "period": 2, "cap": 8}, seed=1)
    assert int(periodic._draw((1,), "cpu")) == 0
    assert int(periodic._draw((1,), "cpu")) == 0
    assert periodic.regime == 0
    assert int(periodic._draw((1,), "cpu")) == 2
    assert periodic.regime == 1

    env = DelayedObservationWrapper(ToyEnv(), max_delay=8)
    env.delay_model = EvaluationDelay({"kind": "fixed", "value": 2, "cap": 8}, seed=1)
    env.training = False
    env.reset()
    assert np.allclose(env.get_obs()[0], 0)
    assert env.get_obs_generation_time()[0] == -1
    env.step([0, 0])
    env.step([0, 0])
    assert np.allclose(env.get_obs()[0], 0)
    assert env.get_obs_generation_time()[0] == 0

    env.training = True
    env.reset()
    env.step([0, 0])
    assert np.allclose(env.get_obs()[0], 1)
    assert env.get_obs_generation_time()[0] == 1


def _write_batch(root, model, condition, kind, env="delayed_mpe"):
    directory = root / "runs" / model / condition / "batch_0000"
    directory.mkdir(parents=True)
    job = {"condition": {"kind": kind, "cap": 8}, "parallel": 1, "seed": 1}
    (directory / "job.json").write_text(json.dumps(job), encoding="utf-8")
    (directory / "effective_config.json").write_text(json.dumps({
        "env": env,
        "env_args": {"map_name": "simple_spread_v3"},
        "seed": 1,
    }), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({
        "seconds": 1.0, "device_name": "CPU", "diagnostic_inclusive_peak_bytes": 0,
    }), encoding="utf-8")
    episode = dict(episode=0, won=False, episode_return=3.0, length=2, decision_count=0,
                   sample_count=2, missing_count=1, never_arrived_count=0, clipped_count=0,
                   delay_histogram={"0": 1, "2": 1})
    (directory / "episodes.jsonl").write_text(json.dumps(episode) + "\n", encoding="utf-8")
    rows = [
        dict(episode=0, step=0, agent=0, missing=False, age=0, regime=0, regime_age=0, decision_ms=1.5),
        dict(episode=0, step=1, agent=0, missing=True, age=1, regime=1, regime_age=0, decision_ms=2.5),
    ]
    with gzip.open(directory / "decisions.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def test_offline_figures(tmp_path):
    study = tmp_path / "study"
    for condition, kind in (
        ("fixed_0", "fixed"), ("gaussian_0_1", "gaussian"),
        ("uniform_0_1", "uniform"), ("periodic_16", "periodic"),
    ):
        _write_batch(study, "seed1", condition, kind)
    cells = {
        "gaussian": [{"mean": 0, "std": 1, "condition_id": "gaussian_0_1"}],
        "uniform": [{"mean": 0, "std": 1, "condition_id": "uniform_0_1"}],
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"env": "mpe", "env_args": {"map_name": "simple_spread_v3"}}), encoding="utf-8")
    (study / "manifest.json").write_text(json.dumps({
        "models": [{"id": "seed1", "config": str(config)}],
        "gaussian_cells": cells["gaussian"], "uniform_cells": cells["uniform"],
    }), encoding="utf-8")
    scripts = ROOT / "scripts" / "eval_scripts"
    import subprocess
    subprocess.run([sys.executable, str(scripts / "draw_all.py"), str(study)], check=True)
    figures = study / "figures" / "evaluation"
    for name in (
        "return_gaussian_simple_spread_v3.png",
        "return_uniform_simple_spread_v3.png",
        "distributions_simple_spread_v3.png",
        "efficiency_simple_spread_v3.png",
        "dynamic_simple_spread_v3_periodic_16.png",
    ):
        assert (figures / name).is_file(), name
    assert not list(figures.glob("policy_*"))
    assert not list(figures.glob("reconstruction_*"))
    assert not list(figures.glob("error_action_*"))
