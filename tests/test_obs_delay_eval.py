"""Observation-delay regression tests without launching StarCraft II."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from controllers.vil2c_controller import VIL2CMAC
from scripts.smoke_tests.vil2c_ymappo_smoke import _args, _batch
from scripts.eval_scripts.evaluate_obs_delay import (
    resolve_checkpoint, find_config, make_config, evaluate, delay_pair,
)


def test_delay_reaches_encoder_only_in_test_and_keeps_current_action_mask():
    args = _args(obs_delay_enabled=True, obs_delay_apply_train=False,
                 obs_delay_apply_test=True, obs_gaussian_delay_mean=2,
                 obs_gaussian_delay_std=0, obs_last_action=True)
    batch, groups = _batch(args)
    batch.scheme["actions_onehot"] = {"vshape": (args.n_actions,), "group": "agents"}
    batch.data.transition_data["actions_onehot"] = torch.zeros(
        batch.batch_size, batch.max_seq_length, args.n_agents, args.n_actions
    )
    batch["actions_onehot"][:, 2, :, 1] = 1
    batch["avail_actions"][:, 3] = 0
    batch["avail_actions"][:, 3, :, 2] = 1
    mac = VIL2CMAC(batch.scheme, groups, args)
    captured = []
    hook = mac.agent.encoder.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].clone()))
    mac.init_hidden(batch.batch_size)
    mac.forward(batch, 3, test_mode=False)
    assert torch.equal(captured[-1][:, :5], batch["obs"][:, 3].reshape(-1, 5))
    mac.init_hidden(batch.batch_size)
    pi = mac.forward(batch, 3, test_mode=True)
    assert torch.equal(captured[-1][:, :5], batch["obs"][:, 1].reshape(-1, 5))
    assert torch.equal(captured[-1][:, 5:9], batch["actions_onehot"][:, 2].reshape(-1, 4))
    assert torch.all(pi[:, :, 2] == 1)
    assert torch.all(mac.observation_delay_model.last_source_times == 1)
    mac.init_hidden(batch.batch_size)
    mac.forward(batch, 0, test_mode=True)
    assert torch.equal(captured[-1][:, :5], batch["obs"][:, 0].reshape(-1, 5))
    hook.remove()


def test_zero_delay_matches_disabled_policy():
    args = _args(obs_delay_enabled=False)
    batch, groups = _batch(args)
    mac = VIL2CMAC(batch.scheme, groups, args)
    mac.init_hidden(batch.batch_size)
    torch.manual_seed(8)
    expected = mac.forward(batch, 3, test_mode=True)
    mac.observation_delay_model.enabled = True
    mac.observation_delay_model.apply_test = True
    mac.init_hidden(batch.batch_size)
    torch.manual_seed(8)
    actual = mac.forward(batch, 3, test_mode=True)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("kind,first,second,expected", [
    ("fixed", 2, 0, {2}), ("uniform", 1, 3, {1, 2, 3}),
    ("uniform", 2, 2, {2}),
])
def test_distribution_configuration_and_historical_observations(kind, first, second, expected):
    from types import SimpleNamespace
    from components.observation_delay_model import ObservationDelayModel
    training = dict(mac="vil2c_mac", env="sc2", env_args={"map_name": "MMM2"}, n_agents=3)
    config = make_config(training, Path("best_model"), first, second, 1, 32, 1, "cpu", "round", kind)
    model = ObservationDelayModel(SimpleNamespace(**config))
    torch.manual_seed(12)
    observations = torch.arange(10).float().reshape(1, 10, 1, 1).expand(500, 10, 3, 1)
    delayed = model.apply(observations, slice(5, 6), training=False)
    assert set(model.last_delays.flatten().tolist()) == expected
    assert torch.equal(delayed.squeeze(-1), 5 - model.last_delays)
    assert torch.equal(model.apply(observations, slice(5, 6), training=True), observations[:, 5:6])


@pytest.mark.parametrize("kind,first,second", [("uniform", 4, 2), ("uniform", 0.5, 3),
                                               ("fixed", 2, 1), ("fixed", 1.5, 0)])
def test_invalid_distribution_parameters(kind, first, second):
    training = dict(mac="vil2c_mac", env="sc2", env_args={"map_name": "MMM2"})
    with pytest.raises(ValueError):
        make_config(training, Path("best_model"), first, second, 1, 32, 1, "cpu", "round", kind)


def test_checkpoint_and_matching_config(tmp_path):
    run = tmp_path / "models" / "training_run"
    for step in ("20", "100", "best_model"):
        (run / step).mkdir(parents=True)
        (run / step / "agent.th").touch()
    config = tmp_path / "sacred" / "training_run" / "1" / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    assert resolve_checkpoint(run).name == "100"
    assert resolve_checkpoint(run, "best_model").name == "best_model"
    assert resolve_checkpoint(run, "20").name == "20"
    assert find_config(resolve_checkpoint(run)) == config
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(run, "21")


def test_eval_config_preserves_architecture_and_rejects_episode_rounding():
    training = dict(mac="vil2c_mac", env="sc2", hidden_dim=128,
                    env_args={"map_name": "MMM2", "seed": 1})
    config = make_config(training, Path("100"), 2, 1, 3, 32, 4, "cpu", "round")
    assert config["hidden_dim"] == 128
    assert config["evaluate"] and not config["save_model"]
    assert config["env_args"]["seed"] == 3
    assert training["env_args"]["seed"] == 1
    with pytest.raises(ValueError):
        make_config(training, Path("100"), 2, 1, 3, 33, 4, "cpu", "round")
    assert delay_pair("-1:0") == (-1, 0)
    for value in ("1:-1", "nan:1", "1:inf"):
        with pytest.raises(Exception):
            delay_pair(value)


def test_gaussian_grid_cli_order_and_total(tmp_path, capsys):
    import json
    from scripts.eval_scripts.evaluate_obs_delay import main
    checkpoint = tmp_path / "best_model"
    checkpoint.mkdir()
    (checkpoint / "agent.th").touch()
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(mac="vil2c_mac", env="sc2", env_args={"map_name": "MMM2"})))
    main([str(checkpoint), "--config", str(config), "--means", "-2", "-1", "0", "1", "2",
          "--stds", "0", "0.5", "1", "1.5", "2", "--episodes", "100",
          "--seeds", "101", "201", "301", "--dry-run"])
    output = capsys.readouterr().out
    assert "Evaluations: 75; episodes each: 100; total episodes: 7500" in output
    rows = [line for line in output.splitlines() if "obs=gaussian" in line]
    assert len(rows) == 75
    assert "mean=-2.0,std=0.0" in rows[0]
    assert "mean=-2.0,std=0.5" in rows[3]
    assert "mean=-1.0,std=0.0" in rows[15]
    assert "mean=2.0,std=2.0" in rows[-1]


@pytest.mark.parametrize("options", [
    ["--means", "1"], ["--stds", "1"],
    ["--means", "1", "--stds", "-1"],
    ["--means", "nan", "--stds", "1"],
    ["--means", "1", "--stds", "inf"],
    ["--means", "1", "--stds", "1", "--delays", "1:1"],
    ["--means", "1", "--stds", "1", "--delay-type", "uniform"],
])
def test_invalid_grid_cli(options):
    from scripts.eval_scripts.evaluate_obs_delay import main
    with pytest.raises(SystemExit) as error:
        main(["unused", *options, "--dry-run"])
    assert error.value.code == 2


def test_negative_gaussian_mean_clamps_samples_not_mean():
    from types import SimpleNamespace
    from components.observation_delay_model import ObservationDelayModel
    training = dict(mac="vil2c_mac", env="sc2", env_args={"map_name": "MMM2"}, n_agents=3)
    config = make_config(training, Path("best_model"), -2, 0, 1, 32, 1, "cpu", "round")
    model = ObservationDelayModel(SimpleNamespace(**config))
    assert torch.all(model._sample_delays(100, 1, "cpu") == 0)
    model.delay_std = 2
    torch.manual_seed(42)
    sampled = model._sample_delays(1000, 1, "cpu")
    assert torch.all(sampled >= 0)
    assert torch.any(sampled > 0)


@pytest.mark.parametrize("env,common_reward", [("sc2", True), ("mpe", True), ("mpe", False)])
def test_eval_loads_frozen_weights_and_never_creates_learner(tmp_path, monkeypatch, env, common_reward):
    from utils.maker import RunnerMaker, LearnerMaker
    from run import parse_buffer_scheme
    from components.episode_buffer import EpisodeBatch
    from components.transforms import OneHot
    args = _args()
    config = vars(args).copy()
    config.update(mac="vil2c_mac", env=env, common_reward=common_reward, runner="parallel", batch_size_run=1,
                  test_nepisode=2, seed=1, device="cpu", state_shape=8,
                  learner="vil2c_ymappo_learner")
    info = dict(n_agents=3, n_actions=4, state_shape=8, obs_shape=5, episode_limit=5)
    scheme = parse_buffer_scheme(info, True, args)
    groups = {"agents": 3}
    batch = EpisodeBatch(scheme, groups, 1, 6, preprocess={"actions": ("actions_onehot", [OneHot(4)])})
    batch["obs"][:] = torch.randn_like(batch["obs"])
    batch["avail_actions"][:] = 1
    original = VIL2CMAC(batch.scheme, groups, args)
    original.save_models(tmp_path)
    original_weights = {key: value.clone() for key, value in original.state_dict().items()}
    calls = []

    class Runner:
        ps, parent_conns, worker_conns = [], [], []
        def __init__(self, args, logger):
            self.logger = logger
        def get_env_info(self):
            return info
        def setup(self, scheme, groups, preprocess, mac):
            self.mac = mac
        def run(self, test_mode):
            assert test_mode and not torch.is_grad_enabled()
            self.mac.init_hidden(1)
            self.mac.forward(batch, 1, test_mode=True)
            calls.append(test_mode)
            for key, value in self.mac.state_dict().items():
                assert torch.equal(value, original_weights[key])
            if env == "sc2":
                self.logger.log_stat("metric/test_battle_won_mean", 0.5, 0)
            key = "metric/test_return_mean" if common_reward else "metric/test_total_return_mean"
            self.logger.log_stat(key, 10, 0)
        def close_env(self):
            calls.append("closed")

    def forbidden(*args, **kwargs):
        pytest.fail("An evaluation must never construct a learner")
    monkeypatch.setattr(LearnerMaker, "make", forbidden)
    monkeypatch.setattr(RunnerMaker, "make", lambda name, **kwargs: Runner(**kwargs))
    metrics = evaluate(config, tmp_path)
    if env == "sc2":
        assert metrics["metric/test_battle_won_mean"] == 0.5
    else:
        assert "metric/test_battle_won_mean" not in metrics
    assert calls == [True, True, "closed"]


@pytest.mark.parametrize("common_reward", [True, False])
def test_mpe_config_and_summary_preserve_reward_semantics(common_reward):
    from scripts.eval_scripts.evaluate_obs_delay import summary_row, return_metric_prefix
    training = dict(mac="vil2c_mac", env="mpe", common_reward=common_reward,
                    reward_scalarisation="mean", env_args=dict(map_name="simple_spread_v3",
                    time_limit=25, scenario_args={"N": 3}, render_mode=None))
    config = make_config(training, Path("best_model"), 2, 1, 1, 4, 2, "cpu", "round")
    assert config["env"] == "mpe"
    assert config["env_args"]["scenario_args"] == {"N": 3}
    assert config["env_args"]["time_limit"] == 25
    prefix = return_metric_prefix(config)
    metrics = {prefix + "_mean": -12.5, prefix + "_std": 2.0, "metric/test_ep_length_mean": 25}
    row = summary_row(config, Path("best_model"), metrics)
    assert row["win_rate"] is None
    assert row["return_mean"] == -12.5
    assert row["return_std"] == 2.0
    assert row["reward_scalarisation"] == "mean"
    assert row["return_metric"] == prefix + "_mean"
