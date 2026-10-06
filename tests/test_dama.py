"""DAMA temporal alignment, learning masks, independent actors and checkpoints."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from run import parse_buffer_scheme
from utils.maker import MACMaker, LearnerMaker


class Logger:
    def __init__(self):
        self.stats = {}

    def log_stat(self, key, value, t):
        self.stats[key] = value


def setup(common_reward=True, **overrides):
    config = yaml.safe_load((ROOT / "src/config/algs/dama_ddpg.yaml").read_text())
    config.update(n_agents=2, n_actions=3, hidden_dim=16, dama_history_length=3,
                  learner_log_interval=1, device="cpu", common_reward=common_reward)
    config.update(overrides)
    args = SimpleNamespace(**config)
    info = dict(n_agents=2, n_actions=3, obs_shape=4, state_shape=8)
    scheme = parse_buffer_scheme(info, common_reward, args)
    batch = EpisodeBatch(scheme, {"agents": 2}, 2, 5,
                         preprocess={"actions": ("actions_onehot", [OneHot(3)])}, device="cpu")
    for t in range(5):
        available = torch.ones(2, 2, 3, dtype=torch.int)
        available[:, 1, 2] = 0
        batch.update(dict(obs=torch.full((2, 2, 4), float(t)),
                          state=torch.full((2, 8), float(t)),
                          avail_actions=available,
                          obs_gen_t=torch.full((2, 2, 1), t),
                          obs_delay=torch.zeros(2, 2, 1, dtype=torch.long),
                          obs_fresh_mask=torch.ones(2, 2, 1)), ts=t)
        if t < 4:
            batch.update(dict(actions=torch.full((2, 2, 1), t % 2),
                              reward=torch.ones(2, 1 if common_reward else 2),
                              terminated=torch.zeros(2, 1, dtype=torch.uint8)), ts=t)
    mac = MACMaker.make(args.mac, batch.scheme, {"agents": 2}, args)
    logger = Logger()
    learner = LearnerMaker.make(args.learner, mac, batch.scheme, logger, args)
    return args, batch, mac, learner, logger


def test_history_uses_only_executed_local_actions():
    _, batch, mac, _, _ = setup()
    inputs = mac._build_inputs(batch, 2)
    # obs[4], oldest-to-newest actions[3*3], valid[3], alignment[3], arrived[1], id[2]
    assert inputs.shape == (2, 2, 22)
    history = inputs[..., 4:13].reshape(2, 2, 3, 3)
    assert torch.equal(history[..., 0, :], torch.zeros(2, 2, 3))
    assert (history[..., 1, 0] == 1).all()
    assert (history[..., 2, 1] == 1).all()
    assert inputs[0, 0, 13:16].tolist() == [0, 1, 1]
    before = inputs.clone()
    batch.update({"actions": torch.full((2, 2, 1), 2)}, ts=2)
    assert torch.equal(before, mac._build_inputs(batch, 2))


def test_timestamp_alignment_missing_and_overwindow():
    _, batch, mac, _, _ = setup()
    batch.update({"obs_gen_t": torch.tensor([[[1], [-1]], [[0], [2]]])}, ts=4)
    inputs = mac._build_inputs(batch, 4)
    assert inputs[0, 0, 16:19].tolist() == [1, 1, 1]  # a1,a2,a3 after o1
    assert inputs[0, 1, 16:20].tolist() == [0, 0, 0, 0]  # nothing arrived
    assert inputs[1, 1, 16:19].tolist() == [0, 1, 1]  # a2,a3 after o2
    assert inputs[1, 0, 16:19].tolist() == [1, 1, 1]  # beyond window still finite


def test_no_future_or_other_agent_observations_in_actor():
    _, batch, mac, _, _ = setup()
    before = mac.forward(batch, 2).detach().clone()
    batch.data.transition_data["obs"][:, 3:] += 1000
    batch.data.transition_data["obs"][:, 2, 1] += 1000
    after = mac.forward(batch, 2)
    assert torch.equal(before[:, 0], after[:, 0])


def test_terminal_reward_kept_padding_and_timeout_excluded():
    _, batch, _, learner, _ = setup()
    batch.data.transition_data["terminated"][0, 1] = 1
    batch.data.transition_data["filled"][1, 3:] = 0  # timeout final observation at t2
    mask = learner.transition_mask(batch)
    assert mask[0, :, 0].tolist() == [1, 1, 0, 0]
    assert mask[1, :, 0].tolist() == [1, 1, 0, 0]


@pytest.mark.parametrize("common_reward", [True, False])
def test_update_masked_actions_target_and_full_checkpoint(tmp_path, common_reward):
    torch.manual_seed(7)
    args, batch, mac, learner, logger = setup(common_reward)
    before = [param.detach().clone() for param in mac.parameters()]
    target_before = [param.detach().clone() for param in learner.target_mac.parameters()]
    batch.data.transition_data["terminated"][:, 3] = 1
    learner.train(batch, 10, 0)
    assert all(torch.isfinite(torch.tensor(value)) for value in logger.stats.values())
    assert any(not torch.equal(old, new) for old, new in zip(before, mac.parameters()))
    for old, online, target in zip(target_before, mac.parameters(), learner.target_mac.parameters()):
        assert torch.allclose(target, old.lerp(online.detach(), args.target_update_interval_or_tau))
    for _ in range(20):
        assert (mac.select_actions(batch, 2, test_mode=False)[:, 1] != 2).all()
    assert (mac.target_actions(batch, 2)[:, 1, 2] == 0).all()
    expected = mac.select_actions(batch, 2, test_mode=True)
    learner.save_models(tmp_path)
    _, _, restored, restored_learner, _ = setup(common_reward)
    restored_learner.load_models(tmp_path)
    assert torch.equal(expected, restored.select_actions(batch, 2, test_mode=True))
    assert restored_learner.training_steps == learner.training_steps
    for original, loaded in zip(learner.critic.parameters(), restored_learner.critic.parameters()):
        assert torch.equal(original, loaded)
    # Optimizer state AND exact target networks permit the same next update.
    torch.manual_seed(9)
    learner.train(batch, 11, 1)
    torch.manual_seed(9)
    restored_learner.train(batch, 11, 1)
    for original, loaded in zip(mac.parameters(), restored.parameters()):
        assert torch.equal(original, loaded)


def test_competitive_actor_gradient_is_local():
    _, batch, mac, learner, _ = setup(False)
    inputs = mac.inputs.sequence(batch)[:, :-1]
    logits = mac.logits(inputs, batch["avail_actions"][:, :-1])
    chosen = torch.nn.functional.gumbel_softmax(logits, hard=True)
    joint = torch.stack((chosen[..., 0, :], chosen[..., 1, :].detach()), dim=-2)
    learner.critic.agent_value(inputs, joint, 0).mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mac.agent.actors[0].parameters())
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in mac.agent.actors[1].parameters())


def test_reject_delayed_training_and_ablation_shapes():
    _, batch, _, learner, _ = setup()
    batch.data.transition_data["obs_gen_t"][:, 2] = 0
    with pytest.raises(ValueError, match="current observations"):
        learner.train(batch, 0, 0)
    for settings, size in ((dict(dama_use_history=False), 6),
                           (dict(dama_use_alignment=False), 18),
                           (dict(dama_history_length=0), 7)):
        _, batch, mac, _, _ = setup(**settings)
        assert mac._build_inputs(batch, 2).shape[-1] == size


def test_padding_rewards_cannot_change_update():
    _, batch, mac, learner, _ = setup()
    other = copy.deepcopy(learner)
    batch.data.transition_data["filled"][:, 3:] = 0
    modified = copy.deepcopy(batch)
    modified.data.transition_data["reward"][:, 2:] = 1e6
    torch.manual_seed(101)
    learner.train(batch, 10, 0)
    torch.manual_seed(101)
    other.train(modified, 10, 0)
    for p, q in zip(mac.parameters(), other.mac.parameters()):
        assert torch.equal(p, q)


def test_tag_checkpoint_uses_selected_team_not_reward_cancellation():
    from runners.dama_parallel_runner import DAMAParallelRunner
    runner = DAMAParallelRunner.__new__(DAMAParallelRunner)
    runner.args = SimpleNamespace(common_reward=False, n_agents=3, dama_checkpoint_agents=[0, 1])
    runner.logger, runner.t_env = Logger(), 0
    runner._log([[10, 10, -20], [12, 12, -24]], {"n_episodes": 2}, "test_")
    assert runner.logger.stats["metric/test_total_return_mean"] == 0
    assert runner.logger.stats["metric/test_return_mean"] == 11


@pytest.mark.parametrize("scenario, common_reward", [("simple_spread_v3", True), ("simple_tag_v3", False)])
def test_real_mpe_shared_runner_train_then_frozen_delays(tmp_path, scenario, common_reward):
    pytest.importorskip("mpe2")
    sys.path.insert(0, str(ROOT / "scripts/eval_scripts"))
    from dama_obs_delay import create_runner, evaluate_condition, summarize
    config = yaml.safe_load((ROOT / "src/config/default.yaml").read_text())
    config.update(yaml.safe_load((ROOT / "src/config/algs/dama_ddpg.yaml").read_text()))
    config.update(env="mpe", seed=1, common_reward=common_reward, hidden_dim=16,
                  dama_history_length=2, learner_log_interval=1,
                  env_args=dict(map_name=scenario, seed=1, time_limit=4, scenario_args={}))
    info = dict(n_agents=3 if common_reward else 4, n_actions=5,
                obs_shape=18 if common_reward else 16, state_shape=54 if common_reward else 64)
    # Query actual adapter shapes; tag has heterogeneous padded observations.
    from envs.mpe_wrapper import MPEWrapper
    env = MPEWrapper(**config["env_args"], common_reward=common_reward)
    try:
        info = env.get_env_info()
    finally:
        env.close()
    args = SimpleNamespace(**config)
    args.n_agents, args.n_actions = info["n_agents"], info["n_actions"]
    args.device = "cpu"
    scheme = parse_buffer_scheme(info, common_reward, args)
    template = EpisodeBatch(scheme, {"agents": args.n_agents}, 1, 1,
                            preprocess={"actions": ("actions_onehot", [OneHot(args.n_actions)])})
    mac = MACMaker.make(args.mac, template.scheme, {"agents": args.n_agents}, args)
    mac.save_models(tmp_path)
    runner, loaded, logger = create_runner(config, tmp_path, "cpu")
    learner = LearnerMaker.make(args.learner, loaded, template.scheme, logger, args)
    try:
        batch = runner.run(test_mode=False)
        assert (batch["obs_delay"][0, :runner.t] == 0).all()
        learner.train(batch, runner.t, 0)
        assert learner.training_steps == 1
        frozen = {key: value.clone() for key, value in loaded.state_dict().items()}
        rows = []
        for delay in (0, 3):
            rows += evaluate_condition(runner, dict(id=f"fixed_{delay}", kind="fixed", value=delay, cap=4),
                                       episodes=1, seed=101, logger=logger)
        assert rows[0]["unarrived_fraction"] == 0
        assert rows[1]["unarrived_fraction"] == 0.75
        assert rows[1]["history_overflow_fraction"] == 0.25
        assert all(torch.equal(frozen[key], value) for key, value in loaded.state_dict().items())
        assert len(summarize(rows)) == 2
        repeated = evaluate_condition(runner, dict(id="fixed_0", kind="fixed", value=0, cap=4),
                                      episodes=1, seed=101, logger=logger)
        assert repeated == rows[:1]  # Paired environment seeds are reproducible.
        if not common_reward:
            assert "return_predator_mean" in rows[0] and "return_prey_mean" in rows[0]
    finally:
        runner.close_env()


def test_delay_grid_and_smac_seed_uses_constructor(monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts/eval_scripts"))
    import dama_obs_delay as evaluation
    cases = evaluation.delay_conditions([0, 1], [0, 1], [0, 2, 2], 4)
    assert len(cases) == 6 and cases[0]["id"] == "fixed_0"
    with pytest.raises(ValueError):
        evaluation.delay_conditions([], [-1], [], 4)
    created_seeds = []

    class Env:
        def __init__(self):
            self.env = SimpleNamespace(agents=[])

        def seed(self, *values):
            pytest.fail("SMAC getter must not be used as a seed setter")

        def close(self):
            pass

    def make_env(kind, **kwargs):
        assert kind == "delayed_sc2"
        created_seeds.append(kwargs["seed"])
        return Env()

    _, batch, mac, _, _ = setup()
    runner = SimpleNamespace(env=Env(), args=SimpleNamespace(env="delayed_sc2", env_args={},
                             common_reward=True, reward_scalarisation="sum"), mac=mac, t=4)
    logger = evaluation.EvaluationLogger()

    def run_episode(test_mode):
        logger.log_stat("running/test_battle_won_mean", 1, 0)
        return batch

    runner.run = run_episode
    monkeypatch.setattr(evaluation.EnvMaker, "make", make_env)
    rows = evaluation.evaluate_condition(runner, dict(id="fixed_0", kind="fixed", value=0, cap=4),
                                         episodes=2, seed=101, logger=logger)
    assert created_seeds == [101, 102]
    assert [row["win"] for row in rows] == [1, 1]


@pytest.mark.parametrize("terminal, expected_loss", [(True, 3.25), (False, 4.0)])
def test_td_target_terminal_reward_and_timeout_bootstrap(terminal, expected_loss):
    _, batch, _, learner, logger = setup(gamma=0.5)
    with torch.no_grad():
        for parameter in learner.critic.parameters():
            parameter.zero_()
        for parameter in learner.target_critic.parameters():
            parameter.zero_()
        for critic in learner.target_critic.critics:
            critic[-1].bias.fill_(2)
    batch.data.transition_data["terminated"][:, 3] = int(terminal)
    learner.train(batch, 10, 0)
    # r=1, gamma=.5, Q_target=2: three targets=2, terminal last target=1.
    # Timeout instead has four targets=2. Both last rewards must remain included.
    assert logger.stats["critic_loss"] == pytest.approx(expected_loss)
