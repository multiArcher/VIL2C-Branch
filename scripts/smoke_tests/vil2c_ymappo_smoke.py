"""CPU smoke test for VIL2C-on-YMAPPO (no StarCraft)."""
from types import SimpleNamespace as SN

import torch as th

from components.episode_buffer import EpisodeBatch
from controllers.vil2c_controller import VIL2CMAC
from learners.vil2c_ymappo_learner import VIL2CYMAPPOLearner
from modules.agents.vil2c_ymappo_agent import VIL2CYMAPPOAgent


class _Logger:
    def __init__(self):
        self.stats = {}

    def log_stat(self, key, val, t):
        self.stats.setdefault(key, []).append((t, val))


def _args(**overrides):
    args = SN(
        n_agents=3,
        n_actions=4,
        hidden_dim=16,
        msg_dim=8,
        device=th.device("cpu"),
        lr=5e-4,
        resonet_lr=3e-5,
        entropy_coef=0.01,
        epochs=2,
        eps_clip=0.05,
        gamma=0.99,
        gae_lambda=0.95,
        huber_delta=10.0,
        use_clipped_value_loss=True,
        use_huber_loss=True,
        use_value_norm=True,
        use_value_active_masks=True,
        standardise_rewards=False,
        common_reward=True,
        use_rnn=True,
        actor_gain=0.01,
        grad_norm_clip=10,
        data_chunk_length=5,
        num_mini_batch=1,
        layer_n=1,
        use_feature_normalization=True,
        critic_use_local_obs=True,
        use_death_mask=True,
        learner_log_interval=1,
        mask_before_softmax=True,
        obs_agent_id=True,
        obs_last_action=False,
        obs_individual_obs=False,
        agent="vil2c_ymappo",
        critic_type="ymappo_critic",
        action_selector="ymappo_soft_policies",
        agent_output_type="pi_logits",
        test_greedy=True,
        use_resonet=False,
        use_progressive=True,
        max_wait_steps=3,
        entropy_threshold=0.8,
        comm_delay_type="gaussian",
        comm_gaussian_delay_mean=1.0,
        comm_gaussian_delay_std=1.0,
        comm_max_delay=16,
        rollout_length=20,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _batch(args, T=10):
    n_agents, n_actions = args.n_agents, args.n_actions
    bs = 2
    scheme = {
        "state": {"vshape": 8, "dtype": th.float32},
        "obs": {"vshape": 5, "group": "agents", "dtype": th.float32},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (n_actions,), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "episode_end": {"vshape": (1,), "dtype": th.uint8},
        "active_masks": {"vshape": (1,), "group": "agents", "dtype": th.float32},
        "actor_hidden": {"vshape": (args.hidden_dim,), "group": "agents", "dtype": th.float32},
    }
    groups = {"agents": n_agents}
    batch = EpisodeBatch(scheme, groups, bs, T + 1, device="cpu")
    batch.update(
        {
            "state": th.randn(bs, T + 1, 8),
            "obs": th.randn(bs, T + 1, n_agents, 5),
            "avail_actions": th.ones(bs, T + 1, n_agents, n_actions, dtype=th.int),
            "reward": th.randn(bs, T + 1, 1),
            "terminated": th.zeros(bs, T + 1, 1, dtype=th.uint8),
            "episode_end": th.zeros(bs, T + 1, 1, dtype=th.uint8),
            "active_masks": th.ones(bs, T + 1, n_agents, 1),
            "actor_hidden": th.zeros(bs, T + 1, n_agents, args.hidden_dim),
            "actions": th.zeros(bs, T + 1, n_agents, 1, dtype=th.long),
        },
        ts=slice(None),
        mark_filled=True,
    )
    return batch, groups


def test_forward_shapes():
    args = _args()
    agent = VIL2CYMAPPOAgent(10, args)
    h = agent.init_hidden().expand(6, -1)
    out, h2 = agent(th.zeros(6, 10), h)
    assert out.shape == (6, 4)
    assert h2.shape == (6, 16)


def test_mac_train_full_messages():
    args = _args()
    batch, groups = _batch(args)
    mac = VIL2CMAC(batch.scheme, groups, args)
    mac.init_hidden(batch.batch_size)
    pi = mac.forward(batch, t=0, test_mode=False)
    assert pi.shape == (2, 3, 4)
    assert th.allclose(pi.sum(dim=-1), th.ones(2, 3), atol=1e-5)


def test_learner_one_update():
    args = _args()
    batch, groups = _batch(args)
    mac = VIL2CMAC(batch.scheme, groups, args)
    mac.to("cpu")
    logger = _Logger()
    learner = VIL2CYMAPPOLearner(mac, batch.scheme, logger, args)
    learner.cuda()
    learner.train(batch, t_env=100, episode_num=1)
    assert "pg_loss" in logger.stats
    assert "critic_loss" in logger.stats
    assert "voi_loss" in logger.stats
    assert logger.stats["voi_loss"][-1][1] == 0.0


def test_comm_delay_train_zero_eval_progressive():
    args = _args(
        comm_gaussian_delay_mean=2.0,
        comm_gaussian_delay_std=0.0,
        comm_max_delay=2,
        max_wait_steps=2,
        entropy_threshold=-1.0,
    )
    batch, groups = _batch(args, T=4)
    mac = VIL2CMAC(batch.scheme, groups, args)
    mac.init_hidden(batch.batch_size)
    mac.forward(batch, t=0, test_mode=False)
    assert th.all(mac._cache["wait_steps"] == 0)
    assert mac._cache["arrival"] is None
    assert th.allclose(mac._cache["recv_mask"], th.ones_like(mac._cache["recv_mask"]))

    mac.init_hidden(batch.batch_size)
    mac.forward(batch, t=0, test_mode=True)
    off = ~th.eye(args.n_agents, dtype=th.bool)
    arrival = mac._cache["arrival"]
    assert th.all(arrival[:, off] == 2)
    assert th.all(mac._cache["wait_steps"] == 2)
    recv = mac._cache["recv_mask"].view(2, 3, 3)
    assert th.all(recv[:, off] == 1.0)


if __name__ == "__main__":
    test_forward_shapes()
    test_mac_train_full_messages()
    test_learner_one_update()
    test_comm_delay_train_zero_eval_progressive()
    print("vil2c_ymappo smoke tests ok")
