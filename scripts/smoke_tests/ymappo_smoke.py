"""CPU smoke test for YMAPPO modules (no StarCraft)."""
from types import SimpleNamespace as SN

import torch as th

from components.episode_buffer import EpisodeBatch
from components.value_norm import ValueNorm
from controllers.basic_controller import BasicMAC
from learners.ymappo_learner import YMAPPOLearner
from modules.agents.ymappo_rnn_agent import YMAPPORNNAgent
from modules.critics.ymappo_critic import YMAPPOCritic


class _Logger:
    def __init__(self):
        self.stats = {}

    def log_stat(self, key, val, t):
        self.stats.setdefault(key, []).append((t, val))


def _args():
    return SN(
        n_agents=3,
        n_actions=4,
        hidden_dim=16,
        device=th.device("cpu"),
        lr=5e-4,
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
        agent="ymappo_rnn",
        critic_type="ymappo_critic",
        action_selector="ymappo_soft_policies",
        agent_output_type="pi_logits",
        test_greedy=True,
    )


def test_value_norm_roundtrip():
    vn = ValueNorm(1)
    x = th.arange(20, dtype=th.float32).view(20, 1)
    vn.update(x)
    y = vn.denormalize(vn.normalize(x))
    assert th.allclose(y, x, atol=1e-3), (y - x).abs().max()


def test_forward_shapes():
    args = _args()
    agent = YMAPPORNNAgent(10, args)
    h = agent.init_hidden().expand(6, -1)
    out, h2 = agent(th.zeros(6, 10), h)
    assert out.shape == (6, 4)
    assert h2.shape == (6, 16)


def test_learner_one_update():
    args = _args()
    n_agents, n_actions = args.n_agents, args.n_actions
    bs, T = 2, 10
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
        "filled": {"vshape": (1,), "dtype": th.long},
    }
    # EpisodeBatch adds filled itself
    scheme.pop("filled")
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
    mac = BasicMAC(batch.scheme, groups, args)
    mac.to("cpu")
    logger = _Logger()
    learner = YMAPPOLearner(mac, batch.scheme, logger, args)
    learner.cuda()
    learner.train(batch, t_env=100, episode_num=1)
    assert "pg_loss" in logger.stats
    assert "critic_loss" in logger.stats


if __name__ == "__main__":
    test_value_norm_roundtrip()
    test_forward_shapes()
    test_learner_one_update()
    print("ymappo smoke tests ok")
