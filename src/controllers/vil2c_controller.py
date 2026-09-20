"""VIL2C MAC: message passing + progressive reception.

Delay *distribution* matches BCRBC (train=0, eval gaussian/fixed/uniform).
ResoNet/Shannon latency is not used. Progressive wait is kept: within one
decision, teammate messages are revealed by sampled arrival step, and the
policy may stop early when entropy is low.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch as th
import torch.nn.functional as F

from components.delay_model import DelayModel
from components.observation_delay_model import ObservationDelayModel
from controllers.mac import MAC
from utils.maker import AgentMaker, ActionSelectorMaker


def _categorical_entropy(logits: th.Tensor) -> th.Tensor:
    probs = F.softmax(logits, dim=-1)
    return -(probs * th.log(probs.clamp(min=1e-10))).sum(dim=-1)


def progressive_receive(arrival_steps, compute_policy_fn, entropy_threshold: float, max_wait: int):
    """Reveal messages as they arrive within one decision; stop if entropy is low.

    `arrival_steps` is [M, N]; self should be -1 (always available via own_msg).
    """
    m, n = arrival_steps.shape
    device = arrival_steps.device
    received = th.zeros(m, n, device=device)
    wait_steps = th.full((m,), max_wait, device=device, dtype=th.long)
    done = th.zeros(m, dtype=th.bool, device=device)
    final_logits = None

    for step in range(max_wait + 1):
        newly = (arrival_steps <= step) & (arrival_steps >= 0)
        received = th.maximum(received, newly.float())
        logits = compute_policy_fn(received)
        stop_now = (_categorical_entropy(logits) <= entropy_threshold) & (~done)
        wait_steps = th.where(stop_now, th.full_like(wait_steps, step), wait_steps)
        done = done | stop_now
        final_logits = logits if final_logits is None else th.where(
            done.unsqueeze(-1), final_logits, logits
        )
        # Do not bool(done.all()): that syncs CUDA every wait slot and dominates eval.

    if final_logits is None:
        final_logits = compute_policy_fn(received)
    return received, wait_steps, final_logits


class VIL2CMAC(MAC):
    def __init__(self, scheme: dict, groups: dict, args: SimpleNamespace):
        super().__init__(scheme, groups, args)
        self.n_agents = args.n_agents
        self.args = args
        self.device = args.device
        self.agent_output_type = args.agent_output_type
        self.use_progressive = bool(getattr(args, "use_progressive", True))
        self.max_wait = int(getattr(args, "max_wait_steps", 3))
        self.entropy_threshold = float(getattr(args, "entropy_threshold", 0.8))

        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.action_selector = ActionSelectorMaker.make(args.action_selector, args)
        self.observation_delay_model = ObservationDelayModel(args)

        delay_type = getattr(args, "comm_delay_type", "gaussian")
        if delay_type == "fixed":
            delay_type = "gaussian"
        max_delay = int(
            getattr(args, "comm_max_delay", getattr(args, "bcrbc_max_delay", 16))
        )
        self.delay_model = DelayModel(
            delay_type=delay_type,
            delay_mean=float(getattr(args, "comm_gaussian_delay_mean", 1.0)),
            delay_std=float(getattr(args, "comm_gaussian_delay_std", 1.0)),
            max_delay=max_delay,
            delay_per_source=True,
        )
        self.hidden_states = None
        self.last_wait_steps = None
        self._cache = {}

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        return self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )

    def forward(self, ep_batch, t, test_mode=False, **kwargs):
        agent_inputs = self._build_inputs(ep_batch, t, test_mode=test_mode)
        avail_actions = ep_batch["avail_actions"][:, t]
        bs = ep_batch.batch_size
        n = self.n_agents
        device = agent_inputs.device

        own_msg = self.agent.encode(agent_inputs)
        all_msgs = own_msg.view(bs, n, -1)
        h0 = self.hidden_states

        if not test_mode:
            recv_mask = th.ones(bs * n, n, device=device)
            wait_steps = th.zeros(bs * n, dtype=th.long, device=device)
            logits, self.hidden_states = self.agent.policy_from_messages(
                own_msg, all_msgs, recv_mask, h0
            )
            arrival = None
        else:
            arrival = self._sample_arrival(bs, n, device, test_mode=True)
            if not self.use_progressive:
                recv_mask = ((arrival >= 0) & (arrival <= self.max_wait)).float()
                eye = th.eye(n, device=device).unsqueeze(0).expand(bs, -1, -1)
                recv_mask = th.maximum(recv_mask, eye).reshape(bs * n, n)
                wait_steps = th.zeros(bs * n, dtype=th.long, device=device)
                logits, self.hidden_states = self.agent.policy_from_messages(
                    own_msg, all_msgs, recv_mask, h0
                )
            else:
                arrival_flat = arrival.reshape(bs * n, n)

                def compute_policy(received_mask):
                    logits_t, _ = self.agent.policy_from_messages(
                        own_msg, all_msgs, received_mask, h0
                    )
                    if getattr(self.args, "mask_before_softmax", True):
                        logits_t = logits_t.clone()
                        logits_t[avail_actions.reshape(bs * n, -1) == 0] = -1e10
                    return logits_t

                recv_mask, wait_steps, logits = progressive_receive(
                    arrival_flat,
                    compute_policy,
                    entropy_threshold=self.entropy_threshold,
                    max_wait=self.max_wait,
                )
                _, self.hidden_states = self.agent.policy_from_messages(
                    own_msg, all_msgs, recv_mask, h0
                )

        self.last_wait_steps = wait_steps.view(bs, n).detach()
        self._cache = {
            "recv_mask": recv_mask.detach(),
            "wait_steps": self.last_wait_steps,
            "arrival": None if arrival is None else arrival.detach(),
        }

        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                logits = logits.clone()
                logits[avail_actions.reshape(bs * n, -1) == 0] = -1e10
            agent_outs = F.softmax(logits, dim=-1)
        else:
            agent_outs = logits
        return agent_outs.view(bs, n, -1)

    def _sample_arrival(self, bs: int, n: int, device, test_mode: bool) -> th.Tensor:
        """Per (receiver, sender) arrival slot in [0, max_wait]; diagonal = -1.

        Train: all zeros (immediate). Test: BCRBC DelayModel draw, then clamp.
        """
        eye = th.eye(n, device=device, dtype=th.bool)
        if not test_mode:
            arrival = th.zeros(bs, n, n, device=device, dtype=th.long)
            arrival = arrival.masked_fill(eye.unsqueeze(0), -1)
            return arrival
        delays = self.delay_model._sample_delay((bs, n, n), device, training=False)
        delays = delays.clamp(0, self.max_wait)
        arrival = delays.masked_fill(eye.unsqueeze(0), -1)
        return arrival

    def compute_voi_loss(self, ep_batch, t: int, hidden=None):
        return th.zeros((), device=self.device)

    def init_hidden(self, batch_size):
        self.hidden_states = (
            self.agent.init_hidden()
            .unsqueeze(0)
            .expand(batch_size, self.n_agents, -1)
            .contiguous()
        )
        self.last_wait_steps = th.zeros(batch_size, self.n_agents, dtype=th.long)
        self._cache = {}

    def resonet_parameters(self):
        if hasattr(self.agent, "resonet"):
            return self.agent.resonet.parameters()
        return []

    def actor_parameters(self):
        return [p for name, p in self.agent.named_parameters() if not name.startswith("resonet.")]

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage)
        )

    def _build_agents(self, input_shape):
        self.agent = AgentMaker.make(self.args.agent, input_shape, self.args)

    def _build_inputs(self, batch, t, test_mode=None):
        bs = batch.batch_size
        training = self.training if test_mode is None else not test_mode
        obs = self.observation_delay_model.apply(
            batch["obs"], slice(t, t + 1), training=training
        ).squeeze(1)
        inputs = [obs]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)
            )
        return th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
