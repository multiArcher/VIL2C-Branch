"""VIL2C policy: local encoder, attention aggregator, RNN actor, optional ResoNet."""

from __future__ import annotations

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.agents.agent import Agent


class MessageEncoder(nn.Module):
    def __init__(self, input_dim: int, msg_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden = hidden_dim or msg_dim
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, msg_dim)

    def forward(self, obs):
        return self.fc2(F.relu(self.fc1(obs)))


class MessageAttentionAggregator(nn.Module):
    """Query from own message; keys/values from buffered teammate messages."""

    def __init__(self, msg_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden = hidden_dim or msg_dim
        self.query = nn.Linear(msg_dim, hidden)
        self.key = nn.Linear(msg_dim, hidden)
        self.value = nn.Linear(msg_dim, hidden)
        self.out = nn.Linear(hidden, msg_dim)
        self.scale = hidden ** -0.5

    def forward(self, own_msg: th.Tensor, other_msgs: th.Tensor, mask: th.Tensor | None = None):
        if other_msgs.size(1) == 0:
            return th.zeros_like(own_msg)

        q = self.query(own_msg).unsqueeze(1)
        k = self.key(other_msgs)
        v = self.value(other_msgs)
        logits = th.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            valid = mask.sum(dim=-1, keepdim=True) > 0
            logits = logits.masked_fill(mask.unsqueeze(1) == 0, -1e9)
            alpha = F.softmax(logits, dim=-1) * valid.unsqueeze(-1).float()
        else:
            alpha = F.softmax(logits, dim=-1)
        return self.out(th.matmul(alpha, v).squeeze(1))


class ResoNet(nn.Module):
    """Decentralized bandwidth/power allocation. Softmax + optional uniform floor."""

    def __init__(
        self,
        obs_dim: int,
        n_agents: int,
        hidden_dim: int = 128,
        bandwidth_budget: float = 1.0,
        power_budget: float = 1.0,
        alloc_floor: float = 0.2,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_others = max(n_agents - 1, 1)
        self.bandwidth_budget = float(bandwidth_budget)
        self.power_budget = float(power_budget)
        self.alloc_floor = float(alloc_floor)

        self.fc1 = nn.Linear(obs_dim + self.n_others, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.b_head = nn.Linear(hidden_dim, self.n_others)
        self.p_head = nn.Linear(hidden_dim, self.n_others)

    def forward(self, obs: th.Tensor, channel_feats: th.Tensor):
        x = th.cat([obs, channel_feats], dim=-1)
        h = F.relu(self.fc2(F.relu(self.fc1(x))))
        b_share = F.softmax(self.b_head(h), dim=-1)
        p_share = F.softmax(self.p_head(h), dim=-1)
        if self.alloc_floor > 0.0:
            uniform = 1.0 / self.n_others
            mix = 1.0 - self.alloc_floor
            b_share = mix * b_share + self.alloc_floor * uniform
            p_share = mix * p_share + self.alloc_floor * uniform
        return (
            self._scatter_to_full(b_share * self.bandwidth_budget),
            self._scatter_to_full(p_share * self.power_budget),
        )

    def _scatter_to_full(self, alloc_others: th.Tensor) -> th.Tensor:
        m = alloc_others.size(0)
        bs = m // self.n_agents
        device = alloc_others.device
        full = th.zeros(m, self.n_agents, device=device, dtype=alloc_others.dtype)
        for i in range(self.n_agents):
            recipients = [j for j in range(self.n_agents) if j != i]
            rows = th.arange(bs, device=device) * self.n_agents + i
            full[rows[:, None], th.tensor(recipients, device=device)] = alloc_others[rows]
        return full


class VIL2CAgent(Agent):
    def __init__(self, input_shape: int, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.msg_dim = int(getattr(args, "msg_dim", 64))
        self.hidden_dim = args.hidden_dim

        self.encoder = MessageEncoder(input_shape, self.msg_dim, args.hidden_dim)
        self.aggregator = MessageAttentionAggregator(self.msg_dim, args.hidden_dim)
        actor_in = self.msg_dim * 2
        self.fc1 = nn.Linear(actor_in, args.hidden_dim)
        if args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

        self.resonet = ResoNet(
            obs_dim=input_shape,
            n_agents=args.n_agents,
            hidden_dim=args.hidden_dim,
            bandwidth_budget=getattr(args, "bandwidth_budget", 1.0),
            power_budget=getattr(args, "power_budget", 1.0),
            alloc_floor=getattr(args, "resonet_alloc_floor", 0.2),
        )

    def init_hidden(self):
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def encode(self, inputs: th.Tensor) -> th.Tensor:
        return self.encoder(inputs)

    def allocate(self, inputs: th.Tensor, channel_feats: th.Tensor):
        return self.resonet(inputs, channel_feats)

    def policy_from_messages(
        self,
        own_msg: th.Tensor,
        all_msgs: th.Tensor,
        recv_mask: th.Tensor,
        hidden_state: th.Tensor,
    ):
        if all_msgs.dim() == 4:
            bs, n = all_msgs.size(0), all_msgs.size(1)
            msgs_exp = all_msgs.reshape(bs * n, n, -1)
        else:
            bs = all_msgs.size(0)
            n = self.n_agents
            msgs_exp = all_msgs.unsqueeze(1).expand(bs, n, n, -1).reshape(bs * n, n, -1)
        eye = th.eye(n, device=recv_mask.device).unsqueeze(0).expand(bs, -1, -1).reshape(bs * n, n)
        other_mask = recv_mask * (1.0 - eye)
        masked_msgs = msgs_exp * other_mask.unsqueeze(-1)

        agg = self.aggregator(own_msg, masked_msgs, mask=other_mask)
        x = F.relu(self.fc1(th.cat([own_msg, agg], dim=-1)))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        return self.fc2(h), h

    def forward(self, inputs, hidden_state, all_msgs=None, recv_mask=None):
        own_msg = self.encode(inputs)
        m = inputs.size(0)
        n = self.n_agents
        bs = m // n
        if all_msgs is None:
            all_msgs = own_msg.view(bs, n, -1)
        if recv_mask is None:
            recv_mask = th.zeros(m, n, device=inputs.device)
        return self.policy_from_messages(own_msg, all_msgs, recv_mask, hidden_state)
