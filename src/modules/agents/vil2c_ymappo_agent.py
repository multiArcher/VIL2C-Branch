"""VIL2C communication head on the Yu MAPPO actor backbone."""

from __future__ import annotations

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.agents.agent import Agent
from modules.agents.vil2c_agent import MessageAttentionAggregator, MessageEncoder, ResoNet
from modules.agents.ymappo_base import YMAPPOBase
from utils.th_utils import orthogonal_init_


class VIL2CYMAPPOAgent(Agent):
    def __init__(self, input_shape: int, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.msg_dim = int(getattr(args, "msg_dim", 64))
        self.hidden_dim = args.hidden_dim
        layer_n = int(getattr(args, "layer_n", 1))
        use_fn = getattr(args, "use_feature_normalization", True)
        gain = getattr(args, "actor_gain", 0.01)

        self.encoder = MessageEncoder(input_shape, self.msg_dim, args.hidden_dim)
        self.aggregator = MessageAttentionAggregator(self.msg_dim, args.hidden_dim)
        self.base = YMAPPOBase(
            self.msg_dim * 2, args.hidden_dim, layer_n=layer_n, use_feature_norm=use_fn
        )
        self.fc_out = nn.Linear(args.hidden_dim, args.n_actions)
        orthogonal_init_(self.fc_out, gain=gain)

        self.resonet = ResoNet(
            obs_dim=input_shape,
            n_agents=args.n_agents,
            hidden_dim=args.hidden_dim,
            bandwidth_budget=getattr(args, "bandwidth_budget", 1.0),
            power_budget=getattr(args, "power_budget", 1.0),
            alloc_floor=getattr(args, "resonet_alloc_floor", 0.2),
        )

    def init_hidden(self):
        return self.fc_out.weight.new(1, self.hidden_dim).zero_()

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
        feat, h = self.base(th.cat([own_msg, agg], dim=-1), hidden_state)
        return self.fc_out(feat), h

    def forward(self, inputs, hidden_state, all_msgs=None, recv_mask=None):
        own_msg = self.encode(inputs)
        m = inputs.size(0)
        n = self.n_agents
        bs = m // n
        if all_msgs is None:
            all_msgs = own_msg.view(bs, n, -1)
        if recv_mask is None:
            recv_mask = th.ones(m, n, device=inputs.device)
        return self.policy_from_messages(own_msg, all_msgs, recv_mask, hidden_state)
