"""DAMA adapted to delayed observations and immediately executed discrete actions."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from components.dama_history import ActionHistoryInputs
from utils.maker import AgentMaker


class DAMAMAC(nn.Module):
    def __init__(self, scheme, groups, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.action_selector = None  # Public runners inspect its optional epsilon.
        self.inputs = ActionHistoryInputs(scheme, args)
        self.agent = AgentMaker.make(args.agent, self.inputs.input_dim, args)
        self.temperature = float(args.dama_gumbel_temperature)
        if self.temperature <= 0:
            raise ValueError("dama_gumbel_temperature must be positive")

    @staticmethod
    def mask_logits(logits, available):
        available = available.bool().clone()
        # Padding slots may have no available actions. Give them a finite dummy
        # action; learner masks exclude them from both losses.
        available[..., 0] |= ~available.any(dim=-1)
        return logits.masked_fill(~available, -1e9)

    def logits(self, inputs, available):
        return self.mask_logits(self.agent(inputs), available)

    def _build_inputs(self, batch, t):
        return self.inputs.build(batch, t)

    def forward(self, ep_batch, t, test_mode=False):
        return self.logits(self._build_inputs(ep_batch, t), ep_batch["avail_actions"][:, t])

    def select_actions(self, ep_batch, t_ep, t_env=0, bs=slice(None), test_mode=False):
        logits = self.forward(ep_batch, t_ep)[bs]
        if test_mode:
            return logits.argmax(dim=-1)
        return F.gumbel_softmax(logits, tau=self.temperature, hard=True).argmax(dim=-1)

    def target_actions(self, ep_batch, t_ep):
        return F.one_hot(self.forward(ep_batch, t_ep).argmax(-1), self.args.n_actions).float()

    def init_hidden(self, batch_size):
        pass  # Explicit history, no recurrent state.

    def load_state(self, other_mac):
        self.load_state_dict(other_mac.state_dict())

    def save_models(self, path):
        torch.save(self.agent.state_dict(), Path(path) / "agent.th")

    def load_models(self, path):
        self.agent.load_state_dict(torch.load(Path(path) / "agent.th",
                                             map_location=next(self.parameters()).device,
                                             weights_only=True))
