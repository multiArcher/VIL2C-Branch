import torch as th
import torch.nn as nn

from modules.agents.ymappo_base import YMAPPOBase
from utils.th_utils import orthogonal_init_


class YMAPPOCritic(nn.Module):
    """Centralised recurrent V with agent-specific (AS) input: EP state ⊕ local obs ⊕ id."""

    def __init__(self, scheme, args):
        super(YMAPPOCritic, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.output_type = "v"
        hidden_dim = args.hidden_dim
        input_shape = self._get_input_shape(scheme)
        layer_n = int(getattr(args, "layer_n", 1))
        use_fn = getattr(args, "use_feature_normalization", True)

        self.base = YMAPPOBase(input_shape, hidden_dim, layer_n=layer_n, use_feature_norm=use_fn)
        self.fc_out = nn.Linear(hidden_dim, 1)
        orthogonal_init_(self.fc_out, gain=1)

    def init_hidden(self, batch_size, device=None):
        device = device if device is not None else self.fc_out.weight.device
        return self.fc_out.weight.new_zeros(batch_size, self.n_agents, self.args.hidden_dim)

    def forward(self, batch, t, hidden_state):
        inputs, bs = self._build_inputs(batch, t)
        feat, h = self.base(inputs, hidden_state)
        v = self.fc_out(feat)
        v = v.view(bs, self.n_agents, 1)
        h = h.view(bs, self.n_agents, self.args.hidden_dim)
        return v, h

    def forward_tensor(self, inputs, hidden_state):
        feat, h = self.base(inputs, hidden_state)
        return self.fc_out(feat), h

    def build_inputs_all(self, batch, t_end):
        """AS critic inputs for t in [0, t_end): (B, T, A, in_dim)."""
        inputs = []
        for t in range(t_end):
            x, _ = self._build_inputs(batch, t)
            bs = batch.batch_size
            inputs.append(x.view(bs, self.n_agents, -1))
        return th.stack(inputs, dim=1)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        ts = slice(t, t + 1)
        parts = []
        parts.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))
        if getattr(self.args, "critic_use_local_obs", True):
            parts.append(batch["obs"][:, ts])
        elif self.args.obs_individual_obs:
            parts.append(
                batch["obs"][:, ts].view(bs, 1, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1)
            )
        agent_id = (
            th.eye(self.n_agents, device=batch.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(bs, 1, -1, -1)
        )
        parts.append(agent_id)
        inputs = th.cat(parts, dim=-1)
        if getattr(self.args, "use_death_mask", True) and "active_masks" in batch.scheme:
            active = batch["active_masks"][:, ts]
            dead = (active <= 0).expand_as(inputs)
            masked = inputs.clone()
            masked[dead] = 0.0
            masked[..., -self.n_agents :] = agent_id
            inputs = masked
        return inputs.reshape(bs * self.n_agents, -1), bs

    def _get_input_shape(self, scheme):
        input_shape = scheme["state"]["vshape"]
        if getattr(self.args, "critic_use_local_obs", True):
            input_shape += scheme["obs"]["vshape"]
        elif self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"] * self.n_agents
        input_shape += self.n_agents
        return input_shape
