"""Local executed-action history for the observation-delay DAMA adaptation.

History is ordered oldest to newest and excludes the action being decided.
Timestamp metadata is used only for alignment; no future observation is read.
"""
import math

import torch


class ActionHistoryInputs:
    def __init__(self, scheme, args):
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.obs_dim = math.prod(scheme["obs"]["vshape"] if isinstance(
            scheme["obs"]["vshape"], (tuple, list)) else (scheme["obs"]["vshape"],))
        self.history_length = int(args.dama_history_length)
        if self.history_length < 0:
            raise ValueError("dama_history_length must be nonnegative")
        self.use_history = bool(args.dama_use_history)
        self.use_alignment = bool(args.dama_use_alignment) and self.use_history
        self.agent_id = bool(args.obs_agent_id)
        self.input_dim = self.obs_dim + (self.n_agents if self.agent_id else 0)
        if self.use_history:
            self.input_dim += self.history_length * (self.n_actions + 1)
        if self.use_alignment:
            self.input_dim += self.history_length + 1  # after-generation flags + arrived

    def build(self, batch, t):
        obs = batch["obs"][:, t].reshape(batch.batch_size, self.n_agents, -1)
        parts = [obs]
        if self.use_history:
            history = obs.new_zeros(batch.batch_size, self.n_agents,
                                    self.history_length, self.n_actions)
            valid = obs.new_zeros(batch.batch_size, self.n_agents, self.history_length)
            start = max(0, t - self.history_length)
            count = t - start
            if count:
                history[:, :, -count:] = batch["actions_onehot"][:, start:t].transpose(1, 2)
                valid[:, :, -count:] = batch["filled"][:, start:t, 0].unsqueeze(1)
            parts.extend((history.flatten(-2), valid))
            if self.use_alignment:
                gen = batch["obs_gen_t"][:, t, :, 0]
                arrived = gen >= 0
                times = torch.arange(t - self.history_length, t, device=obs.device)
                after = ((times >= gen.unsqueeze(-1)) & arrived.unsqueeze(-1)).to(obs.dtype)
                parts.extend((after * valid, arrived.unsqueeze(-1).to(obs.dtype)))
        if self.agent_id:
            parts.append(torch.eye(self.n_agents, device=obs.device, dtype=obs.dtype)
                         .unsqueeze(0).expand(batch.batch_size, -1, -1))
        return torch.cat(parts, dim=-1)

    def sequence(self, batch):
        return torch.stack([self.build(batch, t) for t in range(batch.max_seq_length)], dim=1)
