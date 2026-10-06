"""One centralized critic per agent; supports cooperative and general-sum rewards."""
import torch
from torch import nn

from components.dama_history import ActionHistoryInputs
from modules.agents.dama_agent import mlp


class DAMACritic(nn.Module):
    def __init__(self, scheme, args):
        super().__init__()
        input_dim = args.n_agents * (ActionHistoryInputs(scheme, args).input_dim + args.n_actions)
        self.critics = nn.ModuleList([
            mlp(input_dim, args.hidden_dim, 1) for _ in range(args.n_agents)
        ])

    def agent_value(self, inputs, actions, agent):
        joint = torch.cat((inputs.flatten(-2), actions.flatten(-2)), dim=-1)
        return self.critics[agent](joint)

    def forward(self, inputs, actions):
        return torch.stack([self.agent_value(inputs, actions, i)
                            for i in range(len(self.critics))], dim=-2)
