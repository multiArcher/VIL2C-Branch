"""Independent feed-forward actors, as in the DAMA/MADDPG formulation."""
import torch
from torch import nn


def mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                         nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                         nn.Linear(hidden_dim, output_dim))


class DAMAAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.actors = nn.ModuleList([
            mlp(input_shape, args.hidden_dim, args.n_actions) for _ in range(args.n_agents)
        ])

    def forward(self, inputs):
        return torch.stack([actor(inputs[..., i, :])
                            for i, actor in enumerate(self.actors)], dim=-2)
