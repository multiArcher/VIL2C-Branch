import torch.nn as nn

from modules.agents.ymappo_base import YMAPPOBase
from utils.th_utils import orthogonal_init_


class YMAPPORNNAgent(nn.Module):
    """Yu et al. actor: feature LN, 2-layer MLP, GRU, LN, then action head."""

    def __init__(self, input_shape, args):
        super(YMAPPORNNAgent, self).__init__()
        self.args = args
        hidden_dim = args.hidden_dim
        gain = getattr(args, "actor_gain", 0.01)
        layer_n = int(getattr(args, "layer_n", 1))
        use_fn = getattr(args, "use_feature_normalization", True)

        self.base = YMAPPOBase(input_shape, hidden_dim, layer_n=layer_n, use_feature_norm=use_fn)
        self.fc_out = nn.Linear(hidden_dim, args.n_actions)
        orthogonal_init_(self.fc_out, gain=gain)

    def init_hidden(self):
        return self.fc_out.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        feat, h = self.base(inputs, hidden_state)
        return self.fc_out(feat), h
