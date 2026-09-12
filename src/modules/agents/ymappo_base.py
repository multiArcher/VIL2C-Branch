import torch.nn as nn

from utils.th_utils import orthogonal_init_


class YMAPPOBase(nn.Module):
    """RMAPPO MLP + GRUCell: input LN, Linear-ReLU-LN x (layer_n+1), GRU, LN."""

    def __init__(self, input_shape, hidden_dim, layer_n=1, use_feature_norm=True):
        super(YMAPPOBase, self).__init__()
        self.hidden_dim = hidden_dim
        self.feature_norm = nn.LayerNorm(input_shape) if use_feature_norm else nn.Identity()

        n_mlp = int(layer_n) + 1
        relu_gain = nn.init.calculate_gain("relu")
        self.mlp = nn.ModuleList()
        in_dim = input_shape
        for _ in range(n_mlp):
            lin = nn.Linear(in_dim, hidden_dim)
            orthogonal_init_(lin, gain=relu_gain)
            self.mlp.append(nn.Sequential(lin, nn.ReLU(), nn.LayerNorm(hidden_dim)))
            in_dim = hidden_dim

        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.rnn_norm = nn.LayerNorm(hidden_dim)
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(self, x, hidden_state):
        x = self.feature_norm(x)
        for layer in self.mlp:
            x = layer(x)
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        return self.rnn_norm(h), h
