import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, batch_size, in_dim, out_dim, hidden_dim=64, n_layers=7):
        super(MLP, self).__init__()
        dim = 64
        layers = []
        for i in range(n_layers):
            out_features = out_dim if i == n_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_features=dim, out_features=out_features, bias=False))
            if i != n_layers - 1:
                layers.append(nn.ReLU(inplace=True))

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        output = self.model(input)

        return output
