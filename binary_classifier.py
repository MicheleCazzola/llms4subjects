import torch.nn as nn
import torch.nn.init as init


class BinaryClassifier(nn.Module):
    def __init__(self, in_features, hidden_dims):
        super(BinaryClassifier, self).__init__()

        self.in_features = in_features
        self.hidden_dims = hidden_dims

        self.input = nn.Sequential(
            nn.Linear(in_features, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        init.kaiming_normal_(self.input[0].weight)

        layers_list = []
        for prev, curr in zip(hidden_dims[:-1], hidden_dims[1:]):
            layer = nn.Sequential(
                nn.Linear(prev, curr),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            init.kaiming_normal_(layer[0].weight)
            layers_list.append(layer)

        self.layers = nn.Sequential(*layers_list)
        self.output = nn.Linear(hidden_dims[-1], 1)
        init.kaiming_normal_(self.output.weight)

    def forward(self, x):
        out = self.input(x)
        out = self.layers(out)
        out = self.output(out)

        return out
