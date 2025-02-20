import torch.nn as nn


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

        layers_list = [
            nn.Sequential(
                nn.Linear(prev, curr),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            for prev, curr in zip(hidden_dims[:-1], hidden_dims[1:])
        ]

        self.layers = nn.Sequential(*layers_list)
        self.output = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        out = self.input(x)
        out = self.layers(out)
        out = self.output(out)

        return out
