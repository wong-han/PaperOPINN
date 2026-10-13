'''
Supervised reference network for the numerical optimal solution used in the three-axis attitude-control experiment.
'''

import torch
import torch.nn as nn
import torch.optim as optim


class PINNNetwork(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=64, num_layers=3):
        super(PINNNetwork, self).__init__()

        # Backbone network
        layers = []

        # Hidden layers
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim)) 

        self.backbone = nn.Sequential(*layers)

    def forward(self, x):
        # Backbone network
        backbone_output = self.backbone(x)
        zero_output = self.backbone(torch.zeros_like(x))
        lambda_output = backbone_output - zero_output

        return lambda_output
