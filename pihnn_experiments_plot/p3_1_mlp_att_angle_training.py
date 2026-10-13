'''
Supervised reference network for the numerical optimal solution used in the three-axis attitude-control experiment.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class MLP(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=32, num_layers=3):
        super(MLP, self).__init__()

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




if __name__ == "__main__":

    data = np.load("data/DATA.npz")
    dataset = data["dataset"]  # shape: (N, 10)


    X = dataset[:, :3]   # [phi, theta, psi]
    y = dataset[:, 3:6]  # [lam1, lam2, lam3]
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)


    model = MLP(input_dim=3, output_dim=3, hidden_dim=32, num_layers=3)
    model.train()


    lr = 1e-3
    batch_size = 256
    iteration_times = 10000
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    N = X.shape[0]

    save_path = "model/supervised_angle3d_model.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


    idx = torch.randperm(N)
    X = X[idx]
    y = y[idx]

    loss_tracker = []

    for it in range(iteration_times):

        start_idx = (it * batch_size) % N
        end_idx = start_idx + batch_size
        if end_idx <= N:
            xb = X[start_idx:end_idx]
            yb = y[start_idx:end_idx]
        else:

            tail = N - start_idx
            xb = torch.cat([X[start_idx:], X[:end_idx-N]], dim=0)
            yb = torch.cat([y[start_idx:], y[:end_idx-N]], dim=0)

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        loss_tracker.append(loss.item())


        if (it+1) % 500 == 0 or it == 0:
            print(f"Iter {it+1}/{iteration_times}: Loss={loss.item():.6f}")

    # Save the model
    torch.save(model.state_dict(), save_path)
    print("Training completed. Final training loss: {:.6f}".format(loss_tracker[-1]))
    print("Model saved to", save_path)
