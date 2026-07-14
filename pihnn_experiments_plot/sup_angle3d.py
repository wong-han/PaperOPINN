'''
理论最优解监督学习网络：模型来自2025-08-18王翰编程/6-三轴角度控制
'''

import torch
import torch.nn as nn
import torch.optim as optim


class PINNNetwork(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=64, num_layers=3):
        super(PINNNetwork, self).__init__()

        # 主干网络
        layers = []

        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        # 隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim)) 

        self.backbone = nn.Sequential(*layers)

    def forward(self, x):
        # 主干网络输出
        backbone_output = self.backbone(x)
        zero_output = self.backbone(torch.zeros_like(x))
        lambda_output = backbone_output - zero_output

        return lambda_output