"""MLP world model for Snake.

Input:  flattened obs (4*H*W=400) + one-hot action (4)  →  404 dims
Output: logits (4, H, W) — one 4-class distribution per cell

Loss: F.cross_entropy(logits, next_obs.argmax(dim=1))
"""

import torch
import torch.nn as nn
from env import GRID_SIZE

OBS_DIM   = 4 * GRID_SIZE * GRID_SIZE  # 400
INPUT_DIM = OBS_DIM + 4                # 404


class SnakeWorldModel(nn.Module):
    def __init__(self, hidden=512, depth=4):
        super().__init__()
        layers = []
        in_dim = INPUT_DIM
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers.append(nn.Linear(hidden, OBS_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, obs, action):
        # obs: (B, 4, H, W)  action: (B, 4)  →  logits: (B, 4, H, W)
        x = torch.cat([obs.flatten(1), action], dim=1)
        return self.net(x).view(obs.shape[0], 4, GRID_SIZE, GRID_SIZE)
