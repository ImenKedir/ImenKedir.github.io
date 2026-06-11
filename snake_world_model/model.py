"""MLP world model for Snake.

Input:  flattened obs (4*H*W) + one-hot action (4)  →  404 dims
Output: logits of shape (4, H, W) — one 4-class distribution per cell

Loss: cross-entropy over the 4 channels at every spatial position,
      i.e. F.cross_entropy(logits, next_obs.argmax(dim=1)).
"""

import torch
import torch.nn as nn

from env import GRID_SIZE

OBS_DIM    = 4 * GRID_SIZE * GRID_SIZE   # 400
ACTION_DIM = 4
INPUT_DIM  = OBS_DIM + ACTION_DIM        # 404
OUTPUT_DIM = OBS_DIM                     # 400  →  reshaped to (4, H, W)


class SnakeWorldModel(nn.Module):
    def __init__(self, hidden: int = 512, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = INPUT_DIM
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers.append(nn.Linear(hidden, OUTPUT_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        obs:    (B, 4, H, W)  float32
        action: (B, 4)        float32  one-hot
        returns (B, 4, H, W)  logits
        """
        B = obs.shape[0]
        x = torch.cat([obs.flatten(1), action], dim=1)   # (B, 404)
        return self.net(x).view(B, 4, GRID_SIZE, GRID_SIZE)
