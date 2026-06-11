"""A drop-in SnakeEnv that runs on the trained world model instead of the real
rules — each step advances the game by *predicting* the next frame.

Swap it in wherever SnakeEnv is used (e.g. `python play.py --model`) to see how
convincingly the model has learned Snake's dynamics.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from env import SnakeEnv, BODY, HEAD, FOOD, EMPTY, RIGHT, _OPPOSITE
from model import SnakeWorldModel


class WorldModelEnv:
    def __init__(self, checkpoint="world_model.pt"):
        self._model = SnakeWorldModel()
        path = Path(__file__).parent / checkpoint
        self._model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        self._model.eval()

    def reset(self, seed: int | None = None) -> np.ndarray:
        # The model predicts transitions, not initial states, so borrow the
        # opening frame from the real environment and dream forward from there.
        obs = SnakeEnv().reset(seed)
        self._obs = torch.from_numpy(obs).permute(2, 0, 1)
        self._direction = RIGHT
        return obs

    @torch.no_grad()
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        one_hot_action = F.one_hot(torch.tensor(action), 4).float().unsqueeze(0)
        labels = self._model(self._obs.unsqueeze(0), one_hot_action).argmax(dim=1)[0]

        prev = self._obs.argmax(0)
        prev_len = ((prev == BODY) | (prev == HEAD)).sum()
        new_len = ((labels == BODY) | (labels == HEAD)).sum()

        self._obs = F.one_hot(labels, 4).permute(2, 0, 1).float()
        if action != _OPPOSITE[self._direction]:
            self._direction = action

        reward = 0.0
        if new_len > prev_len:  # the snake grew, so it ate food
            reward = 1.0

        # The model only ever saw live transitions, so a frame without exactly
        # one head means the prediction has collapsed — call that game over.
        done = int((labels == HEAD).sum()) != 1
        return self._obs.permute(1, 2, 0).numpy(), reward, done, {}

    def render_ascii(self) -> str:
        symbols = {EMPTY: ".", BODY: "o", HEAD: "H", FOOD: "*"}
        rows = ["".join(symbols[int(cell)] for cell in row) for row in self._obs.argmax(0)]
        return "\n".join(rows)
