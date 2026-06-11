"""Collect a transition dataset for training a Snake world model.

Saves three tensors to transitions.pt:
  obs:       (N, 4, H, W)  float32  – state before action
  actions:   (N, 4)         float32  – one-hot action
  next_obs:  (N, 4, H, W)  float32  – resulting state

Terminal transitions are skipped so every next_obs is a valid live state.
Episodes cycle through epsilons [1.0, 0.75, 0.4, 0.1, 0.0] for diversity.
Duplicate (obs, action) pairs are hashed and skipped.

Usage:
    python collect.py          # 100k transitions
    python collect.py --n 50000
"""

import argparse
import hashlib
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from env import SnakeEnv, UP, DOWN, LEFT, RIGHT, GRID_SIZE, _OPPOSITE

EPSILONS = [1.0, 0.75, 0.4, 0.1, 0.0]


def greedy_action(snake, food, direction):
    hr, hc = snake[0]
    fr, fc = food
    if abs(fr - hr) >= abs(fc - hc):
        preferred = DOWN if fr > hr else UP
        other = RIGHT if fc > hc else LEFT
    else:
        preferred = RIGHT if fc > hc else LEFT
        other = DOWN if fr > hr else UP
    if preferred != _OPPOSITE[direction]:
        return preferred
    return other


def to_tensor(obs):
    return torch.from_numpy(obs).permute(2, 0, 1).contiguous()


def collect(n):
    obs_buf = torch.zeros(n, 4, GRID_SIZE, GRID_SIZE)
    act_buf = torch.zeros(n, 4)
    next_obs_buf = torch.zeros(n, 4, GRID_SIZE, GRID_SIZE)

    env = SnakeEnv()
    seen = set()
    stored = 0
    episodes = 0
    dupes = 0

    while stored < n:
        epsilon = EPSILONS[episodes % len(EPSILONS)]
        obs_t = to_tensor(env.reset(seed=random.randint(0, 2**31 - 1)))
        done = False

        while not done and stored < n:
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = greedy_action(env._snake, env._food, env._direction)

            # Hash (obs, action) to 16 bytes so the dedup set stays small.
            key = hashlib.md5(obs_t.numpy().tobytes() + action.to_bytes(1, "little")).digest()

            raw_next, _reward, done, _ = env.step(action)
            next_obs_t = to_tensor(raw_next)

            if not done and key not in seen:
                seen.add(key)
                obs_buf[stored] = obs_t
                act_buf[stored] = F.one_hot(torch.tensor(action), num_classes=4).float()
                next_obs_buf[stored] = next_obs_t
                stored += 1
                if stored % 10_000 == 0:
                    print(f"  {stored:>9,} / {n:,}  (episodes: {episodes:,}  dupes: {dupes:,})")
            else:
                dupes += 1

            obs_t = next_obs_t

        episodes += 1

    print(f"\n{stored:,} transitions from {episodes:,} episodes ({dupes:,} dupes skipped)")
    return {"obs": obs_buf, "actions": act_buf, "next_obs": next_obs_buf}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100_000)
    args = p.parse_args()

    out = Path(__file__).parent / "transitions.pt"
    print(f"Collecting {args.n:,} transitions → {out}")

    dataset = collect(args.n)
    torch.save(dataset, out)
    print(f"Saved ({out.stat().st_size / 1e6:.1f} MB)")
