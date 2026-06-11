"""Collect a transition dataset for training a Snake world model.

Produces three parallel tensors saved to transitions.pt:
  obs:       (N, 4, H, W)  float32  – state before action
  actions:   (N, 4)         float32  – one-hot action
  next_obs:  (N, 4, H, W)  float32  – resulting state

Terminal transitions (where the snake dies) are never stored, so every
next_obs is a valid live game state.

Diversity strategy: episodes cycle through four epsilon values so the dataset
contains a mix of near-random, exploratory, and near-optimal play.

Deduplication: each (obs, action) pair is hashed with MD5 before storing;
duplicates are silently dropped.

Usage:
    cd snake_world_model
    python collect.py                  # 100 k transitions → transitions.pt
    python collect.py --n 50000 --out my_data.pt
"""

import argparse
import hashlib
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from env import SnakeEnv, UP, DOWN, LEFT, RIGHT

# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

_OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

# Epsilons cycled across episodes: 1.0=fully random, 0.0=fully greedy.
# This spreads the dataset across early deaths, mid-game, and long snakes.
_EPSILONS = [1.0, 0.75, 0.4, 0.1, 0.0]


def _greedy_action(snake: list, food: tuple, direction: int) -> int:
    """One-step Manhattan-distance move toward food; never reverses."""
    hr, hc = snake[0]
    fr, fc = food
    dr, dc = fr - hr, fc - hc

    # Rank candidates by how much they close the gap
    candidates = sorted(
        [UP, DOWN, LEFT, RIGHT],
        key=lambda a: (
            abs(dr - {UP: -1, DOWN: 1, LEFT: 0, RIGHT: 0}[a])
            + abs(dc - {UP: 0, DOWN: 0, LEFT: -1, RIGHT: 1}[a])
        ),
    )
    for action in candidates:
        if action != _OPPOSITE[direction]:
            return action
    return direction  # fallback (shouldn't happen)


def _choose_action(env: SnakeEnv, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randint(0, 3)
    return _greedy_action(env._snake, env._food, env._direction)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor(obs) -> torch.Tensor:
    """(H, W, 4) numpy float32  →  (4, H, W) torch float32."""
    return torch.from_numpy(obs).permute(2, 0, 1).contiguous()


def _hash(obs_t: torch.Tensor, action: int) -> bytes:
    """Stable 16-byte digest of an (obs, action) pair."""
    return hashlib.md5(obs_t.numpy().tobytes() + action.to_bytes(1, "little")).digest()


# ---------------------------------------------------------------------------
# Collection loop
# ---------------------------------------------------------------------------

def collect(target: int, print_every: int = 10_000) -> dict:
    env = SnakeEnv()
    seen: set[bytes] = set()

    obs_list:      list[torch.Tensor] = []
    act_list:      list[torch.Tensor] = []
    next_obs_list: list[torch.Tensor] = []

    episodes  = 0
    dup_count = 0
    next_milestone = print_every

    while len(obs_list) < target:
        epsilon = _EPSILONS[episodes % len(_EPSILONS)]
        raw_obs = env.reset(seed=random.randint(0, 2**31 - 1))
        obs_t   = _to_tensor(raw_obs)

        done = False
        while not done and len(obs_list) < target:
            action  = _choose_action(env, epsilon)
            key     = _hash(obs_t, action)

            raw_next, _reward, done, _ = env.step(action)
            next_obs_t = _to_tensor(raw_next)

            if not done and key not in seen:
                seen.add(key)
                obs_list.append(obs_t)
                act_list.append(F.one_hot(torch.tensor(action), num_classes=4).float())
                next_obs_list.append(next_obs_t)

                n = len(obs_list)
                if n >= next_milestone:
                    print(f"  {n:>9,} / {target:,}  "
                          f"(episodes: {episodes:,}  dupes skipped: {dup_count:,})")
                    next_milestone += print_every
            else:
                dup_count += 1

            obs_t = next_obs_t

        episodes += 1

    print(f"\nDone. {len(obs_list):,} unique transitions from {episodes:,} episodes "
          f"({dup_count:,} duplicates skipped).")

    return {
        "obs":      torch.stack(obs_list),       # (N, 4, H, W)
        "actions":  torch.stack(act_list),        # (N, 4)
        "next_obs": torch.stack(next_obs_list),   # (N, 4, H, W)
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Snake transition dataset")
    parser.add_argument("--n",   type=int,  default=100_000, help="number of unique transitions")
    parser.add_argument("--out", type=str,  default="transitions.pt", help="output path")
    args = parser.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).parent / out

    print(f"Collecting {args.n:,} unique transitions → {out}")
    dataset = collect(args.n)

    print("\nTensor shapes:")
    for k, v in dataset.items():
        print(f"  {k:10s}  {str(tuple(v.shape)):25s}  {v.dtype}")

    torch.save(dataset, out)
    print(f"\nSaved to {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
