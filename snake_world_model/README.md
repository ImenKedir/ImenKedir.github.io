# Snake World Model — Milestone 1: The Simulator

This is the first milestone of the **Build Your First World Model** project. The
goal of the larger project is to learn a neural network that can *predict* the
next state of an environment (a "world model"). Before any machine learning, we
need a clean, deterministic environment to generate data from.

This milestone is **only the simulator and its tests** — a minimal, CPU-only,
graphics-free 10x10 Snake game. No PyTorch, no rendering libraries, no training
code yet.

## What's here

- `env.py` — `SnakeEnv`, a deterministic Snake environment with a small,
  Gym-like API.
- `test_env.py` — tests covering reset, grid invariants, stepping, wall
  collisions, and seeded determinism.
- `README.md` — this file.

## The environment

State is a `10x10` NumPy array of integers:

| Value | Meaning      |
| ----- | ------------ |
| `0`   | empty        |
| `1`   | snake body   |
| `2`   | snake head   |
| `3`   | food         |

Actions:

| Value | Meaning |
| ----- | ------- |
| `0`   | up      |
| `1`   | down    |
| `2`   | left    |
| `3`   | right   |

Rules:

- The snake starts at length 3, centered and facing right.
- Food is placed on a random empty cell.
- Moving into a wall or into the snake's own body ends the episode.
- Eating food grows the snake and gives reward `+1`.
- A normal move gives reward `0`.
- Death gives reward `-1`.
- Reverse-direction actions (180° turns) are ignored — the snake keeps moving in
  its current direction.
- Given a seed, the environment is fully deterministic.

## API

```python
reset(seed: int | None = None) -> np.ndarray
step(action: int) -> tuple[np.ndarray, float, bool, dict]
render_ascii() -> str
```

## How to run the tests

From inside the `snake_world_model/` directory:

```bash
cd snake_world_model
pytest
```

Or from the project root:

```bash
pytest snake_world_model/test_env.py
```

## Tiny usage example

```python
from env import SnakeEnv

env = SnakeEnv()
grid = env.reset(seed=0)
print(env.render_ascii())

done = False
total_reward = 0.0
while not done:
    action = 3  # always move right (a very short-lived snake)
    grid, reward, done, info = env.step(action)
    total_reward += reward

print("episode finished, total reward:", total_reward)
```

Example `render_ascii()` output (head `H`, body `o`, food `*`, empty `.`):

```
..........
..........
..........
..........
.....*....
...oooH...
..........
..........
..........
..........
```
