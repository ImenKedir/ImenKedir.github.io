# Snake World Model

A tiny, CPU-only project that **learns to predict the future**: first we build a
deterministic Snake simulator, then we train a neural network (a "world model")
to predict the next game state from the current state and an action.

## What's here

- `env.py` — `SnakeEnv`, a deterministic 10x10 Snake simulator.
- `collect.py` — roll out episodes and save a transition dataset (`transitions.pt`).
- `model.py` — `SnakeWorldModel`, an MLP that predicts the next state.
- `train.py` — train the model on the dataset (saves `world_model.pt`).
- `play.py` — play Snake yourself in the terminal.
- `world_model_env.py` — `WorldModelEnv`, a drop-in `SnakeEnv` that runs on the
  trained model so you can play *inside* its predictions.

## The environment

The observation is a `(10, 10, 4)` one-hot `float32` grid — each cell has four
channels, exactly one of which is hot:

| Channel | Meaning    |
| ------- | ---------- |
| `0`     | empty      |
| `1`     | snake body |
| `2`     | snake head |
| `3`     | food       |

Actions: `0` up, `1` down, `2` left, `3` right.

Rules:

- The snake starts at length 3, centered and facing right.
- Food is placed on a random empty cell.
- Moving into a wall or the snake's own body ends the episode (reward `-1`).
- Eating food grows the snake (reward `+1`); a normal move gives `0`.
- Reverse-direction actions are ignored — the snake keeps its current direction.
- Given a seed, the environment is fully deterministic.

API:

```python
reset(seed: int | None = None) -> np.ndarray            # (10, 10, 4) one-hot
step(action: int) -> tuple[np.ndarray, float, bool, dict]
render_ascii() -> str
```

## The pipeline

```bash
python collect.py        # generate transitions.pt
python train.py          # train the world model -> world_model.pt
python play.py           # play Snake in the terminal (arrow keys, q to quit)
python play.py --model   # play inside the model's predictions (the "dream")
```

`collect.py` and `train.py` take optional flags (`--n`, `--epochs`, `--batch`,
`--lr`); run either with `-h` to see them.

## Usage example

```python
from env import SnakeEnv

env = SnakeEnv()
obs = env.reset(seed=0)               # (10, 10, 4) one-hot float32
print(env.render_ascii())

obs, reward, done, _ = env.step(3)    # move right
```

`render_ascii()` shows the grid with `H` head, `o` body, `*` food, `.` empty:

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
