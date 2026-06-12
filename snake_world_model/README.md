# Snake World Model

A tiny project that **learns to predict the future**: a deterministic Snake simulator,
a transition dataset, an MLP world model, and scaling experiments.

Training uses PyTorch with the best available backend: CUDA, then Apple Metal
(`mps`), then CPU.

## Setup

From the repo root:

```bash
uv sync
uv run python snake_world_model/collect.py
uv run python snake_world_model/train.py
```

On macOS, `torch` comes from PyPI with Metal/MPS support. On Linux, uv installs
CPU-only wheels from the PyTorch index.

## What's here

- `env.py` — `SnakeEnv`, a deterministic 10x10 Snake simulator.
- `collect.py` — roll out episodes and save a transition dataset (`transitions.pt`).
- `model.py` — `SnakeWorldModel`, an MLP that predicts the next state.
- `train.py` — train the model on the dataset (saves `world_model.pt`).
- `device.py` — pick CUDA / MPS / CPU.
- `experiments/` — scaling-law grids, extended data-axis runs, plots, logs.

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

## Pipeline

```bash
cd snake_world_model
uv run python collect.py
uv run python train.py
uv run python experiments/run_experiments.py
```

`collect.py` and `train.py` take optional flags (`--n`, `--epochs`, `--batch`,
`--lr`); run either with `-h` to see them.
