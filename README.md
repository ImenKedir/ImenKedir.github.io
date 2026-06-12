# Snake World Model

Train a tiny MLP to predict the next frame of a deterministic 10×10 Snake game, and run scaling-law experiments on it.

## Setup

```bash
uv sync
```

## Train

```bash
uv run python snake_world_model/collect.py
uv run python snake_world_model/train.py
```

PyTorch picks the best device: CUDA → Apple Metal (`mps`) → CPU.

## Experiments

```bash
uv run python snake_world_model/experiments/run_experiments.py   # 2×2 + 3×3 grids
uv run python snake_world_model/experiments/run_scaling_data.py  # extend data axis
```

Results land in `snake_world_model/experiments/` (JSON, PNGs, `EXPERIMENT_LOG.md`).

See [snake_world_model/README.md](snake_world_model/README.md) for env/model details.
