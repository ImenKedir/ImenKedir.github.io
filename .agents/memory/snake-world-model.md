---
name: Snake world-model scaling experiment
description: Non-obvious metric/sandboxing/optimization lessons for the tiny Snake next-frame predictor and its scaling study under snake_world_model/experiments/
---

# Snake world-model — durable lessons

## Metrics: naive accuracy is a trap here
A 10x10 Snake frame is ~95% empty cells. Per-cell accuracy and whole-frame
exact-match are therefore dominated by the trivial "predict empty everywhere"
solution (≈94% cell acc, exact≈0 for a model that learned nothing). They sit at
that floor across an entire data/params/compute sweep and do NOT separate runs.
**Use held-out cross-entropy loss (primary, graded everywhere) + active-cell
accuracy (accuracy restricted to non-empty target cells).**
**Why:** loss is the only metric that moves continuously across the sweep;
active-cell acc is a phase-transition metric — it stays 0 until both data AND
training compute are raised, then jumps (in the study it only lit up at 2000
examples × 100 epochs ≈ 35%).
**How to apply:** when judging "did scaling help?", read loss; treat any
near-floor exact-match/head-accuracy as expected, not as a bug. Base any
"which knob won" headline on loss drops, not on active-acc deltas (those are
mostly 0 and tie degenerately).

## Under-optimization gotcha (recipe-faithful, not a bug)
The deployed recipe uses batch 512, which is ≥ the dataset size for the small
cells, so training runs only a few hundred optimizer steps. Result: the model
learns coarse "where the body roughly is" structure but never precise one-cell
head/food localization, regardless of scale. Multi-step "dreaming" diverges
almost immediately. Scaling data/compute 10× of any single knob does not fix
this — closing the gap needs far more optimizer steps and/or a spatially-aware
architecture. Don't inflate the compute budget to make numbers look good; the
honest finding is that the regime is data-limited and under-optimized.

## Sandboxing rule for the experiment runner
`snake_world_model/experiments/run_experiments.py` must stay non-destructive:
it defines its own model, imports only `env.py`/`collect.py`, and writes ONLY
under `experiments/` (results JSONs, PNGs, EXPERIMENT_LOG.md, data/, _cache.json).
Never let it modify `model.py`, the deployed baseline `transitions.pt`, or
`public/world_model.*` — those drive the live web app.

## Cache caveat
Per-cell results are cached in `experiments/_cache.json` keyed by cell label
only — the key does NOT encode hyperparameters (epochs, widths, eval size).
**If you change the grid config, delete `_cache.json` first**, or it will
silently serve stale results from the previous configuration.

## FLOPs framing
Training FLOPs ≈ 6 · params · examples · epochs. At fixed params/data, 10×
epochs = 10× FLOPs, which is why the 3×3's epochs axis is framed as a
training-compute axis. Grid 2 is iso-FLOP-fair across both axes (10× data at
fixed epochs is also 10× FLOPs).
