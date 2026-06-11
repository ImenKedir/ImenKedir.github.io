---
name: Snake World Model demo
description: Architecture and empirical quirks of the in-browser Snake world-model demo (artifacts/snake-world-model)
---

# Snake World Model demo

In-browser "try the trained world model" demo. The Python project `snake_world_model/`
trains a tiny MLP (404->512x4->400) that predicts the next Snake frame given the current
frame + action. The demo runs inference fully client-side — no Python server.

## Architecture / regeneration workflow
- Weights are exported from PyTorch by `snake_world_model/export_weights.py` into
  `artifacts/snake-world-model/public/world_model.bin` (float32 LE) + `world_model.json` (manifest).
- The forward pass is re-implemented in TypeScript under `artifacts/snake-world-model/src/engine/`.
- After ANY retrain/re-export, run `pnpm --filter @workspace/snake-world-model verify` — it checks
  the TS forward pass against PyTorch using 40 exported fixtures. Treat this as the source of truth
  for port correctness; do not trust the TS port after a weight change until verify passes.

## Empirical quirks of the trained model (NOT derivable from code — measured)
**Why this matters:** these shape any honest UI framing of the demo.
- The model does NOT model eating/growth: even when the real snake eats, the dreamed snake stays
  constant length. So a "dream score" stays ~0 forever — never headline it as a score. The honest
  headline is cell-agreement / "fidelity" vs the real frame (~90%+ for the first dozen+ steps).
- The "collapse" signal (dream frame has != 1 head) rarely fires; degraded frames usually keep
  exactly one head while sprouting extra food / body fragments. So rounds typically end on the REAL
  snake crashing (autoplay greedy traps itself in ~10-40 steps), not on a detected dream collapse.

**How to apply:** if asked to improve the demo, lean on Compare mode + fidelity %, and present the
no-growth behavior as an honest finding, not a bug. Don't add a dream-score leaderboard.
