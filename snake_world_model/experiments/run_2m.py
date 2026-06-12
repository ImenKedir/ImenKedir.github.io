"""2M-example training run at a chosen parameter multiplier.

Usage: python run_2m.py [multiplier]   (default 1)

Single cell at the data axis's far end with the benchmarked fast recipe:
batch 2048 (throughput winner on MPS) and sqrt-scaled lr 2e-3 (vs the curve's
batch 512 / lr 1e-3). NOT comparable to the batch-512 curve cells; results go
to results_2m_{mult}x.json.

Reuses the frozen eval set and the cached seed-42 pool2100k collection.
"""

import sys
import json
import time
from pathlib import Path

import torch

EXPDIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPDIR.parent))
sys.path.insert(0, str(EXPDIR))

import run_experiments as rx  # noqa: E402
from run_scaling_data import filter_against, POOL_N  # noqa: E402

DATA_N = 2_000_000
EPOCHS = 100
BATCH = 2048
LR = 2e-3  # 1e-3 sqrt-scaled for the 4x larger batch


def main() -> None:
    mult = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    tag = f"{mult:g}x"
    hidden = rx.hidden_for_multiplier(mult)
    print(f"2M x {tag}-params run  device={rx.DEVICE}")
    print(f"  data={DATA_N:,}  hidden={hidden} ({rx.n_params(hidden):,} params)  "
          f"epochs={EPOCHS}  batch={BATCH}  lr={LR}")

    pool2000 = rx.get_pool(rx.BASE_N * 10, seed=42, name="pool2000")
    eval_ds = rx.get_eval_set(pool2000)
    pool = rx.get_pool(POOL_N, seed=42, name="pool2100k")
    clean, dropped = filter_against(pool, rx.transition_hashes(eval_ds))
    print(f"  dropped {dropped} eval-colliding transitions; "
          f"{clean['obs'].shape[0]:,} clean")
    data = rx.subset(clean, DATA_N)

    t0 = time.time()
    model, history = rx.train(data, hidden, EPOCHS, eval_ds=eval_ds,
                              label=f"2M / {tag} params / batch {BATCH}",
                              batch=BATCH, lr=LR)
    train_s = time.time() - t0

    m = rx.evaluate_onestep(model, eval_ds)
    m.update(rx.evaluate_rollout(model))
    m.update(params=rx.n_params(hidden), hidden=hidden, train_n=DATA_N,
             epochs=EPOCHS, batch=BATCH, lr=LR, seconds=round(train_s, 1),
             history=history)
    (EXPDIR / f"results_2m_{tag}.json").write_text(json.dumps(m, indent=2))
    torch.save(model.state_dict(), EXPDIR / f"world_model_2m_{tag}.pt")

    print(f"\ntrained in {train_s / 60:.1f} min")
    print(f"  held-out loss     = {m['loss']:.4f}")
    print(f"  active-cell acc   = {m['active_acc']:.1%}")
    print(f"  head acc          = {m['head_acc']:.1%}")
    print(f"  food acc          = {m['food_acc']:.1%}")
    print(f"  exact-frame match = {m['exact_match']:.1%}")
    print(f"  dream head-track  = {m['rollout_head_track']:.1%}")
    print(f"  steps to diverge  = {m['steps_head_diverge']:.1f}")
    print(f"\nwrote results_2m_{tag}.json + world_model_2m_{tag}.pt")


if __name__ == "__main__":
    main()
