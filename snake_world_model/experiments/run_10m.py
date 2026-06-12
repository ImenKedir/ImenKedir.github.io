"""10M-transition x 1000-epoch run: weight saturation + double descent probe.

5x the data of run_2m.py and 10x its epochs, same 1.2M-param model and fast
recipe (batch 2048, lr 2e-3). Two questions: does the 1.2M model finally
saturate with this much data, and does training far past 100 epochs show a
second descent? Watch live_loss.png while it runs.

Collecting the seed-42 10.1M pool happens on first run and is cached.
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
from run_scaling_data import filter_against  # noqa: E402

DATA_N = 10_000_000
POOL_N = 10_100_000  # surplus so >=10M survive eval-collision filtering
EPOCHS = 1000
BATCH = 2048
LR = 2e-3


def main() -> None:
    hidden = rx.BASE_HIDDEN
    print(f"10M x 1000-epoch run  device={rx.DEVICE}")
    print(f"  data={DATA_N:,}  hidden={hidden} ({rx.n_params(hidden):,} params)  "
          f"epochs={EPOCHS}  batch={BATCH}  lr={LR}")

    pool2000 = rx.get_pool(rx.BASE_N * 10, seed=42, name="pool2000")
    eval_ds = rx.get_eval_set(pool2000)
    pool = rx.get_pool(POOL_N, seed=42, name="pool10m")
    clean, dropped = filter_against(pool, rx.transition_hashes(eval_ds))
    del pool  # ~8GB; free before the training copy
    print(f"  dropped {dropped} eval-colliding transitions; "
          f"{clean['obs'].shape[0]:,} clean")
    data = rx.subset(clean, DATA_N)
    del clean

    t0 = time.time()
    model, history = rx.train(data, hidden, EPOCHS, eval_ds=eval_ds,
                              label=f"10M / 1.2M params / {EPOCHS} epochs",
                              batch=BATCH, lr=LR)
    train_s = time.time() - t0

    m = rx.evaluate_onestep(model, eval_ds)
    m.update(rx.evaluate_rollout(model))
    m.update(params=rx.n_params(hidden), hidden=hidden, train_n=DATA_N,
             epochs=EPOCHS, batch=BATCH, lr=LR, seconds=round(train_s, 1),
             history=history)
    (EXPDIR / "results_10m.json").write_text(json.dumps(m, indent=2))
    torch.save(model.state_dict(), EXPDIR / "world_model_10m.pt")

    print(f"\ntrained in {train_s / 3600:.1f} h")
    print(f"  held-out loss     = {m['loss']:.4f}")
    print(f"  active-cell acc   = {m['active_acc']:.1%}")
    print(f"  head acc          = {m['head_acc']:.1%}")
    print(f"  food acc          = {m['food_acc']:.1%}")
    print(f"  exact-frame match = {m['exact_match']:.1%}")
    print(f"  dream head-track  = {m['rollout_head_track']:.1%}")
    print(f"  steps to diverge  = {m['steps_head_diverge']:.1f}")
    print("\nwrote results_10m.json + world_model_10m.pt")


if __name__ == "__main__":
    main()
