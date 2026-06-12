"""Fill the data-scaling curve between 16k and 2M at the original recipe.

Runs 50k/100k/200k/500k/1M/2M at batch 512, lr 1e-3, 100 epochs — directly
comparable to the curve_512h_100ep points in results_scaling_data.json.
Appends to results_fill_curve.json after each cell so partial runs are kept.
"""

import sys
import json
import time
from pathlib import Path

EXPDIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPDIR.parent))
sys.path.insert(0, str(EXPDIR))

import run_experiments as rx  # noqa: E402
from run_scaling_data import filter_against, POOL_N  # noqa: E402

POINTS = [50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000]
OUT = EXPDIR / "results_fill_curve.json"


def main() -> None:
    pool2000 = rx.get_pool(rx.BASE_N * 10, seed=42, name="pool2000")
    eval_ds = rx.get_eval_set(pool2000)
    pool = rx.get_pool(POOL_N, seed=42, name="pool2100k")
    clean, dropped = filter_against(pool, rx.transition_hashes(eval_ds))
    print(f"dropped {dropped} eval-colliding transitions")

    results = json.loads(OUT.read_text()) if OUT.exists() else []
    done_ns = {r["train_n"] for r in results}

    for n in POINTS:
        if n in done_ns:
            print(f"skip {n:,} (already done)")
            continue
        data = rx.subset(clean, n)
        t0 = time.time()
        model, history = rx.train(data, rx.BASE_HIDDEN, 100, eval_ds=eval_ds,
                                  label=f"fill curve: {n:,} examples")
        m = rx.evaluate_onestep(model, eval_ds)
        m.update(rx.evaluate_rollout(model))
        m.update(train_n=n, params=rx.n_params(rx.BASE_HIDDEN), epochs=100,
                 batch=rx.BATCH, lr=rx.LR, seconds=round(time.time() - t0, 1))
        results.append(m)
        OUT.write_text(json.dumps(results, indent=2))
        print(f"{n:>9,}: loss={m['loss']:.4f} active={m['active_acc']:.1%} "
              f"track={m['rollout_head_track']:.1%} ({m['seconds']:.0f}s)")

    print("done")


if __name__ == "__main__":
    main()
