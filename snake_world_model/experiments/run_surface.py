"""Train the data x epochs grid behind the blog's 3D surfaces.

7 data sizes x 3 epoch budgets at the baseline recipe (512h, batch 512,
lr 1e-3). Appends to results_surface.json after each cell.
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

DATA_NS = [200, 632, 2_000, 6_325, 20_000, 63_246, 200_000]
EPOCHS = [10, 32, 100]
OUT = EXPDIR / "results_surface.json"


def main() -> None:
    pool2000 = rx.get_pool(rx.BASE_N * 10, seed=42, name="pool2000")
    eval_ds = rx.get_eval_set(pool2000)
    pool = rx.get_pool(POOL_N, seed=42, name="pool2100k")
    clean, _ = filter_against(pool, rx.transition_hashes(eval_ds))

    results = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {(r["train_n"], r["epochs"]) for r in results}

    for n in DATA_NS:
        data = rx.subset(clean, n)
        for ep in EPOCHS:
            if (n, ep) in done:
                continue
            t0 = time.time()
            model, _ = rx.train(data, rx.BASE_HIDDEN, ep)
            m = rx.evaluate_onestep(model, eval_ds)
            m.update(rx.evaluate_rollout(model))
            m.update(train_n=n, epochs=ep, seconds=round(time.time() - t0, 1))
            results.append(m)
            OUT.write_text(json.dumps(results, indent=2))
            print(f"n={n:>7,} ep={ep:>3}: loss={m['loss']:.4f} "
                  f"active={m['active_acc']:.1%} ({m['seconds']:.0f}s)")

    print("done")


if __name__ == "__main__":
    main()
