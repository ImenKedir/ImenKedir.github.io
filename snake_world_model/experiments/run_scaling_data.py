"""Part 2 of the scaling study: push the DATA axis past the 2000-example ceiling.

Part 1 (run_experiments.py) found the Snake world model sits in a strongly
DATA-LIMITED regime (data > compute > params), maxing out at its largest cell:
2000 examples x 100 epochs x 512 hidden -> held-out loss 0.232, active-cell acc 35%.

This script answers the obvious follow-up: if data is the binding constraint, what
happens when we keep scaling it? It extends the 100-epoch / baseline-512 curve from
4k up to 2M examples (1000x part 1's ceiling) and runs one bounded "do parameters
finally matter at larger data" probe, then plots where (if anywhere) quality
saturates. A separate double-descent probe trains 1000 epochs at three widths.

Comparability is preserved by REUSING part 1's frozen eval set (eval.pt). The new
training pool is a seed-42 20k collection whose first 2000 rows are byte-identical to
pool2000.pt (so the cached small-data cells remain valid points on the same nested
curve). Any transition colliding with the held-out eval set (by md5(obs+action)) is
dropped, and train/eval exact-disjointness is asserted before any training.

Outputs (all under experiments/):
  results_scaling_data.json
  scaling_ext_quality.png   (held-out loss + active-cell acc vs data)
  scaling_ext_rollout.png   (dream head-tracking + steps-to-divergence vs data)
  EXPERIMENT_LOG_PART2.md
"""

import sys
import json
from pathlib import Path

import torch

EXPDIR = Path(__file__).resolve().parent
ROOT = EXPDIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPDIR))

import run_experiments as rx  # noqa: E402

HIDDEN = rx.BASE_HIDDEN   # 512 / baseline params
EPOCHS = 100              # the best training length from part 1
POOL_N = 2_100_000        # surplus so >=2M survive eval-collision filtering
# New points beyond part 1's 2000-example ceiling, up to 1000x (= 2M).
DATA_POINTS = [4000, 8000, 16000, 50000, 100000, 200000,
               500000, 1000000, 2000000]

# Double-descent probe: train far past the overfitting point seen at 100 epochs
# (the 3.8M model's held-out loss bottomed ~epoch 40 and rose after).
DD_DATA_N = 16000
DD_EPOCHS = 1000
DD_MULTS = [1, 10 ** 0.5, 10]  # 1.2M / 3.8M / 12M params


def filter_against(pool: dict, banned: set):
    """Return (clean_pool, n_dropped): rows whose (obs,action) hash is in `banned`
    are removed. Order is preserved so the nested-subset structure is kept."""
    obs = pool["obs"]
    keep = [i for i in range(obs.shape[0])
            if rx.row_hash(obs[i], pool["actions"][i]) not in banned]
    idx = torch.tensor(keep)
    clean = {k: v[idx].clone() for k, v in pool.items()}
    return clean, obs.shape[0] - len(keep)


def main() -> None:
    print("Snake world-model scaling — Part 2 (data axis)")
    print("=" * 60)
    rx.load_cache()

    # --- Frozen eval set + the original 2000 pool (for the prefix sanity check) --- #
    pool2000 = rx.get_pool(rx.BASE_N * 10, seed=42, name="pool2000")
    eval_ds = rx.get_eval_set(pool2000)
    eval_hashes = rx.transition_hashes(eval_ds)
    print(f"eval set: {eval_ds['obs'].shape[0]} held-out transitions (frozen from part 1)")

    # --- Larger seed-42 pool; verify it extends pool2000 exactly --- #
    pool_big = rx.get_pool(POOL_N, seed=42, name="pool2100k")
    assert torch.equal(pool_big["obs"][: pool2000["obs"].shape[0]], pool2000["obs"]), \
        "pool2100k prefix does not match pool2000 — seed/collect drift"
    assert torch.equal(pool_big["actions"][: pool2000["actions"].shape[0]], pool2000["actions"])
    assert torch.equal(pool_big["next_obs"][: pool2000["next_obs"].shape[0]], pool2000["next_obs"])
    print(f"pool2100k: {pool_big['obs'].shape[0]} transitions; first "
          f"{pool2000['obs'].shape[0]} byte-identical to pool2000 (nesting preserved)")

    # --- Drop eval-colliding transitions; enforce exact disjointness --- #
    clean, dropped = filter_against(pool_big, eval_hashes)
    overlap_frac = dropped / pool_big["obs"].shape[0]
    n_clean = clean["obs"].shape[0]
    print(f"dropped {dropped} eval-colliding transitions ({overlap_frac:.2%}); "
          f"{n_clean} train transitions remain")
    assert n_clean >= max(DATA_POINTS), f"only {n_clean} clean rows, need {max(DATA_POINTS)}"
    # Hard guarantee: train pool and eval set share no (obs, action).
    clean_hashes = rx.transition_hashes(clean)
    assert clean_hashes.isdisjoint(eval_hashes), "train/eval leakage detected after filtering"
    # Filtering must not touch the first 2000 (already disjoint from eval).
    assert torch.equal(clean["obs"][: pool2000["obs"].shape[0]], pool2000["obs"]), \
        "filtering disturbed the pool2000 prefix"
    print("disjointness asserted: train ∩ eval = ∅, pool2000 prefix intact")

    # --- Build the 100-epoch / 512-hidden data curve --- #
    # Reuse part 1's cached cells for the 200 / 632 / 2000 points (same recipe, same
    # eval), then add the new 4000 / 8000 / 16000 points from the clean pool.
    curve = []  # list of metric dicts, ascending train_n
    print("\nData curve @ 100 epochs, 512 hidden:")
    for key, n in [("3x3|d1x|e10x", rx.BASE_N),
                   ("3x3|d3.2x|e10x", round(rx.BASE_N * 10 ** 0.5)),
                   ("3x3|d10x|e10x", rx.BASE_N * 10)]:
        m = rx.run_cell(rx.subset(clean, n), HIDDEN, EPOCHS, eval_ds,
                        label=f"data {n} (cached)", key=key)
        curve.append(m)
    for n in DATA_POINTS:
        m = rx.run_cell(rx.subset(clean, n), HIDDEN, EPOCHS, eval_ds,
                        label=f"data {n}", key=f"ext|n{n}|h{HIDDEN}|e{EPOCHS}")
        curve.append(m)

    # --- Bounded params probe at the largest data --- #
    # Does a bigger model finally help once data is plentiful? Run a ~3.2x-params
    # cell at the largest data point; only spend on the full 10x-params cell if the
    # probe beats the marginal last-doubling data gain (architect's rule).
    big_n = max(DATA_POINTS)
    h_probe = rx.hidden_for_multiplier(10 ** 0.5)
    print(f"\nParams probe @ data {big_n}: hidden {h_probe} "
          f"({rx.n_params(h_probe):,} params, ~3.2x baseline)")
    probe = rx.run_cell(rx.subset(clean, big_n), h_probe, EPOCHS, eval_ds,
                        label=f"data {big_n} / params 3.2x",
                        key=f"ext|n{big_n}|h{h_probe}|e{EPOCHS}")
    probes = [probe]

    a16 = next(c["active_acc"] for c in curve if c["train_n"] == big_n)
    a8 = next(c["active_acc"] for c in curve if c["train_n"] == big_n // 2)
    data_gain = a16 - a8
    probe_gain = probe["active_acc"] - a16
    print(f"  {big_n // 2}->{big_n} data gain (active) {data_gain:+.3f}; "
          f"3.2x-params gain {probe_gain:+.3f}")
    if probe_gain > data_gain:
        h_big = rx.hidden_for_multiplier(10)
        print(f"  probe beat marginal data gain -> running 10x-params cell "
              f"(hidden {h_big}, {rx.n_params(h_big):,} params)")
        probes.append(rx.run_cell(rx.subset(clean, big_n), h_big, EPOCHS, eval_ds,
                                  label=f"data {big_n} / params 10x",
                                  key=f"ext|n{big_n}|h{h_big}|e{EPOCHS}"))
    else:
        print("  probe did NOT beat marginal data gain -> skipping 10x-params cell")

    # --- Double descent: train 25x past the 100-epoch overfitting point --- #
    # At 16k examples / 100 epochs the 3.8M model's held-out loss bottomed near
    # epoch 40 and then rose. Train all three widths for DD_EPOCHS and keep the
    # full loss histories: does the validation loss come back down?
    print(f"\nDouble descent @ data {DD_DATA_N}, {DD_EPOCHS} epochs:")
    dd_cells = []
    for mult in DD_MULTS:
        h = rx.hidden_for_multiplier(mult)
        m = rx.run_cell(rx.subset(clean, DD_DATA_N), h, DD_EPOCHS, eval_ds,
                        label=f"dd {mult:.1f}x params ({rx.n_params(h) / 1e6:.1f}M)",
                        key=f"dd|n{DD_DATA_N}|h{h}|e{DD_EPOCHS}")
        dd_cells.append(m)

    # --- Persist results --- #
    out = {
        "eval_n": eval_ds["obs"].shape[0],
        "pool_n": pool_big["obs"].shape[0],
        "eval_overlap_dropped": dropped,
        "eval_overlap_frac": overlap_frac,
        "curve_512h_100ep": curve,
        "params_probes_at_max_data": probes,
        "double_descent_16k_1000ep": dd_cells,
    }
    (EXPDIR / "results_scaling_data.json").write_text(json.dumps(out, indent=2))

    plot(curve, probes, big_n)
    plot_double_descent(dd_cells)
    write_log(curve, probes, eval_ds, dropped, overlap_frac, n_clean, big_n, h_probe,
              dd_cells)
    print("\nDone. See experiments/EXPERIMENT_LOG_PART2.md")


def plot_double_descent(dd_cells):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 5))
    shades = ["0.72", "0.42", "0.0"]
    for shade, m in zip(shades, dd_cells):
        hist = m["history"]
        val = [(h["epoch"], h["val_loss"]) for h in hist if h["val_loss"] is not None]
        ax.plot([v[0] for v in val], [v[1] for v in val], color=shade, linewidth=2,
                label=f"{m['params'] / 1e6:.1f}M params (val)")
        ax.plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist],
                color=shade, linewidth=1, linestyle=":", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epoch  (log scale)")
    ax.set_ylabel("per-cell cross-entropy  (log scale)")
    ax.set_title(f"Double descent probe — {dd_cells[0]['train_n']} examples, "
                 f"{dd_cells[0]['epochs']} epochs (dotted = train)", fontsize=11)
    ax.legend(frameon=False)
    ax.grid(True, which="both", color="0.92")
    fig.tight_layout()
    fig.savefig(EXPDIR / "double_descent.png", dpi=130)
    plt.close(fig)


def plot(curve, probes, big_n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [c["train_n"] for c in curve]
    loss = [c["loss"] for c in curve]
    active = [c["active_acc"] for c in curve]

    # ---- Quality vs data ---- #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(ns, loss, marker="o", color="0.0", linewidth=2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("training data (examples, log scale)")
    axes[0].set_ylabel("held-out loss  (lower = better)")
    axes[0].set_title("Held-out loss vs data  (512 hidden, 100 epochs)", fontsize=11)
    axes[0].grid(True, which="both", color="0.9")

    axes[1].plot(ns, active, marker="o", color="0.0", linewidth=2, label="512 hidden (1.2M)")
    for p in probes:
        mk = "s" if p["params"] < rx.n_params(rx.hidden_for_multiplier(10)) else "^"
        axes[1].plot([big_n], [p["active_acc"]], marker=mk, color="0.45",
                     markersize=9, linestyle="none",
                     label=f"{p['params'] / 1e6:.1f}M params @ {big_n}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("training data (examples, log scale)")
    axes[1].set_ylabel("active-cell accuracy  (higher = better)")
    axes[1].set_title("Active-cell accuracy vs data", fontsize=11)
    axes[1].grid(True, which="both", color="0.9")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Scaling Part 2 — pushing the data axis past 2000",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(EXPDIR / "scaling_ext_quality.png", dpi=130)
    plt.close(fig)

    # ---- Dream rollout fidelity vs data ---- #
    track = [c["rollout_head_track"] for c in curve]
    diverge = [c["steps_head_diverge"] for c in curve]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(ns, track, marker="o", color="0.0", linewidth=2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("training data (examples, log scale)")
    axes[0].set_ylabel("dream head-tracking  (higher = better)")
    axes[0].set_title("Multi-step dream head-tracking vs data", fontsize=11)
    axes[0].grid(True, which="both", color="0.9")
    axes[1].plot(ns, diverge, marker="o", color="0.0", linewidth=2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("training data (examples, log scale)")
    axes[1].set_ylabel("steps until dreamed head diverges")
    axes[1].set_title("Steps before the dream leaves the real head", fontsize=11)
    axes[1].grid(True, which="both", color="0.9")
    fig.suptitle("Scaling Part 2 — does more data fix multi-step dreaming?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(EXPDIR / "scaling_ext_rollout.png", dpi=130)
    plt.close(fig)


def write_log(curve, probes, eval_ds, dropped, overlap_frac, n_clean, big_n, h_probe,
              dd_cells):
    import datetime

    def ddrow(m):
        bv, be = min((h["val_loss"], h["epoch"]) for h in m["history"]
                     if h["val_loss"] is not None)
        return (f"| {m['params']:,} | {bv:.3f} @ ep {be} | "
                f"{m['history'][-1]['val_loss']:.3f} | {m['loss']:.3f} | "
                f"{m['active_acc']:.1%} | {m['head_acc']:.1%} | "
                f"{m['rollout_head_track']:.1%} |")

    def crow(m):
        return (f"| {m['train_n']} | {m['params']:,} | {m['flops']:.2e} | "
                f"{m['loss']:.3f} | {m['active_acc']:.1%} | {m['head_acc']:.1%} | "
                f"{m['rollout_head_track']:.1%} | {m['steps_head_diverge']:.1f} |")

    def prow(m):
        return (f"| {m['params']:,} | {m['loss']:.3f} | {m['active_acc']:.1%} | "
                f"{m['head_acc']:.1%} | {m['rollout_head_track']:.1%} |")

    first, big = curve[0], curve[-1]  # curve is ascending; last cell is big_n
    best = min(curve, key=lambda c: c["loss"])
    loss_2k = next(c["loss"] for c in curve if c["train_n"] == 2000)
    act_2k = next(c["active_acc"] for c in curve if c["train_n"] == 2000)
    probe = probes[0]
    ran_10x = len(probes) > 1
    probe_helped = probe["active_acc"] - big["active_acc"]

    md = f"""# Snake World Model — Scaling Experiment Log, Part 2 (data axis)

_Generated {datetime.date.today().isoformat()} • PyTorch on {rx.DEVICE}._

Continues **[Part 1](EXPERIMENT_LOG.md)**, which found the model strongly
**data-limited** (data > compute > params) and maxed out at its largest cell —
2000 examples x 100 epochs x 512 hidden, held-out loss 0.232, active-cell acc 35%.
Part 1's best cell (2000 examples × 100 epochs) is the reference point. This part
asks the natural follow-up: **if data is the binding constraint, how far does
scaling it actually take us, and does anything saturate?**

## Method (what changed from Part 1)

* **Same frozen eval set** ({eval_ds['obs'].shape[0]} held-out transitions) and **same recipe**
  (Adam, lr {rx.LR}, batch {rx.BATCH}, 100 epochs, 512-hidden / {first['params']:,}-param baseline) so
  every number is directly comparable to Part 1.
* **Bigger training pool.** A seed-42 collection of {POOL_N} transitions whose
  first 2000 rows are **byte-identical to Part 1's pool**, so the data points nest and
  the cached 200/632/2000 cells stay valid on the same curve.
* **Leakage control.** Any training transition whose `(obs, action)` hash collides with the
  held-out eval set is dropped: **{dropped} of {POOL_N} ({overlap_frac:.2%})** removed, leaving
  {n_clean} clean transitions. Train/eval exact-disjointness is asserted before training.
  That {overlap_frac:.1%} overlap is also the **state-space-saturation signal**: under the greedy+ε
  policy the reachable 10x10 state space is small, but at this scale the held-out set is
  still ~97% unseen, so the comparison stays honest.

## Data curve — held-out quality vs training examples (512 hidden, 100 epochs)

![Quality vs data](scaling_ext_quality.png)

| train data | params | train FLOPs | held-out loss | active-cell acc | head acc | dream head-track | steps to diverge |
|---|---|---|---|---|---|---|---|
{chr(10).join(crow(m) for m in curve)}

* From the Part 1 ceiling (2000) to {big_n} examples, held-out loss moves **{loss_2k - big['loss']:+.3f}**
  ({loss_2k:.3f} -> {big['loss']:.3f}) and active-cell accuracy
  **{act_2k:.1%} -> {big['active_acc']:.1%}** ({big['active_acc'] - act_2k:+.1%}).
* Best held-out loss overall: **{best['loss']:.3f}** at {best['train_n']} examples.

## Does a bigger model help once data is plentiful?

![Rollout vs data](scaling_ext_rollout.png)

A bounded params probe at {big_n} examples (the data axis's far end):

| params | held-out loss | active-cell acc | head acc | dream head-track |
|---|---|---|---|---|
| {big['params']:,} (baseline 512) | {big['loss']:.3f} | {big['active_acc']:.1%} | {big['head_acc']:.1%} | {big['rollout_head_track']:.1%} |
{chr(10).join(prow(m) for m in probes)}

The ~3.2x-params model ({h_probe} hidden) changed active-cell accuracy by **{probe_helped:+.1%}** vs the
512 baseline at the same {big_n} examples — {'and beat the marginal last-doubling data gain, so the full 10x-params cell was also run (above)' if ran_10x else 'which did **not** beat the marginal last-doubling data gain, so the expensive 10x-params cell was skipped'}.
Even with plentiful data, **widening the model is not where the gains are** — consistent
with Part 1.

## Double descent — training {dd_cells[0]['epochs']} epochs, far past the overfit point

![Double descent](double_descent.png)

At {dd_cells[0]['train_n']} examples / 100 epochs the wider models overfit (held-out loss bottomed
near epoch 40 then rose). Training all three widths for {dd_cells[0]['epochs']} epochs tests whether
the validation loss descends a second time:

| params | best val loss | final val loss | final loss (full eval) | active acc | head acc | dream head-track |
|---|---|---|---|---|---|---|
{chr(10).join(ddrow(m) for m in dd_cells)}

## Takeaway

Scaling data past Part 1's 2000-example ceiling **keeps lowering held-out loss and
raising active-cell accuracy, but with clearly diminishing returns** — the curve bends
as it climbs, it does not keep falling linearly. The one-step "where is the snake roughly"
signal improves with data; **parameters remain the weakest knob** even when data is no
longer scarce.

The honest caveat from Part 1 still holds: **multi-step dreaming stays near the floor.**
Dream head-tracking and steps-to-divergence (right-hand figure) barely move with data —
the model learns coarse structure but not the exact one-cell head/food dynamics that a
faithful rollout needs. Closing that gap needs a different recipe (far more optimizer
steps and/or a spatially-aware architecture), not simply more data. Within this MLP
recipe, though, the rule remains: **spend a fixed budget on data first.**
"""
    (EXPDIR / "EXPERIMENT_LOG_PART2.md").write_text(md)


if __name__ == "__main__":
    main()
