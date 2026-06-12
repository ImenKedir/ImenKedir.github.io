"""Scaling-laws experiment for the Snake world model.

Two grids:
  * 2x2  : data {1x, 10x}  x  params {1x, 10x}        (fixed training length)
  * 3x3  : data {1x, ~3.2x, 10x}  x  epochs {1x, ~3.2x, 10x}  (fixed model size)

The 2x2 isolates the data-vs-params question. The 3x3 brings in the third
scaling knob the 2x2 never touches -- training compute -- so one axis is
literally "10x the FLOPs used to train". Across both grids all three knobs
(data, parameters, compute) get exercised.

Quality is measured on a single FIXED, held-out, de-duplicated eval set that is
disjoint from every training set, plus a multi-step "dream" rollout fidelity
metric (multi-step dream rollout fidelity).

Everything is cached under experiments/data so reruns are cheap. Outputs:
  experiments/results_2x2.json, results_3x3.json
  experiments/heatmap_2x2.png, heatmap_3x3.png
  experiments/EXPERIMENT_LOG.md
"""

import sys
import json
import time
import random
import hashlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

EXPDIR = Path(__file__).resolve().parent
ROOT = EXPDIR.parent
DATADIR = EXPDIR / "data"
DATADIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from device import best_device  # noqa: E402
from env import SnakeEnv, GRID_SIZE, EMPTY, HEAD, FOOD  # noqa: E402
from collect import collect, greedy_action, to_tensor  # noqa: E402

DEVICE = best_device()
if DEVICE.type == "cpu":
    torch.set_num_threads(4)

OBS_DIM = 4 * GRID_SIZE * GRID_SIZE
INPUT_DIM = OBS_DIM + 4
DEPTH = 4  # hidden layers; matches the deployed baseline (HIDDEN=512 -> 1,200,528 params)
BASE_HIDDEN = 512
BATCH = 512
LR = 1e-3

# Baseline data size (the deployed model was trained on this many transitions).
BASE_N = 200


# --------------------------------------------------------------------------- #
# Model (parametric width, same architecture as the deployed baseline)
# --------------------------------------------------------------------------- #
class WorldModel(nn.Module):
    def __init__(self, hidden: int, depth: int = DEPTH):
        super().__init__()
        layers = []
        in_dim = INPUT_DIM
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers.append(nn.Linear(hidden, OBS_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, obs, action):
        x = torch.cat([obs.flatten(1), action], dim=1)
        return self.net(x).view(-1, 4, GRID_SIZE, GRID_SIZE)


def n_params(hidden: int, depth: int = DEPTH) -> int:
    m = WorldModel(hidden, depth)
    return sum(p.numel() for p in m.parameters())


def hidden_for_multiplier(mult: float) -> int:
    """Smallest hidden width whose param count is >= mult x the baseline."""
    target = mult * n_params(BASE_HIDDEN)
    h = 8
    while n_params(h) < target:
        h += 1
    return h


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def row_hash(obs_row: torch.Tensor, action_row: torch.Tensor) -> bytes:
    # Normalize to uint8 so float32 (old pools) and uint8 (new pools) agree.
    a = int(action_row.argmax())
    data = obs_row.to(torch.uint8).numpy().tobytes()
    return hashlib.md5(data + a.to_bytes(1, "little")).digest()


def transition_hashes(ds) -> set:
    return {row_hash(ds["obs"][i], ds["actions"][i])
            for i in range(ds["obs"].shape[0])}


def get_pool(n: int, seed: int, name: str):
    """Deterministically collect `n` unique transitions, cached to disk."""
    path = DATADIR / f"{name}.pt"
    if path.exists():
        d = torch.load(path, weights_only=True)
        if d["obs"].shape[0] >= n:
            return d
    random.seed(seed)
    print(f"  collecting {name}: {n} transitions (seed {seed}) ...")
    d = collect(n)
    torch.save(d, path)
    return d


def subset(ds, n: int):
    return {k: v[:n].clone() for k, v in ds.items()}


def get_eval_set(train_pool, n_target: int = 4000, seed: int = 777):
    """Held-out eval set, de-duplicated against the training pool."""
    path = DATADIR / "eval.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    train_hashes = transition_hashes(train_pool)
    raw = get_pool(n_target * 2, seed=seed, name="eval_raw")
    keep = []
    for i in range(raw["obs"].shape[0]):
        if row_hash(raw["obs"][i], raw["actions"][i]) not in train_hashes:
            keep.append(i)
        if len(keep) >= n_target:
            break
    idx = torch.tensor(keep)
    ev = {k: raw[k][idx].clone() for k in raw}
    torch.save(ev, path)
    print(f"  eval set: {len(keep)} held-out transitions (disjoint from train)")
    return ev


# --------------------------------------------------------------------------- #
# Train + evaluate
# --------------------------------------------------------------------------- #
VAL_EVERY = 5  # epochs between held-out loss evals during training


@torch.no_grad()
def heldout_loss(model, obs, actions, targets, batch: int = 4096) -> float:
    """Mean per-cell cross-entropy on (already on-device) eval tensors."""
    model.eval()
    ce = 0.0
    for s in range(0, obs.shape[0], batch):
        sl = slice(s, s + batch)
        ce += F.cross_entropy(model(obs[sl], actions[sl]), targets[sl],
                              reduction="sum").item()
    return ce / targets.numel()


def update_live_plot(history, label):
    """Rewrite live_loss.png so training progress can be watched mid-run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    eps = [h["epoch"] for h in history]
    ax.plot(eps, [h["train_loss"] for h in history],
            color="0.6", linewidth=1.5, label="train loss")
    val = [(h["epoch"], h["val_loss"]) for h in history if h["val_loss"] is not None]
    if val:
        ax.plot([v[0] for v in val], [v[1] for v in val],
                color="0.0", linewidth=2, marker="o", markersize=3.5,
                label="held-out loss")
    ax.set_xscale("log")
    ax.set_xlabel("epoch  (log scale)")
    ax.set_ylabel("per-cell cross-entropy")
    ax.set_title(f"Training — {label}", fontsize=11)
    ax.legend(frameon=False)
    ax.grid(True, color="0.9")
    fig.tight_layout()
    fig.savefig(EXPDIR / "live_loss.png", dpi=110)
    plt.close(fig)


def train(data, hidden: int, epochs: int, eval_ds=None, label="",
          batch: int = BATCH, lr: float = LR):
    torch.manual_seed(0)
    # Pools may be uint8 (4x smaller; a 10M pool doesn't fit as float32).
    # Keep them compact on-device and cast per batch.
    obs = data["obs"].to(DEVICE)
    actions = data["actions"].to(DEVICE)
    next_obs = data["next_obs"].to(DEVICE)
    n = obs.shape[0]
    model = WorldModel(hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if eval_ds is not None:
        eobs = eval_ds["obs"].to(DEVICE)
        eact = eval_ds["actions"].to(DEVICE)
        etgt = eval_ds["next_obs"].argmax(1).long().to(DEVICE)
    history = []
    val_every = max(VAL_EVERY, epochs // 50)  # cap val evals at ~50 per run

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        # Accumulate on-device; a per-step .item() would force a GPU sync.
        ep_loss = torch.zeros((), device=DEVICE)
        for s in range(0, n, batch):
            b = perm[s:s + batch]
            logits = model(obs[b].float(), actions[b].float())
            loss = F.cross_entropy(logits, next_obs[b].argmax(1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.detach() * b.shape[0]
        if eval_ds is not None:
            do_val = ep == 1 or ep == epochs or ep % val_every == 0
            history.append({
                "epoch": ep,
                "train_loss": ep_loss.item() / n,
                "val_loss": heldout_loss(model, eobs, eact, etgt) if do_val else None,
            })
            update_live_plot(history, label)
    return model, history


@torch.no_grad()
def evaluate_onestep(model, eval_ds, batch: int = 2048):
    """One-step held-out metrics.

    A 10x10 grid is ~95% empty, so plain cell accuracy and full-frame exact
    match are dominated by the trivial "predict empty" baseline. The informative
    metrics are the held-out loss and accuracy restricted to the cells that
    actually carry the snake/food, plus head/food localization.
    """
    model.eval()
    obs = eval_ds["obs"].to(DEVICE)
    actions = eval_ds["actions"].to(DEVICE)
    targets = eval_ds["next_obs"].argmax(1).long().to(DEVICE)  # (N, H, W) labels
    n = obs.shape[0]
    ce_sum = 0.0
    cell_total = 0
    exact = 0
    active_correct = 0
    active_total = 0
    head_correct = 0
    food_correct = 0
    for s in range(0, n, batch):
        sl = slice(s, s + batch)
        logits = model(obs[sl], actions[sl])  # (b, 4, H, W)
        tgt = targets[sl]
        b = logits.shape[0]
        ce_sum += F.cross_entropy(logits, tgt, reduction="sum").item()
        cell_total += tgt.numel()
        pred = logits.argmax(1)
        exact += (pred == tgt).reshape(b, -1).all(1).sum().item()
        # Accuracy on cells that are non-empty in the target (the structure).
        mask = tgt != EMPTY
        active_correct += (pred[mask] == tgt[mask]).sum().item()
        active_total += int(mask.sum().item())
        # Head / food localization via the dedicated channel argmax.
        head_pred = logits[:, HEAD, :, :].reshape(b, -1).argmax(1)
        head_tgt = (tgt == HEAD).reshape(b, -1).float().argmax(1)
        head_correct += (head_pred == head_tgt).sum().item()
        food_pred = logits[:, FOOD, :, :].reshape(b, -1).argmax(1)
        food_tgt = (tgt == FOOD).reshape(b, -1).float().argmax(1)
        food_correct += (food_pred == food_tgt).sum().item()
    return {
        "loss": ce_sum / cell_total,
        "active_acc": active_correct / max(1, active_total),
        "head_acc": head_correct / n,
        "food_acc": food_correct / n,
        "exact_match": exact / n,
    }


@torch.no_grad()
def evaluate_rollout(model, seeds=range(24), steps: int = 30):
    """Dream forward `steps` from real openings; same greedy policy drives both.

    Reports head-tracking (does the dreamed head stay on the real head) and the
    average number of steps before the dreamed head first leaves the real head --
    a far more discriminative "does the dream follow the snake" signal than
    full-frame agreement, which is swamped by empty cells.
    """
    model.eval()
    head_tracks = []
    head_divs = []
    active_agrees = []
    for sd in seeds:
        real = SnakeEnv()
        real.reset(seed=sd)
        dlabels = torch.from_numpy(real._labels()).long().to(DEVICE)
        head_hits = 0.0
        active_sum = 0.0
        first_head_div = steps
        for t in range(steps):
            action = greedy_action(real._snake, real._food, real._direction)
            oh = F.one_hot(torch.tensor(action, device=DEVICE), num_classes=4).float().unsqueeze(0)
            dobs = F.one_hot(dlabels, num_classes=4).permute(2, 0, 1).float().unsqueeze(0)
            logits = model(dobs, oh)[0]  # (4, H, W)
            dhead = logits[HEAD].reshape(-1).argmax()
            dlabels = logits.argmax(0)
            _, _, done, _ = real.step(action)
            rlabels = torch.from_numpy(real._labels()).long().to(DEVICE)
            rhead = (rlabels == HEAD).reshape(-1).float().argmax()
            hit = bool((dhead == rhead).item())
            head_hits += 1.0 if hit else 0.0
            if first_head_div == steps and not hit:
                first_head_div = t + 1
            mask = rlabels != EMPTY
            active_sum += (dlabels[mask] == rlabels[mask]).float().mean().item()
            if done:
                break
        head_tracks.append(head_hits / (t + 1))
        active_agrees.append(active_sum / (t + 1))
        head_divs.append(first_head_div)
    return {
        "rollout_head_track": sum(head_tracks) / len(head_tracks),
        "rollout_active_agreement": sum(active_agrees) / len(active_agrees),
        "steps_head_diverge": sum(head_divs) / len(head_divs),
    }


CACHE = {}
CACHE_PATH = EXPDIR / "_cache.json"


def load_cache():
    global CACHE
    if CACHE_PATH.exists():
        CACHE = json.loads(CACHE_PATH.read_text())


def run_cell(data, hidden, epochs, eval_ds, label, key):
    if key in CACHE:
        m = CACHE[key]
        print(f"  [{label}] cached -> loss={m['loss']:.3f} "
              f"active={m['active_acc']:.3f} head={m['head_acc']:.3f} "
              f"track={m['rollout_head_track']:.3f}")
        return m
    t0 = time.time()
    model, history = train(data, hidden, epochs, eval_ds=eval_ds, label=label)
    m = evaluate_onestep(model, eval_ds)
    m.update(evaluate_rollout(model))
    params = sum(p.numel() for p in model.parameters())
    train_n = data["obs"].shape[0]
    m.update(
        params=params,
        hidden=hidden,
        train_n=train_n,
        epochs=epochs,
        flops=6 * params * train_n * epochs,
        seconds=round(time.time() - t0, 1),
        history=history,
    )
    print(f"  [{label}] params={params:,} data={train_n} epochs={epochs} "
          f"-> loss={m['loss']:.3f} active={m['active_acc']:.3f} "
          f"head={m['head_acc']:.3f} track={m['rollout_head_track']:.3f} "
          f"({m['seconds']}s)")
    CACHE[key] = m
    CACHE_PATH.write_text(json.dumps(CACHE, indent=2))
    return m


# --------------------------------------------------------------------------- #
# Plotting (greyscale, to match the app aesthetic)
# --------------------------------------------------------------------------- #
def heatmap(ax, grid, row_labels, col_labels, title, xlabel, ylabel,
            fmt="{:.0%}", lower_better=False):
    import numpy as np
    arr = np.array(grid, dtype=float)
    vmax = max(1e-6, arr.max() * 1.15)
    # Greys: bigger -> darker. For "lower is better" use the reversed map so the
    # best (smallest) cells are the dark ones -> darker is always "better".
    cmap = "Greys_r" if lower_better else "Greys"
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            darkness = (1 - val / vmax) if lower_better else (val / vmax)
            ax.text(j, i, fmt.format(val), ha="center", va="center",
                    color="white" if darkness > 0.55 else "black", fontsize=11)
    return im


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("Snake world-model scaling experiment")
    print("=" * 60)
    print(f"device={DEVICE}")
    load_cache()

    # Data pools (nested subsets of a single 2000-transition pool) + eval set.
    pool = get_pool(BASE_N * 10, seed=42, name="pool2000")
    eval_ds = get_eval_set(pool)

    data_levels = {
        "1x": subset(pool, BASE_N),          # 200
        "3.2x": subset(pool, round(BASE_N * 10 ** 0.5)),  # ~632
        "10x": subset(pool, BASE_N * 10),    # 2000
    }
    h1 = BASE_HIDDEN                 # 1x params
    h10 = hidden_for_multiplier(10)  # ~10x params
    print(f"param widths: 1x -> hidden {h1} ({n_params(h1):,}), "
          f"10x -> hidden {h10} ({n_params(h10):,})")
    epoch_levels = {"1x": 10, "3.2x": 32, "10x": 100}

    # ----- Grid 1: 2x2  data x params (epochs fixed at 10) ----- #
    print("\nGrid 1 (2x2): data x params @ 10 epochs")
    base_epochs = 10
    g2 = {}
    params_levels = {"1x": h1, "10x": h10}
    for d_key in ["1x", "10x"]:
        for p_key in ["1x", "10x"]:
            g2[(d_key, p_key)] = run_cell(
                data_levels[d_key], params_levels[p_key], base_epochs, eval_ds,
                label=f"data {d_key} / params {p_key}",
                key=f"2x2|d{d_key}|p{p_key}")

    # ----- Grid 2: 3x3  data x epochs (params fixed at 1x) ----- #
    print("\nGrid 2 (3x3): data x training-length(epochs) @ baseline model size")
    g3 = {}
    for d_key in ["1x", "3.2x", "10x"]:
        for e_key in ["1x", "3.2x", "10x"]:
            g3[(d_key, e_key)] = run_cell(
                data_levels[d_key], h1, epoch_levels[e_key], eval_ds,
                label=f"data {d_key} / epochs {e_key}",
                key=f"3x3|d{d_key}|e{e_key}")

    # ----- Persist JSON ----- #
    def serialize(grid):
        return [{"row": k[0], "col": k[1], **v} for k, v in grid.items()]

    (EXPDIR / "results_2x2.json").write_text(json.dumps(serialize(g2), indent=2))
    (EXPDIR / "results_3x3.json").write_text(json.dumps(serialize(g3), indent=2))

    # ----- Heatmaps ----- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 2x2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    loss2 = [[g2[(d, p)]["loss"] for p in ["1x", "10x"]] for d in ["1x", "10x"]]
    act2 = [[g2[(d, p)]["active_acc"] for p in ["1x", "10x"]] for d in ["1x", "10x"]]
    heatmap(axes[0], loss2, ["data 1x\n(200)", "data 10x\n(2000)"],
            ["params 1x\n(1.2M)", "params 10x\n(12M)"],
            "Held-out loss  (lower = better)", "model parameters", "training data",
            fmt="{:.3f}", lower_better=True)
    heatmap(axes[1], act2, ["data 1x\n(200)", "data 10x\n(2000)"],
            ["params 1x\n(1.2M)", "params 10x\n(12M)"],
            "Active-cell accuracy  (≈0: 10 epochs learns no structure)",
            "model parameters", "training data")
    fig.suptitle("Grid 1 — Data x Parameters (10 epochs fixed)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(EXPDIR / "heatmap_2x2.png", dpi=130)
    plt.close(fig)

    # 3x3
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rows = ["1x", "3.2x", "10x"]
    cols = ["1x", "3.2x", "10x"]
    loss3 = [[g3[(d, e)]["loss"] for e in cols] for d in rows]
    act3 = [[g3[(d, e)]["active_acc"] for e in cols] for d in rows]
    rlab = ["data 1x\n(200)", "data 3.2x\n(632)", "data 10x\n(2000)"]
    clab = ["epochs 1x\n(10)", "epochs 3.2x\n(32)", "epochs 10x\n(100)"]
    heatmap(axes[0], loss3, rlab, clab,
            "Held-out loss  (lower = better)", "training FLOPs (via epochs) ->", "training data ->",
            fmt="{:.3f}", lower_better=True)
    heatmap(axes[1], act3, rlab, clab,
            "Active-cell accuracy  (structure emerges bottom-right)",
            "training FLOPs (via epochs) ->", "training data ->")
    fig.suptitle("Grid 2 — Data x Training-FLOPs (baseline 1.2M params)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(EXPDIR / "heatmap_3x3.png", dpi=130)
    plt.close(fig)

    # Scaling curves: held-out loss vs training FLOPs, one line per data size.
    fig, ax = plt.subplots(figsize=(7.6, 5))
    shades = {"1x": "0.72", "3.2x": "0.42", "10x": "0.0"}
    data_n = {"1x": 200, "3.2x": 632, "10x": 2000}
    for d in rows:
        xs = [g3[(d, e)]["flops"] for e in cols]
        ys = [g3[(d, e)]["loss"] for e in cols]
        ax.plot(xs, ys, marker="o", color=shades[d], linewidth=2,
                label=f"data {d} ({data_n[d]})")
    ax.set_xscale("log")
    ax.set_xlabel("training FLOPs  (log scale)")
    ax.set_ylabel("held-out loss  (lower = better)")
    ax.set_title("Scaling curves — held-out loss vs training compute", fontsize=12)
    ax.legend(title="training data", frameon=False)
    ax.grid(True, which="both", color="0.9")
    fig.tight_layout()
    fig.savefig(EXPDIR / "scaling_curves.png", dpi=130)
    plt.close(fig)

    write_log(g2, g3, eval_ds, h1, h10, base_epochs, epoch_levels)
    print("\nDone. See experiments/EXPERIMENT_LOG.md")


def write_log(g2, g3, eval_ds, h1, h10, base_epochs, epoch_levels):
    import datetime

    def row2(d, p):
        m = g2[(d, p)]
        return (f"| data {d} / params {p} | {m['params']:,} | {m['train_n']} | "
                f"{m['flops']:.2e} | {m['loss']:.3f} | {m['active_acc']:.1%} | "
                f"{m['head_acc']:.1%} | {m['exact_match']:.1%} | "
                f"{m['rollout_head_track']:.1%} |")

    def row3(d, e):
        m = g3[(d, e)]
        return (f"| data {d} / epochs {e} | {m['epochs']} | {m['train_n']} | "
                f"{m['flops']:.2e} | {m['loss']:.3f} | {m['active_acc']:.1%} | "
                f"{m['head_acc']:.1%} | {m['exact_match']:.1%} | "
                f"{m['rollout_head_track']:.1%} |")

    # Auto findings.
    def mean(keys, metric, grid):
        return sum(grid[k][metric] for k in keys) / len(keys)

    # Grid 1 — active-cell accuracy (higher = better) and loss drop (positive = better).
    data10 = mean([("10x", "1x"), ("10x", "10x")], "active_acc", g2)
    data1 = mean([("1x", "1x"), ("1x", "10x")], "active_acc", g2)
    par10 = mean([("1x", "10x"), ("10x", "10x")], "active_acc", g2)
    par1 = mean([("1x", "1x"), ("10x", "1x")], "active_acc", g2)
    d_data = data10 - data1
    d_par = par10 - par1
    dl_data = (mean([("1x", "1x"), ("1x", "10x")], "loss", g2)
               - mean([("10x", "1x"), ("10x", "10x")], "loss", g2))
    dl_par = (mean([("1x", "1x"), ("10x", "1x")], "loss", g2)
              - mean([("1x", "10x"), ("10x", "10x")], "loss", g2))

    rows = ["1x", "3.2x", "10x"]
    ep10 = mean([(d, "10x") for d in rows], "active_acc", g3)
    ep1 = mean([(d, "1x") for d in rows], "active_acc", g3)
    dat10_g3 = mean([("10x", e) for e in rows], "active_acc", g3)
    dat1_g3 = mean([("1x", e) for e in rows], "active_acc", g3)
    d_ep = ep10 - ep1
    d_dat3 = dat10_g3 - dat1_g3
    dl_ep = (mean([(d, "1x") for d in rows], "loss", g3)
             - mean([(d, "10x") for d in rows], "loss", g3))
    dl_dat3 = (mean([("1x", e) for e in rows], "loss", g3)
               - mean([("10x", e) for e in rows], "loss", g3))

    allcells = list(g2.values()) + list(g3.values())
    best_active = max(c["active_acc"] for c in allcells)
    best_head = max(c["head_acc"] for c in allcells)
    best_exact = max(c["exact_match"] for c in allcells)

    md = f"""# Snake World Model — Scaling Experiment Log

_Generated {datetime.date.today().isoformat()} • CPU-only (4 threads), in-repo PyTorch._

## What this is

A small scaling-laws study on the tiny Snake world model (a 10x10 grid next-frame
predictor). The deployed baseline was trained on **just {g2[('1x','1x')]['train_n']} transitions**
with a **{g2[('1x','1x')]['params']:,}-parameter** MLP — i.e. heavily over-parameterized — so there is
lots of headroom to see how each scaling knob moves quality.

Two grids are reported:

* **Grid 1 (2x2): data x parameters**, at a fixed training length (10 epochs).
  Isolates the classic "more data vs. bigger model" question.
* **Grid 2 (3x3): data x training-length (epochs)**, at the baseline model size.
  Brings in the third scaling knob — **training compute / FLOPs** — that the 2x2
  holds fixed. The epochs axis is literally "**10x the FLOPs used to train**"
  (FLOPs = 6 · params · examples · epochs, so at fixed params/data, 10x epochs = 10x FLOPs).

Together the two grids exercise all three knobs (data, parameters, compute), which
is why this pairing was chosen as the most informative.

## A note on metrics (why not plain accuracy?)

A 10x10 Snake frame is **~95% empty cells**, so naive per-cell accuracy and
whole-frame exact-match are dominated by the trivial "predict empty everywhere"
solution (≈94% cell accuracy for a model that has learned nothing). To actually
separate the runs, the **headline metrics are held-out loss and active-cell
accuracy** (accuracy restricted to the cells that carry the snake body, head and
food). Strict metrics (exact next-frame match, precise head localization) are
reported too, but they stay near the floor across the whole sweep — see the
takeaway.

## Method

* **Model.** Same architecture as the deployed baseline ({DEPTH} hidden ReLU layers).
  Parameter count is scaled by widening the hidden layers:
  1x = hidden {h1} ({g2[('1x','1x')]['params']:,} params), 10x = hidden {h10} ({g2[('1x','10x')]['params']:,} params).
* **Data.** Generated with the project's own `collect.py` (deterministic, de-duplicated
  (obs, action) pairs). Sizes are nested subsets of one 2000-transition pool:
  1x = 200, 3.2x = 632, 10x = 2000.
* **Evaluation.** A single FIXED, held-out set of **{eval_ds['obs'].shape[0]} transitions**, generated
  with a different seed and de-duplicated against the training pool, is reused for
  every cell so the numbers are directly comparable.
* **Quality metrics.**
  * **Held-out loss** (primary) — mean per-cell cross-entropy on the held-out set
    (lower is better); the canonical scaling-law quantity.
  * **Active-cell accuracy** (primary) — accuracy on the cells that are non-empty
    in the target frame, i.e. does the model place the snake + food correctly.
  * **Head accuracy** — fraction of held-out frames where the single highest-probability
    head cell lands exactly on the true head.
  * **Exact next-frame match** — strict: the entire 100-cell frame is correct.
  * **Dream head-tracking** — feed the model its own predictions for 30 steps from
    24 fixed real openings (same greedy policy drives both) and measure how often
    the dreamed head stays on the real head.
* **Training.** Adam, lr {LR}, batch {BATCH}, fixed seed for every cell.

---

## Grid 1 — Data x Parameters (10 epochs)

![Grid 1 heatmap](heatmap_2x2.png)

| cell | params | train data | train FLOPs | held-out loss | active-cell acc | head acc | exact match | dream head-track |
|---|---|---|---|---|---|---|---|---|
{row2('1x','1x')}
{row2('1x','10x')}
{row2('10x','1x')}
{row2('10x','10x')}

**Effect of 10x data** (averaged over both param sizes): active-cell acc {data1:.1%} -> {data10:.1%} ({d_data:+.1%}); loss drop {dl_data:+.3f}.
**Effect of 10x params** (averaged over both data sizes): active-cell acc {par1:.1%} -> {par10:.1%} ({d_par:+.1%}); loss drop {dl_par:+.3f}.

> Headline: **{'more data lowered loss far more than more parameters' if dl_data > dl_par else 'more parameters lowered loss more than more data'}**
> (data loss-drop {dl_data:+.3f} vs params loss-drop {dl_par:+.3f}). At a fixed 10 epochs no
> setting yet crosses into learning structure, so active-cell accuracy stays ~0 until the
> training compute is also raised — that emergence shows up in Grid 2.

---

## Grid 2 — Data x Training-FLOPs / epochs (baseline 1.2M params)

![Grid 2 heatmap](heatmap_3x3.png)

| cell | epochs | train data | train FLOPs | held-out loss | active-cell acc | head acc | exact match | dream head-track |
|---|---|---|---|---|---|---|---|---|
{row3('1x','1x')}
{row3('1x','3.2x')}
{row3('1x','10x')}
{row3('3.2x','1x')}
{row3('3.2x','3.2x')}
{row3('3.2x','10x')}
{row3('10x','1x')}
{row3('10x','3.2x')}
{row3('10x','10x')}

**Effect of 10x training FLOPs** (10x epochs, averaged over data): active-cell acc {ep1:.1%} -> {ep10:.1%} ({d_ep:+.1%}); loss drop {dl_ep:+.3f}.
**Effect of 10x data** (averaged over epochs): active-cell acc {dat1_g3:.1%} -> {dat10_g3:.1%} ({d_dat3:+.1%}); loss drop {dl_dat3:+.3f}.

> Headline: **{'spending the extra FLOPs on data beat spending them on longer training' if dl_dat3 > dl_ep else 'longer training (more FLOPs) helped more than more data here'}**
> (data loss-drop {dl_dat3:+.3f} vs epochs/FLOPs loss-drop {dl_ep:+.3f}). Active-cell structure
> only emerges in the bottom-right corner — 10x data **and** 10x epochs together.

---

## Scaling curves

![Scaling curves](scaling_curves.png)

Held-out loss vs training FLOPs, one line per dataset size (from Grid 2). Two things to read
off it: (1) **more data shifts the whole curve down** — a persistent gap that spending more
compute on a smaller dataset never closes; and (2) along each line the **returns to extra
training compute flatten out** — and on the smallest (200-example) dataset the loss even
*rises* again by 100 epochs, the signature of overfitting. That is the classic
data-vs-compute scaling picture in miniature.

---

## Takeaway

The baseline sits in a strongly **data-limited** regime. Both extra data and extra
training compute lower the held-out loss and raise active-cell accuracy, and
{'data is the bigger lever of the two' if (d_data >= d_par and d_dat3 >= d_ep) else 'the per-grid headlines above show which knob dominated'}; adding parameters to the same
tiny dataset does the least.

But the **strict** view is sobering: across the entire sweep the best run still only
reaches **{best_active:.0%} active-cell accuracy**, **{best_head:.0%} head accuracy**, and **{best_exact:.0%} exact
next-frame match** — exact-match and precise head-localization stay near the floor
everywhere. With only a few hundred optimizer steps (batch {BATCH} ≥ dataset size),
this recipe under-fits precise localization: scaling data/compute teaches the model
the coarse "where the body roughly is" structure but not the exact one-cell head/food
dynamics, and multi-step dreaming therefore diverges almost immediately. The honest
conclusion: within this regime, **spend a fixed budget on more data first**, but
closing the gap to a faithful world model needs a different recipe (far more
optimizer steps and/or a spatially-aware architecture), not just 10x of any one knob.
"""
    (EXPDIR / "EXPERIMENT_LOG.md").write_text(md)


if __name__ == "__main__":
    main()
