"""Render the blog's scaling figures: black-on-white, mono type, no color.

Reads results_scaling_data.json (curve + params probes), results_fill_curve.json
(50k..2M), and results_surface.json (data x epochs grid).
Writes PNGs to web/static/plots/.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPDIR = Path(__file__).resolve().parent
OUT = EXPDIR.parent.parent / "web" / "static" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "monospace",
    "font.size": 9,
    "text.color": "#111111",
    "axes.edgecolor": "#111111",
    "axes.labelcolor": "#111111",
    "xtick.color": "#111111",
    "ytick.color": "#111111",
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 200,
})

INK = "#111111"
FAINT = "#999999"


def load_curve():
    base = json.loads((EXPDIR / "results_scaling_data.json").read_text())
    pts = list(base["curve_512h_100ep"])
    fill = EXPDIR / "results_fill_curve.json"
    if fill.exists():
        pts += json.loads(fill.read_text())
    pts.sort(key=lambda p: p["train_n"])
    return pts


def plot_data_curve(pts):
    n = [p["train_n"] for p in pts]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))

    ax1.plot(n, [p["loss"] for p in pts], "o-", color=INK, lw=1.2, ms=3.5)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("training transitions")
    ax1.set_ylabel("held-out loss")

    series = [
        ("active_acc", "active-cell acc", "o-"),
        ("head_acc", "head acc", "s--"),
        ("exact_match", "exact frame", "^:"),
        ("rollout_head_track", "dream head-track", "d-."),
    ]
    for key, label, style in series:
        ax2.plot(n, [p[key] for p in pts], style, color=INK, lw=1.0,
                 ms=3.2, markerfacecolor="white", label=label)
    ax2.set_xscale("log")
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_xlabel("training transitions")
    ax2.set_ylabel("accuracy")
    ax2.legend(frameon=False, fontsize=7.5, loc="upper left")

    # The structural transition: active-cell acc leaves 0 at 2k.
    ax2.axvline(2000, color=FAINT, lw=0.8, ls=":")
    ax2.annotate("structure\nappears", xy=(2000, 0.02), xytext=(290, 0.30),
                 fontsize=7.5, color=INK,
                 arrowprops=dict(arrowstyle="-", color=FAINT, lw=0.8))

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "scaling_data.png", bbox_inches="tight")
    plt.close(fig)


def plot_surfaces():
    cells = json.loads((EXPDIR / "results_surface.json").read_text())
    ns = sorted({c["train_n"] for c in cells})
    eps = sorted({c["epochs"] for c in cells})
    loss = np.zeros((len(ns), len(eps)))
    act = np.zeros((len(ns), len(eps)))
    for c in cells:
        i, j = ns.index(c["train_n"]), eps.index(c["epochs"])
        loss[i, j] = c["loss"]
        act[i, j] = c["active_acc"]

    X, Y = np.meshgrid(np.log10(eps), np.log10(ns))
    fig = plt.figure(figsize=(8.6, 3.8))
    for k, (Z, title) in enumerate([
        (np.log10(loss), "held-out loss (log10)"),
        (act, "active-cell accuracy"),
    ]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot_wireframe(X, Y, Z, color=INK, lw=0.9)
        ax.scatter(X, Y, Z, color=INK, s=8)
        ax.set_xticks(np.log10(eps), [str(e) for e in eps])
        ax.set_yticks(np.log10(ns), [f"{n:,}" for n in ns])
        ax.set_xlabel("epochs", labelpad=-2)
        ax.set_ylabel("transitions", labelpad=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7, pad=-1)
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
        ax.grid(False)
        ax.view_init(elev=22, azim=-55)
        if k == 0:
            ax.invert_yaxis()  # loss falls away from the viewer
    fig.tight_layout()
    fig.savefig(OUT / "scaling_surface.png", bbox_inches="tight")
    plt.close(fig)


def plot_params():
    base = json.loads((EXPDIR / "results_scaling_data.json").read_text())
    probes = base["params_probes_at_max_data"]
    at16k = [p for p in base["curve_512h_100ep"] if p["train_n"] == 16000] + probes
    at16k.sort(key=lambda p: p["params"])
    x = [p["params"] for p in at16k]

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for key, label, style in [
        ("active_acc", "active-cell acc", "o-"),
        ("head_acc", "head acc", "s--"),
        ("rollout_head_track", "dream head-track", "d-."),
    ]:
        ax.plot(x, [p[key] for p in at16k], style, color=INK, lw=1.0,
                ms=3.2, markerfacecolor="white", label=label)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("parameters (at 16k transitions)")
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "scaling_params.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pts = load_curve()
    plot_data_curve(pts)
    plot_surfaces()
    plot_params()
    print(f"wrote plots for {len(pts)} curve points -> {OUT}")
