"""Drive the snake and watch the world model dream alongside reality.

Usage:
    python snake_world_model/dream.py [checkpoint.pt] [--seed N]

Controls: w/a/s/d to steer, enter to repeat the last direction, q to quit.
Left grid is the real env, right grid is the model's rollout (fed its own
predictions each step). Divergent cells are marked in red.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from device import best_device
from env import HEAD, UP, DOWN, LEFT, RIGHT, GRID_SIZE, SnakeEnv

DEVICE = best_device()

KEYS = {"w": UP, "s": DOWN, "a": LEFT, "d": RIGHT}
SYMBOLS = {0: "·", 1: "o", 2: "H", 3: "*"}
RED, DIM, RESET = "\033[31m", "\033[2m", "\033[0m"


def load_model(path: Path) -> torch.nn.Module:
    state = torch.load(path, weights_only=True, map_location=DEVICE)
    hidden = state["net.0.weight"].shape[0]
    depth = sum(1 for k in state if k.endswith(".weight")) - 1
    from run_experiments import WorldModel
    model = WorldModel(hidden, depth).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded {path.name}  hidden={hidden} depth={depth} ({n:,} params)  device={DEVICE}")
    return model


def dream_step(model, dlabels: torch.Tensor, action: int) -> torch.Tensor:
    obs = F.one_hot(dlabels, num_classes=4).permute(2, 0, 1).float().unsqueeze(0)
    act = F.one_hot(torch.tensor(action, device=DEVICE), num_classes=4).float().unsqueeze(0)
    with torch.no_grad():
        return model(obs, act)[0].argmax(0)


def render(real: np.ndarray, dream: np.ndarray) -> str:
    lines = [f"  {'REAL':<{GRID_SIZE * 2}}  {'DREAM'}"]
    for r in range(GRID_SIZE):
        left = " ".join(SYMBOLS[int(v)] for v in real[r])
        right = []
        for c in range(GRID_SIZE):
            ch = SYMBOLS[int(dream[r, c])]
            right.append(ch if dream[r, c] == real[r, c] else f"{RED}{ch}{RESET}")
        lines.append(f"  {left}   {' '.join(right)}")
    return "\n".join(lines)


def read_action(last: int) -> int | None:
    """Returns an action, or None to quit. Enter repeats the last direction."""
    while True:
        try:
            raw = input(f"{DIM}[wasd, enter=repeat, q=quit]>{RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw == "q":
            return None
        if raw == "":
            return last
        if raw[0] in KEYS:
            return KEYS[raw[0]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    opts = parser.parse_args()

    default = ROOT / "experiments" / "world_model_2m_1x.pt"
    ckpt = opts.checkpoint or (default if default.exists() else ROOT / "world_model.pt")
    model = load_model(ckpt)

    env = SnakeEnv()
    env.reset(seed=opts.seed)
    dlabels = torch.from_numpy(env._labels()).long().to(DEVICE)

    step, matched, action = 0, 0, RIGHT
    print(f"\nseed {opts.seed} — snake starts facing right\n")
    print(render(env._labels(), dlabels.cpu().numpy()))

    while True:
        action = read_action(action)
        if action is None:
            break
        dlabels = dream_step(model, dlabels, action)
        _, reward, done, _ = env.step(action)
        rlabels = env._labels()

        step += 1
        dnp = dlabels.cpu().numpy()
        rhead = (rlabels == HEAD).reshape(-1).argmax()
        dhead = (dnp == HEAD).reshape(-1).argmax()
        head_ok = bool((rlabels == HEAD).any()) and rhead == dhead
        matched += head_ok

        print(f"\nstep {step}  reward={reward:+.0f}  "
              f"head {'tracks' if head_ok else RED + 'DIVERGED' + RESET}  "
              f"({matched}/{step} tracked)")
        print(render(rlabels, dnp))
        if done:
            print("\nreal snake died — game over")
            break

    print(f"\n{step} steps, dream head tracked {matched}/{step}")


if __name__ == "__main__":
    main()
