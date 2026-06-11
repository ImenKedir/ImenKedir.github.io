"""Train the Snake MLP world model on the collected transition dataset.

Usage:
    cd snake_world_model
    python train.py                        # defaults
    python train.py --epochs 20 --lr 3e-4
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import SnakeWorldModel


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_path: str  = "transitions.pt",
    epochs:    int  = 10,
    batch:     int  = 512,
    lr:        float = 1e-3,
    val_split: float = 0.1,
    save_path: str  = "world_model.pt",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---------------------------------------------------------------
    root = Path(__file__).parent
    data = torch.load(root / data_path, weights_only=True)
    obs      = data["obs"]       # (N, 4, H, W)
    actions  = data["actions"]   # (N, 4)
    next_obs = data["next_obs"]  # (N, 4, H, W)

    # Targets are class indices (H*W independent 4-class problems)
    targets = next_obs.argmax(dim=1).long()  # (N, H, W)

    dataset = TensorDataset(obs, actions, targets)
    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, pin_memory=True)
    print(f"Train: {n_train:,}  Val: {n_val:,}")

    # --- Model --------------------------------------------------------------
    model = SnakeWorldModel(hidden=512, depth=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Loop ---------------------------------------------------------------
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for obs_b, act_b, tgt_b in train_loader:
            obs_b, act_b, tgt_b = obs_b.to(device), act_b.to(device), tgt_b.to(device)
            logits = model(obs_b, act_b)           # (B, 4, H, W)
            loss   = F.cross_entropy(logits, tgt_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(obs_b)
        train_loss /= n_train

        # Validate
        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for obs_b, act_b, tgt_b in val_loader:
                obs_b, act_b, tgt_b = obs_b.to(device), act_b.to(device), tgt_b.to(device)
                logits = model(obs_b, act_b)
                val_loss += F.cross_entropy(logits, tgt_b).item() * len(obs_b)
                preds    = logits.argmax(dim=1)    # (B, H, W)
                correct  += (preds == tgt_b).sum().item()
                total    += tgt_b.numel()
        val_loss /= n_val
        acc = correct / total * 100

        marker = "  ←" if val_loss < best_val_loss else ""
        print(f"Epoch {epoch:>3}/{epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  acc={acc:.2f}%{marker}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), root / save_path)

    print(f"\nBest val loss: {best_val_loss:.4f}  →  saved to {root / save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="transitions.pt")
    p.add_argument("--epochs", type=int,   default=10)
    p.add_argument("--batch",  type=int,   default=512)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--out",    default="world_model.pt")
    args = p.parse_args()

    train(
        data_path = args.data,
        epochs    = args.epochs,
        batch     = args.batch,
        lr        = args.lr,
        save_path = args.out,
    )


if __name__ == "__main__":
    main()
