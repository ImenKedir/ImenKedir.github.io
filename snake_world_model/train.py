"""Train the Snake MLP world model.

Usage:
    python train.py
    python train.py --epochs 20 --lr 3e-4
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import SnakeWorldModel

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int,   default=10)
    p.add_argument("--batch",  type=int,   default=512)
    p.add_argument("--lr",     type=float, default=1e-3)
    args = p.parse_args()

    root   = Path(__file__).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data    = torch.load(root / "transitions.pt", weights_only=True)
    obs     = data["obs"]
    actions = data["actions"]
    targets = data["next_obs"].argmax(dim=1).long()  # (N, H, W)

    n_val   = len(obs) // 10
    n_train = len(obs) - n_val
    train_ds, val_ds = random_split(
        TensorDataset(obs, actions, targets), [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch)

    model     = SnakeWorldModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"device={device}  train={n_train:,}  val={n_val:,}  params={n_params:,}\n")

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for obs_b, act_b, tgt_b in train_loader:
            obs_b, act_b, tgt_b = obs_b.to(device), act_b.to(device), tgt_b.to(device)
            loss = F.cross_entropy(model(obs_b, act_b), tgt_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(obs_b)
        train_loss /= n_train

        model.eval()
        val_loss = correct = total = 0
        with torch.no_grad():
            for obs_b, act_b, tgt_b in val_loader:
                obs_b, act_b, tgt_b = obs_b.to(device), act_b.to(device), tgt_b.to(device)
                logits = model(obs_b, act_b)
                val_loss += F.cross_entropy(logits, tgt_b).item() * len(obs_b)
                correct  += (logits.argmax(dim=1) == tgt_b).sum().item()
                total    += tgt_b.numel()
        val_loss /= n_val

        marker = "  ←" if val_loss < best_val else ""
        print(f"epoch {epoch:>3}  train={train_loss:.4f}  val={val_loss:.4f}  acc={correct/total*100:.2f}%{marker}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), root / "world_model.pt")

    print(f"\nbest val={best_val:.4f}  →  world_model.pt")
