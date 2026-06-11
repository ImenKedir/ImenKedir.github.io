"""Export the trained Snake world model for in-browser (TypeScript) inference.

Produces three files in the react artifact's public/ directory:

  world_model.bin   raw float32 LE weights, per layer: weight (out*in) then
                    bias (out), for the 5 Linear layers in forward order.
  world_model.json  manifest describing layer shapes + grid constants.
  fixtures.json     (obs, action) -> predicted labels test cases so the TS
                    forward pass can be verified against PyTorch exactly.

The browser never reproduces numpy's RNG; only the forward pass must match,
which these fixtures pin down.
"""

import json
import struct
from pathlib import Path

import numpy as np
import torch

from env import SnakeEnv, GRID_SIZE, _OPPOSITE
from collect import greedy_action, to_tensor
from model import SnakeWorldModel, OBS_DIM

HERE = Path(__file__).parent
OUT = HERE.parent / "artifacts" / "snake-world-model" / "public"
LAYER_KEYS = ["net.0", "net.2", "net.4", "net.6", "net.8"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    model = SnakeWorldModel()
    model.load_state_dict(
        torch.load(HERE / "world_model.pt", weights_only=True, map_location="cpu")
    )
    model.eval()
    sd = model.state_dict()

    # --- binary weights -----------------------------------------------------
    layers = []
    buf = bytearray()
    for key in LAYER_KEYS:
        w = sd[f"{key}.weight"].contiguous().cpu().numpy().astype(np.float32)
        b = sd[f"{key}.bias"].contiguous().cpu().numpy().astype(np.float32)
        out_dim, in_dim = w.shape
        layers.append({"in": int(in_dim), "out": int(out_dim)})
        buf += w.reshape(-1).tobytes()  # row-major: index = o*in + i
        buf += b.reshape(-1).tobytes()

    (OUT / "world_model.bin").write_bytes(bytes(buf))

    manifest = {
        "dtype": "float32",
        "gridSize": GRID_SIZE,
        "channels": 4,
        "obsDim": OBS_DIM,
        "inputDim": OBS_DIM + 4,
        "layers": layers,  # forward order; ReLU after all but the last
    }
    (OUT / "world_model.json").write_text(json.dumps(manifest, indent=2))

    # --- verification fixtures ---------------------------------------------
    # Roll out the real env with a mix of greedy + random actions to get a
    # diverse set of realistic obs, then record the model's argmax prediction.
    rng = np.random.default_rng(1234)
    fixtures = []
    env = SnakeEnv()
    episodes = 0
    while len(fixtures) < 40:
        obs = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        steps = 0
        while not done and steps < 60 and len(fixtures) < 40:
            if rng.random() < 0.4:
                action = int(rng.integers(0, 4))
            else:
                action = greedy_action(env._snake, env._food, env._direction)

            obs_t = to_tensor(obs)  # (4, H, W)
            one_hot = torch.zeros(1, 4)
            one_hot[0, action] = 1.0
            with torch.no_grad():
                logits = model(obs_t.unsqueeze(0), one_hot)
                labels = logits.argmax(dim=1)[0].cpu().numpy().astype(int)

            # Flatten obs the same way the TS port will: index = c*100 + h*10 + w
            obs_flat = obs_t.reshape(-1).cpu().numpy().astype(np.float32).tolist()
            fixtures.append(
                {
                    "obs": obs_flat,
                    "action": action,
                    "labels": labels.reshape(-1).tolist(),  # row-major h*10 + w
                }
            )

            obs, _r, done, _ = env.step(action)
            steps += 1
        episodes += 1

    (OUT / "fixtures.json").write_text(json.dumps(fixtures))

    size_mb = (OUT / "world_model.bin").stat().st_size / 1e6
    print(f"Wrote world_model.bin ({size_mb:.2f} MB), world_model.json, "
          f"fixtures.json ({len(fixtures)} cases) to {OUT}")


if __name__ == "__main__":
    main()
