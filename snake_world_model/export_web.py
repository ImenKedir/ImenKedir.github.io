"""Export a trained checkpoint for the web demo.

Writes web/static/model/weights.bin (all tensors as little-endian float32,
concatenated in manifest order) and weights.json (shapes + byte offsets).

Usage: python snake_world_model/export_web.py [checkpoint.pt]
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "web" / "static" / "model"

DEFAULT = ROOT / "experiments" / "world_model_2m_1x.pt"


def main() -> None:
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    state = torch.load(ckpt, weights_only=True, map_location="cpu")

    OUT.mkdir(parents=True, exist_ok=True)
    manifest, blobs, offset = [], [], 0
    for name, t in state.items():
        arr = t.numpy().astype("<f4")
        manifest.append({"name": name, "shape": list(arr.shape), "offset": offset})
        blobs.append(arr.tobytes())
        offset += arr.nbytes

    (OUT / "weights.bin").write_bytes(b"".join(blobs))
    (OUT / "weights.json").write_text(json.dumps({
        "checkpoint": ckpt.name,
        "total_bytes": offset,
        "tensors": manifest,
    }, indent=2))
    print(f"exported {ckpt.name}: {offset // 4:,} params, "
          f"{offset / 1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
