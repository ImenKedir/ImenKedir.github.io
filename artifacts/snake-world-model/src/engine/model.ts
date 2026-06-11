// Port of the trained MLP world model's forward pass (snake_world_model/model.py).
//
// Architecture: 5 Linear layers (404→512→512→512→512→400) with ReLU between
// every layer except the last. Weights are loaded from a flat float32 binary
// (per layer: weight [out*in] row-major, then bias [out]) exported by
// snake_world_model/export_weights.py.
//
// The pure functions below (parseLayers / buildInput / forward / argmaxLabels)
// are deliberately framework-free so they can be verified against PyTorch in
// Node (see scripts/verify.ts).

import { CHANNELS, CELLS } from "./constants";
import { type Labels } from "./constants";

export interface ModelManifest {
  dtype: string;
  gridSize: number;
  channels: number;
  obsDim: number;
  inputDim: number;
  layers: { in: number; out: number }[];
}

export interface Layer {
  inDim: number;
  outDim: number;
  weight: Float32Array; // length out*in, row-major: index = o*in + i
  bias: Float32Array; // length out
}

/** Slice the flat weight buffer into per-layer weight/bias views. */
export function parseLayers(
  manifest: ModelManifest,
  buffer: ArrayBuffer,
): Layer[] {
  const all = new Float32Array(buffer);
  const layers: Layer[] = [];
  let offset = 0; // in float32 units
  for (const { in: inDim, out: outDim } of manifest.layers) {
    const wCount = inDim * outDim;
    const weight = all.subarray(offset, offset + wCount);
    offset += wCount;
    const bias = all.subarray(offset, offset + outDim);
    offset += outDim;
    layers.push({ inDim, outDim, weight, bias });
  }
  return layers;
}

/**
 * Build the 404-dim input vector: the observation flattened channel-major
 * (index = c*CELLS + cell, matching torch's obs.flatten(1) on a (4,H,W)
 * tensor) followed by the one-hot action.
 */
export function buildInput(labels: Labels, action: number): Float32Array {
  const input = new Float32Array(CHANNELS * CELLS + 4);
  for (let cell = 0; cell < CELLS; cell++) {
    input[labels[cell] * CELLS + cell] = 1;
  }
  input[CHANNELS * CELLS + action] = 1;
  return input;
}

/** Run the MLP forward pass. Returns the 400-dim output logits. */
export function forward(layers: Layer[], input: Float32Array): Float32Array {
  let x = input;
  for (let li = 0; li < layers.length; li++) {
    const { inDim, outDim, weight, bias } = layers[li];
    const y = new Float32Array(outDim);
    const isLast = li === layers.length - 1;
    for (let o = 0; o < outDim; o++) {
      const base = o * inDim;
      let sum = bias[o];
      for (let i = 0; i < inDim; i++) {
        sum += weight[base + i] * x[i];
      }
      y[o] = isLast ? sum : sum > 0 ? sum : 0; // ReLU on all but the last layer
    }
    x = y;
  }
  return x;
}

/**
 * Argmax over the channel dimension per cell (mirrors logits.argmax(dim=1)).
 * Logits are channel-major: value for channel c at cell = logits[c*CELLS + cell].
 */
export function argmaxLabels(logits: Float32Array): Uint8Array {
  const labels = new Uint8Array(CELLS);
  for (let cell = 0; cell < CELLS; cell++) {
    let best = 0;
    let bestVal = logits[cell];
    for (let c = 1; c < CHANNELS; c++) {
      const v = logits[c * CELLS + cell];
      if (v > bestVal) {
        bestVal = v;
        best = c;
      }
    }
    labels[cell] = best;
  }
  return labels;
}

export class WorldModel {
  readonly layers: Layer[];
  readonly manifest: ModelManifest;

  constructor(layers: Layer[], manifest: ModelManifest) {
    this.layers = layers;
    this.manifest = manifest;
  }

  /** Fetch the manifest + binary weights relative to the given base URL. */
  static async load(baseUrl: string): Promise<WorldModel> {
    const [manifest, buffer] = await Promise.all([
      fetch(`${baseUrl}world_model.json`).then((r) => {
        if (!r.ok) throw new Error(`Failed to load model manifest (${r.status})`);
        return r.json() as Promise<ModelManifest>;
      }),
      fetch(`${baseUrl}world_model.bin`).then((r) => {
        if (!r.ok) throw new Error(`Failed to load model weights (${r.status})`);
        return r.arrayBuffer();
      }),
    ]);
    return new WorldModel(parseLayers(manifest, buffer), manifest);
  }

  /** Predict the next frame's labels given current labels + an action. */
  predict(labels: Labels, action: number): Uint8Array {
    return argmaxLabels(forward(this.layers, buildInput(labels, action)));
  }
}
