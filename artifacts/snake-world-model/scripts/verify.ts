// Verify the TypeScript forward-pass port matches PyTorch exactly.
//
// Loads the exported weights + manifest from public/ and the fixtures
// (obs, action -> predicted labels) produced by snake_world_model/export_weights.py,
// then asserts the TS argmax labels match Python for every case.
//
// Run:  pnpm --filter @workspace/snake-world-model verify

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  parseLayers,
  buildInput,
  forward,
  argmaxLabels,
  type ModelManifest,
} from "../src/engine/model";
import { CELLS, CHANNELS } from "../src/engine/constants";

const here = dirname(fileURLToPath(import.meta.url));
const pub = join(here, "..", "public");

interface Fixture {
  obs: number[]; // length CHANNELS*CELLS, channel-major
  action: number;
  labels: number[]; // length CELLS, row-major
}

const manifest: ModelManifest = JSON.parse(
  readFileSync(join(pub, "world_model.json"), "utf8"),
);
const binBuf = readFileSync(join(pub, "world_model.bin"));
const arrayBuffer = binBuf.buffer.slice(
  binBuf.byteOffset,
  binBuf.byteOffset + binBuf.byteLength,
);
const fixtures: Fixture[] = JSON.parse(
  readFileSync(join(here, "fixtures.json"), "utf8"),
);

const layers = parseLayers(manifest, arrayBuffer);

// Rebuild the input from the fixture's channel-major obs to feed forward()
// directly (this exercises forward + argmax, the parts that must match torch).
function inputFromObs(obs: number[], action: number): Float32Array {
  const input = new Float32Array(CHANNELS * CELLS + 4);
  for (let i = 0; i < CHANNELS * CELLS; i++) input[i] = obs[i];
  input[CHANNELS * CELLS + action] = 1;
  return input;
}

let failures = 0;
let totalCellMismatches = 0;
for (let f = 0; f < fixtures.length; f++) {
  const { obs, action, labels } = fixtures[f];
  const logits = forward(layers, inputFromObs(obs, action));
  const predicted = argmaxLabels(logits);
  let mismatches = 0;
  for (let i = 0; i < CELLS; i++) {
    if (predicted[i] !== labels[i]) mismatches++;
  }
  if (mismatches > 0) {
    failures++;
    totalCellMismatches += mismatches;
    console.error(`Fixture ${f}: ${mismatches}/${CELLS} cells mismatch`);
  }
}

// Also exercise buildInput (the path the app uses) on the first fixture by
// reconstructing labels from the channel-major obs and confirming parity.
const first = fixtures[0];
const reconstructed = new Uint8Array(CELLS);
for (let cell = 0; cell < CELLS; cell++) {
  for (let c = 0; c < CHANNELS; c++) {
    if (first.obs[c * CELLS + cell] === 1) reconstructed[cell] = c;
  }
}
const viaBuildInput = argmaxLabels(
  forward(layers, buildInput(reconstructed, first.action)),
);
let buildInputMismatch = 0;
for (let i = 0; i < CELLS; i++) {
  if (viaBuildInput[i] !== first.labels[i]) buildInputMismatch++;
}

if (failures === 0 && buildInputMismatch === 0) {
  console.log(
    `PASS: all ${fixtures.length} fixtures match PyTorch exactly (buildInput path OK).`,
  );
  process.exit(0);
} else {
  console.error(
    `FAIL: ${failures}/${fixtures.length} fixtures mismatched ` +
      `(${totalCellMismatches} cells), buildInput mismatch=${buildInputMismatch}`,
  );
  process.exit(1);
}
