// Browser-side inference for the exported MLP world model.
// Input: one-hot labels (4*100, class-major) + one-hot action (4) = 404 dims.
// Output: 400 logits — 4-class distribution per cell, laid out class-major.

import { GRID, EMPTY, FOOD } from './env';

const N = GRID * GRID;

interface Layer {
	w: Float32Array; // (out, in) row-major, as in torch.nn.Linear
	b: Float32Array;
	inDim: number;
	outDim: number;
}

export class WorldModel {
	constructor(private layers: Layer[]) {}

	/** labels: Uint8Array(100) row-major; returns Float32Array(400) logits. */
	forward(labels: Uint8Array, action: number): Float32Array {
		let x = new Float32Array(4 * N + 4);
		for (let i = 0; i < N; i++) x[labels[i] * N + i] = 1;
		x[4 * N + action] = 1;

		for (let li = 0; li < this.layers.length; li++) {
			const { w, b, inDim, outDim } = this.layers[li];
			const out = new Float32Array(outDim);
			const last = li === this.layers.length - 1;
			for (let j = 0; j < outDim; j++) {
				let acc = b[j];
				const row = j * inDim;
				for (let i = 0; i < inDim; i++) acc += w[row + i] * x[i];
				out[j] = last || acc > 0 ? acc : 0;
			}
			x = out;
		}
		return x;
	}
}

export async function loadModel(base: string): Promise<WorldModel> {
	const [manifest, bin] = await Promise.all([
		fetch(`${base}/model/weights.json`).then((r) => r.json()),
		fetch(`${base}/model/weights.bin`).then((r) => r.arrayBuffer())
	]);
	const tensor = (name: string): Float32Array => {
		const t = manifest.tensors.find((t: { name: string }) => t.name === name);
		const size = t.shape.reduce((a: number, b: number) => a * b, 1);
		return new Float32Array(bin, t.offset, size);
	};
	const layers: Layer[] = [];
	for (const t of manifest.tensors) {
		const m = t.name.match(/^net\.(\d+)\.weight$/);
		if (!m) continue;
		const [outDim, inDim] = t.shape;
		layers.push({ w: tensor(t.name), b: tensor(`net.${m[1]}.bias`), inDim, outDim });
	}
	return new WorldModel(layers);
}

/** Per-cell softmax probability of class `cls` at cell `i`. */
function cellProb(logits: Float32Array, i: number, cls: number): number {
	let max = -Infinity;
	for (let c = 0; c < 4; c++) max = Math.max(max, logits[c * N + i]);
	let sum = 0;
	const exps = new Float32Array(4);
	for (let c = 0; c < 4; c++) {
		exps[c] = Math.exp(logits[c * N + i] - max);
		sum += exps[c];
	}
	return exps[cls] / sum;
}

/**
 * Per-cell argmax decode. With `sampleFood`, if no cell decodes to food but the
 * model predicts ~one food somewhere (mass > 0.5), sample a food cell from the
 * model's own distribution — the env's spawn is uniform-random, so a point
 * prediction is impossible and argmax alone drops the food entirely.
 */
export function decode(logits: Float32Array, sampleFood: boolean, rng: () => number): Uint8Array {
	const labels = new Uint8Array(N);
	let hasFood = false;
	for (let i = 0; i < N; i++) {
		let best = 0;
		for (let c = 1; c < 4; c++) if (logits[c * N + i] > logits[best * N + i]) best = c;
		labels[i] = best;
		if (best === FOOD) hasFood = true;
	}
	if (!sampleFood || hasFood) return labels;

	const foodP = new Float32Array(N);
	let foodMass = 0;
	for (let i = 0; i < N; i++) {
		foodP[i] = cellProb(logits, i, FOOD);
		foodMass += foodP[i];
	}
	if (foodMass < 0.5) return labels;

	// Sample among cells that decoded empty, proportional to p(food).
	let total = 0;
	for (let i = 0; i < N; i++) if (labels[i] === EMPTY) total += foodP[i];
	let r = rng() * total;
	for (let i = 0; i < N; i++) {
		if (labels[i] !== EMPTY) continue;
		r -= foodP[i];
		if (r <= 0) {
			labels[i] = FOOD;
			break;
		}
	}
	return labels;
}
