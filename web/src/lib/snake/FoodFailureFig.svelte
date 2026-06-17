<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import { EMPTY, BODY, HEAD, FOOD, GRID } from './env';

	type Snapshot = {
		seed: number;
		step: number;
		food_mass: number;
		food_max: number;
		food_mean_valid: number;
		food_cells_argmax: number;
		true_new_food_prob: number;
		new_food: [number, number];
		probs: number[][];
	};

	const classes = [
		{ name: 'empty', cls: EMPTY },
		{ name: 'body', cls: BODY },
		{ name: 'head', cls: HEAD },
		{ name: 'food', cls: FOOD }
	];

	let snapshot: Snapshot | null = $state(null);
	let loadError = $state('');

	onMount(() => {
		fetch(`${base}/model/food_snapshot.json`).then(
			async (r) => (snapshot = await r.json()),
			(e) => (loadError = String(e))
		);
	});

	const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
	const newFoodIndex = (s: Snapshot) => s.new_food[0] * GRID + s.new_food[1];
	const shade = (p: number, cls: number) => {
		const scale = cls === FOOD ? 18 : 1.15;
		return Math.min(1, Math.max(0.08, p * scale));
	};
</script>

<figure class="foodfig">
	{#if loadError}
		<p class="status">failed to load food snapshot: {loadError}</p>
	{:else if !snapshot}
		<p class="status">loading final-layer snapshot …</p>
	{:else}
		<div class="stack-wrap">
			<div class="stack">
				{#each classes as ch, z (ch.cls)}
					<div class="layer" style={`transform: translateZ(${z * 68}px)`}>
						<div class="grid plane">
							{#each snapshot.probs[ch.cls] as p, i (i)}
								<div
									class="cell"
									class:actual={ch.cls === FOOD && i === newFoodIndex(snapshot)}
									style={`background: rgba(17, 17, 17, ${shade(p, ch.cls)})`}
								></div>
							{/each}
						</div>
						<span class="tag">{ch.name}: mass {sum(snapshot.probs[ch.cls]).toFixed(2)}</span>
					</div>
				{/each}
			</div>
		</div>

		<figcaption>
			actual exported checkpoint, run on a transition where the snake has just eaten. The dashed
			cell is where the simulator put the next food. The food layer is not empty; it is smeared
			across the board. But decoding asks each cell to pick one class, so the sharper empty/body/head
			planes beat food everywhere.
		</figcaption>
	{/if}
</figure>

<style>
	.foodfig {
		margin: 2.5rem 0;
		font-family: var(--mono);
	}
	.status {
		margin: 0;
		font-size: 0.75rem;
		color: var(--faint);
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(10, 1fr);
		border: 1px solid var(--ink);
		aspect-ratio: 1;
	}
	.cell {
		position: relative;
		box-shadow: inset 0 0 0 0.5px #f0f0f0;
	}
	.actual::after {
		content: '';
		position: absolute;
		inset: 2px;
		border: 1px dashed var(--ink);
		pointer-events: none;
	}
	.stack-wrap {
		display: flex;
		justify-content: center;
		margin-top: 12rem;
		perspective: 1100px;
	}
	.stack {
		position: relative;
		width: 170px;
		height: 170px;
		transform-style: preserve-3d;
		transform: rotateX(58deg) rotateZ(-42deg);
		margin: 0 3rem 1rem 0;
	}
	.layer {
		position: absolute;
		inset: 0;
		transform-style: preserve-3d;
	}
	.plane {
		width: 170px;
		background: rgba(255, 255, 255, 0.88);
	}
	.tag {
		position: absolute;
		left: 104%;
		bottom: 2px;
		font-size: 0.58rem;
		text-transform: uppercase;
		letter-spacing: 0.14em;
		color: var(--ink);
		white-space: nowrap;
		font-variant-numeric: tabular-nums;
		transform: rotateZ(42deg) rotateX(-58deg);
		transform-origin: left center;
	}
	figcaption {
		margin-top: 1rem;
		font-size: 0.7rem;
		line-height: 1.6;
		color: var(--faint);
	}
</style>
