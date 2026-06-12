<script lang="ts">
	import { onMount } from 'svelte';
	import { SnakeEnv, greedyAction, mulberry32, GRID, EMPTY, BODY, HEAD, FOOD, UP, DOWN, LEFT, RIGHT } from './env';
	import { loadModel, decode, type WorldModel } from './model';
	import { base } from '$app/paths';

	const N = GRID * GRID;
	const TICK_MS = 170;
	const KEYMAP: Record<string, number> = {
		arrowup: UP, w: UP, arrowdown: DOWN, s: DOWN,
		arrowleft: LEFT, a: LEFT, arrowright: RIGHT, d: RIGHT
	};

	let model: WorldModel | null = $state(null);
	let loadError = $state('');

	const env = new SnakeEnv();
	let seed = $state(7);
	let real: number[] = $state([]);
	let dream: number[] = $state([]);
	let dreamLabels: Uint8Array<ArrayBufferLike> = new Uint8Array(N);

	let playing = $state(true);
	let autopilot = $state(true);
	let sampleFood = $state(false);
	let gameOver = $state(false);
	let step = $state(0);
	let tracked = $state(0);
	let dreamFoodGone = $state(false);
	let pendingAction = RIGHT;
	let dreamRng = mulberry32(1234);

	function reset(newSeed: number) {
		seed = newSeed;
		env.reset(seed);
		dreamLabels = env.labels();
		real = Array.from(env.labels());
		dream = Array.from(dreamLabels);
		step = 0;
		tracked = 0;
		gameOver = false;
		dreamFoodGone = false;
		pendingAction = RIGHT;
		dreamRng = mulberry32(seed * 7919 + 1);
		playing = true;
	}

	function tick() {
		if (!model || gameOver) return;
		const action = autopilot ? greedyAction(env.snake, env.food, env.direction) : pendingAction;

		const logits = model.forward(dreamLabels, action);
		dreamLabels = decode(logits, sampleFood, dreamRng);
		dreamFoodGone = !dreamLabels.includes(FOOD);

		const done = env.step(action);
		const rl = env.labels();

		step += 1;
		if (rl.indexOf(HEAD) === dreamLabels.indexOf(HEAD)) tracked += 1;

		real = Array.from(rl);
		dream = Array.from(dreamLabels);
		if (done) {
			gameOver = true;
			playing = false;
			setTimeout(() => {
				if (gameOver) reset(seed + 1);
			}, 1400);
		}
	}

	function onKey(e: KeyboardEvent) {
		const a = KEYMAP[e.key.toLowerCase()];
		if (a === undefined) return;
		e.preventDefault();
		autopilot = false;
		pendingAction = a;
		if (!playing && !gameOver) playing = true;
	}

	onMount(() => {
		reset(seed);
		loadModel(base).then(
			(m) => (model = m),
			(e) => (loadError = String(e))
		);
		const id = setInterval(() => playing && tick(), TICK_MS);
		return () => clearInterval(id);
	});

	const cellClass = (v: number) =>
		v === BODY ? 'body' : v === HEAD ? 'head' : v === FOOD ? 'food' : 'empty';
</script>

<svelte:window onkeydown={onKey} />

<figure class="demo">
	{#if loadError}
		<p class="status">failed to load model: {loadError}</p>
	{:else if !model}
		<p class="status">loading weights (4.8 MB) …</p>
	{:else}
		<div class="grids">
			<div class="panel">
				<div class="label">simulation</div>
				<div class="grid">
					{#each real as v, i (i)}
						<div class="cell {cellClass(v)}"></div>
					{/each}
				</div>
			</div>
			<div class="panel">
				<div class="label">dream</div>
				<div class="grid">
					{#each dream as v, i (i)}
						<div class="cell {cellClass(v)} {v !== real[i] ? 'diverged' : ''}"></div>
					{/each}
				</div>
			</div>
		</div>

		<div class="bar">
			<button onclick={() => (gameOver ? reset(seed + 1) : (playing = !playing))}>
				{gameOver ? 'reset' : playing ? 'pause' : 'play'}
			</button>
			<button onclick={() => reset(seed + 1)}>new game</button>
			<label><input type="checkbox" bind:checked={autopilot} /> autopilot</label>
			<label><input type="checkbox" bind:checked={sampleFood} /> sample food</label>
		</div>

		<div class="stats">
			<span>step {step}</span>
			<span>head tracked {tracked}/{step}</span>
			<span class:alert={dreamFoodGone}>
				{dreamFoodGone ? 'no food in dream' : 'food in dream'}
			</span>
			{#if gameOver}<span class="alert">game over</span>{/if}
		</div>
		<figcaption>
			arrows / wasd to steer (disables autopilot). the dream is fed only its own
			predictions. It never sees the simulation after step 0.
		</figcaption>
	{/if}
</figure>

<style>
	.demo {
		margin: 2.5rem 0;
		font-family: var(--mono);
	}
	.status {
		margin: 0;
		font-size: 0.75rem;
		color: var(--faint);
	}
	.grids {
		display: flex;
		gap: 1.5rem;
		flex-wrap: wrap;
	}
	.panel {
		flex: 1;
		min-width: 220px;
	}
	.label {
		text-transform: uppercase;
		font-size: 0.62rem;
		letter-spacing: 0.18em;
		color: var(--faint);
		margin-bottom: 0.5rem;
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(10, 1fr);
		border: 1px solid var(--ink);
		aspect-ratio: 1;
	}
	.cell {
		box-shadow: inset 0 0 0 0.5px #f0f0f0;
	}
	.cell.body {
		background: var(--ink);
	}
	.cell.head {
		background: var(--ink);
		box-shadow: inset 0 0 0 2px var(--ink), inset 0 0 0 4px #fff;
	}
	.cell.food {
		box-shadow: inset 0 0 0 2px var(--ink);
	}
	.cell.diverged {
		background-image: repeating-linear-gradient(
			45deg,
			#c9c9c9 0,
			#c9c9c9 1.5px,
			transparent 1.5px,
			transparent 5px
		);
	}
	.cell.body.diverged,
	.cell.head.diverged {
		background-color: #666;
	}
	.bar {
		display: flex;
		gap: 1.25rem;
		align-items: center;
		margin-top: 1.25rem;
		flex-wrap: wrap;
	}
	button {
		font: inherit;
		font-size: 0.68rem;
		text-transform: uppercase;
		letter-spacing: 0.14em;
		background: #fff;
		color: var(--ink);
		border: 1px solid var(--ink);
		padding: 0.32rem 0.85rem;
		cursor: pointer;
		transition: background 0.1s, color 0.1s;
	}
	button:hover {
		background: var(--ink);
		color: #fff;
	}
	label {
		font-size: 0.68rem;
		text-transform: uppercase;
		letter-spacing: 0.14em;
		display: flex;
		gap: 0.45rem;
		align-items: center;
		cursor: pointer;
	}
	input[type='checkbox'] {
		accent-color: #000;
		margin: 0;
	}
	.stats {
		display: flex;
		gap: 1.75rem;
		margin-top: 1rem;
		font-size: 0.68rem;
		text-transform: uppercase;
		letter-spacing: 0.14em;
		color: var(--faint);
		flex-wrap: wrap;
		font-variant-numeric: tabular-nums;
	}
	.alert {
		color: var(--ink);
		font-weight: 600;
	}
	figcaption {
		margin-top: 1rem;
		font-size: 0.7rem;
		line-height: 1.6;
		color: var(--faint);
	}
</style>
