<script lang="ts">
	// A frame and its four one-hot channel planes. Purely illustrative.
	const G = 10;
	const EMPTY = 0, BODY = 1, HEAD = 2, FOOD = 3;

	const labels = new Uint8Array(G * G);
	labels[5 * G + 3] = BODY;
	labels[5 * G + 4] = BODY;
	labels[5 * G + 5] = HEAD;
	labels[2 * G + 7] = FOOD;

	const channels = [
		{ name: 'empty', cls: EMPTY },
		{ name: 'body', cls: BODY },
		{ name: 'head', cls: HEAD },
		{ name: 'food', cls: FOOD }
	];

	const frameClass = (v: number) =>
		v === BODY ? 'body' : v === HEAD ? 'head' : v === FOOD ? 'food' : 'empty';
</script>

<figure class="tensorfig">
	<div class="row">
		<div class="item">
			<div class="grid frame">
				{#each labels as v, i (i)}
					<div class="cell {frameClass(v)}"></div>
				{/each}
			</div>
			<div class="label">frame</div>
		</div>
		<div class="eq">=</div>
		{#each channels as ch (ch.cls)}
			<div class="item">
				<div class="grid plane">
					{#each labels as v, i (i)}
						<div class="cell {v === ch.cls ? 'on' : ''}"></div>
					{/each}
				</div>
				<div class="label">{ch.name}</div>
			</div>
		{/each}
	</div>
	<div class="stack-wrap">
		<div class="stack">
			{#each channels as ch, z (ch.cls)}
				<div class="layer" style="transform: translateZ({z * 30}px)">
					<div class="grid plane stacked">
						{#each labels as v, i (i)}
							<div class="cell {v === ch.cls ? 'on' : ''}"></div>
						{/each}
					</div>
					<span class="tag">{ch.name}</span>
				</div>
			{/each}
		</div>
	</div>
	<div class="legend">
		<span class="swatch on"></span><span>= 1</span>
		<span class="swatch"></span><span>= 0</span>
	</div>
	<figcaption>
		one frame, unstacked into its four one-hot planes: each cell is 1 in exactly one of them.
		stacked back up, they're the 10×10×4 tensor we can just pass directly into the model.
	</figcaption>
</figure>

<style>
	.tensorfig {
		margin: 2.5rem 0;
		font-family: var(--mono);
	}
	.row {
		display: flex;
		align-items: center;
		gap: 0.9rem;
		flex-wrap: wrap;
	}
	.item {
		text-align: center;
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(10, 1fr);
		border: 1px solid var(--ink);
		aspect-ratio: 1;
	}
	.frame {
		width: 130px;
	}
	.plane {
		width: 88px;
	}
	.cell {
		box-shadow: inset 0 0 0 0.5px #f0f0f0;
	}
	.frame .cell.body {
		background: var(--ink);
	}
	.frame .cell.head {
		background: var(--ink);
		box-shadow: inset 0 0 0 2px var(--ink), inset 0 0 0 3.5px #fff;
	}
	.frame .cell.food {
		box-shadow: inset 0 0 0 2px var(--ink);
	}
	.plane .cell.on {
		background: var(--ink);
	}
	.eq {
		font-size: 1rem;
		color: var(--faint);
	}
	.stack-wrap {
		display: flex;
		justify-content: center;
		margin-top: 2.5rem;
		perspective: 1100px;
	}
	.stack {
		position: relative;
		width: 150px;
		height: 150px;
		transform-style: preserve-3d;
		transform: rotateX(58deg) rotateZ(-42deg);
		margin: 1rem 2rem 3.5rem 0;
	}
	.layer {
		position: absolute;
		inset: 0;
		transform-style: preserve-3d;
	}
	.plane.stacked {
		width: 150px;
		background: rgba(255, 255, 255, 0.88);
	}
	.tag {
		position: absolute;
		left: 103%;
		bottom: 2px;
		font-size: 0.6rem;
		text-transform: uppercase;
		letter-spacing: 0.14em;
		color: var(--faint);
		white-space: nowrap;
	}
	.label {
		margin-top: 0.45rem;
		font-size: 0.62rem;
		text-transform: uppercase;
		letter-spacing: 0.16em;
		color: var(--faint);
	}
	.legend {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		margin-top: 1rem;
		font-size: 0.65rem;
		color: var(--faint);
	}
	.swatch {
		width: 10px;
		height: 10px;
		border: 1px solid var(--ink);
	}
	.swatch.on {
		background: var(--ink);
	}
	.legend span + .swatch {
		margin-left: 0.9rem;
	}
	figcaption {
		margin-top: 1rem;
		font-size: 0.7rem;
		line-height: 1.6;
		color: var(--faint);
	}
</style>
