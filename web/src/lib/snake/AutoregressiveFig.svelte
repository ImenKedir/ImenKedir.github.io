<script lang="ts">
	const G = 10;
	const EMPTY = 0, BODY = 1, HEAD = 2, FOOD = 3;

	const labels = new Uint8Array(G * G);
	labels[5 * G + 3] = BODY;
	labels[5 * G + 4] = BODY;
	labels[5 * G + 5] = HEAD;
	labels[2 * G + 7] = FOOD;

	const vocab = ['empty', 'body', 'head', 'food'];

	const frameClass = (value: number) =>
		value === BODY ? 'body' : value === HEAD ? 'head' : value === FOOD ? 'food' : 'empty';

	const tokenName = (value: number) =>
		value === BODY ? 'body' : value === HEAD ? 'head' : value === FOOD ? 'food' : 'empty';

	const tokenShort = (value: number) =>
		value === BODY ? 'B' : value === HEAD ? 'H' : value === FOOD ? 'F' : 'E';

	const rowMajorOrder = Array.from({ length: G * G }, (_, i) => i);

	const sequenceTokens = rowMajorOrder.map((cellIndex, step) => {
		const value = labels[cellIndex];
		return {
			label: `cell ${cellIndex + 1}`,
			value: tokenName(value),
			short: tokenShort(value),
			step,
			kind: frameClass(value)
		};
	});
</script>

<figure class="arfig">
	<div class="vocab">
		<div class="vocab-label">vocab</div>
		{#each vocab as token (token)}
			<span>{token}</span>
		{/each}
	</div>

	<div class="transform">
		<div class="board-wrap">
			<div class="grid">
				{#each labels as value, i (i)}
					<div class="cell {frameClass(value)}" style:--scan-index={i}></div>
				{/each}
			</div>
			<div class="label">10×10 board</div>
		</div>

		<div class="arrow">→</div>

		<div class="sequence-wrap">
			<div class="lane-label">row-major token sequence</div>
			<div class="sequence">
				{#each sequenceTokens as token, i (`token-${i}`)}
					<div
						aria-label={`${token.label}: ${token.value}`}
						class="token {token.kind}"
						style:--token-step={token.step}
						title={`${token.label}: ${token.value}`}
					>
						{token.short}
					</div>
				{/each}
			</div>
		</div>
	</div>

	<figcaption>
		One simple serialization: start in the top-left, scan left to right, and keep going row by
		row until the 10×10 board becomes 100 cell tokens.
	</figcaption>
</figure>

<style>
	.arfig {
		margin: 2.5rem 0;
		font-family: var(--mono);
	}
	.vocab {
		display: flex;
		align-items: center;
		gap: 0.45rem;
		flex-wrap: wrap;
		margin-bottom: 1.25rem;
	}
	.vocab-label,
	.lane-label {
		font-size: 0.58rem;
		text-transform: uppercase;
		letter-spacing: 0.16em;
		color: var(--faint);
	}
	.vocab span {
		border: 1px solid var(--ink);
		padding: 0.22rem 0.42rem;
		font-size: 0.62rem;
		background: #fff;
	}
	.transform {
		display: grid;
		grid-template-columns: auto 2rem minmax(0, 1fr);
		align-items: center;
		gap: 0.9rem;
	}
	.board-wrap {
		text-align: center;
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(10, 1fr);
		width: 122px;
		border: 1px solid var(--ink);
		aspect-ratio: 1;
		background: #fff;
	}
	.cell {
		box-shadow: inset 0 0 0 0.5px #f0f0f0;
		animation: scan-cell 5.6s linear infinite;
		animation-delay: calc(var(--scan-index) * 22ms);
	}
	.cell.body {
		background: var(--ink);
	}
	.cell.head {
		background: var(--ink);
		box-shadow: inset 0 0 0 2px var(--ink), inset 0 0 0 3.5px #fff;
	}
	.cell.food {
		box-shadow: inset 0 0 0 2px var(--ink);
	}
	.label {
		margin-top: 0.45rem;
		font-size: 0.58rem;
		text-transform: uppercase;
		letter-spacing: 0.16em;
		color: var(--faint);
	}
	.sequence-wrap {
		min-width: 0;
		width: 100%;
	}
	.sequence {
		display: grid;
		grid-template-columns: repeat(25, minmax(0, 1fr));
		gap: 0.16rem;
		margin-top: 0.45rem;
	}
	.token {
		border: 1px solid var(--ink);
		background: #fff;
		aspect-ratio: 1;
		display: grid;
		place-items: center;
		text-align: center;
		font-size: 0.52rem;
		color: var(--faint);
		opacity: 0.22;
		transform: translateY(2px);
		animation: reveal-token 5.6s linear infinite;
		animation-delay: calc(var(--token-step) * 22ms);
	}
	.token.body,
	.token.head {
		background: var(--ink);
		color: #fff;
	}
	.token.head {
		box-shadow: inset 0 0 0 2px var(--ink), inset 0 0 0 3.5px #fff;
	}
	.token.food {
		box-shadow: inset 0 0 0 2px var(--ink);
		color: var(--ink);
	}
	.arrow {
		text-align: center;
		font-size: 1.15rem;
		color: var(--faint);
	}
	figcaption {
		margin-top: 1rem;
		font-size: 0.7rem;
		line-height: 1.6;
		color: var(--faint);
	}
	@keyframes scan-cell {
		0%,
		7% {
			outline: 0 solid transparent;
			outline-offset: 0;
			filter: none;
		}
		8%,
		12% {
			outline: 2px solid var(--ink);
			outline-offset: -2px;
			filter: invert(1);
		}
		18%,
		100% {
			outline: 0 solid transparent;
			outline-offset: 0;
			filter: none;
		}
	}
	@keyframes reveal-token {
		0%,
		7% {
			opacity: 0.22;
			transform: translateY(3px);
		}
		8%,
		100% {
			opacity: 1;
			transform: translateY(0);
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.cell,
		.token {
			animation: none;
		}
		.token {
			opacity: 1;
			transform: none;
		}
	}
	@media (max-width: 680px) {
		.transform {
			grid-template-columns: 1fr;
			justify-items: center;
		}
		.arrow {
			transform: rotate(90deg);
		}
		.sequence-wrap {
			width: 100%;
		}
	}
</style>
