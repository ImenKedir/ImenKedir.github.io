<figure class="archfig">
	<svg viewBox="0 0 760 200" role="img" aria-label="MLP architecture diagram">
		<defs>
			<marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
				<path d="M0,0 L8,4 L0,8" fill="none" stroke="#111" stroke-width="1.1" />
			</marker>
		</defs>

		<!-- inputs -->
		<rect x="8" y="32" width="116" height="44" />
		<text x="66" y="50">frame</text>
		<text x="66" y="65" class="dim">10×10×4 = 400</text>

		<rect x="8" y="124" width="116" height="44" />
		<text x="66" y="142">action</text>
		<text x="66" y="157" class="dim">one-hot = 4</text>

		<!-- concat -->
		<rect x="186" y="78" width="84" height="44" />
		<text x="228" y="96">concat</text>
		<text x="228" y="111" class="dim">404</text>

		<!-- hidden -->
		<rect x="332" y="78" width="150" height="44" />
		<text x="407" y="96">Linear(512) + ReLU</text>
		<text x="407" y="111" class="dim">× 4 layers</text>

		<!-- output head -->
		<rect x="544" y="78" width="92" height="44" />
		<text x="590" y="96">Linear(400)</text>
		<text x="590" y="111" class="dim">logits</text>

		<!-- reshape -->
		<rect x="668" y="78" width="84" height="44" />
		<text x="710" y="96">reshape</text>
		<text x="710" y="111" class="dim">(4,10,10)</text>

		<!-- arrows -->
		<path d="M124,54 H156 V92 H180" marker-end="url(#arr)" />
		<path d="M124,146 H156 V108 H180" marker-end="url(#arr)" />
		<path d="M270,100 H326" marker-end="url(#arr)" />
		<path d="M482,100 H538" marker-end="url(#arr)" />
		<path d="M636,100 H662" marker-end="url(#arr)" />
	</svg>
	<figcaption>
		the whole model: concatenate, four hidden layers, read out 4 logits per cell, a softmax
		over {'{'}empty, body, head, food{'}'} for each of the 100 cells.
	</figcaption>
</figure>

<style>
	.archfig {
		margin: 2.5rem 0;
		font-family: var(--mono);
	}
	svg {
		width: 100%;
		height: auto;
		display: block;
	}
	rect {
		fill: #fff;
		stroke: var(--ink);
		stroke-width: 1;
	}
	path {
		fill: none;
		stroke: var(--ink);
		stroke-width: 1;
	}
	text {
		font-family: var(--mono);
		font-size: 11px;
		fill: var(--ink);
		text-anchor: middle;
	}
	text.dim {
		font-size: 10px;
		fill: var(--faint);
	}
	figcaption {
		margin-top: 1rem;
		font-size: 0.7rem;
		line-height: 1.6;
		color: var(--faint);
	}
</style>
