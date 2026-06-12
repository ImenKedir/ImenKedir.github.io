<script lang="ts">
	import '../app.css';
	import favicon from '$lib/assets/favicon.svg';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<main>
	{@render children()}
	<footer>
		<a href="https://github.com/OWNER/world-model">code on github</a>
	</footer>
</main>

<style>
	footer {
		margin-top: 4rem;
		padding-top: 1.25rem;
		border-top: 1px solid var(--hairline);
	}
	footer a {
		font-family: var(--mono);
		font-size: 0.65rem;
		text-transform: uppercase;
		letter-spacing: 0.16em;
		color: var(--faint);
		text-decoration: none;
	}
	footer a:hover {
		color: var(--ink);
		text-decoration: underline;
		text-underline-offset: 3px;
	}
</style>
