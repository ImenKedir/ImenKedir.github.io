<script lang="ts">
	import { base } from '$app/paths';
</script>

<svelte:head>
	<title>Imen Kedir</title>
	<meta
		name="description"
		content="Imen Kedir's personal site. Notes from tinkering with neural networks."
	/>
</svelte:head>

<h1>Imen Kedir</h1>
<p class="lede">
	I'm Imen. Stanford CS '25. I think neural networks are the coolest thing humanity has, and this
	is where I document my tinkering with them.
</p>
<ul class="links">
	<li><a href="https://x.com/ImenKedir">twitter</a></li>
	<li><a href="https://www.linkedin.com/in/imenkedir">linkedin</a></li>
</ul>

<h2>Posts</h2>
<ul class="posts">
	<li>
		<a href="{base}/snake">Trying to train a tiny world model</a>
		<span class="meta">June 2026</span>
	</li>
</ul>

<style>
	.lede {
		color: #444;
	}
	.links {
		list-style: none;
		display: flex;
		gap: 1.4rem;
		padding: 0;
		margin: 1.5rem 0 0;
	}
	.links li {
		margin: 0;
	}
	.links a {
		font-family: var(--mono);
		font-size: 0.68rem;
		text-transform: uppercase;
		letter-spacing: 0.16em;
		color: var(--faint);
		text-decoration: none;
	}
	.links a:hover {
		color: var(--ink);
		text-decoration: underline;
		text-underline-offset: 3px;
	}
	.posts {
		list-style: none;
		padding: 0;
		margin: 0;
	}
	.posts li {
		display: flex;
		justify-content: space-between;
		gap: 1.5rem;
		align-items: baseline;
		padding: 0.8rem 0;
		border-bottom: 1px solid var(--hairline);
		margin: 0;
	}
	.posts a {
		text-decoration: none;
	}
	.posts a:hover {
		text-decoration: underline;
		text-decoration-thickness: 1px;
		text-underline-offset: 3px;
	}
	.posts .meta {
		white-space: nowrap;
	}
</style>
