<script lang="ts">
	import { base } from '$app/paths';
</script>

<svelte:head>
	<title>we have world models at home</title>
	<meta name="description" content="Notes from building world models, starting stupidly small." />
</svelte:head>

<p class="meta">"mom, can we get a world model?"</p>
<h1>We have world models at home</h1>
<p class="lede">
	Notes from building world models, starting stupidly small and scaling up until something
	interesting breaks.
</p>

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
