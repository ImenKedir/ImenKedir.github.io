import adapter from '@sveltejs/adapter-static';
import { mdsvex } from 'mdsvex';
import { mdsvexHighlighter } from './src/lib/highlight.js';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	extensions: ['.svelte', '.svx'],
	preprocess: [
		mdsvex({
			extensions: ['.svx'],
			highlight: { highlighter: mdsvexHighlighter },
			smartypants: { dashes: 'oldschool' }
		})
	],
	compilerOptions: {
		// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
		runes: ({ filename }) => (filename.split(/[/\\]/).includes('node_modules') ? undefined : true)
	},
	kit: {
		// SPA mode: every route is served via the fallback page (see +layout.ts ssr=false).
		// 404.html doubles as the SPA fallback on GitHub Pages.
		adapter: adapter({ fallback: '404.html' }),
		// Set by CI when deploying under a subpath (e.g. /repo-name on GitHub Pages).
		paths: { base: process.env.BASE_PATH || '' },
		// With ssr=false the crawler can't discover routes; list them explicitly.
		prerender: { entries: ['/', '/snake'] }
	}
};

export default config;
