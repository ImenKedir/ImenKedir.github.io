// Minimal grayscale Python highlighter — enough for short snippets, no deps.
// Plain JS so svelte.config.js (mdsvex) can import it too.

const KEYWORDS =
	/\b(class|def|return|for|in|if|else|elif|while|import|from|as|with|not|and|or|None|True|False|self|super|lambda|pass|raise|try|except)\b/g;

const esc = (/** @type {string} */ s) =>
	s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');

/** @param {string} code */
export function highlightPython(code) {
	// Tokenize comments and strings first so keywords inside them stay untouched.
	const parts = code.split(/(#[^\n]*|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g);
	return parts
		.map((/** @type {string} */ part) => {
			if (part.startsWith('#')) return `<span class="tok-c">${esc(part)}</span>`;
			if (part.startsWith('"') || part.startsWith("'"))
				return `<span class="tok-s">${esc(part)}</span>`;
			return esc(part)
				.replace(KEYWORDS, '<span class="tok-k">$1</span>')
				.replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="tok-n">$1</span>');
		})
		.join('');
}

/**
 * mdsvex highlighter: fenced code blocks, curlies escaped for Svelte.
 * @param {string} code
 * @param {string | undefined} lang
 */
export function mdsvexHighlighter(code, lang) {
	const html = lang === 'python' || lang === 'py' ? highlightPython(code) : esc(code);
	const safe = html.replaceAll('{', '&#123;').replaceAll('}', '&#125;');
	return `<pre><code>${safe}</code></pre>`;
}
