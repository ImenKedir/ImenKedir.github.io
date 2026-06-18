<script lang="ts">
	import { base } from '$app/paths';
	import { onMount } from 'svelte';

	type InputEvent = {
		timestampMs: number;
		type: string;
		key?: string;
		button?: string;
		x?: number;
		y?: number;
		deltaY?: number;
	};

	type Clip = {
		id: string;
		game?: string;
		video: string;
		dataUrl: string;
		eventsUrl: string;
		recordedAt?: string;
		durationMs?: number;
		resolution?: { width?: number; height?: number };
		fps?: number;
		codec?: string;
		videoSizeBytes?: number;
	};

	type Session = Clip & {
		game: string;
		description: string;
		durationMs: number;
		eventCount: number;
		resolution: { width?: number; height?: number };
		cursorBounds: { minX: number | null; maxX: number | null; minY: number | null; maxY: number | null };
		events: InputEvent[];
	};

	type DemoData = {
		sourceManifest: string;
		bucket: string;
		baseUrl: string;
		game: string;
		totalSessions: number;
		filesPerSession: string[];
		generatedFromSessions: string[];
		clipLibrary: Clip[];
		sessions: Session[];
	};

	let data = $state<DemoData | null>(null);
	let loadError = $state('');
	let selectedId = $state('');
	let loadedSessions = $state<Record<string, Session>>({});
	let search = $state('');
	let videoEl = $state<HTMLVideoElement | null>(null);
	let currentMs = $state(0);
	let durationMs = $state(0);
	let playing = $state(false);
	let animationFrame = 0;

	const selected = $derived(loadedSessions[selectedId] ?? data?.sessions[0] ?? null);
	const effectiveDuration = $derived(durationMs || selected?.durationMs || lastEventTime(selected));
	const filteredClips = $derived.by(() => {
		const clips = data?.clipLibrary ?? [];
		const query = search.trim().toLowerCase();
		const filtered = query ? clips.filter((clip) => clip.id.toLowerCase().includes(query)) : clips;
		return [...filtered].sort((a, b) => Number(isInteractive(b)) - Number(isInteractive(a)));
	});
	const activeEvents = $derived.by(() => recentEvents(selected, currentMs));
	const activeKeys = $derived.by(() => new Set(activeEvents.filter((event) => event.type === 'keydown').map((event) => event.key)));
	const activeMouse = $derived.by(() => new Set(activeEvents.filter((event) => event.type === 'mousedown').map((event) => event.button)));
	const cursor = $derived.by(() => latestCursor(selected, currentMs));
	const timelineEvents = $derived.by(() => {
		const events = (selected?.events ?? []).filter((event) => event.type !== 'mousemove');
		const stride = Math.max(1, Math.ceil(events.length / 900));
		return events.filter((_, index) => index % stride === 0);
	});

	onMount(() => {
		fetch(`${base}/data/league-demo.json`)
			.then((response) => {
				if (!response.ok) throw new Error(`HTTP ${response.status}`);
				return response.json();
			})
			.then((json: DemoData) => {
				data = json;
				loadedSessions = Object.fromEntries(json.sessions.map((session) => [session.id, session]));
				selectedId = json.generatedFromSessions[0] ?? json.sessions[0]?.id ?? '';
			})
			.catch((error) => (loadError = error instanceof Error ? error.message : String(error)));

		const tick = () => {
			if (videoEl) {
				currentMs = Math.round(videoEl.currentTime * 1000);
				playing = !videoEl.paused;
				if (Number.isFinite(videoEl.duration) && videoEl.duration > 0) durationMs = Math.round(videoEl.duration * 1000);
			}
			animationFrame = requestAnimationFrame(tick);
		};
		animationFrame = requestAnimationFrame(tick);
		return () => cancelAnimationFrame(animationFrame);
	});

	function selectClip(clip: Clip) {
		if (!loadedSessions[clip.id]) loadedSessions = { ...loadedSessions, [clip.id]: rawSession(clip, data?.game ?? 'League of Legends') };
		selectedId = clip.id;
		currentMs = 0;
		durationMs = 0;
		if (videoEl) {
			videoEl.pause();
			videoEl.currentTime = 0;
			videoEl.load();
		}
	}

	function rawSession(clip: Clip, game: string): Session {
		return {
			...clip,
			game: clip.game ?? game,
			description: 'Raw bucket clip. Video and package links are available; keyboard and mouse overlays are preprocessed for highlighted sessions.',
			durationMs: clip.durationMs ?? 0,
			eventCount: 0,
			resolution: clip.resolution ?? {},
			cursorBounds: { minX: null, maxX: null, minY: null, maxY: null },
			events: []
		};
	}

	function isInteractive(clip: Clip) {
		return Boolean(loadedSessions[clip.id]?.events.length);
	}

	function recentEvents(session: Session | null, timestampMs: number) {
		if (!session) return [];
		return session.events.filter((event) => event.type !== 'mousemove' && event.timestampMs >= timestampMs - 900 && event.timestampMs <= timestampMs + 120).slice(-12);
	}

	function latestCursor(session: Session | null, timestampMs: number) {
		if (!session) return null;
		for (let index = session.events.length - 1; index >= 0; index -= 1) {
			const event = session.events[index];
			if (event.timestampMs > timestampMs || event.type !== 'mousemove' || typeof event.x !== 'number' || typeof event.y !== 'number') continue;
			const bounds = session.cursorBounds;
			const minX = bounds.minX ?? 0;
			const maxX = bounds.maxX ?? session.resolution.width ?? minX + 1;
			const minY = bounds.minY ?? 0;
			const maxY = bounds.maxY ?? session.resolution.height ?? minY + 1;
			return { left: clamp(((event.x - minX) / Math.max(1, maxX - minX)) * 100, 0, 100), top: clamp(((event.y - minY) / Math.max(1, maxY - minY)) * 100, 0, 100) };
		}
		return null;
	}

	function seekTo(percent: number) {
		if (!videoEl || !effectiveDuration) return;
		videoEl.currentTime = (percent / 100) * (effectiveDuration / 1000);
	}

	function jump(seconds: number) {
		if (!videoEl) return;
		videoEl.currentTime = clamp(videoEl.currentTime + seconds, 0, effectiveDuration / 1000);
	}

	function togglePlayback() {
		if (!videoEl) return;
		if (videoEl.paused) videoEl.play();
		else videoEl.pause();
	}

	function lastEventTime(session: Session | null) {
		return session?.events.at(-1)?.timestampMs ?? 0;
	}

	function clamp(value: number, min: number, max: number) {
		return Math.min(max, Math.max(min, value));
	}

	function formatTime(ms = 0) {
		const total = Math.max(0, Math.floor(ms / 1000));
		return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
	}

	function formatDuration(ms = 0) {
		return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
	}

	function formatNumber(value?: number) {
		return value === undefined ? '—' : value.toLocaleString();
	}

	function formatBytes(bytes?: number) {
		return bytes ? `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB` : '—';
	}
</script>

<svelte:head>
	<title>Gameplay data demo</title>
	<meta name="description" content="Interactive data demo for consented gameplay trajectories with synchronized screen, keyboard, mouse, and metadata." />
</svelte:head>

<section class="page">
	<header class="hero">
		<p class="eyebrow">Gameplay telemetry sample</p>
		<h1>Consented gameplay trajectories: synchronized screen video, keyboard, mouse, and metadata from real players.</h1>
		<p class="note">Built on <a href="https://www.tryascent.gg/">Ascent</a>, our game recording software with a consented user community, a 2,000-member Discord, 5,000–6,000 daily active users, and support for recording many different games already.</p>
		<div class="metrics">
			<div><strong>5–6k</strong><span>daily active users</span></div>
			<div><strong>30k</strong><span>total users</span></div>
			<div><strong>2k</strong><span>Discord members</span></div>
		</div>
	</header>

	<section class="viewer">
		<div class="section-heading">
			<p class="eyebrow">Synchronized viewer</p>
			<h2>Video and input stream inspectable at frame time</h2>
		</div>

		{#if loadError}
			<div class="state">Failed to load sample data: {loadError}</div>
		{:else if !data || !selected}
			<div class="state">Loading sample manifest and normalized input events…</div>
		{:else}
			<div class="viewer-grid">
				<div class="player-card">
					<div class="toolbar">
						<div><p class="label">Playback clock</p><strong>{formatTime(currentMs)} / {formatTime(effectiveDuration)}</strong></div>
						<div class="status"><i class:live={playing}></i>{selected.events.length ? (playing ? 'streaming synchronized events' : 'paused at timestamp') : 'raw clip selected; event file linked'}</div>
					</div>
					<div class="video-shell">
						<video bind:this={videoEl} src={selected.video} controls preload="metadata"><track kind="captions" /></video>
						{#if cursor}<span class="cursor" style={`left:${cursor.left}%;top:${cursor.top}%`}></span>{/if}
					</div>
					<div class="transport">
						<button type="button" class="primary" onclick={togglePlayback}>{playing ? 'Pause' : 'Play'}</button>
						<button type="button" onclick={() => jump(-5)}>−5s</button>
						<button type="button" onclick={() => jump(5)}>+5s</button>
						<input type="range" min="0" max="100" step="0.1" value={(currentMs / Math.max(1, effectiveDuration)) * 100} oninput={(event) => seekTo(Number(event.currentTarget.value))} aria-label="seek through synchronized recording" />
					</div>
					<div class="inputs">
						<div><p class="label">Keyboard</p><div class="keyboard">{#each ['Q','W','E','R','A','S','D','F','Ctrl','Shift','Space'] as key}<span class:active={activeKeys.has(key)}>{key}</span>{/each}</div></div>
						<div><p class="label">Mouse</p><div class="mouse">{#each ['Left','Middle','Right'] as button}<span class:active={activeMouse.has(button)}>{button}</span>{/each}</div></div>
						<div><p class="label">Cursor</p><div class="cursor-grid">{#if cursor}<span style={`left:${cursor.left}%;top:${cursor.top}%`}></span>{/if}</div></div>
					</div>
					<div class="timeline">
						<i style={`left:${(currentMs / Math.max(1, effectiveDuration)) * 100}%`}></i>
						{#each timelineEvents as event, index (`${event.timestampMs}-${event.type}-${index}`)}<button type="button" style={`left:${(event.timestampMs / Math.max(1, effectiveDuration)) * 100}%`} title={`${formatTime(event.timestampMs)} · ${event.type}`} onclick={() => seekTo((event.timestampMs / Math.max(1, effectiveDuration)) * 100)}></button>{/each}
					</div>
				</div>
				<aside class="metadata">
					<p class="label">Selected session</p>
					<h3>{selected.game}</h3>
					<p>{selected.description}</p>
					<dl>
						<div><dt>Session ID</dt><dd>{selected.id}</dd></div>
						<div><dt>Duration</dt><dd>{formatDuration(selected.durationMs)}</dd></div>
						<div><dt>Input events</dt><dd>{selected.eventCount ? formatNumber(selected.eventCount) : 'raw file linked'}</dd></div>
						<div><dt>Resolution / FPS</dt><dd>{selected.resolution.width ?? '—'}×{selected.resolution.height ?? '—'} / {selected.fps ?? '—'}</dd></div>
						<div><dt>Video</dt><dd>{selected.codec ?? '—'}, {formatBytes(selected.videoSizeBytes)}</dd></div>
						<div><dt>Recorded</dt><dd>{selected.recordedAt ?? '—'}</dd></div>
					</dl>
					<a href={selected.dataUrl}>metadata.json</a><a href={selected.eventsUrl}>events.csv.gz</a>
				</aside>
			</div>
		{/if}
	</section>

	{#if data}
		<section class="library">
			<div class="section-heading"><p class="eyebrow">Session selector</p><h2>Select any clip in the sample bucket</h2><p>Choose a session to load it into the viewer. Interactive sessions are sorted first; every row links to the raw package files.</p></div>
			<div class="library-tools"><label>Filter <input bind:value={search} type="search" placeholder="session id" /></label><span>{formatNumber(filteredClips.length)} / {formatNumber(data.totalSessions)} clips</span></div>
			<div class="clip-table">
				<div class="clip-row head"><span>Session</span><span>Telemetry</span><span>Video</span><span>Events</span><span>Metadata</span></div>
				{#each filteredClips as clip (clip.id)}
					<div class:selected={clip.id === selectedId} class="clip-row">
						<button type="button" onclick={() => selectClip(clip)}><strong>{clip.id}</strong><small>{formatDuration(clip.durationMs)} · {clip.resolution?.width ?? '—'}×{clip.resolution?.height ?? '—'} · {clip.fps ?? '—'} fps</small></button>
						<span class:interactive={isInteractive(clip)}>{isInteractive(clip) ? 'interactive' : 'raw package'}</span>
						<a href={clip.video}>video.mp4</a><a href={clip.eventsUrl}>events.csv.gz</a><a href={clip.dataUrl}>data.json</a>
					</div>
				{/each}
			</div>
		</section>

		<section class="schema">
			<div class="section-heading"><p class="eyebrow">Delivered format</p><h2>Dataset schema</h2><p>One folder per session: screen recording, session metadata, and timestamped input events.</p></div>
			<div class="schema-grid"><div><p class="label">Per-session package</p><ul><li>video.mp4 <small>screen recording</small></li><li>events.csv.gz <small>input stream</small></li><li>data.json <small>session metadata</small></li><li>manifest.json <small>bucket index</small></li></ul></div><div><p class="label">Event fields</p><table><tbody><tr><td>timestamp_ms</td><td>event time relative to recording start</td></tr><tr><td>event_type</td><td>key_down, key_up, mouse_button_down, cursor_pos, wheel</td></tr><tr><td>key/button</td><td>normalized key label, mouse button, or raw code</td></tr><tr><td>mouse_x / mouse_y</td><td>raw desktop cursor coordinates when present</td></tr><tr><td>session_id</td><td>public session identifier from manifest</td></tr></tbody></table></div></div>
			<p class="source">Source bucket: <a href={data.baseUrl}>{data.bucket}</a> · <a href={data.sourceManifest}>manifest.json</a> · {formatNumber(data.totalSessions)} sessions</p>
		</section>
	{/if}

	<section class="capacity"><div><h2>Data Quality & Provenance</h2><ul><li>First-party collection through Ascent.</li><li>User-consented recording flow.</li><li>2,000-member Discord/community.</li><li>Synchronized screen, keyboard, and mouse streams.</li><li>Session-level metadata and raw files.</li></ul></div><div><h2>Acquisition Capacity</h2><div class="capacity-card"><div class="mini-metrics"><strong>5–6k</strong><strong>30k</strong><strong>2k</strong><span>DAU</span><span>total users</span><span>Discord</span></div><p>This is already a meaningful consented acquisition channel, not a cold-start collection effort.</p></div></div></section>
</section>

<style>
	:global(main){max-width:62rem;padding:0 1.5rem 7rem}.page{color:var(--ink)}.hero{margin:3.5rem 0 3rem}.eyebrow,.label{margin:0 0 .75rem;font-family:var(--mono);font-size:.68rem;letter-spacing:.16em;text-transform:uppercase;color:var(--faint)}h1{max-width:58rem;margin:0;font-size:clamp(2rem,5vw,3.6rem);line-height:1.05;letter-spacing:-.04em}.note{max-width:47rem;margin:1rem 0 0;padding-left:1rem;border-left:1px solid var(--ink);color:#444}.note a,.metadata a,.clip-row a,.source a{color:var(--ink)}.metrics{display:grid;grid-template-columns:repeat(3,1fr);max-width:47rem;margin-top:1.4rem;border:1px solid var(--ink);background:var(--ink);color:#fff}.metrics div{padding:1rem;border-left:1px solid #444}.metrics div:first-child{border-left:0}.metrics strong{display:block;font-size:clamp(2rem,5vw,3.2rem);line-height:.95;letter-spacing:-.05em}.metrics span{display:block;margin-top:.55rem;font-family:var(--mono);font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#ddd}.viewer,.library,.schema,.capacity{margin:0;padding-top:4rem}.viewer{padding-top:0;border-top:1px solid var(--ink)}.section-heading{padding:1.25rem 0 0}.section-heading h2{margin:0 0 .7rem;font-family:var(--mono);font-size:.72rem;font-weight:500;letter-spacing:.18em;text-transform:uppercase}.section-heading p:not(.eyebrow){max-width:48rem;color:#444;line-height:1.65}.viewer-grid{display:grid;grid-template-columns:minmax(0,1fr) 18rem;gap:1rem;padding-top:1.25rem}.player-card,.metadata,.state,.schema-grid>div,.capacity-card{border:1px solid var(--ink);background:#fff}.toolbar{display:flex;justify-content:space-between;gap:1rem;padding:.85rem 1rem;border-bottom:1px solid var(--hairline)}.toolbar strong,.status,.transport button,.library-tools,.clip-row,.metadata dl,.source{font-family:var(--mono)}.status{display:flex;align-items:center;gap:.45rem;font-size:.66rem;letter-spacing:.12em;text-transform:uppercase;color:var(--faint)}.status i{width:.46rem;height:.46rem;border:1px solid var(--ink);border-radius:50%}.status i.live{background:var(--ink)}.video-shell{position:relative;aspect-ratio:16/9;background:#000;overflow:hidden}video{display:block;width:100%;height:100%;object-fit:contain}.cursor{position:absolute;width:1rem;height:1rem;transform:translate(-50%,-50%);border:1px solid #fff;border-radius:50%;box-shadow:0 0 0 4px rgba(0,0,0,.35)}.transport,.inputs{display:flex;gap:.65rem;align-items:center;flex-wrap:wrap;padding:.85rem 1rem;border-top:1px solid var(--hairline)}.transport input{flex:1;accent-color:var(--ink)}button{font:inherit;cursor:pointer}.transport button{border:1px solid var(--ink);background:#fff;padding:.36rem .62rem;font-size:.66rem;letter-spacing:.14em;text-transform:uppercase}.transport .primary,.transport button:hover{background:var(--ink);color:#fff}.inputs{align-items:stretch;background:#fafafa}.inputs>div{flex:1;min-width:12rem;border-left:1px solid var(--hairline);padding-left:.85rem}.inputs>div:first-child{border-left:0;padding-left:0}.keyboard,.mouse{display:grid;grid-template-columns:repeat(4,1fr);gap:.18rem;padding:.32rem;border:1px solid var(--hairline);background:#fff}.keyboard span,.mouse span{display:flex;align-items:center;justify-content:center;min-height:1.25rem;border:1px solid var(--ink);font-family:var(--mono);font-size:.55rem}.keyboard .active,.mouse .active{background:var(--ink);color:#fff}.cursor-grid{position:relative;aspect-ratio:16/9;height:4.3rem;border:1px solid var(--hairline);background:linear-gradient(90deg,#eee 1px,transparent 1px) 0 0/25% 100%,linear-gradient(180deg,#eee 1px,transparent 1px) 0 0/100% 50%,#fff}.cursor-grid span{position:absolute;width:.56rem;height:.56rem;transform:translate(-50%,-50%);border-radius:50%;background:var(--ink)}.timeline{position:relative;height:3.7rem;margin:0 1rem 1rem;border:1px solid var(--ink);background:linear-gradient(90deg,#eee 1px,transparent 1px) 0 0/10% 100%,#fff}.timeline i{position:absolute;top:0;bottom:0;width:2px;background:var(--ink);z-index:2}.timeline button{position:absolute;bottom:.8rem;width:4px;height:1.45rem;padding:0;border:0;background:var(--ink);transform:translateX(-50%);opacity:.86}.metadata{padding:1rem}.metadata h3{margin:.2rem 0 .4rem}.metadata p{color:#444;line-height:1.55}.metadata dl div{display:grid;grid-template-columns:6.5rem 1fr;gap:.8rem;padding:.48rem 0;border-bottom:1px solid var(--hairline)}dt{color:var(--faint)}dd{margin:0;word-break:break-word}.library{padding-top:1.5rem;border-top:1px solid var(--hairline)}.library-tools{display:flex;justify-content:space-between;gap:1rem;margin:.8rem 0 .75rem;font-size:.66rem;color:var(--faint)}.library-tools label{display:flex;align-items:center;gap:.6rem;letter-spacing:.14em;text-transform:uppercase}.library-tools input{border:1px solid var(--ink);padding:.42rem .55rem;font-family:var(--mono)}.clip-table{max-height:21rem;overflow:auto;border:1px solid var(--ink)}.clip-row{display:grid;grid-template-columns:minmax(12rem,1fr) auto repeat(3,auto);gap:1rem;align-items:center;min-width:46rem;padding:.58rem .75rem;border-top:1px solid var(--hairline);font-size:.7rem}.clip-row.head{position:sticky;top:0;background:var(--ink);color:#fff;text-transform:uppercase;letter-spacing:.14em}.clip-row.selected{background:#f1f1f1;box-shadow:inset 3px 0 0 var(--ink)}.clip-row button{display:grid;gap:.15rem;padding:0;border:0;background:transparent;text-align:left}.clip-row small{color:var(--faint)}.clip-row span.interactive{background:var(--ink);color:#fff}.clip-row>span{padding:.22rem .4rem;border:1px solid var(--hairline);font-size:.58rem;text-transform:uppercase;text-align:center}.schema-grid{display:grid;grid-template-columns:17rem 1fr;gap:1rem}.schema-grid>div{padding:1rem}.schema ul{margin:0;padding:0;list-style:none;border-top:1px solid var(--ink)}.schema li{display:grid;padding:.7rem 0;border-bottom:1px solid var(--hairline);font-family:var(--mono)}.schema small{color:var(--faint);text-transform:uppercase;font-size:.58rem}.schema table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.7rem}.schema td{padding:.58rem;border-bottom:1px solid var(--hairline)}.capacity{display:grid;grid-template-columns:1fr 1fr;gap:2rem}.capacity li{margin:.45rem 0;line-height:1.55}.capacity-card{padding:1rem}.mini-metrics{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid var(--ink);text-align:center}.mini-metrics>*{padding:.55rem;border-left:1px solid var(--hairline);font-family:var(--mono)}.mini-metrics>*:nth-child(1),.mini-metrics>*:nth-child(4){border-left:0}.mini-metrics strong{font-size:1.5rem}.mini-metrics span{font-size:.55rem;text-transform:uppercase;color:var(--faint)}@media(max-width:920px){.viewer-grid,.schema-grid,.capacity{grid-template-columns:1fr}.inputs>div{min-width:100%;border-left:0;border-top:1px solid var(--hairline);padding:.75rem 0 0}.inputs>div:first-child{border-top:0;padding-top:0}}@media(max-width:580px){.metrics{grid-template-columns:1fr}.metrics div{border-left:0;border-top:1px solid #444}.metrics div:first-child{border-top:0}}
</style>
