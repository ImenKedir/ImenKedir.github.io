<script lang="ts">
	import { base } from '$app/paths';
	import { onMount } from 'svelte';

	type InputEvent = {
		timestampMs: number;
		type: 'keydown' | 'keyup' | 'mousedown' | 'mouseup' | 'mousemove' | 'wheel' | string;
		key?: string;
		button?: string;
		x?: number;
		y?: number;
		deltaX?: number;
		deltaY?: number;
		rawInput?: string;
		sessionId: string;
		game: string;
	};

	type DemoSession = {
		id: string;
		game: string;
		gameType?: string;
		gameName?: string;
		source?: string;
		description: string;
		recordedAt?: string;
		durationMs: number;
		eventCount: number;
		normalizedEventCount: number;
		video: string;
		dataUrl: string;
		eventsUrl: string;
		resolution: { width?: number; height?: number };
		fps?: number;
		codec?: string;
		videoSizeBytes?: number;
		captureMethod?: string;
		droppedFrames?: number;
		laggedFrames?: number;
		sessionIndex?: number;
		champion?: string;
		queue?: string;
		rawEventCounts: Record<string, number>;
		cursorBounds: { minX: number | null; maxX: number | null; minY: number | null; maxY: number | null };
		events: InputEvent[];
	};

	type ClipSummary = {
		id: string;
		video: string;
		dataUrl: string;
		eventsUrl: string;
		game?: string;
		gameType?: string;
		gameName?: string;
		source?: string;
		recordedAt?: string;
		durationMs?: number;
		resolution?: { width?: number; height?: number };
		fps?: number;
		codec?: string;
		videoSizeBytes?: number;
	};

	type DemoData = {
		sourceManifest: string;
		bucket: string;
		baseUrl: string;
		game: string;
		source: string;
		totalSessions: number;
		filesPerSession: string[];
		generatedFromSessions: string[];
		clipLibrary?: ClipSummary[];
		notes: string[];
		sessions: DemoSession[];
	};

	let data = $state<DemoData | null>(null);
	let loadError = $state('');
	let selectedSessionId = $state('');
	let loadedSessions = $state<Record<string, DemoSession>>({});
	let currentTimeMs = $state(0);
	let videoDurationMs = $state(0);
	let playing = $state(false);
	let videoEl = $state<HTMLVideoElement | null>(null);
	let seekTarget = $state(0);
	let clipSearch = $state('');
	let animationFrame = 0;
	const gameLogoByType: Record<string, string> = {
		LOL: 'lol.png',
		TFT: 'tft.png',
		VAL: 'valorant.png',
		CS2: 'cs2.png',
		DL: 'deadlock.png',
		MR: 'marvel-rivals.png',
		R6: 'r6-siege.png'
	};
	const gameLogoByName: Record<string, string> = {
		'League of Legends': 'lol.png',
		'Teamfight Tactics': 'tft.png',
		Valorant: 'valorant.png',
		'Counter-Strike 2': 'cs2.png',
		Deadlock: 'deadlock.png',
		'Marvel Rivals': 'marvel-rivals.png',
		'Rainbow Six Siege': 'r6-siege.png',
		'R6 Siege': 'r6-siege.png',
		CS2: 'cs2.png',
		TFT: 'tft.png'
	};
	const gameLogos = [
		['League of Legends', 'lol.png'],
		['Valorant', 'valorant.png'],
		['CS2', 'cs2.png'],
		['Apex Legends', 'apex.png'],
		['Dota 2', 'dota2.png'],
		['Rocket League', 'rocket-league.png'],
		['Elden Ring', 'elden-ring.png'],
		["Baldur's Gate 3", 'bg3.png'],
		['Helldivers 2', 'helldivers2.png'],
		['Cyberpunk 2077', 'cyberpunk.png'],
		['Overwatch 2', 'overwatch2.png'],
		['Destiny 2', 'destiny2.png'],
		['GTA V', 'gtav.png'],
		['Deadlock', 'deadlock.png'],
		['Palworld', 'palworld.png'],
		['Marvel Rivals', 'marvel-rivals.png'],
		['Forza Horizon 5', 'forza-horizon5.png'],
		['Monster Hunter', 'monster-hunter.png'],
		['TFT', 'tft.png'],
		['Rust', 'rust.png'],
		['PUBG', 'pubg.png'],
		['R6 Siege', 'r6-siege.png'],
		['Fall Guys', 'fall-guys.png'],
		['The Witcher 3', 'witcher3.png'],
		['Terraria', 'terraria.png'],
		['Among Us', 'among-us.png'],
		['Halo Infinite', 'halo-infinite.png'],
		['Deep Rock Galactic', 'deep-rock.png'],
		['Path of Exile', 'path-of-exile.png'],
		['Sea of Thieves', 'sea-of-thieves.png'],
		['Lethal Company', 'lethal-company.png'],
		['Warframe', 'warframe.png'],
		['War Thunder', 'war-thunder.png'],
		["No Man's Sky", 'no-mans-sky.png'],
		['Dark Souls III', 'dark-souls3.png']
	];
	const keyboardRows = [
		['Q', 'W', 'E', 'R', 'T'],
		['A', 'S', 'D', 'F', 'G'],
		['Z', 'X', 'C', 'V', 'B'],
		['Ctrl', 'Shift', 'Alt', 'Space']
	];
	const mouseButtonLabels = ['Left', 'Middle', 'Right'];

	const sessions = $derived(data?.sessions ?? []);
	const clipLibrary = $derived(data?.clipLibrary ?? []);
	const filteredClips = $derived.by(() => {
		const query = clipSearch.trim().toLowerCase();
		const clips = query
			? clipLibrary.filter((clip) => {
					const haystack = `${clip.id} ${gameLabel(clip)} ${clip.gameType ?? ''}`.toLowerCase();
					return haystack.includes(query);
				})
			: clipLibrary;
		return prioritizeGameDiversity(clips);
	});
	const selectedSession = $derived(loadedSessions[selectedSessionId] ?? sessions[0] ?? null);
	const effectiveDurationMs = $derived(
		videoDurationMs || selectedSession?.durationMs || lastEventTime(selectedSession)
	);
	const activeEvents = $derived.by(() => findActiveEvents(selectedSession, currentTimeMs));
	const activeKeys = $derived.by(() => pressedKeyLabels(selectedSession, currentTimeMs));
	const pressedMouseButtons = $derived.by(() => pressedMouseButtonLabels(selectedSession, currentTimeMs));
	const recentMouseButtons = $derived.by(() => recentMouseButtonLabels(activeEvents, currentTimeMs));
	const cursor = $derived.by(() => latestCursor(selectedSession, currentTimeMs));
	const cursorTrail = $derived.by(() => recentCursorTrail(selectedSession, currentTimeMs));
	const timelineEvents = $derived.by(() => {
		const nonMouseEvents = (selectedSession?.events ?? []).filter((event) => event.type !== 'mousemove');
		const maxMarkers = 900;
		const stride = Math.max(1, Math.ceil(nonMouseEvents.length / maxMarkers));
		return nonMouseEvents.filter((_, index) => index % stride === 0);
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
				selectedSessionId = json.generatedFromSessions[0] ?? json.sessions[0]?.id ?? '';
			})
			.catch((error) => {
				loadError = error instanceof Error ? error.message : String(error);
			});

		const update = () => {
			if (videoEl) {
				currentTimeMs = Math.round(videoEl.currentTime * 1000);
				playing = !videoEl.paused;
				if (Number.isFinite(videoEl.duration) && videoEl.duration > 0) {
					seekTarget = (videoEl.currentTime / videoEl.duration) * 100;
				}
			}
			animationFrame = requestAnimationFrame(update);
		};
		animationFrame = requestAnimationFrame(update);
		return () => cancelAnimationFrame(animationFrame);
	});

	function selectClip(clip: ClipSummary) {
		if (!loadedSessions[clip.id]) {
			loadedSessions = {
				...loadedSessions,
				[clip.id]: clipToSession(clip, data?.game ?? 'Gameplay')
			};
		}
		selectedSessionId = clip.id;
		currentTimeMs = 0;
		videoDurationMs = 0;
		seekTarget = 0;
		if (videoEl) {
			videoEl.pause();
			videoEl.currentTime = 0;
			videoEl.load();
		}
	}

	function clipToSession(clip: ClipSummary, game: string): DemoSession {
		return {
			id: clip.id,
			game: clip.game ?? game,
			gameType: clip.gameType,
			gameName: clip.gameName,
			source: clip.source,
			description: `${gameLabel(clip)} gameplay capture. Video and package links are available; keyboard and mouse overlays are preprocessed for highlighted sessions.`,
			recordedAt: clip.recordedAt,
			durationMs: clip.durationMs ?? 0,
			eventCount: 0,
			normalizedEventCount: 0,
			video: clip.video,
			dataUrl: clip.dataUrl,
			eventsUrl: clip.eventsUrl,
			resolution: clip.resolution ?? {},
			fps: clip.fps,
			codec: clip.codec,
			videoSizeBytes: clip.videoSizeBytes,
			rawEventCounts: {},
			cursorBounds: { minX: null, maxX: null, minY: null, maxY: null },
			events: []
		};
	}

	function hasInteractiveTelemetry(clip: ClipSummary) {
		return Boolean(loadedSessions[clip.id]?.events.length);
	}

	function gameLabel(item: Pick<ClipSummary, 'game' | 'gameName'>) {
		return item.gameName ?? item.game ?? 'Gameplay';
	}

	function gameLogo(item: Pick<ClipSummary, 'game' | 'gameName' | 'gameType'>) {
		const fileName = item.gameType ? gameLogoByType[item.gameType] : undefined;
		const fallbackFileName = gameLogoByName[gameLabel(item)];
		const logo = fileName ?? fallbackFileName;
		return logo ? `https://www.tryascent.gg/games/${logo}` : '';
	}

	function prioritizeGameDiversity(clips: ClipSummary[]) {
		const interactive = clips.filter(hasInteractiveTelemetry);
		const raw = clips.filter((clip) => !hasInteractiveTelemetry(clip));
		return [...roundRobinByGame(interactive), ...roundRobinByGame(raw)];
	}

	function roundRobinByGame(clips: ClipSummary[]) {
		const groups = new Map<string, ClipSummary[]>();
		for (const clip of clips) {
			const label = gameLabel(clip);
			const group = groups.get(label) ?? [];
			group.push(clip);
			groups.set(label, group);
		}
		const result: ClipSummary[] = [];
		let added = true;
		while (added) {
			added = false;
			for (const group of groups.values()) {
				const clip = group.shift();
				if (clip) {
					result.push(clip);
					added = true;
				}
			}
		}
		return result;
	}

	function onLoadedMetadata() {
		if (!videoEl) return;
		videoDurationMs = Math.round(videoEl.duration * 1000);
		seekTarget = 0;
	}

	function seekTo(percent: number) {
		if (!videoEl || !Number.isFinite(percent) || !effectiveDurationMs) return;
		const nextMs = Math.max(0, Math.min(effectiveDurationMs, (percent / 100) * effectiveDurationMs));
		videoEl.currentTime = nextMs / 1000;
		currentTimeMs = Math.round(nextMs);
	}

	function togglePlayback() {
		if (!videoEl) return;
		if (videoEl.paused) videoEl.play();
		else videoEl.pause();
	}

	function jump(ms: number) {
		if (!videoEl) return;
		const next = Math.max(0, Math.min((effectiveDurationMs || 0) / 1000, videoEl.currentTime + ms / 1000));
		videoEl.currentTime = next;
		currentTimeMs = Math.round(next * 1000);
	}

	function pressedKeyLabels(session: DemoSession | null, timestampMs: number) {
		const pressed = new Set<string>();
		if (!session) return pressed;
		const keyboardLabels = new Set(keyboardRows.flat());
		const seen = new Set<string>();
		const start = Math.min(session.events.length - 1, lowerBound(session.events, timestampMs + 80));
		for (let index = start; index >= 0; index -= 1) {
			const event = session.events[index];
			if (timestampMs - event.timestampMs > 10000 || seen.size >= keyboardLabels.size) break;
			if (event.type !== 'keydown' && event.type !== 'keyup') continue;
			const key = event.key;
			if (!key || !keyboardLabels.has(key) || seen.has(key)) continue;
			seen.add(key);
			if (event.type === 'keydown') pressed.add(key);
		}
		return pressed;
	}

	function pressedMouseButtonLabels(session: DemoSession | null, timestampMs: number) {
		const pressed = new Set<string>();
		if (!session) return pressed;
		const seen = new Set<string>();
		const start = Math.min(session.events.length - 1, lowerBound(session.events, timestampMs + 80));
		for (let index = start; index >= 0; index -= 1) {
			const event = session.events[index];
			if (timestampMs - event.timestampMs > 10000 || seen.size === mouseButtonLabels.length) break;
			if (event.type !== 'mousedown' && event.type !== 'mouseup') continue;
			const button = event.button;
			if (!button || seen.has(button)) continue;
			seen.add(button);
			if (event.type === 'mousedown') pressed.add(button);
		}
		return pressed;
	}

	function recentMouseButtonLabels(events: InputEvent[], timestampMs: number) {
		const recent = new Set<string>();
		for (const event of events) {
			if (timestampMs - event.timestampMs > 240) continue;
			if (event.type === 'mousedown' || event.type === 'mouseup') {
				recent.add(event.button ?? 'Unknown');
			}
		}
		return recent;
	}

	function findActiveEvents(session: DemoSession | null, timestampMs: number) {
		if (!session) return [];
		const windowMs = 1400;
		const start = lowerBound(session.events, timestampMs - windowMs);
		const result: InputEvent[] = [];
		for (let index = start; index < session.events.length; index += 1) {
			const event = session.events[index];
			if (event.timestampMs > timestampMs + 120) break;
			if (event.type !== 'mousemove') result.unshift(event);
		}
		return result.slice(0, 16);
	}

	function latestCursor(session: DemoSession | null, timestampMs: number) {
		if (!session) return null;
		const start = Math.min(session.events.length - 1, lowerBound(session.events, timestampMs));
		for (let index = start; index >= 0; index -= 1) {
			const event = session.events[index];
			if (event.type === 'mousemove' && typeof event.x === 'number' && typeof event.y === 'number') {
				const bounds = session.cursorBounds;
				const minX = bounds.minX ?? 0;
				const maxX = bounds.maxX ?? session.resolution.width ?? minX + 1;
				const minY = bounds.minY ?? 0;
				const maxY = bounds.maxY ?? session.resolution.height ?? minY + 1;
				return {
					x: event.x,
					y: event.y,
					left: clamp(((event.x - minX) / Math.max(1, maxX - minX)) * 100, 0, 100),
					top: clamp(((event.y - minY) / Math.max(1, maxY - minY)) * 100, 0, 100)
				};
			}
		}
		return null;
	}

	function recentCursorTrail(session: DemoSession | null, timestampMs: number) {
		if (!session) return [];
		const start = Math.max(0, lowerBound(session.events, timestampMs - 2200));
		const trail: Array<{ left: number; top: number; age: number }> = [];
		for (let index = start; index < session.events.length; index += 1) {
			const event = session.events[index];
			if (event.timestampMs > timestampMs) break;
			if (event.type !== 'mousemove' || typeof event.x !== 'number' || typeof event.y !== 'number') continue;
			const bounds = session.cursorBounds;
			const minX = bounds.minX ?? 0;
			const maxX = bounds.maxX ?? session.resolution.width ?? minX + 1;
			const minY = bounds.minY ?? 0;
			const maxY = bounds.maxY ?? session.resolution.height ?? minY + 1;
			trail.push({
				left: clamp(((event.x - minX) / Math.max(1, maxX - minX)) * 100, 0, 100),
				top: clamp(((event.y - minY) / Math.max(1, maxY - minY)) * 100, 0, 100),
				age: clamp((timestampMs - event.timestampMs) / 2200, 0, 1)
			});
		}
		return trail.slice(-10);
	}

	function lowerBound(events: InputEvent[], timestampMs: number) {
		let low = 0;
		let high = events.length;
		while (low < high) {
			const middle = (low + high) >> 1;
			if (events[middle].timestampMs < timestampMs) low = middle + 1;
			else high = middle;
		}
		return low;
	}

	function lastEventTime(session: DemoSession | null) {
		if (!session || session.events.length === 0) return 0;
		return session.events[session.events.length - 1].timestampMs;
	}

	function clamp(value: number, min: number, max: number) {
		return Math.min(max, Math.max(min, value));
	}

	function formatTime(ms: number) {
		const totalSeconds = Math.max(0, Math.floor(ms / 1000));
		const minutes = Math.floor(totalSeconds / 60);
		const seconds = totalSeconds % 60;
		return `${minutes}:${String(seconds).padStart(2, '0')}`;
	}

	function formatDuration(ms: number) {
		const minutes = Math.floor(ms / 60000);
		const seconds = Math.floor((ms % 60000) / 1000);
		return `${minutes}m ${seconds}s`;
	}

	function formatNumber(value: number | undefined) {
		return value === undefined ? '—' : value.toLocaleString();
	}

	function formatBytes(bytes: number | undefined) {
		if (!bytes) return '—';
		return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
	}

	function eventLabel(event: InputEvent) {
		if (event.key) return event.key;
		if (event.button) return event.button;
		if (event.type === 'wheel') return `wheel ${event.deltaY ?? 0}`;
		return event.type;
	}

	function eventClass(event: InputEvent) {
		if (event.type === 'keydown' || event.type === 'keyup') return 'key';
		if (event.type === 'mousedown' || event.type === 'mouseup') return 'click';
		if (event.type === 'wheel') return 'wheel';
		return 'other';
	}
</script>

<svelte:head>
	<title>Gameplay data demo</title>
	<meta
		name="description"
		content="Interactive data demo for consented gameplay trajectories with synchronized screen, keyboard, mouse, and metadata."
	/>
</svelte:head>

<section class="demo-page">
	<header class="hero">
		<p class="eyebrow">Gameplay telemetry sample</p>
		<h1>Consented gameplay trajectories: synchronized screen video, keyboard, mouse, and metadata from real players.</h1>
		<p class="product-note">
			This is built on <a href="https://www.tryascent.gg/">Ascent</a>, our game recording
			software with a consented user community, a 2,000-member Discord, 5,000–6,000 daily
			active users, and support for recording many different games already.
		</p>
		<div class="hero-metrics" aria-label="Ascent community scale">
			<div>
				<strong>5–6k</strong>
				<span>daily active users</span>
			</div>
			<div>
				<strong>30k</strong>
				<span>total users</span>
			</div>
			<div>
				<strong>2k</strong>
				<span>Discord members</span>
			</div>
		</div>
	</header>

	<section class="viewer-section" aria-labelledby="viewer-title">
		<div class="section-heading">
			<p class="eyebrow">Synchronized viewer</p>
			<h2 id="viewer-title">Video and input stream inspectable at frame time</h2>
		</div>

		{#if loadError}
			<div class="state-card">Failed to load sample data: {loadError}</div>
		{:else if !data}
			<div class="state-card">Loading sample manifest and normalized input events…</div>
		{:else if selectedSession}
			<div class="viewer-grid">
					<div class="player-card">
						<div class="viewer-toolbar">
							<div>
								<p class="panel-label">Playback clock</p>
								<strong>{formatTime(currentTimeMs)} / {formatTime(effectiveDurationMs)}</strong>
							</div>
							<div class="sync-status">
								<span class:playing-dot={playing}></span>
								{selectedSession.events.length
									? playing ? 'streaming synchronized events' : 'paused at timestamp'
									: 'raw clip selected; event file linked'}
							</div>
						</div>
						<div class="video-shell">
							<video
								bind:this={videoEl}
								src={selectedSession.video}
							controls
							preload="metadata"
							onloadedmetadata={onLoadedMetadata}
							onplay={() => (playing = true)}
							onpause={() => (playing = false)}
						>
							<track kind="captions" />
						</video>
						{#if cursor}
							<div class="cursor" style={`left: ${cursor.left}%; top: ${cursor.top}%`}>
								<span></span>
							</div>
							{/if}
						</div>

						<div class="transport">
							<div class="transport-buttons">
								<button type="button" class="primary-action" onclick={togglePlayback}>{playing ? 'Pause' : 'Play'}</button>
								<button type="button" onclick={() => jump(-5000)}>−5s</button>
								<button type="button" onclick={() => jump(5000)}>+5s</button>
							</div>
							<label class="scrubber" aria-label="seek through synchronized recording">
								<input
									type="range"
									min="0"
									max="100"
									step="0.1"
									bind:value={seekTarget}
									oninput={(event) => seekTo(Number(event.currentTarget.value))}
								/>
							</label>
						</div>

						<div class="live-inputs">
							<div>
								<p class="panel-label compact-label">Keyboard</p>
								<div class="keyboard-map" aria-label="mini keyboard input state">
									{#each keyboardRows as row}
										<div>
											{#each row as key}
												<span class:active={activeKeys.has(key)}>{key}</span>
											{/each}
										</div>
									{/each}
								</div>
							</div>
							<div>
								<p class="panel-label compact-label">Mouse</p>
								<div class="mouse-device" aria-label="mini mouse button input state">
									<div
										class="mouse-button left-button"
										class:active={pressedMouseButtons.has('Left')}
										class:recent={recentMouseButtons.has('Left')}
									>Left</div>
									<div
										class="mouse-button wheel-button"
										class:active={pressedMouseButtons.has('Middle')}
										class:recent={recentMouseButtons.has('Middle') || recentMouseButtons.has('Wheel')}
									>Wheel</div>
									<div
										class="mouse-button right-button"
										class:active={pressedMouseButtons.has('Right')}
										class:recent={recentMouseButtons.has('Right')}
									>Right</div>
									<div class="mouse-palm"></div>
								</div>
							</div>
							<div>
								<p class="panel-label compact-label">Cursor</p>
								<div class="cursor-grid" aria-label="mini cursor position grid">
									{#each cursorTrail as point, index (`${point.left}-${point.top}-${index}`)}
										<span
											class="trail-dot"
											style={`left: ${point.left}%; top: ${point.top}%; opacity: ${Math.max(0.18, 1 - point.age)}`}
										></span>
									{/each}
									{#if cursor}
										<span class="cursor-dot" style={`left: ${cursor.left}%; top: ${cursor.top}%`}></span>
									{/if}
								</div>
							</div>
						</div>

						<div class="timeline-panel">
							<div class="timeline-header">
								<p class="panel-label">Event timeline</p>
								<span>{formatNumber(timelineEvents.length)} visible event markers</span>
							</div>
							<div class="timeline" aria-label="input event timeline">
								<div class="playhead" style={`left: ${(currentTimeMs / Math.max(1, effectiveDurationMs)) * 100}%`}></div>
								{#each timelineEvents as event, index (`${event.timestampMs}-${event.type}-${index}`)}
									<button
										type="button"
										class:event-key={eventClass(event) === 'key'}
										class:event-click={eventClass(event) === 'click'}
										class:event-wheel={eventClass(event) === 'wheel'}
										style={`left: ${(event.timestampMs / Math.max(1, effectiveDurationMs)) * 100}%`}
										title={`${formatTime(event.timestampMs)} · ${event.type} · ${eventLabel(event)}`}
										onclick={() => seekTo((event.timestampMs / Math.max(1, effectiveDurationMs)) * 100)}
									></button>
								{/each}
							</div>
							<div class="timeline-legend">
								<span><i class="key-dot"></i> key</span>
								<span><i class="click-dot"></i> mouse button</span>
								<span><i class="wheel-dot"></i> wheel</span>
								<span>cursor overlay from sampled mousemove</span>
							</div>
						</div>
					</div>

				<aside class="metadata-card">
					<p class="panel-label">Selected session</p>
					<h3>{gameLabel(selectedSession)}</h3>
					<p>{selectedSession.description}</p>
					<dl>
						<div><dt>Session ID</dt><dd>{selectedSession.id}</dd></div>
						<div><dt>Duration</dt><dd>{formatDuration(selectedSession.durationMs)}</dd></div>
						<div><dt>Input events</dt><dd>{selectedSession.eventCount ? formatNumber(selectedSession.eventCount) : 'raw file linked'}</dd></div>
						<div><dt>Resolution / FPS</dt><dd>{selectedSession.resolution.width ?? '—'}×{selectedSession.resolution.height ?? '—'} / {selectedSession.fps ?? '—'}</dd></div>
						<div><dt>Video</dt><dd>{selectedSession.codec}, {formatBytes(selectedSession.videoSizeBytes)}</dd></div>
						<div><dt>Recorded</dt><dd>{selectedSession.recordedAt ?? '—'}</dd></div>
					</dl>
					<a href={selectedSession.dataUrl}>metadata.json</a>
					<a href={selectedSession.eventsUrl}>events.csv.gz</a>
				</aside>
			</div>
		{/if}
	</section>

	{#if data}
		<section class="clip-library" aria-labelledby="clip-library-title">
			<div class="section-heading compact">
				<p class="eyebrow">Session selector</p>
				<h2 id="clip-library-title">Select across the multi-game sample bucket</h2>
				<p>
					Choose any game sample to load it into the viewer. Interactive sessions are prioritized first,
					and the list is ordered to show breadth across the bucket.
				</p>
			</div>
			<div class="library-toolbar">
				<label>
					<span>Filter</span>
					<input bind:value={clipSearch} type="search" placeholder="game or session id" />
				</label>
				<span class="library-count">
					{formatNumber(filteredClips.length)} / {formatNumber(data.totalSessions)} clips
				</span>
			</div>
			{#if clipLibrary.length}
				<div class="clip-table" role="table" aria-label="selectable clips in the sample bucket">
					<div class="clip-row clip-head" role="row">
						<span>Session</span>
						<span>Telemetry</span>
						<span>Video</span>
						<span>Events</span>
						<span>Metadata</span>
					</div>
					{#each filteredClips as clip (clip.id)}
						<div class:selected={clip.id === selectedSessionId} class="clip-row" role="row">
							<button type="button" class="clip-select" onclick={() => selectClip(clip)}>
								<span class="game-chip">
									{#if gameLogo(clip)}
										<img src={gameLogo(clip)} alt="" loading="lazy" />
									{/if}
									<span>
										<strong>{gameLabel(clip)}</strong>
										<small>{clip.id}</small>
									</span>
								</span>
								<small>{formatDuration(clip.durationMs ?? 0)} · {clip.resolution?.width ?? '—'}×{clip.resolution?.height ?? '—'} · {clip.fps ?? '—'} fps</small>
							</button>
							<span class:interactive={hasInteractiveTelemetry(clip)} class="telemetry-chip">
								{hasInteractiveTelemetry(clip) ? 'interactive' : 'raw package'}
							</span>
							<a href={clip.video}>video.mp4</a>
							<a href={clip.eventsUrl}>events.csv.gz</a>
							<a href={clip.dataUrl}>data.json</a>
						</div>
					{/each}
				</div>
			{:else}
				<div class="state-card">Clip library unavailable in this static payload.</div>
			{/if}
		</section>
	{/if}


	<section class="schema" aria-labelledby="schema-title">
		<div class="section-heading">
			<p class="eyebrow">Delivered format</p>
			<h2 id="schema-title">Dataset schema</h2>
			<p>
				The sample manifest points to one folder per session. Each folder contains the screen recording,
				session metadata, and an input event stream. The demo loads compact normalized events from
				the sample bucket and plays them against the original MP4 using the video clock.
			</p>
		</div>
		<div class="schema-grid">
			<div class="package-card">
				<p class="panel-label">Per-session package</p>
				<ul class="file-list">
					<li><span>video.mp4</span><small>screen recording</small></li>
					<li><span>events.csv.gz</span><small>timestamped input stream</small></li>
					<li><span>data.json</span><small>session and match metadata</small></li>
					<li><span>manifest.json</span><small>bucket-level index</small></li>
				</ul>
			</div>
			<div class="field-card">
				<div class="field-card-header">
					<p class="panel-label">Event fields</p>
					<span>normalized for demo</span>
				</div>
				<table>
					<tbody>
						<tr><td>timestamp_ms</td><td>event time relative to recording start</td></tr>
						<tr><td>event_type</td><td>key_down, key_up, mouse_button_down, cursor_pos, wheel</td></tr>
						<tr><td>key</td><td>normalized key label plus raw scan code</td></tr>
						<tr><td>mouse_x / mouse_y</td><td>raw desktop cursor coordinates when present</td></tr>
						<tr><td>button</td><td>left, right, middle, or raw button code</td></tr>
						<tr><td>session_id</td><td>public session identifier from manifest</td></tr>
						<tr><td>game</td><td>game label from the multi-game manifest</td></tr>
						<tr><td>recorded_at</td><td>session-level metadata timestamp when available</td></tr>
					</tbody>
				</table>
			</div>
		</div>
		{#if data}
			<p class="source-line">
				Source bucket: <a href={data.baseUrl}>{data.bucket}</a> ·
				<a href={data.sourceManifest}>manifest.json</a> · {formatNumber(data.totalSessions)} sessions ·
				files: {data.filesPerSession.join(', ')}
			</p>
		{/if}
	</section>

	<section class="game-support" aria-labelledby="game-support-title">
		<div class="section-heading compact">
			<p class="eyebrow">Recorder coverage</p>
			<h2 id="game-support-title">Record any game</h2>
			<p>Works with 10,000+ titles across genres. These are representative games already supported by the recorder.</p>
		</div>
		<div class="logo-marquee" aria-label="supported game examples">
			<div class="logo-track">
				{#each [...gameLogos, ...gameLogos] as game, index (`${game[1]}-${index}`)}
					<div class="game-logo">
						<img src={`https://www.tryascent.gg/games/${game[1]}`} alt={game[0]} loading="lazy" />
					</div>
				{/each}
			</div>
		</div>
	</section>

	<section class="two-column">
		<div>
			<h2>Data Quality & Provenance</h2>
			<ul>
				<li>First-party collection through Ascent, our game recording software.</li>
				<li>User-consented recording flow.</li>
				<li>2,000-member Discord/community with users who have consented to gameplay recording.</li>
				<li>Recorder already supports many different games.</li>
				<li>Synchronized screen, keyboard, and mouse streams.</li>
				<li>Session-level metadata.</li>
				<li>Corrupted video filtering.</li>
				<li>Duplicate/session sanity checks.</li>
				<li>PII-aware collection and redaction path can be supported.</li>
				<li>Opt-out/deletion process can be supported.</li>
			</ul>
		</div>
		<div>
			<h2>Acquisition Capacity</h2>
			<div class="capacity-card">
				<h3>Current Ascent network</h3>
				<div class="capacity-metrics">
					<div><strong>5–6k</strong><span>daily active users</span></div>
					<div><strong>30k</strong><span>total users</span></div>
					<div><strong>2k</strong><span>Discord members</span></div>
				</div>
				<p>
					This is already a meaningful consented acquisition channel, not a cold-start data
					collection effort.
				</p>
			</div>
			<div class="capacity-card">
				<h3>Campaign-based expansion</h3>
				<ul>
					<li>Can recruit for specific games.</li>
					<li>Can run ads to source targeted users.</li>
					<li>Can collect around specific tasks, ranks, game modes, or scenarios.</li>
					<li>Game-agnostic recorder already supports many games.</li>
					<li>Custom collection campaigns possible.</li>
				</ul>
			</div>
		</div>
	</section>

	<section class="pilots" aria-labelledby="pilots-title">
		<div class="section-heading compact">
			<p class="eyebrow">Pilot proposal</p>
			<h2 id="pilots-title">Two practical starting points</h2>
		</div>
		<div class="pilot-grid">
			<article>
				<span>Pilot A</span>
				<h3>League trajectory dataset</h3>
				<p>100–500 hours of League gameplay with synchronized screen video, keyboard/mouse events, metadata, and QA.</p>
			</article>
			<article>
				<span>Pilot B</span>
				<h3>Custom game campaign</h3>
				<p>Mercor specifies a game, task, and data schema. We recruit users and deliver targeted sessions.</p>
			</article>
		</div>
	</section>
</section>

<style>
	:global(main) {
		max-width: 62rem;
		padding: 0 1.5rem 7rem;
	}

	.demo-page {
		margin: 0;
		padding: 0;
		background: #fff;
		color: var(--ink);
	}

	.hero {
		margin: 3.5rem 0 3rem;
	}

	.eyebrow,
	.panel-label {
		margin: 0 0 0.75rem;
		font-family: var(--mono);
		font-size: 0.68rem;
		letter-spacing: 0.16em;
		text-transform: uppercase;
		color: var(--faint);
	}

	h1 {
		max-width: 58rem;
		margin: 0;
		font-size: clamp(2rem, 5vw, 3.6rem);
		font-weight: 600;
		line-height: 1.05;
		letter-spacing: -0.04em;
	}


	.product-note {
		max-width: 47rem;
		margin: 1rem 0 0;
		padding-left: 1rem;
		border-left: 1px solid var(--ink);
		color: #444;
	}

	.product-note a {
		color: var(--ink);
	}

	.hero-metrics {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		max-width: 47rem;
		margin-top: 1.4rem;
		border: 1px solid var(--ink);
		background: var(--ink);
		color: #fff;
	}

	.hero-metrics div {
		padding: 1rem;
		border-left: 1px solid #444;
	}

	.hero-metrics div:first-child {
		border-left: 0;
	}

	.hero-metrics strong {
		display: block;
		font-size: clamp(2rem, 5vw, 3.2rem);
		font-weight: 600;
		line-height: 0.95;
		letter-spacing: -0.05em;
	}

	.hero-metrics span {
		display: block;
		margin-top: 0.55rem;
		font-family: var(--mono);
		font-size: 0.62rem;
		letter-spacing: 0.14em;
		text-transform: uppercase;
		color: #d8d8d8;
	}


	.viewer-section,
	.clip-library,
	.schema,
	.two-column,
	.pilots {
		margin: 0;
		padding-top: 4rem;
	}

	.clip-library {
		padding-top: 0.75rem;
		padding-bottom: 2rem;
		border-top: 1px solid var(--hairline);
	}

	.library-toolbar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 1rem;
		margin: 0.8rem 0 0.75rem;
	}

	.library-toolbar label {
		display: inline-flex;
		align-items: center;
		gap: 0.6rem;
		min-width: min(100%, 20rem);
		font-family: var(--mono);
		font-size: 0.64rem;
		letter-spacing: 0.14em;
		text-transform: uppercase;
		color: var(--faint);
	}

	.library-toolbar input {
		width: 100%;
		border: 1px solid var(--ink);
		border-radius: 0;
		padding: 0.42rem 0.55rem;
		background: #fff;
		color: var(--ink);
		font-family: var(--mono);
		font-size: 0.72rem;
		letter-spacing: 0.02em;
	}

	.library-count {
		white-space: nowrap;
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--faint);
	}

	.game-chip {
		display: inline-flex;
		align-items: center;
		gap: 0.6rem;
	}

	.game-chip img {
		display: block;
		width: 2rem;
		height: 2rem;
		padding: 0.32rem;
		border: 1px solid var(--ink);
		background: var(--ink);
		object-fit: contain;
	}

	.game-chip span {
		display: grid;
		gap: 0.12rem;
	}

	.clip-table {
		max-height: 21rem;
		overflow: auto;
		border: 1px solid var(--ink);
		background: #fff;
	}

	.clip-row {
		display: grid;
		grid-template-columns: minmax(12rem, 1fr) auto repeat(3, auto);
		align-items: center;
		gap: 1rem;
		min-width: 46rem;
		padding: 0.58rem 0.75rem;
		border-top: 1px solid var(--hairline);
		font-family: var(--mono);
		font-size: 0.7rem;
	}

	.clip-row:first-child {
		border-top: 0;
	}

	.clip-row.selected {
		background: #f1f1f1;
		box-shadow: inset 3px 0 0 var(--ink);
	}

	.clip-head {
		position: sticky;
		top: 0;
		z-index: 1;
		background: var(--ink);
		color: #fff;
		font-size: 0.62rem;
		letter-spacing: 0.14em;
		text-transform: uppercase;
	}

	.clip-select {
		display: grid;
		gap: 0.28rem;
		width: 100%;
		padding: 0;
		border: 0;
		background: transparent;
		color: var(--ink);
		text-align: left;
	}

	.clip-select strong {
		font-weight: 500;
	}

	.clip-select small {
		color: var(--faint);
		font-size: 0.58rem;
	}

	.clip-row a {
		color: var(--ink);
		text-decoration: underline;
		text-underline-offset: 0.2em;
	}

	.clip-head span {
		color: #fff;
	}

	.telemetry-chip {
		display: inline-flex;
		justify-content: center;
		min-width: 6.5rem;
		padding: 0.22rem 0.4rem;
		border: 1px solid var(--hairline);
		color: var(--faint);
		font-size: 0.58rem;
		letter-spacing: 0.1em;
		text-transform: uppercase;
	}

	.telemetry-chip.interactive {
		border-color: var(--ink);
		background: var(--ink);
		color: #fff;
	}

	.viewer-section {
		padding-top: 0;
		border-top: 1px solid var(--ink);
	}

	.section-heading {
		padding: 1.25rem 0 0;
	}

	.section-heading.compact {
		padding-left: 0;
		padding-right: 0;
	}

	h2 {
		margin: 0 0 0.7rem;
		font-family: var(--mono);
		font-size: 0.72rem;
		font-weight: 500;
		text-transform: uppercase;
		letter-spacing: 0.18em;
		line-height: 1.35;
	}

	h2::before {
		content: '';
		display: block;
		width: 2rem;
		border-top: 1px solid var(--ink);
		margin-bottom: 1.25rem;
	}

	.section-heading h2::before {
		display: none;
	}

	.section-heading p:not(.eyebrow) {
		max-width: 48rem;
		margin: 0 0 1.25rem;
		color: #444;
		line-height: 1.65;
	}

	.viewer-grid {
		display: grid;
		grid-template-columns: minmax(0, 1fr) 18rem;
		gap: 1rem;
		padding: 1.25rem 0 0;
	}

	.player-card,
	.metadata-card,
	.state-card,
	.capacity-card,
	.pilot-grid article {
		border: 1px solid var(--ink);
		background: #fff;
	}

	.player-card {
		overflow: hidden;
	}

	.viewer-toolbar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 1rem;
		padding: 0.85rem 1rem;
		border-bottom: 1px solid var(--hairline);
	}

	.viewer-toolbar .panel-label {
		margin-bottom: 0.2rem;
	}

	.viewer-toolbar strong {
		font-family: var(--mono);
		font-size: 0.86rem;
		font-weight: 500;
		letter-spacing: 0.02em;
	}

	.sync-status {
		display: inline-flex;
		align-items: center;
		gap: 0.45rem;
		white-space: nowrap;
		font-family: var(--mono);
		font-size: 0.66rem;
		letter-spacing: 0.12em;
		text-transform: uppercase;
		color: var(--faint);
	}

	.sync-status span {
		width: 0.46rem;
		height: 0.46rem;
		border: 1px solid var(--ink);
		border-radius: 999px;
		background: #fff;
	}

	.sync-status span.playing-dot {
		background: var(--ink);
	}

	.video-shell {
		position: relative;
		aspect-ratio: 16 / 9;
		background: #000;
		overflow: hidden;
	}

	video {
		display: block;
		width: 100%;
		height: 100%;
		object-fit: contain;
	}

	.cursor {
		position: absolute;
		width: 1rem;
		height: 1rem;
		transform: translate(-50%, -50%);
		border: 1px solid #fff;
		border-radius: 999px;
		box-shadow: 0 0 0 4px rgba(0, 0, 0, 0.35);
		pointer-events: none;
	}

	.cursor span {
		position: absolute;
		left: 50%;
		top: 50%;
		width: 2px;
		height: 2px;
		background: #fff;
		transform: translate(-50%, -50%);
	}

	.transport,
	.live-inputs,
	.timeline-legend {
		display: flex;
		gap: 0.65rem;
		align-items: center;
		flex-wrap: wrap;
		border-top: 1px solid var(--hairline);
	}

	button {
		font: inherit;
		cursor: pointer;
	}

	.transport {
		display: grid;
		grid-template-columns: auto minmax(10rem, 1fr);
		padding: 0.85rem 1rem;
	}

	.transport-buttons {
		display: flex;
		gap: 0.5rem;
		align-items: center;
	}

	.transport button {
		border: 1px solid var(--ink);
		background: #fff;
		color: var(--ink);
		font-family: var(--mono);
		font-size: 0.66rem;
		letter-spacing: 0.14em;
		text-transform: uppercase;
		padding: 0.36rem 0.62rem;
	}

	.transport button.primary-action {
		min-width: 4.8rem;
		background: var(--ink);
		color: #fff;
	}

	.transport button:hover {
		background: var(--ink);
		color: #fff;
	}

	.scrubber {
		display: block;
	}

	.scrubber input {
		width: 100%;
		accent-color: var(--ink);
	}

	.live-inputs {
		align-items: stretch;
		padding: 0.9rem 1rem;
		background: #fafafa;
	}

	.live-inputs > div {
		flex: 1;
		min-width: 12rem;
		border-left: 1px solid var(--hairline);
		padding: 0 0 0 0.85rem;
	}

	.live-inputs > div:first-child {
		border-left: 0;
		padding-left: 0;
	}

	.compact-label {
		margin-bottom: 0.35rem;
	}

	.keyboard-map {
		display: grid;
		gap: 0.18rem;
		padding: 0.32rem;
		border: 1px solid var(--hairline);
		border-radius: 0.25rem;
		background: #fff;
	}

	.keyboard-map > div {
		display: grid;
		grid-template-columns: repeat(5, minmax(0, 1fr));
		gap: 0.18rem;
	}

	.keyboard-map > div:last-child {
		grid-template-columns: 1.1fr 1.2fr 1fr 2.4fr;
	}

	.keyboard-map span {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		min-height: 1rem;
		border: 1px solid var(--ink);
		background: #fff;
		border-radius: 0.16rem;
		font-family: var(--mono);
		font-size: 0.52rem;
		line-height: 1;
		color: var(--ink);
		transition: background 80ms linear, color 80ms linear;
	}

	.keyboard-map span.active,
	.mouse-device .active {
		background: var(--ink);
		color: #fff;
	}

	.mouse-device .recent:not(.active) {
		background: #dcdcdc;
		box-shadow: inset 0 0 0 2px #fff;
	}

	.mouse-device {
		display: grid;
		grid-template-columns: 1fr 0.45fr 1fr;
		grid-template-rows: 1.35rem 2.6rem;
		gap: 0.18rem;
		max-width: 9.5rem;
		height: calc(4 * 1rem + 3 * 0.18rem + 0.64rem);
		margin: 0 auto;
		padding: 0.32rem;
		border: 1px solid var(--hairline);
		border-radius: 2.5rem 2.5rem 2.1rem 2.1rem;
		background: #fff;
	}

	.mouse-button {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		border: 1px solid var(--ink);
		background: #fff;
		font-family: var(--mono);
		font-size: 0.5rem;
		line-height: 1;
		color: var(--ink);
		transition: background 80ms linear, color 80ms linear;
	}

	.left-button {
		border-radius: 1.8rem 0.25rem 0.2rem 0.55rem;
	}

	.right-button {
		border-radius: 0.25rem 1.8rem 0.55rem 0.2rem;
	}

	.wheel-button {
		border-radius: 999px;
		font-size: 0;
	}

	.wheel-button::before {
		content: '';
		width: 0.2rem;
		height: 0.72rem;
		border-radius: 999px;
		background: currentColor;
	}

	.mouse-palm {
		grid-column: 1 / -1;
		border: 1px solid var(--ink);
		border-top: 0;
		border-radius: 0.55rem 0.55rem 2rem 2rem;
		background:
			linear-gradient(90deg, transparent calc(50% - 0.5px), #d8d8d8 calc(50% - 0.5px), #d8d8d8 calc(50% + 0.5px), transparent calc(50% + 0.5px)),
			#fff;
	}

	.cursor-grid {
		position: relative;
		aspect-ratio: 16 / 9;
		height: calc(4 * 1rem + 3 * 0.18rem + 0.64rem);
		max-width: 100%;
		border: 1px solid var(--hairline);
		background:
			linear-gradient(90deg, #e6e6e6 1px, transparent 1px) 0 0 / 25% 100%,
			linear-gradient(180deg, #e6e6e6 1px, transparent 1px) 0 0 / 100% 50%,
			#fff;
		overflow: hidden;
	}

	.cursor-dot,
	.trail-dot {
		position: absolute;
		border-radius: 999px;
		transform: translate(-50%, -50%);
		pointer-events: none;
	}

	.cursor-dot {
		width: 0.56rem;
		height: 0.56rem;
		background: var(--ink);
		box-shadow: 0 0 0 3px #fff;
		z-index: 2;
	}

	.trail-dot {
		width: 0.28rem;
		height: 0.28rem;
		background: #777;
	}

	.timeline-panel {
		border-top: 1px solid var(--hairline);
		padding: 0.9rem 1rem 0.8rem;
	}

	.timeline-header {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: 1rem;
		margin-bottom: 0.55rem;
	}

	.timeline-header .panel-label {
		margin: 0;
	}

	.timeline-header span {
		font-family: var(--mono);
		font-size: 0.64rem;
		color: var(--faint);
	}

	.timeline {
		position: relative;
		height: 3.7rem;
		margin: 0;
		border: 1px solid var(--ink);
		background:
			linear-gradient(90deg, #eee 1px, transparent 1px) 0 0 / 10% 100%,
			#fff;
	}

	.timeline button {
		position: absolute;
		bottom: 0.8rem;
		width: 4px;
		height: 1.45rem;
		padding: 0;
		border: 0;
		border-radius: 0;
		transform: translateX(-50%);
		opacity: 0.86;
	}

	.timeline button:hover {
		width: 7px;
		opacity: 1;
	}

	.timeline button.event-key {
		background: var(--ink);
	}

	.timeline button.event-click {
		background: #777;
	}

	.timeline button.event-wheel {
		background: #bbb;
	}

	.playhead {
		position: absolute;
		top: 0;
		bottom: 0;
		width: 2px;
		background: var(--ink);
		z-index: 2;
	}

	.timeline-legend {
		padding: 0.55rem 0 0;
		border: 0;
		font-family: var(--mono);
		font-size: 0.64rem;
		color: var(--faint);
	}

	.timeline-legend i {
		display: inline-block;
		width: 0.55rem;
		height: 0.55rem;
		margin-right: 0.25rem;
		border-radius: 999px;
	}

	.key-dot { background: var(--ink); }
	.click-dot { background: #777; }
	.wheel-dot { background: #bbb; }

	.metadata-card {
		padding: 1rem;
	}

	.metadata-card h3,
	.capacity-card h3,
	.pilot-grid h3 {
		margin: 0 0 0.4rem;
		font-size: 1.05rem;
		line-height: 1.25;
	}

	.metadata-card p,
	.pilot-grid p {
		margin: 0 0 1rem;
		color: #444;
		font-size: 0.92rem;
		line-height: 1.55;
	}

	dl {
		margin: 1rem 0;
		font-family: var(--mono);
		font-size: 0.7rem;
	}

	dl div {
		display: grid;
		grid-template-columns: 6.5rem 1fr;
		gap: 0.8rem;
		padding: 0.48rem 0;
		border-bottom: 1px solid var(--hairline);
	}

	dt {
		color: var(--faint);
	}

	dd {
		margin: 0;
		color: var(--ink);
		word-break: break-word;
	}

	.metadata-card a,
	.source-line a {
		display: inline-block;
		margin-right: 0.7rem;
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink);
		text-decoration-color: #999;
	}

	.state-card {
		margin: 1.25rem 0 0;
		padding: 1rem;
		font-family: var(--mono);
		font-size: 0.76rem;
		color: var(--faint);
	}

	.schema-grid {
		display: grid;
		grid-template-columns: 17rem minmax(0, 1fr);
		gap: 1rem;
		align-items: stretch;
	}

	.package-card,
	.field-card {
		border: 1px solid var(--ink);
		background: #fff;
	}

	.package-card {
		display: flex;
		flex-direction: column;
		padding: 1rem;
	}

	.package-card .panel-label,
	.field-card .panel-label {
		margin-bottom: 0.75rem;
	}

	.file-list {
		flex: 1;
		display: grid;
		grid-template-rows: repeat(4, 1fr);
		list-style: none;
		margin: 0;
		padding: 0;
		border-top: 1px solid var(--ink);
	}

	.file-list li {
		display: grid;
		grid-template-columns: 1fr;
		align-content: center;
		gap: 0.1rem;
		margin: 0;
		padding: 0.7rem 0;
		border-bottom: 1px solid var(--hairline);
	}

	.file-list li:last-child {
		border-bottom: 0;
	}

	.file-list span {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink);
	}

	.file-list small {
		font-family: var(--mono);
		font-size: 0.58rem;
		letter-spacing: 0.04em;
		text-transform: uppercase;
		color: var(--faint);
	}

	.field-card-header {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: 1rem;
		padding: 1rem 1rem 0;
	}

	.field-card-header span {
		font-family: var(--mono);
		font-size: 0.58rem;
		letter-spacing: 0.12em;
		text-transform: uppercase;
		color: var(--faint);
	}

	.schema table {
		margin: 0;
		border-top: 1px solid var(--ink);
		border-bottom: 0;
		background: #fff;
	}

	.schema td {
		padding: 0.58rem 1rem;
	}

	.schema td:first-child {
		width: 13rem;
		color: var(--ink);
	}

	.schema td:last-child {
		text-align: left;
		color: #555;
	}

	.source-line {
		margin-top: 1rem;
		font-family: var(--mono);
		font-size: 0.7rem;
		color: #555;
	}

	.game-support {
		margin: 0;
		padding-top: 4rem;
	}

	.game-support .section-heading p:not(.eyebrow) {
		max-width: 42rem;
		margin-bottom: 1rem;
	}

	.logo-marquee {
		position: relative;
		overflow: hidden;
		border-top: 1px solid var(--ink);
		border-bottom: 1px solid var(--ink);
		background: var(--ink);
	}

	.logo-marquee::before,
	.logo-marquee::after {
		content: '';
		position: absolute;
		top: 0;
		bottom: 0;
		width: 5rem;
		z-index: 2;
		pointer-events: none;
	}

	.logo-marquee::before {
		left: 0;
		background: linear-gradient(90deg, var(--ink), rgba(17, 17, 17, 0));
	}

	.logo-marquee::after {
		right: 0;
		background: linear-gradient(270deg, var(--ink), rgba(17, 17, 17, 0));
	}

	.logo-track {
		display: flex;
		width: max-content;
		animation: logo-scroll 58s linear infinite;
	}

	.game-logo {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 10.5rem;
		height: 4.6rem;
		padding: 0.9rem 1.2rem;
		border-right: 1px solid #2f2f2f;
	}

	.game-logo img {
		display: block;
		max-width: 100%;
		max-height: 2.35rem;
		object-fit: contain;
		opacity: 1;
	}

	@keyframes logo-scroll {
		from {
			transform: translateX(0);
		}
		to {
			transform: translateX(-50%);
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.logo-track {
			animation: none;
			flex-wrap: wrap;
			width: auto;
		}

		.logo-marquee {
			overflow: visible;
		}
	}

	.two-column {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 2rem;
	}

	.two-column h2 {
		margin-bottom: 1rem;
	}

	.two-column ul,
	.capacity-card ul {
		margin: 0;
		padding-left: 1.1rem;
	}

	.two-column li,
	.capacity-card li {
		margin: 0.45rem 0;
		line-height: 1.55;
	}

	.capacity-card {
		border-color: var(--hairline);
		padding: 1rem;
		margin-bottom: 0.85rem;
	}

	.capacity-card p {
		margin: 0.9rem 0 0;
		color: #444;
		line-height: 1.55;
	}

	.capacity-metrics {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		margin-top: 0.85rem;
		border: 1px solid var(--ink);
	}

	.capacity-metrics div {
		padding: 0.75rem;
		border-left: 1px solid var(--hairline);
	}

	.capacity-metrics div:first-child {
		border-left: 0;
	}

	.capacity-metrics strong,
	.capacity-metrics span {
		display: block;
	}

	.capacity-metrics strong {
		font-size: 1.65rem;
		font-weight: 600;
		line-height: 1;
		letter-spacing: -0.04em;
	}

	.capacity-metrics span {
		margin-top: 0.35rem;
		font-family: var(--mono);
		font-size: 0.54rem;
		letter-spacing: 0.1em;
		text-transform: uppercase;
		color: var(--faint);
	}

	.pilot-grid {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.9rem;
	}

	.pilot-grid article {
		border-color: var(--hairline);
		padding: 1rem;
	}

	.pilot-grid span {
		display: block;
		margin-bottom: 0.55rem;
		font-family: var(--mono);
		font-size: 0.66rem;
		letter-spacing: 0.14em;
		text-transform: uppercase;
		color: var(--faint);
	}

	.pilot-grid p {
		color: #444;
	}


	@media (max-width: 920px) {
		.viewer-grid,
		.schema-grid,
		.two-column,
		.pilot-grid {
			grid-template-columns: 1fr;
		}

		.viewer-toolbar,
		.transport {
			grid-template-columns: 1fr;
		}

		.viewer-toolbar {
			align-items: flex-start;
		}

		.sync-status {
			white-space: normal;
		}

		.live-inputs > div,
		.live-inputs > div:first-child {
			min-width: 100%;
			border-left: 0;
			border-top: 1px solid var(--hairline);
			padding: 0.75rem 0 0;
		}

		.live-inputs > div:first-child {
			border-top: 0;
			padding-top: 0;
		}
	}

	@media (max-width: 580px) {
		.hero-metrics,
		.capacity-metrics {
			grid-template-columns: 1fr;
		}

		.hero-metrics div,
		.capacity-metrics div {
			border-left: 0;
			border-top: 1px solid #444;
		}

		.hero-metrics div:first-child,
		.capacity-metrics div:first-child {
			border-top: 0;
		}

		.transport button {
			flex: 1;
		}
	}
</style>
