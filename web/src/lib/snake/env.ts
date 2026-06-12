// TypeScript port of snake_world_model/env.py — same rules, JS-seeded RNG.

export const GRID = 10;
export const EMPTY = 0, BODY = 1, HEAD = 2, FOOD = 3;
export const UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3;

const MOVES: [number, number][] = [[-1, 0], [1, 0], [0, -1], [0, 1]];
const OPPOSITE = [DOWN, UP, RIGHT, LEFT];

export function mulberry32(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a |= 0; a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

export type Cell = [number, number];

export class SnakeEnv {
	snake: Cell[] = [];
	food: Cell = [0, 0];
	direction = RIGHT;
	private rng: () => number = Math.random;

	reset(seed: number): void {
		this.rng = mulberry32(seed);
		const mid = GRID / 2;
		this.snake = [[mid, mid], [mid, mid - 1], [mid, mid - 2]];
		this.direction = RIGHT;
		this.placeFood();
	}

	/** Advance one step; returns true if the snake died. */
	step(action: number): boolean {
		if (action !== OPPOSITE[this.direction]) this.direction = action;
		const [dr, dc] = MOVES[this.direction];
		const [hr, hc] = this.snake[0];
		const nr = hr + dr, nc = hc + dc;

		const inBounds = nr >= 0 && nr < GRID && nc >= 0 && nc < GRID;
		const hitsBody = this.snake.slice(0, -1).some(([r, c]) => r === nr && c === nc);
		if (!inBounds || hitsBody) return true;

		this.snake.unshift([nr, nc]);
		if (nr === this.food[0] && nc === this.food[1]) {
			this.placeFood();
			return false;
		}
		this.snake.pop();
		return false;
	}

	/** Per-cell class labels, row-major Uint8Array(100). */
	labels(): Uint8Array {
		const g = new Uint8Array(GRID * GRID);
		for (const [r, c] of this.snake.slice(1)) g[r * GRID + c] = BODY;
		g[this.food[0] * GRID + this.food[1]] = FOOD;
		const [hr, hc] = this.snake[0];
		g[hr * GRID + hc] = HEAD;
		return g;
	}

	private placeFood(): void {
		const occupied = new Set(this.snake.map(([r, c]) => r * GRID + c));
		const empty: number[] = [];
		for (let i = 0; i < GRID * GRID; i++) if (!occupied.has(i)) empty.push(i);
		const idx = empty[Math.floor(this.rng() * empty.length)];
		this.food = [Math.floor(idx / GRID), idx % GRID];
	}
}

/** Same data-collection policy as collect.py: walk toward the food. */
export function greedyAction(snake: Cell[], food: Cell, direction: number): number {
	const [hr, hc] = snake[0];
	const [fr, fc] = food;
	let preferred: number, other: number;
	if (Math.abs(fr - hr) >= Math.abs(fc - hc)) {
		preferred = fr > hr ? DOWN : UP;
		other = fc > hc ? RIGHT : LEFT;
	} else {
		preferred = fc > hc ? RIGHT : LEFT;
		other = fr > hr ? DOWN : UP;
	}
	return preferred !== OPPOSITE[direction] ? preferred : other;
}
