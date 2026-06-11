// Port of the real Snake environment (snake_world_model/env.py) plus the greedy
// policy (snake_world_model/collect.py). This is the ground-truth game used to
// seed the dream and to compare against the model's predictions.
//
// numpy RNG parity is NOT needed: only the model's forward pass must match
// Python. We use a small seedable PRNG so episodes are reproducible in-browser.

import {
  GRID_SIZE,
  CELLS,
  BODY,
  HEAD,
  FOOD,
  UP,
  DOWN,
  LEFT,
  RIGHT,
  MOVES,
  OPPOSITE,
} from "./constants";
import { type Dir } from "./constants";

export type Cell = [number, number]; // [row, col]

/** mulberry32 — tiny, fast, seedable PRNG returning floats in [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface StepResult {
  reward: number;
  done: boolean;
}

export class SnakeGame {
  snake: Cell[] = [];
  food: Cell = [0, 0];
  direction: Dir = RIGHT;
  done = false;
  private rand: () => number;

  constructor(seed: number) {
    this.rand = mulberry32(seed);
    this.reset();
  }

  reset(): void {
    const mid = Math.floor(GRID_SIZE / 2);
    // Head first, body trailing to the left; the snake faces right.
    this.snake = [
      [mid, mid],
      [mid, mid - 1],
      [mid, mid - 2],
    ];
    this.direction = RIGHT;
    this.done = false;
    this.placeFood();
  }

  step(action: Dir): StepResult {
    if (this.done) return { reward: 0, done: true };

    // Reverse turns are ignored; the snake keeps its current direction.
    if (action !== OPPOSITE[this.direction]) this.direction = action;

    const [dr, dc] = MOVES[this.direction];
    const [hr, hc] = this.snake[0];
    const newHead: Cell = [hr + dr, hc + dc];
    const [r, c] = newHead;

    const inBounds = r >= 0 && r < GRID_SIZE && c >= 0 && c < GRID_SIZE;
    // The tail vacates its cell this step, so it never counts as a collision.
    let hitsBody = false;
    for (let i = 0; i < this.snake.length - 1; i++) {
      if (this.snake[i][0] === r && this.snake[i][1] === c) {
        hitsBody = true;
        break;
      }
    }
    if (!inBounds || hitsBody) {
      this.done = true;
      return { reward: -1, done: true };
    }

    this.snake.unshift(newHead);
    if (newHead[0] === this.food[0] && newHead[1] === this.food[1]) {
      this.placeFood();
      return { reward: 1, done: false };
    }

    this.snake.pop();
    return { reward: 0, done: false };
  }

  /** Flat row-major label grid (length CELLS). */
  labels(): Uint8Array {
    const grid = new Uint8Array(CELLS); // EMPTY = 0
    for (let i = 1; i < this.snake.length; i++) {
      const [r, c] = this.snake[i];
      grid[r * GRID_SIZE + c] = BODY;
    }
    grid[this.food[0] * GRID_SIZE + this.food[1]] = FOOD;
    grid[this.snake[0][0] * GRID_SIZE + this.snake[0][1]] = HEAD;
    return grid;
  }

  /** Apples eaten so far (snake starts at length 3). */
  get score(): number {
    return this.snake.length - 3;
  }

  private placeFood(): void {
    const occupied = new Set(this.snake.map(([r, c]) => r * GRID_SIZE + c));
    const empty: Cell[] = [];
    for (let r = 0; r < GRID_SIZE; r++) {
      for (let c = 0; c < GRID_SIZE; c++) {
        if (!occupied.has(r * GRID_SIZE + c)) empty.push([r, c]);
      }
    }
    this.food = empty[Math.floor(this.rand() * empty.length)];
  }
}

/** Greedy "head toward the food" policy, ported from collect.py. */
export function greedyAction(snake: Cell[], food: Cell, direction: Dir): Dir {
  const [hr, hc] = snake[0];
  const [fr, fc] = food;
  let preferred: Dir;
  let other: Dir;
  if (Math.abs(fr - hr) >= Math.abs(fc - hc)) {
    preferred = fr > hr ? DOWN : UP;
    other = fc > hc ? RIGHT : LEFT;
  } else {
    preferred = fc > hc ? RIGHT : LEFT;
    other = fr > hr ? DOWN : UP;
  }
  return preferred !== OPPOSITE[direction] ? preferred : other;
}
