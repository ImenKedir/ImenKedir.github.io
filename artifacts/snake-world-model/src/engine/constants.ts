// Shared constants for the Snake world model — ported 1:1 from the Python
// project (env.py / model.py). Do not change these without re-exporting weights.

export const GRID_SIZE = 10;
export const CHANNELS = 4;
export const CELLS = GRID_SIZE * GRID_SIZE; // 100

// Cell labels (channel index in the one-hot observation).
export const EMPTY = 0;
export const BODY = 1;
export const HEAD = 2;
export const FOOD = 3;

// Actions / directions.
export const UP = 0;
export const DOWN = 1;
export const LEFT = 2;
export const RIGHT = 3;

export type Dir = 0 | 1 | 2 | 3;

// Row 0 is the top of the grid, so "up" decreases the row index.
export const MOVES: Record<number, readonly [number, number]> = {
  [UP]: [-1, 0],
  [DOWN]: [1, 0],
  [LEFT]: [0, -1],
  [RIGHT]: [0, 1],
};

export const OPPOSITE: Record<number, Dir> = {
  [UP]: DOWN,
  [DOWN]: UP,
  [LEFT]: RIGHT,
  [RIGHT]: LEFT,
};

// A label grid is a flat Uint8Array of length CELLS, row-major: index = r*W + c.
export type Labels = Uint8Array;
