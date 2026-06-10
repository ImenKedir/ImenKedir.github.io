"""Minimal deterministic 10x10 Snake environment (CPU-only, no graphics, no ML).

This is the simulator for the "Build Your First World Model" project. It exposes a
small Gym-like API (`reset`, `step`, `render_ascii`) over a 10x10 integer grid.
"""

from __future__ import annotations

import numpy as np

GRID_SIZE = 10

EMPTY = 0
BODY = 1
HEAD = 2
FOOD = 3

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# Row/column deltas for each action. Row 0 is the top of the grid, so "up"
# decreases the row index.
_MOVES = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

# Pairs of actions that are direct reverses of each other.
_OPPOSITE = {
    UP: DOWN,
    DOWN: UP,
    LEFT: RIGHT,
    RIGHT: LEFT,
}


class SnakeEnv:
    """A deterministic 10x10 Snake environment.

    The snake is represented as a list of (row, col) cells. The first element is
    the head and the last element is the tail. The environment is fully
    deterministic once seeded: the same seed always produces the same food
    placement sequence.
    """

    def __init__(self, grid_size: int = GRID_SIZE):
        self.grid_size = grid_size
        self._rng = np.random.default_rng()
        # The snake is a list of (row, col) tuples, head first.
        self._snake: list[tuple[int, int]] = []
        self._food: tuple[int, int] | None = None
        self._direction: int = RIGHT
        self._done: bool = True

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the environment and return the initial grid.

        The snake starts with length 3, laid out horizontally in the middle of
        the grid and facing right. Food is placed on a random empty cell.
        """
        self._rng = np.random.default_rng(seed)

        mid = self.grid_size // 2
        # Head is to the right; body extends to the left. All three cells fit on
        # the grid for any reasonable grid size.
        self._snake = [
            (mid, mid),
            (mid, mid - 1),
            (mid, mid - 2),
        ]
        self._direction = RIGHT
        self._done = False
        self._place_food()

        return self._build_grid()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Advance the environment by one step.

        Returns (grid, reward, done, info).

        Reward is +1 for eating food, -1 for dying, and 0 otherwise. Reverse
        actions (180-degree turns) are ignored and the snake keeps its current
        direction.
        """
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset() first")

        # Ignore reverse-direction actions; keep moving in the current direction.
        if action in _MOVES and action != _OPPOSITE[self._direction]:
            self._direction = action

        dr, dc = _MOVES[self._direction]
        head_r, head_c = self._snake[0]
        new_head = (head_r + dr, head_c + dc)

        if self._is_collision(new_head):
            self._done = True
            return self._build_grid(), -1.0, True, {"reason": "collision"}

        ate_food = new_head == self._food

        # Move: add the new head. Remove the tail unless we ate food (growth).
        self._snake.insert(0, new_head)
        if ate_food:
            reward = 1.0
            self._place_food()
        else:
            self._snake.pop()
            reward = 0.0

        info: dict = {"ate_food": ate_food, "length": len(self._snake)}
        # `self._done` is normally False here, but may be set if eating food
        # filled the entire board (a "win"), so report it consistently.
        return self._build_grid(), reward, self._done, info

    def render_ascii(self) -> str:
        """Return a human-readable ASCII rendering of the grid.

        Legend: '.' empty, 'o' body, 'H' head, '*' food.
        """
        symbols = {EMPTY: ".", BODY: "o", HEAD: "H", FOOD: "*"}
        grid = self._build_grid()
        lines = ["".join(symbols[int(cell)] for cell in row) for row in grid]
        return "\n".join(lines)

    # -- internal helpers -------------------------------------------------

    def _build_grid(self) -> np.ndarray:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        # Body first, then head, so the head overwrites if they ever overlap.
        for r, c in self._snake[1:]:
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                grid[r, c] = BODY
        if self._food is not None:
            fr, fc = self._food
            grid[fr, fc] = FOOD
        head_r, head_c = self._snake[0]
        if 0 <= head_r < self.grid_size and 0 <= head_c < self.grid_size:
            grid[head_r, head_c] = HEAD
        return grid

    def _is_collision(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return True
        # The tail will move out of the way on a normal step, so colliding with
        # the current tail cell is allowed (unless the snake is about to grow,
        # but the head can never reach the tail on the same step it grows).
        body_without_tail = set(self._snake[:-1])
        return cell in body_without_tail

    def _place_food(self) -> None:
        occupied = set(self._snake)
        empty_cells = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in occupied
        ]
        if not empty_cells:
            # No room left: the snake fills the board (a "win"). Leave food unset.
            self._food = None
            self._done = True
            return
        idx = int(self._rng.integers(len(empty_cells)))
        self._food = empty_cells[idx]
