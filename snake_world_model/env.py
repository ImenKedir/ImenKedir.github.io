"""Minimal deterministic 10x10 Snake environment for the world-model project.

No graphics, no ML — just the simulator and a small Gym-like API.
"""

import numpy as np

GRID_SIZE = 10

EMPTY, BODY, HEAD, FOOD = 0, 1, 2, 3
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

# Row 0 is the top of the grid, so "up" decreases the row index.
_MOVES = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
_OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


class SnakeEnv:
    """Snake on a 10x10 grid. Seed it for fully deterministic episodes."""

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._rng = np.random.default_rng(seed)
        mid = GRID_SIZE // 2
        # Head first, body trailing to the left; the snake faces right.
        self._snake = [(mid, mid), (mid, mid - 1), (mid, mid - 2)]
        self._direction = RIGHT
        self._place_food()
        return self._build_grid()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        # Reverse turns are ignored; the snake keeps its current direction.
        if action != _OPPOSITE[self._direction]:
            self._direction = action

        dr, dc = _MOVES[self._direction]
        head_r, head_c = self._snake[0]
        new_head = (head_r + dr, head_c + dc)

        # The tail vacates its cell this step, so it never counts as a collision.
        r, c = new_head
        in_bounds = 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE
        if not in_bounds or new_head in set(self._snake[:-1]):
            return self._build_grid(), -1.0, True, {}

        self._snake.insert(0, new_head)
        if new_head == self._food:
            self._place_food()
            return self._build_grid(), 1.0, False, {}

        self._snake.pop()
        return self._build_grid(), 0.0, False, {}

    def render_ascii(self) -> str:
        obs = self._build_grid()
        # Recover a scalar label per cell from the one-hot channels.
        labels = obs.argmax(axis=-1)  # shape (GRID_SIZE, GRID_SIZE)
        symbols = {EMPTY: ".", BODY: "o", HEAD: "H", FOOD: "*"}
        rows = ["".join(symbols[int(cell)] for cell in row) for row in labels]
        return "\n".join(rows)

    def _build_grid(self) -> np.ndarray:
        """Return a (GRID_SIZE, GRID_SIZE, 4) float32 one-hot tensor.

        Channel order matches the constants: EMPTY=0, BODY=1, HEAD=2, FOOD=3.
        Every cell has exactly one channel set to 1.0.
        """
        obs = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.float32)
        obs[:, :, EMPTY] = 1.0          # all cells start as empty
        for cell in self._snake[1:]:
            obs[cell][EMPTY] = 0.0
            obs[cell][BODY] = 1.0
        obs[self._food][EMPTY] = 0.0
        obs[self._food][FOOD] = 1.0
        obs[self._snake[0]][EMPTY] = 0.0
        obs[self._snake[0]][HEAD] = 1.0
        return obs

    def _place_food(self) -> None:
        occupied = set(self._snake)
        empty = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if (r, c) not in occupied]
        self._food = empty[int(self._rng.integers(len(empty)))]
