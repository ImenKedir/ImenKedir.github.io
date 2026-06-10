"""Tests for the minimal Snake environment.

Run with:  pytest snake_world_model/test_env.py
"""

import numpy as np

from env import FOOD, GRID_SIZE, HEAD, SnakeEnv


def test_reset_returns_10x10_grid():
    env = SnakeEnv()
    grid = env.reset(seed=0)
    assert isinstance(grid, np.ndarray)
    assert grid.shape == (10, 10)


def test_grid_contains_exactly_one_head():
    env = SnakeEnv()
    grid = env.reset(seed=0)
    assert np.count_nonzero(grid == HEAD) == 1


def test_grid_contains_exactly_one_food():
    env = SnakeEnv()
    grid = env.reset(seed=0)
    assert np.count_nonzero(grid == FOOD) == 1


def test_step_changes_the_state():
    env = SnakeEnv()
    grid_before = env.reset(seed=0)
    grid_after, _reward, _done, _info = env.step(3)  # move right
    assert not np.array_equal(grid_before, grid_after)


def test_wall_collision_ends_episode():
    env = SnakeEnv()
    env.reset(seed=0)
    # Snake starts in the middle facing right. Drive it straight into the right
    # wall; within grid_size steps it must hit the wall and end the episode.
    done = False
    for _ in range(GRID_SIZE):
        _grid, reward, done, _info = env.step(3)  # keep moving right
        if done:
            break
    assert done is True
    assert reward == -1.0


def test_seeded_resets_are_deterministic():
    env_a = SnakeEnv()
    env_b = SnakeEnv()
    grid_a = env_a.reset(seed=42)
    grid_b = env_b.reset(seed=42)
    assert np.array_equal(grid_a, grid_b)

    # And the same seed should also produce identical trajectories.
    for action in (3, 1, 2, 0):
        out_a = env_a.step(action)[0]
        out_b = env_b.step(action)[0]
        assert np.array_equal(out_a, out_b)


def test_different_seeds_differ():
    env_a = SnakeEnv()
    env_b = SnakeEnv()
    grid_a = env_a.reset(seed=1)
    grid_b = env_b.reset(seed=2)
    # Snake start is identical; food placement should differ for these seeds.
    assert not np.array_equal(grid_a, grid_b)
