"""Terminal Snake — play with arrow keys, q to quit."""

import curses
import time

from env import SnakeEnv, UP, DOWN, LEFT, RIGHT

_KEY_TO_ACTION = {
    curses.KEY_UP: UP,
    curses.KEY_DOWN: DOWN,
    curses.KEY_LEFT: LEFT,
    curses.KEY_RIGHT: RIGHT,
}

TICK = 0.15  # seconds per step


def main(stdscr: curses.window) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)  # non-blocking getch
    stdscr.timeout(int(TICK * 1000))

    env = SnakeEnv()
    env.reset()
    score = 0

    while True:
        # Draw
        stdscr.erase()
        stdscr.addstr(0, 0, f"Score: {score}   (arrow keys to move, q to quit)")
        for i, line in enumerate(env.render_ascii().splitlines()):
            stdscr.addstr(i + 1, 0, line)
        stdscr.refresh()

        # Input
        key = stdscr.getch()
        if key == ord("q"):
            break
        action = _KEY_TO_ACTION.get(key)

        # Step (if no key, keep current direction by passing None-safe value)
        if action is None:
            action = env._direction  # hold direction
        _, reward, done, _ = env.step(action)

        if reward == 1.0:
            score += 1

        if done:
            stdscr.erase()
            stdscr.addstr(0, 0, f"Game over!  Final score: {score}")
            stdscr.addstr(1, 0, "Press any key to play again, q to quit.")
            stdscr.nodelay(False)
            stdscr.refresh()
            key = stdscr.getch()
            if key == ord("q"):
                break
            env.reset()
            score = 0
            stdscr.nodelay(True)


if __name__ == "__main__":
    curses.wrapper(main)
