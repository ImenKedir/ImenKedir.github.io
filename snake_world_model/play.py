"""Terminal Snake — play with arrow keys, q to quit."""

import curses

from env import SnakeEnv, UP, DOWN, LEFT, RIGHT

KEY_TO_ACTION = {
    curses.KEY_UP: UP,
    curses.KEY_DOWN: DOWN,
    curses.KEY_LEFT: LEFT,
    curses.KEY_RIGHT: RIGHT,
}

TICK = 0.15  # seconds per step


def main(stdscr, env):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(int(TICK * 1000))

    env.reset()
    score = 0

    while True:
        stdscr.erase()
        stdscr.addstr(0, 0, f"Score: {score}   (arrow keys to move, q to quit)")
        for i, line in enumerate(env.render_ascii().splitlines()):
            stdscr.addstr(i + 1, 0, line)
        stdscr.refresh()

        key    = stdscr.getch()
        action = KEY_TO_ACTION.get(key, env._direction)

        if key == ord("q"):
            break

        _, reward, done, _ = env.step(action)
        if reward == 1.0:
            score += 1

        if done:
            stdscr.erase()
            stdscr.addstr(0, 0, f"Game over!  Final score: {score}")
            stdscr.addstr(1, 0, "Press any key to play again, q to quit.")
            stdscr.nodelay(False)
            stdscr.refresh()
            if stdscr.getch() == ord("q"):
                break
            env.reset()
            score = 0
            stdscr.nodelay(True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", action="store_true",
                   help="dream inside the world model instead of the real game")
    args = p.parse_args()

    if args.model:
        from world_model_env import WorldModelEnv
        env = WorldModelEnv()
    else:
        env = SnakeEnv()

    curses.wrapper(main, env)
