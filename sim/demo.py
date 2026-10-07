"""Watch the sim with random actions, or measure its speed.

    python sim/demo.py               # pygame window (close it or Ctrl-C to quit)
    python sim/demo.py --benchmark   # 100k random steps, prints steps/sec
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_env import JetRacerEnv  # noqa: E402


def show():
    env = JetRacerEnv(render_mode="human")
    obs, info = env.reset(seed=0)
    try:
        while True:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                print("episode over, laps:", info["lap"])
                obs, info = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def benchmark(n_steps=100_000):
    env = JetRacerEnv()  # no render_mode: the fast path
    env.reset(seed=0)
    env.action_space.seed(0)
    start = time.perf_counter()
    for _ in range(n_steps):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    elapsed = time.perf_counter() - start
    print("%d steps in %.2f s = %.0f steps/sec" % (n_steps, elapsed, n_steps / elapsed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", help="time 100k random steps")
    args = parser.parse_args()
    benchmark() if args.benchmark else show()
