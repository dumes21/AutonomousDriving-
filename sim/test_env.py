"""Validate JetRacerEnv before any training.

    python sim/test_env.py            # check_env + 100 random steps
    python sim/test_env.py --steps 300

Exits non-zero if anything is wrong, so it can be used as a pre-flight check
on SageMaker too. (Named test_env.py but it is a script, not a pytest file:
pytest only looks in tests/.)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_env import JetRacerEnv  # noqa: E402

ACTION_NAMES = {0: "LEFT", 1: "STRAIGHT", 2: "RIGHT"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from stable_baselines3.common.env_checker import check_env

    env = JetRacerEnv()
    problems = []
    try:
        print("== stable_baselines3 check_env ==")
        check_env(env, warn=True)
        print("check_env: PASSED\n")

        print("== %d random steps (seed %d) ==" % (args.steps, args.seed))
        obs, info = env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        if not env.observation_space.contains(obs):
            problems.append("reset obs outside space: %s" % obs)
        print("reset  obs=%s" % np.round(obs, 3))

        episodes, ep_steps = 0, 0
        for t in range(1, args.steps + 1):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_steps += 1
            event = "CRASH (left lane)" if terminated else "TIMEOUT" if truncated else ""
            print("t=%3d  a=%-8s obs=%s  r=%+.3f  off=%+.3fm  %s"
                  % (t, ACTION_NAMES[int(action)], np.round(obs, 3), reward,
                     info["lane_offset_m"], event))
            if not env.observation_space.contains(obs):
                problems.append("step %d obs outside space: %s" % (t, obs))
            if obs.dtype != np.float32 or obs.shape != (3,):
                problems.append("step %d bad obs dtype/shape: %s %s" % (t, obs.dtype, obs.shape))
            if not np.isfinite(reward):
                problems.append("step %d non-finite reward" % t)
            if terminated or truncated:
                episodes += 1
                print("   -> episode %d ended after %d steps; resetting" % (episodes, ep_steps))
                obs, info = env.reset()
                ep_steps = 0
                if not env.observation_space.contains(obs):
                    problems.append("reset obs outside space: %s" % obs)

        # Same seed must give the same start (reproducibility).
        a, _ = env.reset(seed=123)
        b, _ = env.reset(seed=123)
        if not np.array_equal(a, b):
            problems.append("reset(seed=123) not reproducible")
    finally:
        env.close()

    print()
    if problems:
        print("FAILED:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("All observations inside the declared observation space; seeded reset reproducible.")
    print("OK")


if __name__ == "__main__":
    main()
