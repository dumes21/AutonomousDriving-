"""Look at the simulated track and car.

    python sim/view_track.py                 # pygame window if a display exists, else a GIF
    python sim/view_track.py --gif out.gif   # force GIF output (works headless)
    python sim/view_track.py --policy random # straight (default) | random | centre

The 'centre' policy is a hand-written P-controller (steer back toward the
lane centre) - handy to see the car actually complete laps.
Close the window (or press Esc / Ctrl-C) to quit.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_env import JetRacerEnv  # noqa: E402


def make_policy(name, env):
    if name == "straight":
        return lambda obs: 1
    if name == "random":
        return lambda obs: env.action_space.sample()

    def centre(obs):
        # obs = [lane_offset, heading_error, last_steering]; +steer = left.
        # Left of centre (+offset) or pointing left (+heading) -> steer right.
        u = -0.5 * obs[0] - 3.0 * obs[1]
        return 0 if u > 0.05 else 2 if u < -0.05 else 1
    return centre


def has_display():
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True  # macOS / Windows


def run_window(policy_name):
    import pygame

    env = JetRacerEnv(render_mode="human")
    policy = make_policy(policy_name, env)
    obs, _ = env.reset(seed=0)
    try:
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    return
            obs, _, term, trunc, info = env.step(policy(obs))
            if term or trunc:
                print("episode over (%s), laps: %d" % ("crash" if term else "timeout", info["lap"]))
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def run_gif(path, policy_name, steps, every):
    from PIL import Image

    env = JetRacerEnv(render_mode="rgb_array")
    policy = make_policy(policy_name, env)
    obs, _ = env.reset(seed=0)
    frames = [Image.fromarray(env.render())]
    try:
        for t in range(steps):
            obs, _, term, trunc, info = env.step(policy(obs))
            if t % every == 0:
                frames.append(Image.fromarray(env.render()))
            if term or trunc:
                print("episode over at step %d (%s); resetting" % (t, "crash" if term else "timeout"))
                obs, _ = env.reset()
    finally:
        env.close()
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 * every * env.cfg.dt), loop=0)
    print("wrote %s (%d frames)" % (path, len(frames)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["straight", "random", "centre"], default="straight")
    ap.add_argument("--gif", metavar="PATH", help="save a GIF instead of opening a window")
    ap.add_argument("--steps", type=int, default=300, help="GIF length in sim steps")
    ap.add_argument("--every", type=int, default=2, help="GIF: keep every Nth frame")
    args = ap.parse_args()

    if args.gif:
        run_gif(args.gif, args.policy, args.steps, args.every)
    elif has_display():
        run_window(args.policy)
    else:
        print("No display found: writing track_view.gif instead.")
        run_gif("track_view.gif", args.policy, args.steps, args.every)


if __name__ == "__main__":
    main()
