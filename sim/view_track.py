"""Look at the simulated track and car.

    python sim/view_track.py                 # pygame window if a display exists, else a GIF
    python sim/view_track.py --gif out.gif   # force GIF output (works headless)
    python sim/view_track.py --policy random # straight (default) | random | centre | dqn
    python sim/view_track.py --policy dqn    # trained model (--model, default models/dqn/jetracer_final.zip)
    python sim/view_track.py --evaluate      # 20 episodes: DQN vs centre controller, same start states

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

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "dqn", "jetracer_final.zip")


def make_policy(name, env, model_path=DEFAULT_MODEL):
    if name == "dqn":
        from stable_baselines3 import DQN

        model = DQN.load(model_path)
        # deterministic=True: greedy action, no epsilon exploration
        return lambda obs: int(model.predict(obs, deterministic=True)[0])
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


def run_window(policy_name, model_path):
    import pygame

    env = JetRacerEnv(render_mode="human")
    policy = make_policy(policy_name, env, model_path)
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


def run_gif(path, policy_name, steps, every, model_path):
    from PIL import Image

    env = JetRacerEnv(render_mode="rgb_array")
    policy = make_policy(policy_name, env, model_path)
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


def evaluate(policy_name, model_path, episodes, base_seed):
    """Run `episodes` episodes; episode i starts from reset(seed=base_seed + i),
    so every policy sees exactly the same start positions and headings."""
    env = JetRacerEnv()
    policy = make_policy(policy_name, env, model_path)
    rows = []
    for i in range(episodes):
        obs, _ = env.reset(seed=base_seed + i)
        total, abs_off, n = 0.0, 0.0, 0
        while True:
            obs, r, term, trunc, info = env.step(policy(obs))
            total += r
            abs_off += abs(info["lane_offset_m"])
            n += 1
            if term or trunc:
                break
        rows.append({
            "reward": total,
            "progress_m": info["progress_m"],
            "laps": info["progress_m"] / env.perimeter,
            "off_track": bool(term),
            "mean_abs_offset_m": abs_off / n,
            "duration_s": n * env.cfg.dt,
        })
    env.close()
    return rows, env.perimeter


def print_episodes(name, rows):
    print("\n%s - per episode" % name)
    print("ep  reward  progress_m  laps  off_track  mean|off|_m  duration_s")
    for i, r in enumerate(rows):
        print("%2d  %6.1f  %10.2f  %4.2f  %-9s  %11.3f  %10.1f"
              % (i, r["reward"], r["progress_m"], r["laps"], "YES" if r["off_track"] else "no",
                 r["mean_abs_offset_m"], r["duration_s"]))


def summarise(rows):
    m = lambda k: float(np.mean([r[k] for r in rows]))
    return {
        "mean reward": m("reward"),
        "mean progress (m)": m("progress_m"),
        "mean laps": m("laps"),
        ">=1 full lap (%)": 100 * m_bool(rows, lambda r: r["laps"] >= 1.0),
        "off-track / crash (%)": 100 * m_bool(rows, lambda r: r["off_track"]),
        "survived to timeout (%)": 100 * m_bool(rows, lambda r: not r["off_track"]),
        "mean |lane offset| (m)": m("mean_abs_offset_m"),
        "mean duration (s)": m("duration_s"),
    }


def m_bool(rows, f):
    return float(np.mean([f(r) for r in rows]))


def run_evaluate(policies, model_path, episodes, base_seed, show_episodes):
    results = {}
    perimeter = None
    for name in policies:
        rows, perimeter = evaluate(name, model_path, episodes, base_seed)
        results[name] = rows
        if show_episodes:
            print_episodes(name, rows)
    sums = {n: summarise(r) for n, r in results.items()}
    print("\nSummary: %d episodes, seeds %d..%d (identical start states for each policy), "
          "track length %.2f m, max episode 50 s" % (episodes, base_seed, base_seed + episodes - 1, perimeter))
    keys = list(next(iter(sums.values())).keys())
    print("%-26s" % "" + "".join("%12s" % n for n in policies))
    for k in keys:
        print("%-26s" % k + "".join("%12.2f" % sums[n][k] for n in policies))
    print("Notes: laps = progress / track length from a random start point, so a policy that "
          "times out after 50 s can reach at most ~3.4 laps. Mean |offset| includes the final "
          "step of a crash (which overshoots the lane edge slightly).")
    return sums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["straight", "random", "centre", "dqn"], default="straight")
    ap.add_argument("--gif", metavar="PATH", help="save a GIF instead of opening a window")
    ap.add_argument("--steps", type=int, default=300, help="GIF length in sim steps")
    ap.add_argument("--every", type=int, default=2, help="GIF: keep every Nth frame")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="DQN .zip for --policy dqn / --evaluate")
    ap.add_argument("--evaluate", action="store_true",
                    help="no rendering: run --episodes episodes and compare dqn vs centre on the same starts")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000, help="episode i uses reset(seed=SEED+i)")
    ap.add_argument("--per-episode", action="store_true", help="--evaluate: also print one row per episode")
    args = ap.parse_args()

    if args.evaluate:
        run_evaluate(["dqn", "centre"], args.model, args.episodes, args.seed, args.per_episode)
    elif args.gif:
        run_gif(args.gif, args.policy, args.steps, args.every, args.model)
    elif has_display():
        run_window(args.policy, args.model)
    else:
        print("No display found: writing track_view.gif instead.")
        run_gif("track_view.gif", args.policy, args.steps, args.every, args.model)


if __name__ == "__main__":
    main()
