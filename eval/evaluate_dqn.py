"""Standalone evaluation of a trained SB3 DQN on the JetRacer simulator, with a simple lane-following baseline.

    python eval/evaluate_dqn.py                                   # models/dqn/jetracer_final.zip, 100 episodes per condition
    python eval/evaluate_dqn.py --model experiments/dqn/<run>/final.zip --out results/dqn_evaluation_footprint_v1

Read-only with respect to the model and the simulator: the checkpoint is only loaded, `sim/track_env.py` is imported
unchanged with its default Config (same dynamics, reward, observation `[offset/0.13, heading/pi, last_steering]`, actions
0 = left, 1 = straight, 2 = right). The policy is greedy (`model.predict(obs, deterministic=True)`), no exploration.

Conditions (each: `--episodes` rollouts per controller, fixed seeds, identical starts for DQN and baseline):
  standard      the simulator's own reset(): random place on the track, offset within +-30 % of the half-width,
                heading within +-0.2 rad.
  challenging   same limits, but the start is drawn from the OUTER 20 % of both ranges (|offset| >= 0.8 x 0.3 x half-width,
                |heading error| >= 0.8 x 0.2 rad, random signs), and it must still satisfy the simulator's own rule that the
                whole car body starts inside the lane. The environment is not modified: the start state is set on the
                instance after reset(), exactly as reset() sets it.
  robustness    EXTRA (not requested): standard starts with (a) Gaussian noise sigma 0.01 added to the offset and heading
                observations, (b) the chosen action applied one step (50 ms) late. Shows how brittle the policy is to
                the imperfections a real car has; the simulator itself is unchanged.

Definitions: laps_completed = info["lap"] at the end (whole laps; partial laps do not count), laps_fractional =
progress / track perimeter. Crash = `terminated` (any part of the body left the lane). Timeout = `truncated` (survived
max_steps = 1000 steps = 50 s). Per-episode |offset| and |heading error| are averaged over the steps of the episode, then
averaged over episodes.

Outputs (refuses to overwrite existing results): <out>/summary.json, episodes.csv, evaluation_plots.png, simulation.gif.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")        # headless rendering for the GIF

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "sim"))
from track_env import Config, JetRacerEnv  # noqa: E402

DEFAULT_MODEL = os.path.join(ROOT, "models", "dqn", "jetracer_final.zip")
DEFAULT_OUT = os.path.join(ROOT, "results", "dqn_evaluation")
ACTIONS = ("left", "straight", "right")
STANDARD_SEED, CHALLENGING_SEED, ROBUST_SEED = 20000, 30000, 40000
OUTER_BAND = 0.8                                          # challenging: |start| in [0.8, 1.0] x the permitted limit


# --------------------------------------------------------------------------------------------------
# Controllers
# --------------------------------------------------------------------------------------------------
def dqn_policy(model_path):
    from stable_baselines3 import DQN
    model = DQN.load(model_path, device="cpu")
    return (lambda obs: int(model.predict(obs, deterministic=True)[0])), model


def baseline_policy(obs):
    """Simple proportional lane follower on the same observation: steer against the offset and the heading error.
    Left of centre (+offset) or pointing left (+heading) -> steer right (action 2). Dead band 0.05."""
    u = -0.5 * obs[0] - 3.0 * obs[1]
    return 0 if u > 0.05 else 2 if u < -0.05 else 1


# --------------------------------------------------------------------------------------------------
# Starting conditions
# --------------------------------------------------------------------------------------------------
def challenging_start(env, seed):
    """reset(seed) then replace the start state by one from the outer band of the permitted ranges (see module doc).
    Returns (obs, info) with the same bookkeeping reset() performs. Falls back to the plain reset start if no valid
    outer-band start is found (reported via the returned flag)."""
    obs, _ = env.reset(seed=seed)
    c, rng = env.cfg, env.np_random
    for _ in range(500):
        idx = int(rng.integers(len(env.track_s)))
        off = rng.choice([-1, 1]) * rng.uniform(OUTER_BAND, 1.0) * c.start_offset_frac * c.lane_half_width
        her = rng.choice([-1, 1]) * rng.uniform(OUTER_BAND, 1.0) * c.start_heading_noise
        t = env.track_theta[idx]
        env.x = float(env.track_x[idx] - off * math.sin(t))
        env.y = float(env.track_y[idx] + off * math.cos(t))
        env.heading = float(t + her)
        env.idx = idx
        env._update_track_state()
        if env._footprint_margin() >= 0.0:
            env.last_steering, env.steps, env.total_progress = 0.0, 0, 0.0
            env.prev_s = env.s_along
            return env._obs(), True
    obs, _ = env.reset(seed=seed)
    return obs, False


# --------------------------------------------------------------------------------------------------
# One rollout
# --------------------------------------------------------------------------------------------------
def rollout(policy, seed, challenging=False, obs_noise=0.0, delay=0, noise_seed=0):
    env = JetRacerEnv()
    if challenging:
        obs, outer_ok = challenging_start(env, seed)
    else:
        obs, _ = env.reset(seed=seed)
        outer_ok = None
    rng = np.random.default_rng(noise_seed)
    start_off_cm, start_head_deg = 100.0 * env.offset, math.degrees(env.heading_error)
    pending = [1] * delay                                  # STRAIGHT until the first delayed action arrives
    reward, acts, offs, heads, margins = 0.0, [], [], [], []
    terminated = truncated = False
    info = {"lap": 0, "progress_m": 0.0}
    while not (terminated or truncated):
        seen = obs.copy()
        if obs_noise:
            seen[:2] += rng.normal(0.0, obs_noise, 2).astype(np.float32)
        a = policy(seen)
        if delay:
            pending.append(a)
            a = pending.pop(0)
        acts.append(a)
        obs, r, terminated, truncated, info = env.step(a)
        reward += r
        offs.append(abs(info["lane_offset_m"]))
        heads.append(abs(env.heading_error))
        margins.append(info["footprint_margin_m"])
    perimeter = env.perimeter
    acts = np.array(acts)
    return {
        "seed": seed, "reward": reward, "steps": len(acts),
        "laps_completed": int(info["lap"]), "laps_fractional": info["progress_m"] / perimeter,
        "crashed": bool(terminated), "timed_out": bool(truncated and not terminated),
        "mean_abs_offset_cm": 100.0 * float(np.mean(offs)), "mean_abs_heading_deg": math.degrees(float(np.mean(heads))),
        "n_left": int((acts == 0).sum()), "n_straight": int((acts == 1).sum()), "n_right": int((acts == 2).sum()),
        "min_body_clearance_mm": 1000.0 * float(np.min(margins)),
        "start_offset_cm": start_off_cm, "start_heading_deg": start_head_deg, "outer_band_start": outer_ok,
        "_offset_trace_cm": [100.0 * o for o in offs], "dt": env.cfg.dt,
    }


def run_condition(policy, n, base_seed, **kw):
    return [rollout(policy, base_seed + i, noise_seed=base_seed + i, **kw) for i in range(n)]


# --------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------
def summarise(rows):
    n = len(rows)
    steps = sum(r["steps"] for r in rows)
    acts = np.array([sum(r["n_left"] for r in rows), sum(r["n_straight"] for r in rows), sum(r["n_right"] for r in rows)], float)
    switches = None
    return {
        "episodes": n,
        "mean_reward": float(np.mean([r["reward"] for r in rows])),
        "mean_laps_completed": float(np.mean([r["laps_completed"] for r in rows])),
        "mean_laps_fractional": float(np.mean([r["laps_fractional"] for r in rows])),
        "pct_ge_1_lap": 100.0 * float(np.mean([r["laps_completed"] >= 1 for r in rows])),
        "pct_ge_3_laps": 100.0 * float(np.mean([r["laps_completed"] >= 3 for r in rows])),
        "pct_crashed_off_track": 100.0 * float(np.mean([r["crashed"] for r in rows])),
        "pct_survived_to_timeout": 100.0 * float(np.mean([r["timed_out"] for r in rows])),
        "mean_abs_lane_offset_cm": float(np.mean([r["mean_abs_offset_cm"] for r in rows])),
        "mean_abs_heading_error_deg": float(np.mean([r["mean_abs_heading_deg"] for r in rows])),
        "mean_episode_length_steps": steps / float(n),
        "action_share_pct": {k: float(100.0 * v / acts.sum()) for k, v in zip(ACTIONS, acts)},
        "min_body_clearance_mm_over_all_episodes": float(np.min([r["min_body_clearance_mm"] for r in rows])),
    }


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# --------------------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------------------
def pick_representatives(results):
    """(success_row, failure_row) from the DQN: the median-reward surviving episode, and the first crash
    (standard condition first, then challenging, then robustness). None if there is none."""
    order = ["standard", "challenging", "robustness_noise", "robustness_delay"]
    ok = [r for r in results["standard"]["dqn"] if r["timed_out"]]
    succ = sorted(ok, key=lambda r: r["reward"])[len(ok) // 2] if ok else None
    fail = None
    for cond in order:
        crashes = [r for r in results[cond]["dqn"] if r["crashed"]]
        if crashes:
            fail = dict(crashes[0], condition=cond)
            break
    return succ, fail


def make_plots(results, path, succ, fail):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    for cond, ls in (("standard", "-"), ("challenging", "--")):
        for name, col in (("dqn", "tab:blue"), ("baseline", "tab:orange")):
            rows = results[cond][name]
            ax[0, 0].plot([r["reward"] for r in rows], ls, color=col, alpha=0.8, label="%s, %s" % (name, cond))
            ax[0, 1].plot([r["laps_completed"] for r in rows], ls, color=col, alpha=0.8, label="%s, %s" % (name, cond))
    ax[0, 0].set(title="Episode reward (episode index = seed order)", xlabel="episode", ylabel="reward")
    ax[0, 1].set(title="Completed laps per episode (info['lap'])", xlabel="episode", ylabel="whole laps")
    ax[0, 0].legend(fontsize=8)
    for a, row, title in ((ax[0, 2], succ, "DQN, successful episode (median reward, standard start)"),
                          (ax[1, 0], fail, "DQN, failed episode")):
        if row is None:
            a.text(0.5, 0.5, "none available", ha="center", va="center", transform=a.transAxes)
        else:
            t = np.arange(len(row["_offset_trace_cm"])) * row["dt"]
            a.plot(t, row["_offset_trace_cm"], color="tab:blue")
            limit = 100.0 * (Config.lane_half_width - 0.5 * Config.car_width)
            a.axhline(limit, color="r", ls=":", label="%.1f cm: body touches the line if the car is parallel to the lane" % limit)
            sub = "" if "condition" not in row else " [%s]" % row["condition"]
            a.set(title=title + sub + "\nseed %d, %s" % (row["seed"], "crashed" if row["crashed"] else "survived"),
                  xlabel="time (s)", ylabel="|lane offset| of rear-axle centre (cm)")
            a.legend(fontsize=8)
    conds = ["standard", "challenging", "robustness_noise", "robustness_delay"]
    x = np.arange(len(conds))
    for k, (name, col) in enumerate((("dqn", "tab:blue"), ("baseline", "tab:orange"))):
        ax[1, 1].bar(x + 0.38 * k - 0.19, [summarise(results[c][name])["pct_crashed_off_track"] for c in conds], 0.38,
                     color=col, label=name)
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels(["standard", "challenging", "obs noise\n(extra)", "1-step delay\n(extra)"], fontsize=8)
    ax[1, 1].set(title="Crash rate by condition", ylabel="% episodes ending off-track")
    ax[1, 1].legend()
    shares = {n: summarise(results["standard"][n])["action_share_pct"] for n in ("dqn", "baseline")}
    for k, (name, col) in enumerate((("dqn", "tab:blue"), ("baseline", "tab:orange"))):
        ax[1, 2].bar(np.arange(3) + 0.38 * k - 0.19, [shares[name][a] for a in ACTIONS], 0.38, color=col, label=name)
    ax[1, 2].set_xticks(range(3))
    ax[1, 2].set_xticklabels(ACTIONS)
    ax[1, 2].set(title="Action distribution (standard starts)", ylabel="% of steps")
    ax[1, 2].legend()
    fig.suptitle("DQN evaluation in simulation (NOT evidence of real-world performance)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_gif(policy, path, steps=600, every=3, seed=STANDARD_SEED):
    from PIL import Image
    env = JetRacerEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = [Image.fromarray(env.render())]
    for t in range(steps):
        obs, _, term, trunc, _ = env.step(policy(obs))
        if t % every == 0:
            frames.append(Image.fromarray(env.render()))
        if term or trunc:
            break
    env.close()
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 * every * env.cfg.dt), loop=0)
    return len(frames), t + 1


def write_csv(results, path):
    cols = ["controller", "condition", "seed", "reward", "steps", "laps_completed", "laps_fractional", "crashed", "timed_out",
            "mean_abs_offset_cm", "mean_abs_heading_deg", "n_left", "n_straight", "n_right", "min_body_clearance_mm",
            "start_offset_cm", "start_heading_deg", "outer_band_start"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for cond, per in results.items():
            for name, rows in per.items():
                for r in rows:
                    w.writerow([name, cond] + [("%.4f" % r[c] if isinstance(r[c], float) else r[c]) for c in cols[2:]])


def verdict(sm):
    s, ch, nz, dl = sm["standard"]["dqn"], sm["challenging"]["dqn"], sm["robustness_noise"]["dqn"], sm["robustness_delay"]["dqn"]
    lines = []
    sim_ok = s["pct_crashed_off_track"] == 0 and ch["pct_crashed_off_track"] <= 5 and s["pct_ge_3_laps"] >= 95
    lines.append("In the simulator as defined: %s (standard %.0f%% crashes, challenging %.0f%%, >=3 laps in %.0f%% of standard episodes)."
                 % ("GOOD" if sim_ok else "NOT good enough", s["pct_crashed_off_track"], ch["pct_crashed_off_track"], s["pct_ge_3_laps"]))
    brittle = nz["pct_crashed_off_track"] > 20 or dl["pct_crashed_off_track"] > 20
    lines.append("Robustness probes: obs noise sigma 0.01 -> %.0f%% crashes; one step of action delay -> %.0f%% crashes. %s"
                 % (nz["pct_crashed_off_track"], dl["pct_crashed_off_track"],
                    "The policy is BRITTLE to small imperfections a real car has." if brittle else "No strong brittleness seen in these two probes."))
    ready = sim_ok and not brittle
    lines.append("VERDICT: %s." % ("simulation results support a cautious, supervised first physical test (low throttle, e-stop in hand)"
                                   if ready else "NOT ready for unsupervised or confident real-world testing; train with noise/delay/steering randomisation first "
                                                 "and complete camera calibration"))
    lines.append("Caveat: success in simulation does not guarantee sim-to-real transfer. The simulator has exact state, no latency, "
                 "no steering/servo error, no wheel slip and a constant 0.6 m/s; the real car has a camera-based state estimate that is "
                 "not yet calibrated or validated.")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    if not os.path.exists(a.model):
        sys.exit("model not found: %s" % a.model)
    if os.path.exists(os.path.join(a.out, "summary.json")):
        sys.exit("%s already contains results; choose a new --out (nothing is overwritten)" % a.out)
    os.makedirs(a.out, exist_ok=True)

    policy, model = dqn_policy(a.model)
    assert model.observation_space.shape == (3,) and model.action_space.n == 3, "model does not match JetRacerEnv"
    n = a.episodes
    controllers = (("dqn", policy), ("baseline", baseline_policy))
    results = {"standard": {}, "challenging": {}, "robustness_noise": {}, "robustness_delay": {}}
    for name, pol in controllers:
        print("evaluating %s ..." % name, flush=True)
        results["standard"][name] = run_condition(pol, n, STANDARD_SEED)
        results["challenging"][name] = run_condition(pol, n, CHALLENGING_SEED, challenging=True)
        results["robustness_noise"][name] = run_condition(pol, n, ROBUST_SEED, obs_noise=0.01)
        results["robustness_delay"][name] = run_condition(pol, n, ROBUST_SEED, delay=1)

    sm = {c: {name: summarise(rows) for name, rows in per.items()} for c, per in results.items()}
    fallbacks = sum(1 for r in results["challenging"]["dqn"] if r["outer_band_start"] is False)
    succ, fail = pick_representatives(results)
    make_plots(results, os.path.join(a.out, "evaluation_plots.png"), succ, fail)
    frames, gif_steps = make_gif(policy, os.path.join(a.out, "simulation.gif"))
    write_csv(results, os.path.join(a.out, "episodes.csv"))
    cfg = {k: v for k, v in vars(Config).items() if not k.startswith("_") and isinstance(v, (int, float, bool))}
    verdict_lines = verdict(sm)
    summary = {
        "model": os.path.relpath(a.model, ROOT) if a.model.startswith(ROOT) else a.model, "model_sha256": sha256(a.model),
        "model_num_timesteps": int(model.num_timesteps), "episodes_per_condition": n, "policy": "greedy, deterministic=True",
        "seeds": {"standard": STANDARD_SEED, "challenging": CHALLENGING_SEED, "robustness": ROBUST_SEED},
        "simulator_config": cfg,
        "conditions": {"standard": "simulator reset(): random place, offset <= 0.3 x half-width, heading <= 0.2 rad",
                       "challenging": "outer 20%% of both start ranges, body inside the lane (%d of %d episodes fell back to a plain start)" % (fallbacks, n),
                       "robustness_noise": "EXTRA: sigma 0.01 Gaussian noise on offset and heading observations",
                       "robustness_delay": "EXTRA: chosen action applied one step (50 ms) late"},
        "results": sm, "verdict": verdict_lines,
        "representative_episodes": {"success_seed": None if succ is None else succ["seed"],
                                    "failure": None if fail is None else {"seed": fail["seed"], "condition": fail["condition"]}},
        "gif": {"frames": frames, "steps_simulated": gif_steps, "seed": STANDARD_SEED},
    }
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== DQN evaluation: %s (%d training timesteps recorded in the file) ===" % (summary["model"], summary["model_num_timesteps"]))
    hdr = "%-26s %-9s %8s %6s %7s %7s %7s %9s %9s %6s  %s"
    print(hdr % ("condition", "ctrl", "reward", "laps", ">=1lap", ">=3lap", "crash%", "|off|cm", "|head|deg", "steps", "L/S/R %"))
    for c in results:
        for name in ("dqn", "baseline"):
            s = sm[c][name]
            ac = s["action_share_pct"]
            print(hdr % (c, name, "%.0f" % s["mean_reward"], "%.2f" % s["mean_laps_completed"], "%.0f%%" % s["pct_ge_1_lap"],
                         "%.0f%%" % s["pct_ge_3_laps"], "%.0f%%" % s["pct_crashed_off_track"], "%.2f" % s["mean_abs_lane_offset_cm"],
                         "%.2f" % s["mean_abs_heading_error_deg"], "%.0f" % s["mean_episode_length_steps"],
                         "%.0f/%.0f/%.0f" % (ac["left"], ac["straight"], ac["right"])))
    print("\n" + "\n".join(verdict_lines))
    print("\nsaved to %s: summary.json episodes.csv evaluation_plots.png simulation.gif" % a.out)


if __name__ == "__main__":
    main()
