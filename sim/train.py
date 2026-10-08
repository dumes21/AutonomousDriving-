"""Train DQN in the sim, then compare it with two baselines.

    python sim/train.py                  # 300k steps (a few minutes)
    python sim/train.py --steps 50000    # quick test

Outputs (per seed): model_log/dqn_seed0.zip, logs/train_monitor_seed0.monitor.csv,
logs/learning_curve_seed0.png,
and a results table printed at the end (copy it into the report).
"""
import argparse
import os
import sys

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_env import JetRacerEnv  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL_DIR = os.path.join(ROOT, "model_log")
LOG_DIR = os.path.join(ROOT, "logs")


def random_policy(obs, env):
    return env.action_space.sample()


def hand_controller(obs, env):
    """Steer back towards the centre line: a simple rule, no learning."""
    offset, heading_error, _ = obs
    turn = -1.0 * offset - 3.0 * heading_error  # >0 means we want to go left
    if turn > 0.15:
        return 0
    if turn < -0.15:
        return 2
    return 1


def evaluate(policy, n_episodes=20):
    """Run n_episodes with fixed seeds (same starts for every policy)."""
    env = JetRacerEnv()
    env.action_space.seed(0)
    lengths, laps, returns, crashes = [], [], [], 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done, total = False, 0.0
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs, env))
            total += reward
            done = terminated or truncated
        lengths.append(env.steps)
        laps.append(info["progress_m"] / env.perimeter)  # fractional laps
        returns.append(total)
        crashes += int(terminated)
    return {
        "length": np.mean(lengths), "laps": np.mean(laps),
        "return": np.mean(returns), "crash_rate": crashes / n_episodes,
    }


def plot_learning_curve(csv_path, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Monitor csv: 1 comment line, then columns r (return), l (length), t (time)
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=2)
    lengths, steps = data[:, 1], np.cumsum(data[:, 1])
    k = 50  # moving-average window (episodes)
    smooth = np.convolve(lengths, np.ones(k) / k, mode="valid")
    plt.figure(figsize=(6, 3.5))
    plt.plot(steps[k - 1:], smooth)
    plt.xlabel("training steps")
    plt.ylabel("episode length (mean of last %d)" % k)
    plt.title("DQN learning curve (max 1000)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Monitor records the length/return of every training episode to a csv.
    csv_base = os.path.join(LOG_DIR, "train_monitor_seed%d" % args.seed)
    env = Monitor(JetRacerEnv(), csv_base)

    # DQN learns Q(state, action) = expected future reward, with a small neural net.
    # Settings below are tuned for a tiny 3-number observation (SB3 defaults are for Atari).
    model = DQN(
        "MlpPolicy", env,
        learning_rate=1e-3,
        buffer_size=100_000,         # replay memory of past steps
        learning_starts=1_000,       # random steps before learning starts
        batch_size=64,
        gamma=0.99,                  # how much future reward matters
        train_freq=4,                # one learning update every 4 steps
        target_update_interval=500,  # how often the "target" network is refreshed
        exploration_fraction=0.2,    # epsilon falls from 1.0 to 0.05 over the first 20% of training
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [64, 64]},
        seed=args.seed, verbose=0, device="cpu",
    )
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(os.path.join(MODEL_DIR, "dqn_seed%d" % args.seed))
    plot_learning_curve(csv_base + ".monitor.csv", os.path.join(LOG_DIR, "learning_curve_seed%d.png" % args.seed))

    def dqn_policy(obs, env):
        return int(model.predict(obs, deterministic=True)[0])

    rows = [("Random actions", random_policy), ("Hand-written controller", hand_controller),
            ("DQN seed %d (%dk steps)" % (args.seed, args.steps // 1000), dqn_policy)]
    print("\n| Policy | Mean episode length (max 1000) | Mean laps | Crash rate | Mean return |")
    print("|---|---|---|---|---|")
    for name, policy in rows:
        r = evaluate(policy)
        print("| %s | %.0f | %.2f | %d%% | %.0f |" % (
            name, r["length"], r["laps"], round(100 * r["crash_rate"]), r["return"]))


if __name__ == "__main__":
    main()
