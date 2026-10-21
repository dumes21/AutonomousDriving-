"""Train (or resume) the baseline DQN on JetRacerEnv under a wall-clock budget.

Baseline is unchanged: 3 discrete actions, 3 observations, SB3 DQN MlpPolicy [64, 64].

    # 20-minute experiment (the default budget)
    python sim/train_dqn.py --name baseline

    # evaluate a saved model: 20 fixed seeds, DQN vs the scripted centre controller
    python sim/train_dqn.py --eval-only experiments/dqn/<run>/final.zip

    # watch it (window, or --gif out.gif when headless)
    python sim/view_track.py --policy dqn --model experiments/dqn/<run>/final.zip

    # continue later: new experiment folder, timestep count continues
    python sim/train_dqn.py --name baseline_cont --resume experiments/dqn/<run>/final.zip --time-budget-min 20

Each run gets its own folder  experiments/dqn/<YYYYmmdd_HHMMSS>_<name>/  (never reused, so no
earlier model or checkpoint is overwritten):
    config.json            hyperparameters, simulator Config values, git commit/dirty flag, hash of track_env.py
    training_log.csv       every ~30 s: elapsed_s, timesteps, fps, episodes, mean reward, crash rate, laps
    episodes.csv           one row per training episode: reward, length, crashed, progress, laps
    jetracer_<N>_steps.zip + replay buffer   periodic checkpoints
    final.zip (+ final_replay_buffer.pkl)    written at the end of EVERY run: time budget, step cap, Ctrl-C, SIGTERM or error
    status.json            why training stopped
    tb/                    TensorBoard   (tensorboard --logdir experiments/dqn)
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track_env  # noqa: E402
from track_env import Config, JetRacerEnv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments" / "dqn"
EVAL_EPISODES = 20
EVAL_SEED = 1000   # evaluation episode i uses reset(seed=EVAL_SEED + i); same as view_track.py

# Baseline hyperparameters (the ones that worked in sim/train.py). CLI flags default to None so that on
# --resume we can tell "not given" from "given", and never silently change a saved model.
HPARAM_DEFAULTS = {
    "learning_rate": 1e-3,
    "buffer_size": 100_000,
    "learning_starts": 1_000,
    "batch_size": 64,
    "gamma": 0.99,
    "train_freq": 4,
    "target_update_interval": 500,
    "exploration_final_eps": 0.05,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", help="experiment tag (required for training); folder = <timestamp>_<name>")
    p.add_argument("--time-budget-min", type=float, default=20.0, help="wall-clock training budget, minutes")
    p.add_argument("--max-timesteps", type=int, default=20_000_000,
                   help="safety cap on env steps in this invocation (budget normally ends the run first)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int, default=250_000,
                   help="env steps between checkpoints (each ~4 MB with its replay buffer)")
    p.add_argument("--log-every-s", type=float, default=30.0, help="seconds between training_log.csv rows")
    p.add_argument("--exploration-steps", type=int, default=100_000,
                   help="fresh runs: epsilon falls 1.0 -> final over this many env steps")
    for k, v in HPARAM_DEFAULTS.items():
        p.add_argument("--" + k.replace("_", "-"), type=type(v), default=None,
                       help="default %s. On --resume, giving a value that differs from the saved model is an error "
                            "unless --allow-param-change" % v)
    # resuming
    p.add_argument("--resume", metavar="CHECKPOINT.zip", help="continue from this checkpoint (timestep count continues)")
    p.add_argument("--replay-buffer", metavar="PKL", help="resume: replay buffer file (default: the one saved with the checkpoint)")
    p.add_argument("--resume-eps", type=float, default=None,
                   help="resume: epsilon at the start (default: the saved final epsilon, i.e. no re-exploration)")
    p.add_argument("--resume-explore-steps", type=int, default=0,
                   help="resume: steps over which epsilon decays from --resume-eps to final (0 = constant)")
    p.add_argument("--allow-param-change", action="store_true", help="resume: let CLI hyperparameters override the saved ones")
    p.add_argument("--allow-env-change", action="store_true",
                   help="resume: continue even though the simulator Config differs from the one the checkpoint was trained on")
    p.add_argument("--eval-only", metavar="MODEL.zip", help="skip training, run the 20-seed evaluation")
    p.add_argument("--no-eval", action="store_true", help="skip the automatic evaluation after training")
    p.add_argument("--out-root", default=str(EXPERIMENTS), help=argparse.SUPPRESS)
    return p.parse_args()


# --------------------------------------------------------------------------------------------------
# Environment / provenance snapshot
# --------------------------------------------------------------------------------------------------
def sim_config():
    return {k: v for k, v in vars(Config).items() if not k.startswith("_") and isinstance(v, (int, float, bool))}


def provenance():
    def git(*a):
        try:
            return subprocess.check_output(["git"] + list(a), cwd=str(ROOT), stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    src = Path(track_env.__file__).read_bytes()
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty_files": (git("status", "--porcelain", "sim") or "").splitlines(),
            "track_env_sha256": hashlib.sha256(src).hexdigest()[:16]}


def model_hparams(model):
    return {"learning_rate": model.learning_rate if not callable(model.learning_rate) else "schedule",
            "buffer_size": model.buffer_size, "learning_starts": model.learning_starts,
            "batch_size": model.batch_size, "gamma": model.gamma, "train_freq": model.train_freq.frequency,
            "target_update_interval": model.target_update_interval,
            "exploration_final_eps": model.exploration_final_eps}


# --------------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------------
def run_evaluation(model_path, out_json=None):
    """20 fixed seeds, DQN (greedy) vs centre controller on identical starts."""
    import view_track
    results = view_track.run_evaluate(["dqn", "centre"], str(model_path), EVAL_EPISODES, EVAL_SEED,
                                      show_episodes=False)
    if out_json:
        Path(out_json).write_text(json.dumps({"model": str(model_path), "seeds": [EVAL_SEED, EVAL_SEED + EVAL_EPISODES - 1],
                                              "summary": results}, indent=2))
        print("Saved", out_json)


# --------------------------------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------------------------------
def make_callbacks(run_dir, budget_s, log_every_s, checkpoint_every, perimeter, max_dt):
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

    class RunLogger(BaseCallback):
        """Wall-clock budget + episode/crash/lap logging (CSV + TensorBoard + console)."""

        def __init__(self):
            super().__init__()
            self.t0 = None
            self.stopped_by = None
            self.window = deque(maxlen=100)       # (reward, crashed, laps)
            self.episodes = 0
            self.crashes = 0
            self.total_laps = 0.0
            self.next_log = 0.0
            self.last = (0.0, 0)

        def _on_training_start(self):
            self.t0 = time.monotonic()
            self.next_log = log_every_s
            self.last = (0.0, self.model.num_timesteps)
            self.log_f = open(str(run_dir / "training_log.csv"), "w", newline="")
            self.log_w = csv.writer(self.log_f)
            self.log_w.writerow(["elapsed_s", "timesteps", "steps_per_s", "episodes", "crashes_total",
                                 "mean_reward_100", "crash_rate_100", "mean_laps_100", "lap_ge1_rate_100",
                                 "laps_completed_total", "epsilon"])
            self.ep_f = open(str(run_dir / "episodes.csv"), "w", newline="")
            self.ep_w = csv.writer(self.ep_f)
            self.ep_w.writerow(["elapsed_s", "timesteps", "reward", "length_steps", "crashed", "progress_m", "laps"])

        def _on_step(self):
            elapsed = time.monotonic() - self.t0
            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                if not done:
                    continue
                ep = info.get("episode", {})
                crashed = not info.get("TimeLimit.truncated", False)   # env ends only by crash or time limit
                progress = float(info.get("progress_m", 0.0))
                laps = progress / perimeter
                self.episodes += 1
                self.crashes += int(crashed)
                self.total_laps += max(0, int(laps))
                self.window.append((float(ep.get("r", 0.0)), crashed, laps))
                self.ep_w.writerow(["%.1f" % elapsed, self.num_timesteps, "%.3f" % ep.get("r", 0.0),
                                    int(ep.get("l", 0)), int(crashed), "%.3f" % progress, "%.3f" % laps])
            if elapsed >= self.next_log:
                self.next_log = elapsed + log_every_s
                self._log_row(elapsed)
            if elapsed >= budget_s:
                self.stopped_by = "time_budget"
                return False                      # SB3 stops learn() cleanly; the caller then saves final.zip
            return True

        def _stats(self):
            w = self.window
            if not w:
                return float("nan"), float("nan"), float("nan"), float("nan")
            n = float(len(w))
            return (sum(x[0] for x in w) / n, sum(1 for x in w if x[1]) / n, sum(x[2] for x in w) / n,
                    sum(1 for x in w if x[2] >= 1.0) / n)

        def _log_row(self, elapsed):
            rew, crash, laps, ge1 = self._stats()
            steps = self.num_timesteps
            fps = (steps - self.last[1]) / max(1e-9, elapsed - self.last[0])
            self.last = (elapsed, steps)
            eps = getattr(self.model, "exploration_rate", float("nan"))
            self.log_w.writerow(["%.1f" % elapsed, steps, "%.0f" % fps, self.episodes, self.crashes, "%.2f" % rew,
                                 "%.3f" % crash, "%.3f" % laps, "%.3f" % ge1, "%d" % self.total_laps, "%.3f" % eps])
            self.log_f.flush()
            self.ep_f.flush()
            for k, v in (("elapsed_s", elapsed), ("mean_reward_100", rew), ("crash_rate_100", crash),
                         ("mean_laps_100", laps), ("lap_ge1_rate_100", ge1)):
                self.logger.record("sim/" + k, v)
            print("[%5.1f min] steps %9d  %5.0f st/s  eps %.2f  episodes %6d  reward(100) %7.1f  crash(100) %3.0f%%  "
                  "laps(100) %.2f  lap>=1 %3.0f%%" % (elapsed / 60.0, steps, fps, eps, self.episodes, rew,
                                                       100 * crash, laps, 100 * ge1), flush=True)

        def _on_training_end(self):
            if self.t0 is not None:
                self._log_row(time.monotonic() - self.t0)
                self.log_f.close()
                self.ep_f.close()

    logger_cb = RunLogger()
    ckpt = CheckpointCallback(save_freq=checkpoint_every, save_path=str(run_dir), name_prefix="jetracer",
                              save_replay_buffer=True)
    return logger_cb, CallbackList([logger_cb, ckpt])


def resumed_epsilon_fn(start_step, extra_steps, eps_start, eps_final, explore_steps):
    """Epsilon as a function of SB3's progress_remaining for a resumed run.

    With reset_num_timesteps=False SB3 uses total = start_step + extra_steps, so
    progress_remaining = 1 - n / total with n the global step count. The stock schedule
    would be rebuilt from exploration_fraction on load and be misaligned; we replace it.
    """
    total = start_step + extra_steps

    def eps(progress_remaining):
        n = (1.0 - progress_remaining) * total
        if explore_steps <= 0:
            return eps_final
        frac = min(1.0, max(0.0, (n - start_step) / explore_steps))
        return eps_start + frac * (eps_final - eps_start)

    return eps


def replay_buffer_for(ckpt):
    """Path of the replay buffer saved with a checkpoint, or None."""
    name = ckpt.name
    cands = []
    if name == "final.zip":
        cands.append(ckpt.with_name("final_replay_buffer.pkl"))
    if name.startswith("jetracer_") and name.endswith("_steps.zip"):
        cands.append(ckpt.with_name(name.replace("jetracer_", "jetracer_replay_buffer_", 1)[:-4] + ".pkl"))
    for c in cands:
        if c.exists():
            return c
    return None


# --------------------------------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.eval_only:
        mp = Path(args.eval_only)
        run_evaluation(mp, mp.with_name(mp.stem + "_eval.json"))
        return
    if not args.name:
        sys.exit("--name is required for training (e.g. --name baseline)")

    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root) / ("%s_%s" % (stamp, args.name))
    run_dir.mkdir(parents=True, exist_ok=False)      # a run folder is never reused or overwritten

    env = Monitor(JetRacerEnv())
    perimeter = env.unwrapped.perimeter
    env.reset(seed=args.seed)
    cur_cfg = sim_config()
    prov = provenance()
    budget_s = args.time_budget_min * 60.0

    if args.resume:
        ckpt = Path(args.resume)
        # 1. The simulator must be the one the checkpoint was trained on.
        old_cfg_path = ckpt.parent / "config.json"
        if old_cfg_path.exists():
            old = json.loads(old_cfg_path.read_text()).get("sim_config", {})
            old.setdefault("footprint_collision", False)   # runs saved before the footprint rule used the centre-only rule
            diff = {k: (old[k], cur_cfg.get(k)) for k in old if old[k] != cur_cfg.get(k)}
            if not old.get('footprint_collision'):
                diff = {k: v for k, v in diff.items() if not k.startswith('car_') and k != 'footprint_samples'}
            if diff and not args.allow_env_change:
                sys.exit("Simulator Config differs from the checkpoint's run (saved vs now): %s\n"
                         "Resuming would silently train on different dynamics. Use --allow-env-change to force." % diff)
        else:
            print("WARNING: no config.json next to the checkpoint; cannot verify the simulator matches.")
        model = DQN.load(str(ckpt), env=env, device="cpu")
        # 2. Saved hyperparameters win; explicit CLI values that differ are refused.
        saved = model_hparams(model)
        changed = {}
        for k in HPARAM_DEFAULTS:
            given = getattr(args, k)
            if given is not None and given != saved[k]:
                changed[k] = (saved[k], given)
        if changed and not args.allow_param_change:
            sys.exit("Refusing to silently change saved hyperparameters (saved, requested): %s\n"
                     "Drop those flags to keep the saved values, or add --allow-param-change." % changed)
        if changed:
            print("NOTE: overriding saved hyperparameters (--allow-param-change): %s" % changed)
            for k, (_, v) in changed.items():
                if k == "train_freq":
                    model.train_freq = type(model.train_freq)(v, model.train_freq.unit)
                else:
                    setattr(model, k, v)
        start = model.num_timesteps
        eps_final = model.exploration_final_eps
        eps_start = eps_final if args.resume_eps is None else args.resume_eps
        model.exploration_schedule = resumed_epsilon_fn(start, args.max_timesteps, eps_start, eps_final,
                                                        args.resume_explore_steps)
        model.exploration_rate = eps_start
        buf = Path(args.replay_buffer) if args.replay_buffer else replay_buffer_for(ckpt)
        if buf is not None and buf.exists():
            model.load_replay_buffer(str(buf))
            print("Loaded replay buffer (%d transitions) from %s" % (model.replay_buffer.size(), buf))
        else:
            model.learning_starts = start + model.learning_starts
            print("WARNING: no replay buffer found for this checkpoint; starting with an empty one, "
                  "learning resumes after %d fresh steps." % (model.learning_starts - start))
        print("Resuming from %s at step %d (timestep count continues). Saved hyperparameters: %s"
              % (ckpt, start, saved))
        print("Epsilon: %.3f -> %.3f over %d steps" % (eps_start, eps_final, args.resume_explore_steps))
        hp = saved
    else:
        hp = {k: (getattr(args, k) if getattr(args, k) is not None else v) for k, v in HPARAM_DEFAULTS.items()}
        model = DQN(
            "MlpPolicy", env,
            learning_rate=hp["learning_rate"], buffer_size=hp["buffer_size"], learning_starts=hp["learning_starts"],
            batch_size=hp["batch_size"], gamma=hp["gamma"], train_freq=hp["train_freq"],
            target_update_interval=hp["target_update_interval"],
            # exact absolute schedule: fraction of the step cap that equals --exploration-steps
            exploration_fraction=min(1.0, args.exploration_steps / float(args.max_timesteps)),
            exploration_final_eps=hp["exploration_final_eps"],
            policy_kwargs={"net_arch": [64, 64]}, seed=args.seed, verbose=0, device="cpu",
            tensorboard_log=str(run_dir / "tb"),
        )
        start = 0
    if args.resume:
        model.tensorboard_log = str(run_dir / "tb")

    config = {"args": vars(args), "hyperparameters": hp, "sim_config": cur_cfg, "provenance": prov,
              "start_timesteps": start, "perimeter_m": perimeter, "algorithm": "SB3 DQN MlpPolicy [64,64]",
              "observation": "[lane_offset, heading_error, last_steering] (3)", "actions": "left/straight/right (3)"}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    logger_cb, callbacks = make_callbacks(run_dir, budget_s, args.log_every_s, args.checkpoint_every, perimeter,
                                          JetRacerEnv().cfg.dt)

    def on_sigterm(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, on_sigterm)

    print("Run folder: %s\nBudget: %.1f min wall clock (cap %d steps). Ctrl-C stops early and still saves final.zip."
          % (run_dir, args.time_budget_min, args.max_timesteps))
    print("Simulator: %s" % cur_cfg)
    status = {"stopped_by": "step_cap"}
    t0 = time.monotonic()
    try:
        model.learn(total_timesteps=args.max_timesteps, callback=callbacks, reset_num_timesteps=False,
                    tb_log_name="tb", progress_bar=False)
        if logger_cb.stopped_by:
            status["stopped_by"] = logger_cb.stopped_by
    except KeyboardInterrupt:
        status["stopped_by"] = "interrupted"
        print("\nInterrupted: saving what has been learned so far...")
    except BaseException as e:      # noqa: B902 - save the model whatever went wrong, then re-raise
        status["stopped_by"] = "error: %r" % (e,)
        raise
    finally:
        # Always runs: a finished (or interrupted) run is never lost.
        final = run_dir / "final.zip"
        model.save(str(final))
        model.save_replay_buffer(str(run_dir / "final_replay_buffer.pkl"))
        status.update({"elapsed_s": round(time.monotonic() - t0, 1), "timesteps": model.num_timesteps,
                       "episodes": logger_cb.episodes, "crashes": logger_cb.crashes})
        (run_dir / "status.json").write_text(json.dumps(status, indent=2))
        print("Saved %s (total timesteps %d, stopped by: %s)" % (final, model.num_timesteps, status["stopped_by"]))
        env.close()

    if not args.no_eval:
        print("\nEvaluating on %d fixed seeds..." % EVAL_EPISODES)
        run_evaluation(final, run_dir / "eval.json")
    print("\nEvaluate again:  python sim/train_dqn.py --eval-only %s" % final)
    print("Watch it:        python sim/view_track.py --policy dqn --model %s" % final)
    print("Resume:          python sim/train_dqn.py --name %s_cont --resume %s" % (args.name, final))


if __name__ == "__main__":
    main()
