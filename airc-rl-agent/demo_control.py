"""Drive a trained SAC model (like `racer demo`) with START/STOP keys in the same terminal.

    cd ~/AutonomousDriving-/airc-rl-agent
    python3 demo_control.py -model model_g200_cont2.zip

Keys: g = go   SPACE or s = stop (car stops and waits)   q = quit (stops the car first)
Every run (g ... SPACE) is timed, and each step is logged to demo_logs/ for the report.
Uses racer's own image preprocessing, action scaling and steering smoothing (BaseWrappedEnv).
"""
import argparse
import csv
import datetime
import os
import select
import signal
import sys
import termios
import time
import tty

import numpy as np
import torch
from stable_baselines3 import SAC

from learning_racer.agent.interface.base_wrapped_env import BaseWrappedEnv
from learning_racer.config import ConfigReader
from learning_racer.robot.jetracer.jetracer_env import JetRacerEnv  # import BEFORE config.load(): importing it resets ConfigReader
from learning_racer.vae.vae import VAE


class DemoEnv(BaseWrappedEnv):
    """BaseWrappedEnv with no teleoperation: the keys below decide when to drive."""

    def on_training_end(self):
        pass

    def on_rollout_start(self):
        pass

    def on_pre_step_callback(self, action):
        return action

    def on_post_step_callback(self, action, t_img, reward, done, info, z, train):
        return action, t_img, reward, done, info, z

    def on_pre_reset(self):
        pass

    def on_post_reset(self, observe):
        return observe


def say(text):
    print("\r" + text, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-model", "--model-path", required=True, help="trained SAC model, e.g. model_g200_cont2.zip")
    p.add_argument("-vae", "--vae-path", default="../vae_real.torch")
    p.add_argument("-config", "--config-path", default="config.yml")
    p.add_argument("-device", "--device", default="cuda")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions like `racer demo` does (default: deterministic, best for evaluation)")
    p.add_argument("--gui-port", type=int, default=8080, help="live web dashboard port (0 = off)")
    p.add_argument("--max-seconds", type=float, default=0,
                   help="auto-STOP each run after this many seconds (0 = off)")
    args = p.parse_args()

    config = ConfigReader()
    config.load(args.config_path)
    device = args.device if torch.cuda.is_available() else "cpu"
    say("Loading VAE and model on %s ..." % device)
    vae = VAE(image_channels=config.sac_image_channel(), z_dim=config.sac_variants_size())
    vae.load_state_dict(torch.load(args.vae_path, map_location=device))
    vae.to(device).eval()
    model = SAC.load(args.model_path, device=device)

    env = DemoEnv(JetRacerEnv(), vae, config)  # opens the camera and the car
    env.enable_dashboard(args.gui_port, "DEMO")

    def stop_car():
        for _ in range(2):
            try:
                env.env.controller.action(0.0, 0.0)
                return
            except Exception as e:
                say("WARNING: failed to stop the car: %s" % e)

    def on_signal(signum, frame):  # SSH dropped (SIGHUP) or killed (SIGTERM): stop the car on the way out
        raise SystemExit(1)

    signal.signal(signal.SIGHUP, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    os.makedirs("demo_logs", exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = os.path.splitext(os.path.basename(args.model_path))[0]
    step_path = os.path.join("demo_logs", "demo_%s_%s_steps.csv" % (name, stamp))
    runs_path = os.path.join("demo_logs", "demo_%s_%s_runs.csv" % (name, stamp))
    step_file, runs_file = open(step_path, "w", newline=""), open(runs_path, "w", newline="")
    step_log, runs_log = csv.writer(step_file), csv.writer(runs_file)
    step_log.writerow(["run", "step", "t", "steering", "throttle"])
    runs_log.writerow(["run", "steps", "seconds", "steps_per_s", "ended_by"])

    old_tty = termios.tcgetattr(sys.stdin)
    driving, run, steps, t0, obs = False, 0, 0, 0.0, None
    runs = []

    def end_run(ended_by):
        stop_car()
        secs = time.time() - t0
        runs.append((run, steps, secs))
        runs_log.writerow([run, steps, "%.2f" % secs, "%.1f" % (steps / max(secs, 1e-6)), ended_by])
        runs_file.flush()
        step_file.flush()
        say("<-- RUN %d stopped (%s): %d steps in %.1f s (%.1f steps/s). Press g for another run, q to quit."
            % (run, ended_by, steps, secs, steps / max(secs, 1e-6)))

    try:
        # Warm-up so the first run reacts as fast as later ones (first GPU calls are slow).
        # Encodes camera frames and runs the policy WITHOUT sending anything to the car.
        obs = env.reset()
        for _ in range(5):
            env.encode_observe(env.env.observer.observation())
            model.predict(obs, deterministic=True)
        stop_car()
        tty.setcbreak(sys.stdin.fileno())
        say("READY. Put the car on the track.  g = go   SPACE/s = stop   q = quit  (%s actions)"
            % ("stochastic" if args.stochastic else "deterministic"))

        while True:
            key = ""
            if select.select([sys.stdin], [], [], 0 if driving else 0.1)[0]:
                key = sys.stdin.read(1).lower()
            if key == "q":
                break
            if key == "g" and not driving:
                obs = env.reset()  # zero the car, clear the action history, fresh camera frame
                driving, run, steps, t0 = True, run + 1, 0, time.time()
                say("--> RUN %d driving (SPACE = stop)" % run)
            elif key in (" ", "s") and driving:
                driving = False
                end_run("stop key")
            if driving:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                action = np.clip(action, env.action_space.low, env.action_space.high)  # safety net
                obs, _, _, _ = env.step(action)
                steps += 1
                steer, throttle = env.action_history[-2:]  # what was actually sent (after scaling/smoothing)
                step_log.writerow([run, steps, "%.3f" % (time.time() - t0), "%.4f" % steer, "%.4f" % throttle])
                if args.max_seconds and time.time() - t0 >= args.max_seconds:
                    driving = False
                    end_run("max-seconds")
    except KeyboardInterrupt:
        pass
    finally:
        if driving:
            end_run("quit")
        stop_car()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
        step_file.close()
        runs_file.close()
        say("Car stopped. %d run(s). Logs: %s, %s" % (len(runs), step_path, runs_path))
        sys.stdout.flush()
        os._exit(0)  # skip the camera release, which can hang for minutes


if __name__ == "__main__":
    main()
