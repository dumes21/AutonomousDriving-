"""Validate the simulator for a given vehicle geometry (no training).

    python sim/check_footprint.py                                   # uses the numbers in sim/track_env.py Config
    python sim/check_footprint.py --front-overhang 0.03 --rear-overhang 0.04 --wheelbase 0.15 --width 0.17
                                                                    # what-if, without editing Config

Prints the footprint, how much room the car has in the lane (straight and bend), runs the scripted centre
controller on 20 fixed seeds under the footprint rule, and (if present) the old baseline DQN for comparison.
Then run the full checks:   python -m pytest -q && python sim/test_env.py
Exit code 1 if the scripted controller cannot complete laps with this geometry.
"""
import argparse
import glob
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_env import Config, JetRacerEnv  # noqa: E402
import view_track  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SEEDS = range(1000, 1020)


def margin_at(cfg, x, y, heading):
    env = JetRacerEnv(config=cfg)
    env.reset(seed=0)
    env.x, env.y, env.heading = x, y, heading
    env.idx = int(np.argmin((env.track_x - x) ** 2 + (env.track_y - y) ** 2))
    env._update_track_state()
    return env._footprint_margin()


def corridor_cm(cfg, where):
    """Width (cm) of the band of rear-axle positions, perfectly aligned with the track, whose body stays inside the lane."""
    R, Lh = cfg.corner_radius, cfg.straight_length / 2
    ok = []
    for d in np.arange(-0.10, 0.1001, 0.001):          # d > 0 = towards the inside of the (left) bend
        if where == "straight":
            m = margin_at(cfg, 0.0, -R + d, 0.0)
        else:                                            # apex of the right-hand bend, heading +y
            m = margin_at(cfg, Lh + R - d, 0.0, math.pi / 2)
        if m >= 0:
            ok.append(d)
    return (100 * (max(ok) - min(ok)), 100 * min(ok), 100 * max(ok)) if ok else (0.0, 0.0, 0.0)


def evaluate(cfg, policy_name, model=None):
    env = JetRacerEnv(config=cfg)
    if policy_name == "dqn":
        from stable_baselines3 import DQN
        m = DQN.load(model)
        pol = lambda o: int(m.predict(o, deterministic=True)[0])  # noqa: E731
    else:
        pol = view_track.make_policy(policy_name, env)
    laps, crashes, offs = [], 0, []
    for sd in SEEDS:
        o, _ = env.reset(seed=sd)
        while True:
            o, r, t, tr, i = env.step(pol(o))
            offs.append(abs(i["lane_offset_m"]))
            if t or tr:
                break
        laps.append(i["progress_m"] / env.perimeter)
        crashes += int(t)
    n = len(SEEDS)
    return float(np.mean(laps)), 100.0 * crashes / n, float(np.mean(offs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wheelbase", type=float)
    ap.add_argument("--width", type=float)
    ap.add_argument("--front-overhang", type=float, help="front axle -> front bumper, m")
    ap.add_argument("--rear-overhang", type=float, help="rear axle -> rear bumper, m")
    args = ap.parse_args()

    cfg = Config()
    for attr, val in (("wheelbase", args.wheelbase), ("car_width", args.width),
                      ("car_front_overhang", args.front_overhang), ("car_rear_overhang", args.rear_overhang)):
        if val is not None:
            setattr(cfg, attr, val)
    cfg.footprint_collision = True

    length = cfg.car_rear_overhang + cfg.wheelbase + cfg.car_front_overhang
    print("Geometry (m): width %.3f | wheelbase %.3f | rear overhang %.3f | front overhang %.3f | body length %.3f"
          % (cfg.car_width, cfg.wheelbase, cfg.car_rear_overhang, cfg.car_front_overhang, length))
    print("Track (m): lane width %.2f (half %.2f) | bend centre-line radius %.2f | straight %.2f"
          % (2 * cfg.lane_half_width, cfg.lane_half_width, cfg.corner_radius, cfg.straight_length))
    print("Min turning radius at full lock (rear axle): %.3f m (assumed max steer %.2f rad)"
          % (cfg.wheelbase / math.tan(cfg.max_steer), cfg.max_steer))

    R = cfg.corner_radius
    ms = margin_at(cfg, 0.0, -R, 0.0)
    mb = margin_at(cfg, cfg.straight_length / 2 + R, 0.0, math.pi / 2)
    cs, lo_s, hi_s = corridor_cm(cfg, "straight")
    cb, lo_b, hi_b = corridor_cm(cfg, "bend")
    print("\nRoom in the lane with the car perfectly aligned on the centre line:")
    print("  straight: clearance %+.1f cm each side; usable band for the rear axle %.1f cm wide" % (100 * ms, cs))
    print("  bend    : clearance %+.1f cm; usable band %.1f cm wide (rear axle %.1f..%.1f cm from the centre line, + = inside)"
          % (100 * mb, cb, lo_b, hi_b))
    if mb < 0:
        print("  -> the body does not fit in the bend even when perfectly placed: the track is infeasible for this geometry.")

    laps, crash, off = evaluate(cfg, "centre")
    print("\nScripted centre controller, 20 seeds, footprint rule: %.2f laps, %.0f%% crashes, mean |offset| %.3f m"
          % (laps, crash, off))
    ok = crash == 0 and laps >= 4.0
    print("  -> %s" % ("PASS: the 3-action controller can drive this geometry." if ok else
                       "FAIL: even the scripted controller cannot stay in the lane with this geometry; check the "
                       "measurements before training (bend clearance %+.1f cm; it worked above ~2.2 cm in earlier tests)."
                       % (100 * mb)))

    baseline = sorted(glob.glob(os.path.join(ROOT, "experiments", "dqn", "*baseline", "final.zip")))
    if baseline:
        laps, crash, off = evaluate(cfg, "dqn", baseline[0])
        print("\nOld baseline DQN (trained under the centre-only rule) under this geometry: %.2f laps, %.0f%% crashes, "
              "mean |offset| %.3f m\n  (%s)" % (laps, crash, off, os.path.relpath(baseline[0], ROOT)))
    print("\nNext: python -m pytest -q && python sim/test_env.py")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
