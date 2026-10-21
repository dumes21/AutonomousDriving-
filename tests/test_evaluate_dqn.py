"""Light checks of eval/evaluate_dqn.py: rollouts are reproducible, challenging starts obey the simulator's own start rule
and stay within the permitted limits, laps are counted as whole laps, and the model file is only read."""
import hashlib
import math
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "eval"))
import evaluate_dqn as ev  # noqa: E402


def test_rollouts_are_reproducible_and_flags_are_consistent():
    a = ev.rollout(ev.baseline_policy, 5)
    b = ev.rollout(ev.baseline_policy, 5)
    assert a["reward"] == b["reward"] and a["steps"] == b["steps"] == 1000
    assert a["timed_out"] and not a["crashed"]                       # truncated, not terminated
    assert a["laps_completed"] == int(a["laps_fractional"]) and a["laps_fractional"] >= a["laps_completed"]
    assert a["n_left"] + a["n_straight"] + a["n_right"] == a["steps"]


@pytest.mark.parametrize("seed", range(8))
def test_challenging_start_is_in_the_outer_band_inside_the_lane_and_within_limits(seed):
    env = ev.JetRacerEnv()
    obs, ok = ev.challenging_start(env, 1000 + seed)
    c = env.cfg
    assert ok and env._footprint_margin() >= 0.0
    assert ev.OUTER_BAND * c.start_offset_frac * c.lane_half_width - 1e-6 <= abs(env.offset) <= c.start_offset_frac * c.lane_half_width + 1e-6
    assert ev.OUTER_BAND * c.start_heading_noise - 1e-6 <= abs(env.heading_error) <= c.start_heading_noise + 1e-6
    assert obs.shape == (3,) and obs[2] == 0.0 and env.steps == 0 and env.total_progress == 0.0


def test_crash_is_reported_as_terminated_not_timeout():
    r = ev.rollout(lambda obs: 0, 3)                                 # always steer left: leaves the lane
    assert r["crashed"] and not r["timed_out"] and r["steps"] < 1000


def test_model_file_is_not_modified(tmp_path):
    m = os.path.join(ROOT, "models", "dqn", "jetracer_final.zip")
    if not os.path.exists(m):
        pytest.skip("no model file")
    before = hashlib.sha256(open(m, "rb").read()).hexdigest()
    policy, _ = ev.dqn_policy(m)
    ev.rollout(policy, 1)
    assert hashlib.sha256(open(m, "rb").read()).hexdigest() == before
