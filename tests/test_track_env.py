import os
import sys

import numpy as np
from stable_baselines3.common.env_checker import check_env

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
from track_env import JetRacerEnv  # noqa: E402


def place_car(env, idx, sideways):
    """Put the car on waypoint idx, moved `sideways` metres to the left (+) or right (-)
    of the centre line, pointing along the track."""
    t = env.track_theta[idx]
    env.idx = idx
    env.x = env.track_x[idx] - sideways * np.sin(t)
    env.y = env.track_y[idx] + sideways * np.cos(t)
    env.heading = t
    env._update_track_state()
    return env._obs()


def test_check_env():
    check_env(JetRacerEnv(), skip_render_check=True)


def test_offset_sign():
    env = JetRacerEnv()
    env.reset(seed=0)
    # idx 50 is on the bottom straight (heading +x), so left of centre is +y
    assert place_car(env, 50, +0.1)[0] > 0
    assert place_car(env, 50, -0.1)[0] < 0
    assert abs(place_car(env, 50, 0.0)[0]) < 1e-6
    # same check in a bend (idx 150 is in the right-hand bend)
    assert place_car(env, 150, +0.1)[0] > 0
    assert place_car(env, 150, -0.1)[0] < 0


def test_heading_error_sign():
    env = JetRacerEnv()
    env.reset(seed=0)
    place_car(env, 50, 0.0)
    env.heading += 0.3  # pointing left of the track direction
    env._update_track_state()
    assert env._obs()[1] > 0


def test_leaving_lane_terminates():
    env = JetRacerEnv()
    env.reset(seed=0)
    for _ in range(300):  # always steering left drives in a small circle -> leaves the lane
        _, reward, terminated, truncated, _ = env.step(0)
        if terminated:
            break
    assert terminated and reward == env.cfg.crash_penalty


def test_seeded_reset_is_reproducible():
    a, b = JetRacerEnv(), JetRacerEnv()
    obs_a, _ = a.reset(seed=123)
    obs_b, _ = b.reset(seed=123)
    assert np.array_equal(obs_a, obs_b)
    for action in [0, 1, 2, 2, 1, 0, 1, 1]:
        assert np.array_equal(a.step(action)[0], b.step(action)[0])


def test_observations_stay_in_bounds():
    env = JetRacerEnv()
    env.reset(seed=0)
    for _ in range(5000):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            env.reset()


def test_driving_straight_on_straight_makes_progress():
    env = JetRacerEnv()
    env.reset(seed=0)
    place_car(env, 20, 0.0)
    env.prev_s = env.s_along
    _, reward, _, _, info = env.step(1)
    assert reward > 0.9 and info["progress_m"] > 0
