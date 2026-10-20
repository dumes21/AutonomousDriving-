import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
from track_env import Config, JetRacerEnv  # noqa: E402


def place(env, sideways=0.0, yaw=0.0):
    """Rear axle on the bottom straight, `sideways` metres left of the centre line, yawed by `yaw`."""
    env.reset(seed=0)
    env.x, env.y, env.heading = 0.0, -env.cfg.corner_radius + sideways, yaw
    env.idx = int(np.argmin((env.track_x - env.x) ** 2 + (env.track_y - env.y) ** 2))
    env.prev_s = None
    env._update_track_state()
    env.prev_s = env.s_along


def test_centred_margin_is_lane_minus_body_half_width():
    env = JetRacerEnv()
    place(env)
    assert abs(env._footprint_margin() - (env.cfg.lane_half_width - 0.5 * env.cfg.car_width)) < 2e-3


def test_body_over_the_line_ends_the_episode_even_if_the_centre_is_inside():
    c = Config()
    env = JetRacerEnv(config=c)
    # Centre 2 cm further out than the body can tolerate, yet still inside the lane itself.
    sideways = c.lane_half_width - 0.5 * c.car_width + 0.02
    place(env, sideways=sideways)
    assert abs(env.offset) < c.lane_half_width
    _, _, terminated, _, info = env.step(1)
    assert terminated and info["footprint_margin_m"] < 0

    old = Config()
    old.footprint_collision = False        # the previous rule: centre point only
    env = JetRacerEnv(config=old)
    place(env, sideways=sideways)
    assert not env.step(1)[2]


def test_yaw_reduces_clearance():
    env = JetRacerEnv()
    place(env)
    straight = env._footprint_margin()
    place(env, yaw=0.3)
    assert env._footprint_margin() < straight - 0.02


def test_every_reset_starts_with_the_whole_body_inside_the_lane():
    env = JetRacerEnv()
    for seed in range(300):
        env.reset(seed=seed)
        assert env._footprint_margin() >= 0.0


def test_footprint_dimensions_follow_the_config():
    c = Config()
    c.car_width, c.car_length = 0.17, 0.22
    env = JetRacerEnv(config=c)
    xs, ys = env._fp[:, 0], env._fp[:, 1]
    assert abs((xs.max() - xs.min()) - 0.22) < 1e-9 and abs((ys.max() - ys.min()) - 0.17) < 1e-9


def test_render_rgb_array_draws_the_footprint():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = JetRacerEnv(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3
    red = np.all(frame == (220, 50, 50), axis=2).sum()
    px = env.cfg.pixels_per_metre
    expected = env.cfg.car_width * env.cfg.car_length * px * px
    assert 0.8 * expected < red < 1.1 * expected
    env.close()
