"""Lightweight 2D simulator for JetRacer lane following (Level 1).

A kinematic-bicycle car drives around a closed stadium (oval) track.
Everything is in metres and radians. Rendering (pygame) is optional and
completely separate from the fast step() path.

Observation (3 numbers, roughly in [-1, 1]) - this is deliberately ONLY what
the real car's camera pipeline can later give us:
    lane_offset    signed distance from the lane centre / lane half-width
                   (+ = car is LEFT of the centre line)
    heading_error  (car heading - track direction) / pi
                   (+ = car points LEFT of the track direction)
    last_steering  previous action's steering: +1 = left, 0 = straight, -1 = right

Actions: 0 = steer left, 1 = straight, 2 = steer right (fixed speed).
"""
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Config:
    """All tunables live here. Change numbers here, not inside the code."""

    # --- track (a stadium: two straights joined by two half-circles) ---
    straight_length = 2.0      # m, length of each straight
    corner_radius = 0.8        # m, radius of the centre line in the bends
    lane_half_width = 0.20     # m, car must stay within this of the centre line
    n_points = 400             # number of centre-line waypoints

    # --- car (kinematic bicycle) ---
    dt = 0.05                  # s per step (20 Hz, similar to a camera loop)
    speed = 0.6                # m/s, constant
    wheelbase = 0.17           # m, JetRacer is about 17 cm
    max_steer = 0.4            # rad (~23 deg) at full left/right

    # --- episode ---
    max_steps = 1000           # 1000 * 0.05 s = 50 s, about 3 laps
    start_offset_frac = 0.3    # start within +-30% of the half-width of the centre
    start_heading_noise = 0.2  # rad, start heading error is within +-this
    random_start_along_track = True  # start anywhere on the track, not always at one spot

    # --- reward ---
    progress_weight = 1.0      # x (distance driven along track / max possible per step)
    centre_weight = 0.1        # x (1 - |lane_offset|), small bonus for staying central
    crash_penalty = -10.0      # when the car leaves the lane (episode ends)

    # --- rendering ---
    pixels_per_metre = 120
    fps = 20


class JetRacerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": Config.fps}

    def __init__(self, render_mode=None, config=None):
        super().__init__()
        assert render_mode in (None, "human", "rgb_array")
        self.cfg = config if config is not None else Config()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        self._build_track()
        self._window = None  # pygame objects are created lazily in render()
        self._clock = None

    # ------------------------------------------------------------------
    # Track: built once with numpy. Counter-clockwise, so "left" = inside.
    # ------------------------------------------------------------------
    def _build_track(self):
        c = self.cfg
        L, R, N = c.straight_length, c.corner_radius, c.n_points
        arc = math.pi * R                      # length of one half-circle
        perimeter = 2 * L + 2 * arc
        s = np.arange(N) * (perimeter / N)     # arc length of each waypoint

        # Four pieces: bottom straight, right bend, top straight, left bend.
        # Each piece gives x, y and the track direction (tangent angle).
        x = np.zeros(N)
        y = np.zeros(N)
        theta = np.zeros(N)

        a = s < L                              # bottom straight, heading +x
        x[a] = -L / 2 + s[a]
        y[a] = -R
        theta[a] = 0.0

        b = (s >= L) & (s < L + arc)           # right bend, turning left (CCW)
        ang = (s[b] - L) / R                   # 0 -> pi
        x[b] = L / 2 + R * np.sin(ang)
        y[b] = -R * np.cos(ang)
        theta[b] = ang

        d = (s >= L + arc) & (s < 2 * L + arc)  # top straight, heading -x
        x[d] = L / 2 - (s[d] - L - arc)
        y[d] = R
        theta[d] = math.pi

        e = s >= 2 * L + arc                   # left bend
        ang = (s[e] - 2 * L - arc) / R         # 0 -> pi
        x[e] = -L / 2 - R * np.sin(ang)
        y[e] = R * np.cos(ang)
        theta[e] = math.pi + ang

        self.track_x = x
        self.track_y = y
        self.track_theta = theta               # direction of travel at each waypoint
        self.track_s = s
        self.track_cos = np.cos(theta)
        self.track_sin = np.sin(theta)
        self.perimeter = perimeter

        # Nearest-point search only looks at this many waypoints either side
        # of the previous nearest one. The car moves ~3 cm/step and waypoints
        # are ~2 cm apart, so 10 either side is plenty.
        self._near = np.arange(-10, 11)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # sets up self.np_random from the seed
        c = self.cfg
        rng = self.np_random

        # Pick a start point on the centre line, then nudge it sideways/rotate it.
        if c.random_start_along_track:
            idx = int(rng.integers(len(self.track_s)))
        else:
            idx = 0
        offset = rng.uniform(-1, 1) * c.start_offset_frac * c.lane_half_width
        heading_err = rng.uniform(-1, 1) * c.start_heading_noise
        t = self.track_theta[idx]
        # left normal of the track is (-sin t, cos t)
        self.x = float(self.track_x[idx] - offset * math.sin(t))
        self.y = float(self.track_y[idx] + offset * math.cos(t))
        self.heading = float(t + heading_err)

        self.idx = idx              # nearest waypoint (kept between steps)
        self.last_steering = 0.0    # -1..+1
        self.steps = 0
        self.total_progress = 0.0   # metres driven along the track (can go negative)
        self._update_track_state()
        self.prev_s = self.s_along

        if self.render_mode == "human":
            self.render()
        return self._obs(), {}

    def step(self, action):
        c = self.cfg
        # action 0,1,2 -> steering +1 (left), 0, -1 (right)
        self.last_steering = float(1 - action)
        delta = self.last_steering * c.max_steer

        # Kinematic bicycle model, fixed speed.
        self.x += c.speed * math.cos(self.heading) * c.dt
        self.y += c.speed * math.sin(self.heading) * c.dt
        self.heading += c.speed / c.wheelbase * math.tan(delta) * c.dt
        self.steps += 1

        self._update_track_state()

        # Distance moved along the track this step. Wrap so that crossing the
        # start/finish line doesn't look like a huge jump.
        ds = self.s_along - self.prev_s
        ds = (ds + self.perimeter / 2) % self.perimeter - self.perimeter / 2
        self.prev_s = self.s_along
        self.total_progress += ds

        off = abs(self.offset) / c.lane_half_width
        terminated = off > 1.0
        truncated = (not terminated) and self.steps >= c.max_steps

        if terminated:
            reward = c.crash_penalty
        else:
            # progress is divided by the most we could drive in one step, so it is ~1
            reward = c.progress_weight * ds / (c.speed * c.dt) + c.centre_weight * (1.0 - off)

        info = {
            "lap": max(0, int(self.total_progress // self.perimeter)),
            "lane_offset_m": self.offset,
            "progress_m": self.total_progress,
        }
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_track_state(self):
        """Find the nearest waypoint (small window only) and work out where the
        car is relative to the track: signed offset, heading error, distance along."""
        n = len(self.track_s)
        cand = (self.idx + self._near) % n
        d2 = (self.track_x[cand] - self.x) ** 2 + (self.track_y[cand] - self.y) ** 2
        self.idx = int(cand[np.argmin(d2)])

        i = self.idx
        dx = self.x - self.track_x[i]
        dy = self.y - self.track_y[i]
        tc, ts = self.track_cos[i], self.track_sin[i]
        # Split the vector (waypoint -> car) into along-track and sideways parts.
        along = dx * tc + dy * ts
        self.offset = float(-dx * ts + dy * tc)   # + = left of the centre line
        self.s_along = float(self.track_s[i] + along)
        # Heading error wrapped to [-pi, pi)
        err = self.heading - self.track_theta[i]
        self.heading_error = (err + math.pi) % (2 * math.pi) - math.pi

    def _obs(self):
        obs = np.array(
            [
                self.offset / self.cfg.lane_half_width,
                self.heading_error / math.pi,
                self.last_steering,
            ],
            dtype=np.float32,
        )
        # The crash step can overshoot the lane edge slightly, so clip to the space.
        return np.clip(obs, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Rendering (pygame, only used when asked)
    # ------------------------------------------------------------------
    def _to_px(self, x, y):
        """World metres -> pixel coordinates (y flipped, track centred)."""
        w, h = self._size
        s = self.cfg.pixels_per_metre
        return int(w / 2 + x * s), int(h / 2 - y * s)

    def render(self):
        if self.render_mode is None:
            return None
        import pygame  # imported here so the fast path never needs pygame

        c = self.cfg
        if self._window is None:
            margin = 0.5
            w = (c.straight_length + 2 * c.corner_radius + 2 * margin) * c.pixels_per_metre
            h = (2 * c.corner_radius + 2 * margin) * c.pixels_per_metre
            self._size = (int(w), int(h))
            if self.render_mode == "human":
                pygame.init()
                self._window = pygame.display.set_mode(self._size)
                pygame.display.set_caption("JetRacer sim")
                self._clock = pygame.time.Clock()
            else:
                self._window = pygame.Surface(self._size)

        surf = self._window
        surf.fill((40, 120, 60))
        pts = [self._to_px(px, py) for px, py in zip(self.track_x, self.track_y)]
        lane_px = int(2 * c.lane_half_width * c.pixels_per_metre)
        # Lane: draw a thick grey dot on every waypoint (looks like a road), then the centre line.
        for p in pts:
            pygame.draw.circle(surf, (90, 90, 90), p, lane_px // 2)
        pygame.draw.lines(surf, (230, 230, 230), True, pts, 1)

        # Car: a rotated rectangle (0.25 m x 0.15 m) plus a nose marker.
        cx, cy = self.x, self.y
        cos_h, sin_h = math.cos(self.heading), math.sin(self.heading)
        corners = []
        for lx, ly in [(0.125, 0.075), (0.125, -0.075), (-0.125, -0.075), (-0.125, 0.075)]:
            corners.append(self._to_px(cx + lx * cos_h - ly * sin_h, cy + lx * sin_h + ly * cos_h))
        pygame.draw.polygon(surf, (220, 50, 50), corners)
        pygame.draw.circle(surf, (255, 255, 0), self._to_px(cx + 0.125 * cos_h, cy + 0.125 * sin_h), 4)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(c.fps)
            return None
        return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2))

    def close(self):
        if self._window is not None and self.render_mode == "human":
            import pygame

            pygame.quit()
        self._window = None
