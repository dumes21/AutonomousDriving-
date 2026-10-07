import numpy as np
import gymnasium as gym
from gymnasium import spaces

from action_processor_dqn import DiscreteActionProcessor, ACTION_NAMES
from config import LATENT_DIM, DROPOFF_TARGETS, MAX_STEPS
from delivery_task import DeliveryTask
from observation import build_observation
from reward_dqn import calculate_dqn_reward

class FoodDeliveryDQNEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, simulator, max_steps=MAX_STEPS, fixed_target=None):
        super().__init__()
        self.simulator = simulator
        self.max_steps = int(max_steps)
        self.fixed_target = fixed_target
        self.task = DeliveryTask()
        self.action_processor = DiscreteActionProcessor()

        obs_dim = LATENT_DIM + 1 + len(DROPOFF_TARGETS) + 1 + 2

        self.observation_space = spaces.Box(
            low=np.full((obs_dim,), -10.0, dtype=np.float32),
            high=np.full((obs_dim,), 10.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(9)

        self.previous_action = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.episode_reward = 0.0

    def _obs(self, target_detected=False):
        return build_observation(
            self.simulator.get_latent_vector(),
            self.task.stage,
            self.task.assigned_dropoff,
            target_detected,
            self.previous_action,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.simulator.reset()
        self.action_processor.reset()

        if seed is not None:
            self.task.reseed(seed)

        target = self.fixed_target
        if options:
            target = options.get("assigned_dropoff", target)

        self.task.reset(target)
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.episode_reward = 0.0

        return self._obs(False), {
            "event": "reset",
            "stage": self.task.stage,
            "assigned_dropoff": self.task.assigned_dropoff,
        }

    def step(self, action):
        action_id = int(action)
        steering, throttle = self.action_processor.process(action_id)

        self.simulator.apply_control(steering, throttle)
        self.step_count += 1

        events = self.simulator.get_events(
            self.task.stage,
            self.task.assigned_dropoff,
        )

        reached = events["dropoff_zone"]
        timeout = self.step_count >= self.max_steps
        action_is_stop = action_id == 8

        correct_zone = (
            self.task.stage == "dropoff"
            and reached is not None
            and reached == self.task.assigned_dropoff
        )

        wrong_zone = (
            self.task.stage == "dropoff"
            and reached is not None
            and reached != self.task.assigned_dropoff
        )

        result = calculate_dqn_reward(
            stage=self.task.stage,
            action_is_stop=action_is_stop,
            pickup_zone=events["pickup_zone"],
            correct_dropoff_zone=correct_zone,
            wrong_dropoff_zone=wrong_zone,
            crash=events["crash"],
            off_track=events["off_track"],
            timeout=timeout,
        )

        if result.event == "pickup":
            self.task.complete_pickup()
        elif result.event == "success":
            self.task.complete_dropoff(reached)
        elif result.event == "wrong_dropoff":
            self.task.complete_dropoff(reached)
        elif result.event == "crash_or_offtrack":
            self.task.mark_failure("crash" if events["crash"] else "off_track")
        elif result.event == "timeout":
            self.task.mark_failure("timeout")

        self.previous_action = np.array([steering, throttle], dtype=np.float32)
        self.episode_reward += result.reward

        terminated = bool(result.terminated)
        truncated = bool(result.event == "timeout")
        if truncated:
            terminated = False

        info = {
            "event": result.event,
            "stage": self.task.stage,
            "assigned_dropoff": self.task.assigned_dropoff,
            "pickup_completed": self.task.pickup_completed,
            "success": self.task.success,
            "failure_reason": self.task.failure_reason,
            "step_count": self.step_count,
            "episode_reward": self.episode_reward,
            "action_id": action_id,
            "action_name": ACTION_NAMES[action_id],
            "previous_action": self.previous_action.copy(),
        }

        return self._obs(events["target_detected"]), float(result.reward), terminated, truncated, info
