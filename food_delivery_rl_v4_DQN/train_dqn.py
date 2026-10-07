from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from config import (
    DQN_LEARNING_RATE, DQN_GAMMA, DQN_BATCH_SIZE,
    DQN_BUFFER_SIZE, DQN_LEARNING_STARTS,
    DQN_TARGET_UPDATE_INTERVAL,
    DQN_EXPLORATION_FRACTION, DQN_EXPLORATION_FINAL_EPS,
    DQN_TRAIN_FREQ, DQN_GRADIENT_STEPS,
    TRAIN_TIMESTEPS
)
from fake_simulator_dqn import FakeSimulatorAdapter
from food_delivery_dqn_env import FoodDeliveryDQNEnv

def main():
    env = Monitor(
        FoodDeliveryDQNEnv(
            FakeSimulatorAdapter("success"),
            fixed_target="B",
        )
    )

    Path("outputs_dqn/models").mkdir(parents=True, exist_ok=True)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=DQN_LEARNING_RATE,
        buffer_size=DQN_BUFFER_SIZE,
        learning_starts=DQN_LEARNING_STARTS,
        batch_size=DQN_BATCH_SIZE,
        gamma=DQN_GAMMA,
        train_freq=DQN_TRAIN_FREQ,
        gradient_steps=DQN_GRADIENT_STEPS,
        target_update_interval=DQN_TARGET_UPDATE_INTERVAL,
        exploration_fraction=DQN_EXPLORATION_FRACTION,
        exploration_final_eps=DQN_EXPLORATION_FINAL_EPS,
        verbose=1,
        tensorboard_log="outputs_dqn/tensorboard",
        seed=42,
    )

    model.learn(total_timesteps=TRAIN_TIMESTEPS)
    model.save("outputs_dqn/models/dqn_food_delivery_final")
    print("Saved: outputs_dqn/models/dqn_food_delivery_final.zip")

if __name__ == "__main__":
    main()
