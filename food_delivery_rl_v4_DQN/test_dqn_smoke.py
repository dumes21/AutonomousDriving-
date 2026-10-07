from pathlib import Path
import tempfile

from stable_baselines3 import DQN

from fake_simulator_dqn import FakeSimulatorAdapter
from food_delivery_dqn_env import FoodDeliveryDQNEnv

def main():
    env = FoodDeliveryDQNEnv(
        FakeSimulatorAdapter("success"),
        fixed_target="B",
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_starts=10,
        buffer_size=500,
        batch_size=16,
        target_update_interval=50,
        exploration_fraction=0.5,
        exploration_final_eps=0.05,
        verbose=0,
        seed=42,
        device="cpu",
    )

    model.learn(total_timesteps=200)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dqn_smoke"
        model.save(path)
        loaded = DQN.load(path, env=env)
        obs, info = env.reset(seed=42)
        action, _ = loaded.predict(obs, deterministic=True)
        assert 0 <= int(action) <= 8

    print("[PASS] DQN trains for 200 timesteps.")
    print("[PASS] DQN save/load works.")
    print("[PASS] Valid discrete action returned.")

if __name__ == "__main__":
    main()
