from gymnasium.utils.env_checker import check_env

from fake_simulator_dqn import FakeSimulatorAdapter
from food_delivery_dqn_env import FoodDeliveryDQNEnv

def main():
    env = FoodDeliveryDQNEnv(
        FakeSimulatorAdapter("success"),
        fixed_target="B",
    )

    check_env(env, skip_render_check=True)
    print("[PASS] Gymnasium environment checker")

    obs, info = env.reset(seed=42)

    # Step 1-2: straight
    env.step(2)
    env.step(2)

    # Step 3: stop at pickup
    obs, reward, terminated, truncated, info = env.step(8)
    assert info["event"] == "pickup"
    assert info["stage"] == "dropoff"
    assert reward == 10.01
    print("[PASS] pickup requires STOP")

    # Step 4-6: straight
    env.step(2)
    env.step(2)
    env.step(2)

    # Step 7: stop at correct drop-off
    obs, reward, terminated, truncated, info = env.step(8)
    assert info["event"] == "success"
    assert reward == 30.0
    assert terminated is True
    print("[PASS] correct delivery requires STOP")

    print("\nAll DQN environment tests passed.")

if __name__ == "__main__":
    main()
