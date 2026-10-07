from action_processor_dqn import DiscreteActionProcessor, ACTION_MAP
from reward_dqn import calculate_dqn_reward

def main():
    ap = DiscreteActionProcessor()
    assert len(ACTION_MAP) == 9

    s, t = ap.process(0)
    assert s == -0.8 and t == 0.55

    s, t = ap.process(8)
    assert s == 0.0 and t == 0.0

    r = calculate_dqn_reward(
        stage="pickup",
        action_is_stop=True,
        pickup_zone=True,
    )
    assert r.event == "pickup"
    assert r.reward == 10.01

    r = calculate_dqn_reward(
        stage="pickup",
        action_is_stop=True,
        pickup_zone=False,
    )
    assert r.event == "unnecessary_stop"
    assert r.reward == -0.10

    r = calculate_dqn_reward(
        stage="dropoff",
        action_is_stop=True,
        correct_dropoff_zone=True,
    )
    assert r.event == "success"
    assert r.reward == 30.0
    assert r.terminated is True

    print("All DQN core tests passed.")

if __name__ == "__main__":
    main()
