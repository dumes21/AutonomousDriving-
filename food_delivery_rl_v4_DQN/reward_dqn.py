from dataclasses import dataclass
from config import (
    ALIVE_REWARD, PICKUP_REWARD, DELIVERY_REWARD,
    CRASH_PENALTY, WRONG_DROPOFF_PENALTY,
    TIMEOUT_PENALTY, UNNECESSARY_STOP_PENALTY
)

@dataclass
class RewardResult:
    reward: float
    terminated: bool
    event: str

def calculate_dqn_reward(
    stage,
    action_is_stop=False,
    pickup_zone=False,
    correct_dropoff_zone=False,
    wrong_dropoff_zone=False,
    crash=False,
    off_track=False,
    timeout=False,
):
    if crash or off_track:
        return RewardResult(CRASH_PENALTY, True, "crash_or_offtrack")
    if timeout:
        return RewardResult(TIMEOUT_PENALTY, True, "timeout")
    if stage == "dropoff" and wrong_dropoff_zone and action_is_stop:
        return RewardResult(WRONG_DROPOFF_PENALTY, True, "wrong_dropoff")
    if stage == "dropoff" and correct_dropoff_zone and action_is_stop:
        return RewardResult(DELIVERY_REWARD, True, "success")
    if stage == "pickup" and pickup_zone and action_is_stop:
        return RewardResult(PICKUP_REWARD + ALIVE_REWARD, False, "pickup")
    if action_is_stop:
        return RewardResult(UNNECESSARY_STOP_PENALTY, False, "unnecessary_stop")
    return RewardResult(ALIVE_REWARD, False, "normal")
