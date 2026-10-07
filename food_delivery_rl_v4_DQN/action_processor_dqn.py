from config import FIXED_THROTTLE, SLOW_THROTTLE

ACTION_MAP = {
    0: (-0.8, FIXED_THROTTLE),
    1: (-0.4, FIXED_THROTTLE),
    2: ( 0.0, FIXED_THROTTLE),
    3: ( 0.4, FIXED_THROTTLE),
    4: ( 0.8, FIXED_THROTTLE),
    5: (-0.4, SLOW_THROTTLE),
    6: ( 0.0, SLOW_THROTTLE),
    7: ( 0.4, SLOW_THROTTLE),
    8: ( 0.0, 0.0),
}

ACTION_NAMES = {
    0: "Left Strong",
    1: "Left Mild",
    2: "Straight",
    3: "Right Mild",
    4: "Right Strong",
    5: "Slow Left",
    6: "Slow Straight",
    7: "Slow Right",
    8: "Stop",
}

class DiscreteActionProcessor:
    def reset(self):
        self.previous_action_id = 2

    def process(self, action_id):
        action_id = int(action_id)
        if action_id not in ACTION_MAP:
            raise ValueError(f"Invalid action id: {action_id}")
        return ACTION_MAP[action_id]
