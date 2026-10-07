import numpy as np
from config import LATENT_DIM

class FakeSimulatorAdapter:
    def __init__(self, scenario="success"):
        self.scenario = scenario
        self.reset()

    def reset(self):
        self.step_count = 0
        self.last_steering = 0.0
        self.last_throttle = 0.0

    def apply_control(self, steering, throttle):
        self.last_steering = float(steering)
        self.last_throttle = float(throttle)
        self.step_count += 1

    def get_latent_vector(self):
        z = np.zeros(LATENT_DIM, dtype=np.float32)
        z[0] = self.step_count / 100.0
        z[1] = self.last_steering
        z[2] = self.last_throttle
        if 3 <= self.step_count <= 4:
            z[3] = 1.0
        if 7 <= self.step_count <= 8:
            z[4] = 1.0
        return z

    def get_events(self, stage, assigned_dropoff):
        events = {
            "crash": False,
            "off_track": False,
            "pickup_zone": False,
            "dropoff_zone": None,
            "target_detected": False,
        }

        if self.scenario == "crash" and self.step_count == 2:
            events["crash"] = True
            return events

        if self.scenario == "off_track" and self.step_count == 2:
            events["off_track"] = True
            return events

        if stage == "pickup" and 3 <= self.step_count <= 4:
            events["pickup_zone"] = True
            events["target_detected"] = True

        if stage == "dropoff" and 7 <= self.step_count <= 8:
            events["target_detected"] = True
            if self.scenario == "wrong_dropoff":
                events["dropoff_zone"] = next(
                    x for x in ["A", "B", "C"] if x != assigned_dropoff
                )
            elif self.scenario == "success":
                events["dropoff_zone"] = assigned_dropoff

        return events
