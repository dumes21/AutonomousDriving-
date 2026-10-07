import random
from config import DROPOFF_TARGETS, RANDOM_SEED

class DeliveryTask:
    def __init__(self, targets=None, seed=RANDOM_SEED):
        self.targets = list(targets or DROPOFF_TARGETS)
        self.rng = random.Random(seed)
        self.reset()

    def reseed(self, seed):
        self.rng.seed(seed)

    def reset(self, assigned_dropoff=None):
        self.assigned_dropoff = (
            assigned_dropoff
            if assigned_dropoff is not None
            else self.rng.choice(self.targets)
        )
        self.stage = "pickup"
        self.pickup_completed = False
        self.success = False
        self.failure_reason = None
        return self.get_task_state()

    def complete_pickup(self):
        if self.stage != "pickup":
            return False
        self.pickup_completed = True
        self.stage = "dropoff"
        return True

    def complete_dropoff(self, dropoff_id):
        if self.stage != "dropoff":
            return False
        if dropoff_id == self.assigned_dropoff:
            self.success = True
            self.failure_reason = None
            return True
        self.success = False
        self.failure_reason = "wrong_dropoff"
        return False

    def mark_failure(self, reason):
        self.success = False
        self.failure_reason = reason

    def get_task_state(self):
        return {
            "stage": self.stage,
            "assigned_dropoff": self.assigned_dropoff,
            "pickup_completed": self.pickup_completed,
            "success": self.success,
            "failure_reason": self.failure_reason,
        }
