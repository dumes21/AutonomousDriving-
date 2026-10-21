"""DQN inference for the JetRacer policy (Python 3.6 + numpy only for the .npz form; no hardware).

The trained checkpoint is a Stable-Baselines3 DQN (MlpPolicy, 3 -> 64 -> 64 -> 3, ReLU). SB3 2.x needs Python >= 3.10,
which the Jetson Nano (Python 3.6) cannot run, so the Q-network is also available as a plain numpy forward pass:

    python inference/dqn_policy.py export experiments/dqn/20261020_151620_footprint_v1/final.zip inference/dqn_footprint_v1.npz
    (laptop only; verifies the numpy network against SB3 on random inputs and never touches the checkpoint)

Conventions (read from sim/track_env.py and the training code):
  * observation = [lane_offset / 0.13, heading_error / pi, last_steering], float32, first two in [-1, 1]
  * last_steering = float(1 - action_index): +1 after LEFT, 0 after STRAIGHT, -1 after RIGHT (SIMULATOR convention)
  * actions: 0 = LEFT (+0.4 rad in the sim), 1 = STRAIGHT, 2 = RIGHT (-0.4 rad). Greedy: argmax of the Q-values.
  * the simulator decides at 20 Hz (dt = 0.05 s) at a fixed 0.6 m/s.
The physical car uses the OPPOSITE steering sign (negative = left) and straight is not 0; nothing here converts that:
see inference/vehicle_adapter.py.
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

ACTION_LABELS = {0: "LEFT", 1: "STRAIGHT", 2: "RIGHT"}
OBS_NAMES = ("lane_offset/0.13", "heading_error/pi", "last_steering")


class ObservationError(ValueError):
    pass


def sim_steering_of_action(action):
    """The simulator's last_steering for an action index: +1 left, 0 straight, -1 right."""
    return float(1 - int(action))


def validate_observation(obs):
    """Return a float32 array of shape (3,) or raise ObservationError.
    Checks shape, finiteness and the normalisation ranges the policy was trained on."""
    try:
        arr = np.asarray(obs, dtype=np.float64)
    except (TypeError, ValueError):
        raise ObservationError("observation is not numeric")
    if arr.shape != (3,):
        raise ObservationError("observation must have shape (3,), got %s" % (arr.shape,))
    if not np.all(np.isfinite(arr)):
        raise ObservationError("observation contains NaN or infinity: %s" % arr)
    if abs(arr[0]) > 1.0 or abs(arr[1]) > 1.0:
        raise ObservationError("offset/heading must be normalised into [-1, 1], got %s" % arr[:2])
    if arr[2] not in (-1.0, 0.0, 1.0):
        raise ObservationError("last_steering must be -1, 0 or +1 (simulator convention), got %s" % arr[2])
    return arr.astype(np.float32)


class NumpyQNet(object):
    """3 -> 64 -> 64 -> 3 ReLU MLP, weights as numpy arrays (SB3 layout: W has shape (out, in))."""

    def __init__(self, weights, biases, source=""):
        if len(weights) != 3 or len(biases) != 3:
            raise ValueError("expected 3 linear layers")
        self.weights = [np.asarray(w, np.float32) for w in weights]
        self.biases = [np.asarray(b, np.float32) for b in biases]
        if self.weights[0].shape[1] != 3 or self.weights[-1].shape[0] != 3:
            raise ValueError("network must map 3 inputs to 3 actions")
        self.source = source

    def q_values(self, obs):
        x = np.asarray(obs, np.float32)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = w.dot(x) + b
            if i < 2:
                x = np.maximum(x, 0.0)
        return x

    @classmethod
    def from_npz(cls, path):
        d = np.load(path)
        meta = json.loads(str(d["meta"]))
        return cls([d["w0"], d["w1"], d["w2"]], [d["b0"], d["b1"], d["b2"]], meta.get("source", path))

    @classmethod
    def from_sb3_model(cls, model):
        sd = model.policy.q_net.q_net.state_dict()
        keys = sorted(k for k in sd if k.endswith("weight"))
        w = [sd[k].cpu().numpy() for k in keys]
        b = [sd[k.replace("weight", "bias")].cpu().numpy() for k in keys]
        return cls(w, b, "sb3")


class Action(object):
    def __init__(self, index, q_values, last_steering_before):
        self.index = int(index)
        self.label = ACTION_LABELS[self.index]
        self.sim_steering = sim_steering_of_action(self.index)     # +1 left, 0 straight, -1 right
        self.q_values = [float(q) for q in q_values]
        self.last_steering_before = last_steering_before


class DQNPolicy(object):
    """Greedy (deterministic) DQN with the previous-steering state tracked exactly as in training."""

    def __init__(self, qnet):
        self.qnet = qnet
        self.last_steering = 0.0               # the simulator's reset value

    def reset(self):
        """Back to the training reset state (last_steering = 0). Call after any stop, before resuming."""
        self.last_steering = 0.0

    def build_observation(self, offset_norm, heading_norm):
        return validate_observation([offset_norm, heading_norm, self.last_steering])

    def act(self, obs):
        """Greedy action for a validated observation. obs[2] must equal the tracked previous steering."""
        obs = validate_observation(obs)
        if obs[2] != self.last_steering:
            raise ObservationError("last_steering in the observation (%+.0f) does not match the policy's tracked "
                                   "previous action (%+.0f)" % (obs[2], self.last_steering))
        q = self.qnet.q_values(obs)
        if not np.all(np.isfinite(q)):
            raise ObservationError("network produced non-finite Q-values")
        action = Action(int(np.argmax(q)), q, self.last_steering)
        self.last_steering = action.sim_steering            # same update as JetRacerEnv.step()
        return action


def load_policy(path):
    """Load from an exported .npz (numpy only) or from the SB3 .zip (needs stable-baselines3 on the laptop)."""
    if path.endswith(".npz"):
        return DQNPolicy(NumpyQNet.from_npz(path))
    try:
        from stable_baselines3 import DQN
    except ImportError:
        raise ImportError("loading a .zip checkpoint needs stable-baselines3 (Python >= 3.10). On the car use the .npz "
                          "made by `python inference/dqn_policy.py export ...`")
    return DQNPolicy(NumpyQNet.from_sb3_model(DQN.load(path, device="cpu")))


def export_npz(zip_path, out_path, n_check=3000):
    if os.path.exists(out_path):
        raise IOError("%s already exists; nothing is overwritten" % out_path)
    from stable_baselines3 import DQN
    model = DQN.load(zip_path, device="cpu")
    net = NumpyQNet.from_sb3_model(model)
    rng = np.random.RandomState(0)
    obs = np.column_stack([rng.uniform(-1, 1, n_check), rng.uniform(-1, 1, n_check), rng.choice([-1.0, 0.0, 1.0], n_check)])
    ref = np.array([int(model.predict(o.astype(np.float32), deterministic=True)[0]) for o in obs])
    mine = np.array([int(np.argmax(net.q_values(o))) for o in obs])
    if not np.array_equal(ref, mine):
        raise RuntimeError("numpy network disagrees with SB3 on %d of %d inputs" % (int((ref != mine).sum()), n_check))
    sha = hashlib.sha256(open(zip_path, "rb").read()).hexdigest()
    meta = {"source": os.path.relpath(zip_path), "source_sha256": sha, "observation": list(OBS_NAMES),
            "actions": ACTION_LABELS, "architecture": "3-64-64-3 ReLU", "checked_inputs": n_check,
            "note": "exported read-only from the SB3 checkpoint; greedy argmax of the Q-values"}
    np.savez(out_path, w0=net.weights[0], b0=net.biases[0], w1=net.weights[1], b1=net.biases[1],
             w2=net.weights[2], b2=net.biases[2], meta=json.dumps(meta))
    return meta


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "export":
        print(json.dumps(export_npz(sys.argv[2], sys.argv[3]), indent=2))
    else:
        sys.exit("usage: python inference/dqn_policy.py export CHECKPOINT.zip OUT.npz")
