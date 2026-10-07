import numpy as np
from config import LATENT_DIM, DROPOFF_TARGETS

def encode_stage(stage):
    return 0.0 if stage == "pickup" else 1.0

def encode_target(target):
    out = np.zeros(len(DROPOFF_TARGETS), dtype=np.float32)
    if target in DROPOFF_TARGETS:
        out[DROPOFF_TARGETS.index(target)] = 1.0
    return out

def build_observation(latent_vector, stage, assigned_target, target_detected, previous_action):
    latent = np.asarray(latent_vector, dtype=np.float32).reshape(-1)
    prev = np.asarray(previous_action, dtype=np.float32).reshape(-1)
    obs = np.concatenate([
        latent,
        np.array([encode_stage(stage)], dtype=np.float32),
        encode_target(assigned_target),
        np.array([float(bool(target_detected))], dtype=np.float32),
        prev,
    ]).astype(np.float32)
    if obs.shape != (39,):
        raise ValueError(f"Expected observation shape (39,), got {obs.shape}")
    return obs
