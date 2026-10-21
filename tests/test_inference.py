"""Offline tests for the DQN inference + safety pipeline. No hardware is touched: no jetracer/jetcam import, the car is a fake."""
import math
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for sub in ("tests", "inference", "perception", "car"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import dqn_policy as dp  # noqa: E402
import pipeline as pl  # noqa: E402
import state_estimator as se  # noqa: E402
import vehicle_adapter as va  # noqa: E402
import ground_plane as gp  # noqa: E402
import test_state_estimator as tse  # noqa: E402  (synthetic camera fixture + pose renderer)

CKPT = os.path.join(ROOT, "experiments", "dqn", "20261020_151620_footprint_v1", "final.zip")
NPZ = os.path.join(ROOT, "inference", "dqn_footprint_v1.npz")
INITIAL_CAL = os.path.join(ROOT, "perception", "calibration", "initial_fit_from_recordings.json")
REAL_FRAME = os.path.join(ROOT, "data", "20261013_001444", "000000.jpg")
need_ckpt = pytest.mark.skipif(not os.path.exists(CKPT), reason="checkpoint not present")


def tiny_policy(winner):
    """Numpy net whose argmax is always `winner` (for action-mapping tests)."""
    w0, w1 = np.zeros((64, 3)), np.zeros((64, 64))
    w2, b2 = np.zeros((3, 64)), np.zeros(3)
    b2[winner] = 1.0
    return dp.DQNPolicy(dp.NumpyQNet([w0, w1, w2], [np.zeros(64), np.zeros(64), b2]))


def synthetic_pipeline(policy=None, **kw):
    est = se.StateEstimator(tse.fixture_calibration(), scale_source="model", min_confidence=0.3)
    p = pl.CameraToActionPipeline(est, policy or dp.load_policy(NPZ), **kw)
    return p


def arm(p, frame, t=0.0):
    """Feed enough good frames, then arm like the operator would."""
    for i in range(6):
        p.step(frame, t + 0.1 * i, t + 0.1 * i)
    assert p.arm()
    return t + 0.7


GOOD = tse.pose_image(0.0, 0.0)


# ---------------------------------------------------------------- 1. checkpoint loading / 4. determinism
@need_ckpt
def test_checkpoint_loads_with_expected_architecture_and_spaces():
    from stable_baselines3 import DQN
    m = DQN.load(CKPT, device="cpu")
    assert m.observation_space.shape == (3,) and m.action_space.n == 3
    net = dp.NumpyQNet.from_sb3_model(m)
    assert [w.shape for w in net.weights] == [(64, 3), (64, 64), (3, 64)]


@need_ckpt
def test_numpy_export_matches_sb3_predictions_and_is_deterministic():
    from stable_baselines3 import DQN
    m = DQN.load(CKPT, device="cpu")
    net = dp.NumpyQNet.from_npz(NPZ)
    rng = np.random.RandomState(1)
    for _ in range(500):
        obs = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.choice([-1.0, 0.0, 1.0])], np.float32)
        assert int(np.argmax(net.q_values(obs))) == int(m.predict(obs, deterministic=True)[0])
    p1, p2 = dp.load_policy(NPZ), dp.load_policy(NPZ)
    o = p1.build_observation(0.2, -0.05)
    assert [p1.act(o).index for _ in range(1)] == [p2.act(o).index for _ in range(1)]


def test_checkpoint_file_is_not_modified_by_loading():
    if not os.path.exists(CKPT):
        pytest.skip("no checkpoint")
    import hashlib
    h = lambda: hashlib.sha256(open(CKPT, "rb").read()).hexdigest()  # noqa: E731
    before = h()
    dp.load_policy(NPZ)
    assert h() == before


# ---------------------------------------------------------------- 2/11. observation validation
@pytest.mark.parametrize("bad", [[0, 0], [0, 0, 0, 0], [float("nan"), 0, 0], [0, float("inf"), 0], [1.5, 0, 0],
                                 [0, -2, 0], [0, 0, 0.35], "abc", None])
def test_invalid_observations_are_rejected(bad):
    with pytest.raises(dp.ObservationError):
        dp.validate_observation(bad)


def test_valid_observation_is_float32_shape_3():
    o = dp.validate_observation([0.5, -0.25, 1])
    assert o.dtype == np.float32 and o.shape == (3,)


# ---------------------------------------------------------------- 3. action mapping / 5. previous steering
@pytest.mark.parametrize("winner,label,steer", [(0, "LEFT", 1.0), (1, "STRAIGHT", 0.0), (2, "RIGHT", -1.0)])
def test_action_index_label_and_simulator_steering(winner, label, steer):
    p = tiny_policy(winner)
    a = p.act(p.build_observation(0.0, 0.0))
    assert (a.index, a.label, a.sim_steering) == (winner, label, steer)
    assert p.last_steering == steer == float(1 - winner)


def test_previous_steering_is_tracked_and_reset():
    p = tiny_policy(0)
    p.act(p.build_observation(0, 0))
    assert p.last_steering == 1.0
    assert p.build_observation(0, 0)[2] == 1.0
    with pytest.raises(dp.ObservationError):                    # inconsistent history is refused
        p.act([0.0, 0.0, -1.0])
    p.reset()
    assert p.last_steering == 0.0


@need_ckpt
def test_real_policy_signs_match_the_simulator_convention():
    p = dp.load_policy(NPZ)
    assert p.act(p.build_observation(+0.3, 0.0)).label == "RIGHT"    # car left of centre -> steer right
    p.reset()
    assert p.act(p.build_observation(-0.3, 0.0)).label == "LEFT"     # car right of centre -> steer left
    p.reset()
    assert p.act(p.build_observation(0.0, +0.1)).label == "RIGHT"    # car pointing left of the track -> steer right


# ---------------------------------------------------------------- 6. synthetic calibrated frames -> prediction
@need_ckpt
def test_synthetic_calibrated_frame_produces_a_dqn_action_and_it_has_the_right_sign():
    p = synthetic_pipeline()
    t = arm(p, GOOD)
    left_of_centre = p.step(tse.pose_image(+0.04, 0.0), t, t)
    assert not left_of_centre.stop and left_of_centre.status == "OK", (left_of_centre.reason,)
    assert left_of_centre.observation.shape == (3,) and left_of_centre.label in ("LEFT", "STRAIGHT", "RIGHT")
    assert left_of_centre.label == "RIGHT"                          # car 4 cm left of centre must steer right
    p2 = synthetic_pipeline()
    t = arm(p2, GOOD)
    assert p2.step(tse.pose_image(-0.04, 0.0), t, t).label == "LEFT"


# ---------------------------------------------------------------- 7. real uncalibrated frames -> STOP
def test_real_recorded_frames_with_incomplete_calibration_return_stop():
    if not os.path.exists(REAL_FRAME):
        pytest.skip("recording not present")
    est = se.StateEstimator(gp.CameraCalibration.load(INITIAL_CAL))
    p = pl.CameraToActionPipeline(est, tiny_policy(1))
    frame = cv2.imread(REAL_FRAME)
    for i in range(10):
        c = p.step(frame, 0.1 * i, 0.1 * i)
        assert c.stop and c.label == "STOP" and c.status == "UNCALIBRATED" and c.action is None and c.observation is None
        assert any("UNCALIBRATED" in w for w in c.warnings)             # diagnostics preserved
    assert not p.arm()                                                   # never arms on uncalibrated frames


def test_empty_calibration_also_stops():
    p = pl.CameraToActionPipeline(se.StateEstimator(gp.CameraCalibration(), min_confidence=0.3), tiny_policy(1))
    assert p.step(GOOD, 0.0, 0.0).status == "UNCALIBRATED"


# ---------------------------------------------------------------- 8. missing boundaries -> STOP
@pytest.mark.parametrize("left,right", [(True, False), (False, True), (False, False)])
def test_missing_boundary_returns_stop_without_substituting_observations(left, right):
    p = synthetic_pipeline(tiny_policy(1))
    t = arm(p, GOOD)
    c = p.step(tse.synthetic_image(left=left, right=right), t, t)
    assert c.stop and c.status == "INVALID_DETECTION" and c.observation is None and c.action is None
    assert p.policy.last_steering == 0.0                                 # memory reset on stop


# ---------------------------------------------------------------- 9. stale / missing frames
def test_stale_or_missing_frame_returns_stop():
    p = synthetic_pipeline(tiny_policy(1))
    t = arm(p, GOOD)
    assert p.step(GOOD, t, t).status == "OK"
    c = p.step(GOOD, t + 1.0, t + 0.2)                                   # frame is 0.8 s old
    assert c.stop and c.status == "STALE"
    c = p.step(None, t + 1.1, None)
    assert c.stop and c.status == "STALE"


def test_persistent_loss_latches_until_operator_rearms():
    p = synthetic_pipeline(tiny_policy(1))
    t = arm(p, GOOD)
    for i in range(4):
        p.step(None, t + 0.1 * i, None)
    t += 1.0
    for i in range(8):                                                   # good frames alone do not restart driving
        assert p.step(GOOD, t + 0.1 * i, t + 0.1 * i).stop
    assert p.arm()
    assert not p.step(GOOD, t + 1.0, t + 1.0).stop


def test_not_armed_at_start():
    p = synthetic_pipeline(tiny_policy(1))
    c = p.step(GOOD, 0.0, 0.0)
    assert c.stop and c.status == "GATE"


# ---------------------------------------------------------------- 10. emergency stop
def test_emergency_stop_overrides_everything_until_cleared_and_rearmed():
    p = synthetic_pipeline(tiny_policy(1))
    t = arm(p, GOOD)
    assert not p.step(GOOD, t, t).stop
    p.emergency_stop()
    for i in range(5):
        c = p.step(GOOD, t + 0.1 * (i + 1), t + 0.1 * (i + 1))
        assert c.stop and c.status == "ESTOP"
    p.clear_emergency_stop()
    assert p.step(GOOD, t + 1.0, t + 1.0).stop                           # gate still latched
    assert not p.arm()                                                   # no valid frames since the stop yet
    t2 = arm(p, GOOD, t + 2.0)
    assert not p.step(GOOD, t2, t2).stop


def test_out_of_range_estimate_stops():
    p = synthetic_pipeline(tiny_policy(1))
    t = arm(p, GOOD)
    c = p.step(tse.pose_image(0.2, 0.0), t, t)                            # 20 cm off: far beyond the 13 cm half-width
    assert c.stop and c.status in ("OUT_OF_RANGE", "INVALID_DETECTION", "INVALID_GEOMETRY")


def test_inference_failure_stops_instead_of_raising():
    class Broken(dp.DQNPolicy):
        def act(self, obs):
            raise RuntimeError("boom")

    p = synthetic_pipeline(Broken(tiny_policy(1).qnet))
    t = arm(p, GOOD)
    c = p.step(GOOD, t, t)
    assert c.stop and c.status == "ERROR" and "boom" in c.reason


def test_estimator_exception_stops_instead_of_raising():
    class BrokenEstimator(object):
        def estimate(self, frame):
            raise RuntimeError("camera glitch")

    p = pl.CameraToActionPipeline(BrokenEstimator(), tiny_policy(1))
    c = p.step(GOOD, 0.0, 0.0)
    assert c.stop and c.status == "ERROR"


# ---------------------------------------------------------------- 12. no hardware
class FakeCar(object):
    def __init__(self):
        self.throttle, self.steering = 0.0, 0.0



def test_offline_runs_never_import_or_command_hardware():
    p = synthetic_pipeline()
    t = arm(p, GOOD)
    dry = va.DryRunAdapter()
    for i in range(5):
        dry.apply(p.step(GOOD, t + 0.1 * i, t + 0.1 * i), t)
    dry.stop()
    assert dry.hardware_commands_issued == 0 and len(dry.log) == 6
    assert "jetracer" not in sys.modules and "jetcam" not in sys.modules


def test_motor_adapter_refuses_without_calibration_enable_or_explicit_throttle():
    ok = {"LEFT": "measured", "STRAIGHT": "measured", "RIGHT": "measured"}
    with pytest.raises(va.CalibrationMissing):
        va.SteeringMap(-0.5, 0.3, 0.5, 0.6, {"LEFT": "measured"})                    # uncalibrated entries
    with pytest.raises(va.CalibrationMissing):
        va.SteeringMap(-0.9, 0.3, 0.5, 0.9, ok)                                      # beyond the 0.8 safe range
    with pytest.raises(va.CalibrationMissing):
        va.SteeringMap(0.5, 0.3, -0.5, 0.6, ok)                                      # wrong ordering/sign
    sm = va.SteeringMap(-0.5, 0.3, 0.5, 0.6, ok)
    with pytest.raises(va.CalibrationMissing):
        va.JetRacerAdapter(FakeCar(), sm, 0.2)                                       # not enabled
    with pytest.raises(va.CalibrationMissing):
        va.JetRacerAdapter(FakeCar(), sm, None, enabled=True)                        # no throttle
    with pytest.raises(va.CalibrationMissing):
        va.JetRacerAdapter(FakeCar(), sm, 0.9, enabled=True)                         # above the hard limit


def test_motor_adapter_with_fake_car_stops_on_stop_commands_and_missing_deadman():
    ok = {"LEFT": "measured", "STRAIGHT": "measured", "RIGHT": "measured"}
    sm = va.SteeringMap(-0.5, 0.3, 0.5, 0.6, ok)
    car = FakeCar()
    ad = va.JetRacerAdapter(car, sm, 0.2, enabled=True)
    p = synthetic_pipeline(tiny_policy(0))
    t = arm(p, GOOD)
    go = p.step(GOOD, t, t)
    ad.apply(go, t, deadman_ok=True)
    assert car.steering == -0.5 and car.throttle == 0.2                              # LEFT -> the measured physical value
    ad.apply(go, t, deadman_ok=False)
    assert car.throttle == 0.0                                                       # deadman released
    ad.apply(go, t, deadman_ok=True)
    ad.apply(pl.Command(True, "x", "ESTOP"), t)
    assert car.throttle == 0.0
    ad.stop()
    assert car.throttle == 0.0 and car.steering == 0.0
