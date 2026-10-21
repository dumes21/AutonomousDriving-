"""Tests for the calibrated car/camera_test_dqn.py on SYNTHETIC images with known poses.

The calibration used here is a TEST FIXTURE (the same synthetic pinhole camera as tests/test_calibration_workflow.py), written only
into pytest's temporary folder. It says nothing about the real JetRacer camera and must never be used for driving."""
import csv
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tests"))
for sub in ("car", "perception", "inference"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import test_calibration_workflow as wf  # noqa: E402  (synthetic camera renderer)
import camera_test_dqn as ct  # noqa: E402
import ground_plane as gp  # noqa: E402
import state_estimator as se  # noqa: E402
from dqn_policy import load_policy  # noqa: E402
from pipeline import CameraToActionPipeline  # noqa: E402

SCRIPT = os.path.join(ROOT, "car", "camera_test_dqn.py")
NPZ = os.path.join(ROOT, "inference", "dqn_footprint_v1.npz")
INITIAL = os.path.join(ROOT, "perception", "calibration", "initial_fit_from_recordings.json")


def fixture_calibration():
    c = gp.CameraCalibration(wf.W)
    c.set("horizon_row_px", wf.Y_H, "measured")
    c.set("camera_height_eff_m", wf.K, "measured")
    c.set("depth_scale_F", wf.F, "measured")
    c.set("axis_x_ref_px", wf.AX0, "measured")
    c.set("axis_slope", wf.AXS, "measured")
    c.set("camera_forward_of_rear_axle_m", wf.L_CAM, "measured")
    return c


@pytest.fixture()
def est_and_policy():
    return se.StateEstimator(fixture_calibration()), load_policy(NPZ)


def run_script(args, ok=True):
    p = subprocess.run([sys.executable, SCRIPT] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if ok:
        assert p.returncode == 0, p.stdout
    return p


# ---------------------------------------------------------------------------- sign conventions and the observation format
@pytest.mark.parametrize("offset_cm,yaw_deg", [(2.5, 0), (-2.5, 0), (5, 0), (-5, 0), (0, 5), (0, -5), (2.5, 5), (-2.5, -5)])
def test_known_poses_give_simulator_signs_and_normalisation(est_and_policy, offset_cm, yaw_deg):
    est, policy = est_and_policy
    r = ct.observe(est, policy, wf.render(offset_cm / 100.0, math.radians(yaw_deg)))
    assert r["accepted"] and r["status"] == "OK"
    assert r["est_offset_m"] == pytest.approx(offset_cm / 100.0, abs=0.004)
    assert r["est_heading_rad"] == pytest.approx(math.radians(yaw_deg), abs=0.012)
    assert np.sign(r["est_offset_m"]) == np.sign(offset_cm) or offset_cm == 0           # + = car LEFT of the centre line
    assert np.sign(r["est_heading_rad"]) == np.sign(yaw_deg) or yaw_deg == 0             # + = car pointing LEFT
    o = r["obs"]
    assert o.dtype == np.float32 and o.shape == (3,)
    assert o[0] == pytest.approx(r["est_offset_m"] / 0.13, abs=1e-5)                     # offset_m / 0.13
    assert o[1] == pytest.approx(r["est_heading_rad"] / math.pi, abs=1e-5)               # heading_rad / pi
    assert o[2] == 0.0                                                                   # previous steering 0 for a fresh still


def test_previous_steering_follows_the_simulator_convention_and_resets_on_rejection(est_and_policy):
    est, policy = est_and_policy
    first = ct.observe(est, policy, wf.render(0.02, 0.0))
    assert first["accepted"] and policy.last_steering == float(1 - first["action"])      # +1 after LEFT, 0 STRAIGHT, -1 RIGHT
    second = ct.observe(est, policy, wf.render(0.02, 0.0))
    assert second["obs"][2] == float(1 - first["action"])                                # carried into the next observation
    blank = ct.observe(est, policy, np.full((240, 320, 3), (176, 150, 178), np.uint8))
    assert not blank["accepted"] and policy.last_steering == 0.0                         # reset after a rejection, as in the pipeline


# ---------------------------------------------------------------------------- rejections: never an observation, never an action
def test_blank_image_and_single_line_are_rejected_without_numbers(est_and_policy):
    est, policy = est_and_policy
    blank = ct.observe(est, policy, np.full((240, 320, 3), (176, 150, 178), np.uint8))
    one = wf.render(0.0, 0.0)
    one[:, 165:] = wf.FLOOR                                        # erase everything right of the centre: one boundary left
    one_line = ct.observe(est, policy, one)
    for r in (blank, one_line):
        assert not r["accepted"] and r["status"] == "INVALID_DETECTION"
        assert r["obs"] is None and r["action"] is None and r["est_offset_m"] is None and r["reject_reason"]


def test_low_confidence_is_rejected(est_and_policy):
    _, policy = est_and_policy
    strict = se.StateEstimator(fixture_calibration(), min_confidence=0.999)
    r = ct.observe(strict, policy, wf.render(0.0, 0.0))
    assert not r["accepted"] and "confidence" in r["reject_reason"] and r["action"] is None


class _StubEstimator(object):
    def __init__(self, **kw):
        self.kw = kw

    def estimate(self, bgr):
        e = se.StateEstimate()
        e.detection = None
        for k, v in self.kw.items():
            setattr(e, k, v)
        return e


def test_out_of_range_estimate_is_rejected_not_clipped_into_the_policy():
    policy = load_policy(NPZ)
    stub = _StubEstimator(status=se.OK, valid=True, calibrated=True, confidence=0.9, offset_m=0.20, heading_rad=0.0, extrapolation_m=0.2)
    r = ct.observe(stub, policy, None)
    assert not r["accepted"] and r["status"] == "OUT_OF_RANGE" and r["action"] is None and r["obs"] is None


@pytest.mark.parametrize("status", [se.UNCALIBRATED, se.INVALID_GEOMETRY, se.INVALID_DETECTION])
def test_estimator_failures_are_rejected(status):
    policy = load_policy(NPZ)
    stub = _StubEstimator(status=status, valid=False, warnings=["%s: test" % status])
    r = ct.observe(stub, policy, None)
    assert not r["accepted"] and r["status"] == status and r["action"] is None and status in r["reject_reason"]


# ---------------------------------------------------------------------------- calibration is required, never defaulted
def test_calibration_argument_is_required():
    p = run_script(["whatever.jpg"], ok=False)
    assert p.returncode == 2 and "--calibration" in p.stdout


def test_missing_and_incomplete_calibration_files_are_refused(tmp_path):
    p = run_script(["--calibration", str(tmp_path / "nope.json"), "x.jpg"], ok=False)
    assert p.returncode != 0 and "not found" in p.stdout
    p = run_script(["--calibration", INITIAL, "x.jpg"], ok=False)                     # only horizon / k are set in the repo file
    assert p.returncode != 0 and "INCOMPLETE" in p.stdout and "depth_scale_F" in p.stdout and "no fallback" in p.stdout


# ---------------------------------------------------------------------------- same pipeline as car/drive_dqn.py
@pytest.mark.parametrize("name,img", [
    ("centred", lambda: wf.render(0.0, 0.0)), ("left25", lambda: wf.render(0.025, 0.0)), ("right50", lambda: wf.render(-0.05, 0.0)),
    ("yaw_left", lambda: wf.render(0.0, math.radians(8))), ("yaw_right", lambda: wf.render(0.01, math.radians(-8))),
    ("blank", lambda: np.full((240, 320, 3), (176, 150, 178), np.uint8))])
def test_camera_test_matches_the_drive_dqn_pipeline(name, img):
    frame = img()
    policy_a = load_policy(NPZ)
    r = ct.observe(se.StateEstimator(fixture_calibration()), policy_a, frame)
    pipe = CameraToActionPipeline(se.StateEstimator(fixture_calibration()), load_policy(NPZ))
    pipe.gate.latched = False                                   # armed, so only the perception / observation logic is compared
    cmd = pipe.step(frame, now=1.0)
    assert (not cmd.stop) == r["accepted"]
    if r["accepted"]:
        assert cmd.action.label == ct.ACTIONS[r["action"]]
        assert np.allclose(cmd.observation, r["obs"])
    else:
        assert cmd.action is None and cmd.observation is None


def test_default_model_matches_sb3_greedy_actions():
    zip_path = ct.MODEL_PATH
    if not os.path.exists(zip_path):
        pytest.skip("footprint_v1 checkpoint not present")
    from stable_baselines3 import DQN
    sb3 = DQN.load(zip_path, device="cpu")
    policy = load_policy(zip_path)
    rng = np.random.RandomState(1)
    for _ in range(200):
        obs = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.choice([-1.0, 0.0, 1.0])], np.float32)
        policy.last_steering = float(obs[2])
        assert policy.act(obs).index == int(sb3.predict(obs, deterministic=True)[0])


# ---------------------------------------------------------------------------- batch CLI output
def test_batch_cli_writes_csv_summary_and_annotated_images_and_never_overwrites(tmp_path):
    cal = tmp_path / "FIXTURE_CAL.json"
    fixture_calibration().save(str(cal))
    imgs = []
    for name, off, yaw in (("L25", 2.5, 0), ("R25", -2.5, 0), ("yL5", 0, 5)):
        p = str(tmp_path / (name + ".jpg"))
        cv2.imwrite(p, wf.render(off / 100.0, math.radians(yaw)))
        imgs.append(p)
    blank = str(tmp_path / "blank.jpg")
    cv2.imwrite(blank, np.full((240, 320, 3), (176, 150, 178), np.uint8))
    out = str(tmp_path / "out")
    p = run_script(["--calibration", str(cal), "--out", out] + imgs + [blank])
    rows = list(csv.DictReader(open(os.path.join(out, "predictions.csv"))))
    assert len(rows) == 4
    for col in ("est_offset_m", "est_heading_rad", "confidence", "reject_reason", "status", "obs_offset", "obs_heading",
                "obs_prev_steering", "action_name"):
        assert col in rows[0]
    by = {os.path.basename(r["image"]): r for r in rows}
    assert float(by["L25.jpg"]["est_offset_m"]) > 0.02 and float(by["R25.jpg"]["est_offset_m"]) < -0.02
    assert float(by["yL5.jpg"]["est_heading_rad"]) > 0.07
    assert float(by["L25.jpg"]["obs_offset"]) == pytest.approx(float(by["L25.jpg"]["est_offset_m"]) / 0.13, abs=2e-3)   # NOT the image-space value
    assert by["blank.jpg"]["accepted"] == "False" and by["blank.jpg"]["reject_reason"] and by["blank.jpg"]["action_name"] == ""
    summary = json.load(open(os.path.join(out, "summary.json")))
    assert summary["images"] == 4 and summary["accepted_for_dqn"] == 3 and summary["calibration_assumed_parameters"] == []
    assert "UNVERIFIED" in summary["observation_note"]
    assert len(os.listdir(os.path.join(out, "annotated"))) == 4
    again = run_script(["--calibration", str(cal), "--out", out] + imgs, ok=False)
    assert again.returncode != 0 and "already contains results" in again.stdout


def test_script_contains_no_motor_or_camera_hardware_code():
    src = open(SCRIPT).read()
    for word in ("jetcam", "NvidiaRacecar", "jetracer", "throttle", "CSICamera"):
        assert word not in src
