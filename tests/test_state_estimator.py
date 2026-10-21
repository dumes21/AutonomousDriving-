"""Geometry tests with a SYNTHETIC camera: the numbers below are test fixtures chosen only to exercise the code
(they say nothing about the real JetRacer camera). Ground truth poses are projected forward through the same
pinhole/flat-floor model and the estimator must recover them."""
import math
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "perception"))
import ground_plane as gp  # noqa: E402
import state_estimator as se  # noqa: E402

Y_H, K, F, CX, L_CAM, W = 40.0, 0.12, 14.4, 160.0, 0.10, 0.26     # test fixture camera (f = F/K = 120 px)


def fixture_calibration():
    c = gp.CameraCalibration(W)
    c.set("horizon_row_px", Y_H, "measured")
    c.set("camera_height_eff_m", K, "measured")
    c.set("depth_scale_F", F, "measured")
    c.set("axis_x_ref_px", CX, "measured")
    c.set("axis_slope", 0.0, "measured")
    c.set("camera_forward_of_rear_axle_m", L_CAM, "measured")
    return c


ROWS = list(range(100, 164, 8))


def straight_rows(offset, heading):
    """Rows seen by a car `offset` m LEFT of the lane centre, pointing `heading` rad LEFT of the track."""
    phi = -heading                                         # track direction in the vehicle frame
    out = []
    for y in ROWS:
        x_f = L_CAM + F / (y - Y_H)
        yc = -offset / math.cos(phi) + math.tan(phi) * x_f       # centre line, y left
        half = 0.5 * W / math.cos(phi)
        yl, yr = yc + half, yc - half                            # left / right boundary (y left)
        s = K / (y - Y_H)                                        # m per px
        xl, xr = CX - yl / s, CX - yr / s
        out.append({"y": y, "left_x": xl, "right_x": xr, "centre_x": 0.5 * (xl + xr), "width_px": xr - xl})
    return out


def bend_rows(offset, heading, radius=0.72):
    """Left-hand bend (centre of curvature on the left), boundaries at radius +- W/2."""
    phi = -heading
    n = np.array([-math.sin(phi), math.cos(phi)])
    centre = (radius - offset) * n                         # foot = -offset*n ; C = foot + radius*n
    out = []
    for y in ROWS:
        x_f = L_CAM + F / (y - Y_H)
        s = K / (y - Y_H)
        ys_ = []
        for r in (radius - W / 2, radius + W / 2):
            ys_.append(centre[1] - math.sqrt(r * r - (x_f - centre[0]) ** 2))
        yl, yr = min(ys_), max(ys_)                        # inner (larger r: further left) -> left boundary
        yl, yr = (centre[1] - math.sqrt((radius + W / 2) ** 2 - (x_f - centre[0]) ** 2),
                  centre[1] - math.sqrt((radius - W / 2) ** 2 - (x_f - centre[0]) ** 2))
        # y larger = more left. The outer circle is nearer the car side (smaller y); order the pair by y:
        yl, yr = max(yl, yr), min(yl, yr)
        xl, xr = CX - yl / s, CX - yr / s
        out.append({"y": y, "left_x": xl, "right_x": xr, "centre_x": 0.5 * (xl + xr), "width_px": xr - xl})
    return out


# ---------------------------------------------------------------------------- signs and accuracy
@pytest.mark.parametrize("offset", [-0.04, -0.02, 0.0, 0.02, 0.04])
@pytest.mark.parametrize("heading", [-0.2, -0.05, 0.0, 0.05, 0.2])
def test_straight_lane_recovers_offset_and_heading_with_model_scale(offset, heading):
    o, h, *_ = se.compute_state(straight_rows(offset, heading), fixture_calibration(), "model", curve_model="linear")
    assert o == pytest.approx(offset, abs=1e-3)
    assert h == pytest.approx(heading, abs=1e-3)


def test_offset_sign_positive_means_car_left_of_centre():
    o, _, *_ = se.compute_state(straight_rows(+0.03, 0.0), fixture_calibration(), "model")
    assert o > 0.025
    o, _, *_ = se.compute_state(straight_rows(-0.03, 0.0), fixture_calibration(), "model")
    assert o < -0.025


def test_heading_sign_positive_means_car_points_left_of_track():
    _, h, *_ = se.compute_state(straight_rows(0.0, +0.15), fixture_calibration(), "model")
    assert h > 0.12
    _, h, *_ = se.compute_state(straight_rows(0.0, -0.15), fixture_calibration(), "model")
    assert h < -0.12


def test_measured_width_scale_is_close_to_model_scale_for_modest_yaw():
    o, h, *_ = se.compute_state(straight_rows(0.03, 0.1), fixture_calibration(), "measured_width")
    assert o == pytest.approx(0.03, abs=0.004) and h == pytest.approx(0.1, abs=0.01)


def test_bend_limitation_is_pinned_linear_is_badly_biased_quadratic_is_better():
    """Known, documented limitation: only ~0.12 m of lane is visible, 0.22 m ahead of the rear axle, so in an
    R = 0.72 m bend the lane direction there differs from the direction at the car by ~0.4 rad."""
    cal = fixture_calibration()
    lin = se.compute_state(bend_rows(0.0, 0.0), cal, "model", curve_model="linear")
    quad = se.compute_state(bend_rows(0.0, 0.0), cal, "model", curve_model="quadratic")
    assert abs(lin[1]) > 0.3 and abs(lin[0]) > 0.03              # linear: far off
    assert abs(quad[1]) < 0.15 and abs(quad[0]) < 0.02           # quadratic: ~0.1 rad / ~1 cm even with exact pixels
    assert lin[5]["curve_model"] == "linear" and quad[5]["curve_model"] == "quadratic"


def test_auto_model_uses_the_parabola_only_in_bends():
    cal = fixture_calibration()
    assert se.compute_state(straight_rows(0.02, 0.05), cal, "model")[5]["curve_model"] == "linear"
    assert se.compute_state(bend_rows(0.0, 0.0), cal, "model")[5]["curve_model"] == "quadratic"


# ---------------------------------------------------------------------------- invalid calibration / geometry
def test_missing_calibration_is_reported_not_defaulted():
    c = gp.CameraCalibration(W)
    with pytest.raises(gp.CalibrationError, match="missing calibration"):
        se.compute_state(straight_rows(0.0, 0.0), c, "model")
    assert "depth_scale_F" in c.missing_for_state()


@pytest.mark.parametrize("name,value,status", [("depth_scale_F", -1, "measured"), ("camera_height_eff_m", 0, "fitted"),
                                              ("camera_forward_of_rear_axle_m", 2.0, "measured"),
                                              ("horizon_row_px", float("nan"), "measured"),
                                              ("depth_scale_F", 5, "guess"), ("not_a_param", 1, "measured")])
def test_invalid_calibration_values_are_rejected(name, value, status):
    with pytest.raises(gp.CalibrationError):
        gp.CameraCalibration(W).set(name, value, status)


def test_row_at_or_above_the_horizon_is_rejected():
    c = fixture_calibration()
    c.set("horizon_row_px", 120.0, "measured")
    with pytest.raises(gp.CalibrationError):
        se.compute_state(straight_rows(0.0, 0.0), c, "model")


def test_too_few_rows_is_a_geometry_error():
    with pytest.raises(ValueError):
        se.compute_state(straight_rows(0.0, 0.0)[:2], fixture_calibration(), "model")


def test_calibration_json_round_trip(tmp_path):
    c = fixture_calibration()
    p = str(tmp_path / "c.json")
    c.save(p)
    d = gp.CameraCalibration.load(p)
    assert d.values == c.values and d.status == c.status and d.lane_width_m == c.lane_width_m


# ---------------------------------------------------------------------------- DQN observation
def valid_estimate(offset, heading):
    e = se.StateEstimate()
    e.valid, e.calibrated, e.status, e.offset_m, e.heading_rad = True, True, se.OK, offset, heading
    return e


def test_normalisation_matches_the_simulator_definition():
    obs = se.to_dqn_observation(valid_estimate(0.065, 0.4), 1)
    assert obs.vector.dtype == np.float32
    assert obs.vector.tolist() == pytest.approx([0.5, 0.4 / math.pi, 1.0])
    assert not obs.clipped and obs.warnings == []


def test_out_of_range_values_are_clipped_and_flagged():
    obs = se.to_dqn_observation(valid_estimate(0.2, 0.0), -1)
    assert obs.vector[0] == 1.0 and obs.clipped and any("outside the lane" in w for w in obs.warnings)
    obs = se.to_dqn_observation(valid_estimate(-0.2, 0.7), 0)
    assert obs.vector[0] == -1.0
    assert any("training distribution" in w for w in obs.warnings)


def test_invalid_estimate_gives_no_vector_not_zeros():
    e = se.StateEstimate()
    obs = se.to_dqn_observation(e, 0)
    assert obs.vector is None and obs.warnings


def test_last_steering_must_use_the_simulator_convention():
    with pytest.raises(ValueError):
        se.to_dqn_observation(valid_estimate(0.0, 0.0), 0.35)


# ---------------------------------------------------------------------------- fitting helpers
def test_fit_depth_model_recovers_F_and_horizon():
    rows = np.array([120, 150, 185.0])
    d = F / (rows - Y_H)
    f, yh, rms = gp.fit_depth_model(rows, d)
    assert f == pytest.approx(F, rel=1e-6) and yh == pytest.approx(Y_H, abs=1e-6) and rms < 1e-9


def test_fit_width_model_recovers_horizon_and_height():
    ys = np.arange(100, 200.0)
    yh, k, rms, n = gp.fit_width_model(ys, W * (ys - Y_H) / K, W)
    assert yh == pytest.approx(Y_H, abs=1e-6) and k == pytest.approx(K, rel=1e-6)


def test_fit_vehicle_axis_and_errors():
    ys = np.arange(100, 160.0)
    x0, slope, rms = gp.fit_vehicle_axis(ys, 163 + 0.05 * (ys - 140), 140)
    assert x0 == pytest.approx(163) and slope == pytest.approx(0.05)
    with pytest.raises(gp.CalibrationError):
        gp.fit_vehicle_axis([100, 101, 102], [1, 2, 3], 140)
    with pytest.raises(gp.CalibrationError):
        gp.fit_depth_model([120, 150], [0.3, 0.6])      # distance grows with row: impossible


# ---------------------------------------------------------------------------- end to end with the detector
TAPE, FLOOR = (100, 125, 178), (176, 150, 178)


def synthetic_image(shift=0, left=True, right=True):
    img = np.full((240, 320, 3), FLOOR, np.uint8)
    img[:75] = (60, 60, 60)
    if left:
        cv2.line(img, (110 + shift, 100), (30 + shift, 225), TAPE, 14)
    if right:
        cv2.line(img, (215 + shift, 100), (290 + shift, 225), TAPE, 14)
    return cv2.GaussianBlur(img, (3, 3), 0)


def pose_image(offset, heading):
    """Render the lane boundaries for a car `offset` m left of centre / `heading` rad left of the track, by projecting
    ground lines through the fixture camera, so the detector + estimator are tested together."""
    phi = -heading
    img = np.full((240, 320, 3), FLOOR, np.uint8)
    img[:75] = (60, 60, 60)
    for side in (+1, -1):
        pts = []
        for x_f in np.linspace(L_CAM + 0.16, L_CAM + 0.5, 60):
            yc = -offset / math.cos(phi) + math.tan(phi) * x_f
            yb = yc + side * 0.5 * W / math.cos(phi)
            row = Y_H + F / (x_f - L_CAM)
            pts.append((CX - yb * (row - Y_H) / K, row))
        pts = np.array([p for p in pts if 80 < p[1] < 226 and 2 < p[0] < 317], np.float32)
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, TAPE, 12)
    return cv2.GaussianBlur(img, (3, 3), 0)


@pytest.mark.parametrize("offset,heading", [(0.0, 0.0), (0.03, 0.0), (-0.03, 0.0), (0.0, 0.1), (0.0, -0.1), (0.02, -0.08)])
def test_end_to_end_detector_plus_estimator_recovers_a_projected_pose(offset, heading):
    est = se.StateEstimator(fixture_calibration(), scale_source="model", min_confidence=0.3)
    r = est.estimate(pose_image(offset, heading))
    assert r.valid, (r.status, r.warnings)
    assert r.offset_m == pytest.approx(offset, abs=0.012)       # detection is pixel-quantised: centimetre-level, not mm
    assert r.heading_rad == pytest.approx(heading, abs=0.05)


def test_end_to_end_missing_boundary_is_invalid_with_no_numbers():
    est = se.StateEstimator(fixture_calibration(), min_confidence=0.3)
    for img in (synthetic_image(left=False), synthetic_image(right=False), synthetic_image(left=False, right=False)):
        r = est.estimate(img)
        assert r.status == se.INVALID_DETECTION and not r.valid
        assert r.offset_m is None and r.heading_rad is None and r.ground_points is None
        assert se.to_dqn_observation(r, 0).vector is None


def test_end_to_end_uncalibrated_camera_produces_diagnostic_only():
    r = se.StateEstimator(gp.CameraCalibration(W), min_confidence=0.3).estimate(synthetic_image())
    assert r.status == se.UNCALIBRATED and not r.valid and not r.calibrated
    assert r.offset_m is None and r.heading_rad is None
    assert r.centre_minus_image_centre_m is not None and "depth_scale_F" in r.missing_calibration
    assert se.to_dqn_observation(r, 0).vector is None
