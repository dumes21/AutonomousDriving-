"""Camera -> DQN state estimation: lane offset (m), heading error (rad), validity/confidence. No hardware access.

    python perception/state_estimator.py IMAGE_OR_DIR [...] [--calibration FILE] [--out DIR] [--csv FILE]

Simulator conventions this module reproduces (read from sim/track_env.py, not assumed)
------------------------------------------------------------------------------------
* Reference point: the car's (x, y) is the REAR-AXLE centre (the bicycle model moves that point).
* lane_offset: signed perpendicular distance from the rear axle to the track centre line, in metres.
  POSITIVE = the car is to the LEFT of the centre line (left normal of the travel direction; the track is
  counter-clockwise, so left is the inside of the oval). Observation = offset / lane_half_width (0.13), clipped to [-1, 1].
* heading_error: (car heading - track tangent direction) wrapped to [-pi, pi). POSITIVE = the car points LEFT of the
  track direction. Observation = heading_error / pi, clipped to [-1, 1].
* last_steering: float(1 - action) = +1 (action 0, LEFT), 0 (straight), -1 (RIGHT). The physical car's sign is the
  OPPOSITE (negative = left) and its straight is ~+0.35: converting is the caller's job and is NOT done here.

Pipeline
--------
1. perception/boundary_detector.py (Lab method) finds, on sampled image rows, the left and right boundary columns.
2. Each row's lane centre is mapped to the floor with the calibrated ground-plane model (perception/ground_plane.py),
   giving points in the VEHICLE frame (origin rear axle, x forward, y left).
3. A line / parabola is fitted through those centre points; the foot of the perpendicular from the rear axle gives the
   offset, the tangent angle there gives the heading error (centre line going left => car points right => negative).
4. offset/0.13, heading/pi, last_steering -> the 3-vector the DQN was trained on.

UNCALIBRATED means exactly that: if any required calibration parameter is unset the metric offset/heading are
NOT computed (None), never defaulted. The image midpoint is never used as the vehicle's centre; the vehicle axis
line comes from calibration. `centre_minus_image_centre_m` is a diagnostic only, explicitly NOT the vehicle offset.

Nothing here has been validated against ground-truth poses: even a fully calibrated output is UNVERIFIED until
the validation images in docs/CAMERA_TO_STATE.md have been compared with measured poses.
"""
import argparse
import csv
import datetime
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boundary_detector as bd  # noqa: E402
from ground_plane import CalibrationError, CameraCalibration  # noqa: E402

LANE_HALF_WIDTH_M = 0.13        # the simulator's normalisation (Config.lane_half_width)
OK = "OK"
UNCALIBRATED = "UNCALIBRATED"
INVALID_DETECTION = "INVALID_DETECTION"
INVALID_GEOMETRY = "INVALID_GEOMETRY"


class StateEstimate(object):
    def __init__(self):
        self.status = INVALID_DETECTION
        self.valid = False                       # True only for status OK
        self.calibrated = False                  # all required calibration parameters present
        self.confidence = 0.0
        self.offset_m = None                     # sim convention, + = car LEFT of centre line, at the rear axle
        self.heading_rad = None                  # sim convention, + = car points LEFT of the track
        self.centre_minus_image_centre_m = None  # DIAGNOSTIC (metres, per-row width scale): NOT the vehicle offset
        self.extrapolation_m = None              # distance from the nearest measured lane point back to the rear axle
        self.missing_calibration = []
        self.assumed_calibration = []
        self.warnings = []
        self.detection = None                    # boundary_detector.BoundaryResult
        self.ground_points = None                # (K, 2) lane-centre points, vehicle frame (x forward, y left), metres
        self.fit = None                          # polynomial coefficients of y(x), highest power first
        self.curve_model_used = None
        self.curvature_hint_per_m = None         # noisy; |value| above ~1/m suggests a bend


class DQNObservation(object):
    def __init__(self, vector, clipped, warnings):
        self.vector = vector                     # float32 array of 3, or None if the estimate is not valid
        self.clipped = clipped
        self.warnings = warnings


def _centre_by_offset_curves(xf, y_l, y_r, half_width_m, deg):
    """Lane centre points from the two boundaries, robust to curvature: fit each boundary, shift it by half the
    lane width along ITS normal toward the lane, and merge. (The midpoint of the two boundaries at a fixed forward
    distance is biased inside a bend: ~3 mm at R=0.72 m, which the extrapolation back to the rear axle amplifies.)"""
    xs = np.linspace(xf.min(), xf.max(), 25)
    pts = []
    for ys_, sign in ((y_l, +1.0), (y_r, -1.0)):          # left boundary shifts to the right of travel, right boundary to the left
        p = np.poly1d(np.polyfit(xf, ys_, deg))
        phi = np.arctan(np.polyder(p)(xs))
        # left boundary (+1): move by +hw * right-normal (sin phi, -cos phi); right boundary (-1): by +hw * left-normal
        pts.append(np.column_stack([xs + sign * half_width_m * np.sin(phi),
                                    p(xs) - sign * half_width_m * np.cos(phi)]))
    return np.vstack(pts)


def compute_state(rows, calib, scale_source="measured_width", max_extrapolation_m=0.5, min_rows=3,
                  curve_model="auto", bend_threshold_per_m=1.0):
    """Pure geometry: boundary rows -> (offset_m, heading_rad, points, fit, extrapolation_m, info).

    curve_model: 'linear' (straight-line centre fit; best on straights, but biased by ~curvature x look-ahead in bends),
    'quadratic' (parabola through offset-curve centre points; handles bends but amplifies pixel noise when it
    extrapolates back to the rear axle) or 'auto' (quadratic only when the fitted curvature exceeds
    `bend_threshold_per_m`, else linear). In simulation tests with a 0.12 m visible span, 0.5 px noise, R = 0.72 m
    bends: linear -> heading bias -0.42 rad in bends, quadratic -> +0.11 rad bias; on straights linear gives
    0.005 rad rms while quadratic gives 0.08 rad rms.

    rows: dicts with y, left_x, right_x, centre_x, width_px (as returned by the detector).
    Raises CalibrationError (missing/invalid parameters) or ValueError (geometry unusable)."""
    missing = calib.missing_for_state(scale_source)
    if missing:
        raise CalibrationError("missing calibration: %s" % ", ".join(missing))
    if len(rows) < min_rows:
        raise ValueError("fewer than %d usable rows" % min_rows)
    ys = np.array([r["y"] for r in rows], float)
    xl = np.array([r["left_x"] for r in rows], float)
    xr = np.array([r["right_x"] for r in rows], float)
    wpx = xr - xl
    if scale_source == "measured_width":
        scale = calib.lane_width_m / wpx                  # m per px at each row, from the lane itself
    elif scale_source == "model":
        scale = calib.lateral_scale_m_per_px(ys)
    else:
        raise ValueError("scale_source must be 'measured_width' or 'model'")
    axis = calib.axis_column(ys)
    y_l = -(xl - axis) * scale                            # y is LEFT-positive in the vehicle frame
    y_r = -(xr - axis) * scale
    x_fwd = calib.get("camera_forward_of_rear_axle_m") + calib.depth_m(ys)
    if np.ptp(x_fwd) < 0.05:
        raise ValueError("lane points span less than 5 cm of forward distance")
    enough = len(rows) >= 5 and np.ptp(x_fwd) >= 0.10
    mid = np.column_stack([x_fwd, 0.5 * (y_l + y_r)])
    kappa = float(2.0 * np.polyfit(mid[:, 0], mid[:, 1], 2)[0]) if enough else None   # curvature hint, 1/m (noisy)
    use_quadratic = enough and (curve_model == "quadratic" or
                                (curve_model == "auto" and kappa is not None and abs(kappa) > bend_threshold_per_m))
    if curve_model not in ("linear", "quadratic", "auto"):
        raise ValueError("curve_model must be 'linear', 'quadratic' or 'auto'")
    if use_quadratic:
        deg = 2
        pts = _centre_by_offset_curves(x_fwd, y_l, y_r, calib.lane_width_m / 2.0, 2)
    else:
        deg = 1
        pts = mid
    coef = np.polyfit(pts[:, 0], pts[:, 1], deg)
    poly, dpoly = np.poly1d(coef), np.polyder(np.poly1d(coef))
    # foot of the perpendicular from the rear axle (origin) on the fitted centre line
    lo, hi = x_fwd.min() - max_extrapolation_m, x_fwd.max()
    grid = np.linspace(lo, hi, 2000)
    xs = grid[np.argmin(grid ** 2 + poly(grid) ** 2)]
    extrap = max(0.0, float(x_fwd.min() - xs))
    if extrap > max_extrapolation_m - 1e-9 and xs <= lo + 1e-6:
        raise ValueError("rear axle is more than %.2f m behind the nearest measured lane point" % max_extrapolation_m)
    phi = math.atan(float(dpoly(xs)))                      # track tangent angle in the vehicle frame
    offset = float(xs * math.sin(phi) - poly(xs) * math.cos(phi))   # + = car left of the centre line
    heading = -phi                                         # car heading - track direction
    info = {"curve_model": "quadratic" if deg == 2 else "linear", "curvature_hint_per_m": kappa}
    return offset, heading, pts, coef, extrap, info


def to_dqn_observation(est, last_steering, lane_half_width=LANE_HALF_WIDTH_M, heading_ood_rad=0.5):
    """StateEstimate -> DQN input [offset/0.13, heading/pi, last_steering] (float32), simulator convention.

    last_steering must already be in the SIMULATOR's convention (+1 left, 0 straight, -1 right).
    An invalid or uncalibrated estimate gives vector=None: missing values are never replaced by zeros."""
    if last_steering not in (-1, 0, 1, -1.0, 0.0, 1.0):
        raise ValueError("last_steering must be -1, 0 or +1 in the simulator convention (+1 = left)")
    if not est.valid:
        return DQNObservation(None, False, ["no observation: " + est.status] + list(est.warnings))
    raw = np.array([est.offset_m / lane_half_width, est.heading_rad / math.pi, float(last_steering)], np.float64)
    vec = np.clip(raw, -1.0, 1.0).astype(np.float32)
    warn = []
    clipped = bool(np.any(np.abs(raw[:2]) > 1.0))
    if abs(raw[0]) > 1.0:
        warn.append("offset %.3f m is outside the lane half-width (clipped); implausible or already over the line"
                    % est.offset_m)
    if abs(est.heading_rad) > heading_ood_rad:
        warn.append("heading %.2f rad is far outside the training distribution (+-0.2 rad starts)" % est.heading_rad)
    if not est.calibrated:
        warn.append("calibration incomplete")
    return DQNObservation(vec, clipped, warn)


class StateEstimator(object):
    def __init__(self, calibration, method="lab", min_confidence=0.5, scale_source="measured_width",
                 max_extrapolation_m=0.5, curve_model="auto"):
        self.curve_model = curve_model
        self.calibration = calibration
        self.method = method
        self.min_confidence = min_confidence
        self.scale_source = scale_source
        self.max_extrapolation_m = max_extrapolation_m

    def estimate(self, bgr):
        est = StateEstimate()
        det = bd.detect(bgr, self.method)
        est.detection = det
        est.confidence = det.confidence
        est.warnings += list(det.warnings)
        if not det.valid:
            est.status = INVALID_DETECTION                # no coordinates of any kind are produced
            return est
        if det.confidence < self.min_confidence:
            est.status = INVALID_DETECTION
            est.warnings.append("detection confidence %.2f below the minimum %.2f" % (det.confidence, self.min_confidence))
            return est
        # diagnostic: lane-centre displacement from the image-centre column, in metres (per-row width scale).
        ref = min(det.rows, key=lambda r: abs(r["y"] - det.ref_row))
        est.centre_minus_image_centre_m = (ref["centre_x"] - det.image_centre_x_px) * self.calibration.lane_width_m / ref["width_px"]

        est.missing_calibration = self.calibration.missing_for_state(self.scale_source)
        est.assumed_calibration = self.calibration.assumed_parameters()
        if est.missing_calibration:
            est.status = UNCALIBRATED
            est.warnings.append("UNCALIBRATED: offset and heading not computed; missing %s"
                                % ", ".join(est.missing_calibration))
            return est
        est.calibrated = True
        if est.assumed_calibration:
            est.warnings.append("uses ASSUMED (unmeasured) parameters: %s" % ", ".join(est.assumed_calibration))
        try:
            off, hd, pts, coef, extrap, info = compute_state(det.rows, self.calibration, self.scale_source,
                                                             self.max_extrapolation_m, curve_model=self.curve_model)
        except (CalibrationError, ValueError) as e:
            est.status = INVALID_GEOMETRY
            est.warnings.append("geometry: %s" % e)
            return est
        est.offset_m, est.heading_rad, est.ground_points, est.fit, est.extrapolation_m = off, hd, pts, coef, extrap
        est.curve_model_used, est.curvature_hint_per_m = info["curve_model"], info["curvature_hint_per_m"]
        if info["curve_model"] == "quadratic":
            est.warnings.append("bend suspected (curvature hint %.1f /m): heading/offset extrapolated with a parabola, "
                                "expect errors of ~0.1 rad / ~1.5 cm even with perfect pixels" % info["curvature_hint_per_m"])
        est.status, est.valid = OK, True
        return est


def render_state_debug(bgr, est, scale=2):
    """Boundary overlay + mask (from the detector) with a text panel below: status, offset, heading, warnings."""
    img = bd.render_debug(bgr, est.detection, scale=scale)
    lines = []
    if est.valid:
        lines.append("OFFSET %+.3f m (obs %+.2f)   HEADING %+.3f rad (obs %+.2f)   conf %.2f   [sim convention: + = left]"
                     % (est.offset_m, est.offset_m / LANE_HALF_WIDTH_M, est.heading_rad, est.heading_rad / math.pi,
                        est.confidence))
        lines.append("extrapolated %.2f m back to the rear axle. UNVERIFIED against ground truth." % est.extrapolation_m)
    else:
        lines.append("%s: no offset / heading produced" % est.status)
    if est.centre_minus_image_centre_m is not None:
        lines.append("diagnostic: lane centre is %+.3f m from the IMAGE centre column (NOT the vehicle offset)"
                     % est.centre_minus_image_centre_m)
    lines += est.warnings[:3]
    pad = np.zeros((18 * len(lines) + 8, img.shape[1], 3), np.uint8)
    for i, t in enumerate(lines):
        bad = (not est.valid) and (i == 0 or t.startswith("UNCALIBRATED") or t.startswith("INVALID"))
        cv2.putText(pad, t[:150], (6, 16 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255) if bad else (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([img, pad])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--calibration", help="calibration JSON (see perception/calibrate.py); default: nothing set = UNCALIBRATED")
    ap.add_argument("--out", help="debug folder (default logs/perception_state/<timestamp>); must not exist")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--csv", help="write results to this CSV (must not exist)")
    ap.add_argument("--scale-source", choices=["measured_width", "model"], default="measured_width")
    args = ap.parse_args()

    calib = CameraCalibration.load(args.calibration) if args.calibration else CameraCalibration()
    est_ = StateEstimator(calib, scale_source=args.scale_source)
    files = bd.list_images(args.images)
    if not files:
        sys.exit("no images found")
    out = None
    if not args.no_images:
        out = args.out or os.path.join(bd.ROOT, "logs", "perception_state", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(out)
    if args.csv and os.path.exists(args.csv):
        sys.exit("%s already exists" % args.csv)
    rows = []
    for p in files:
        img = bd.load_image(p)
        e = est_.estimate(img)
        print("%s: %s conf=%.2f offset=%s heading=%s%s" % (
            p, e.status, e.confidence, "-" if e.offset_m is None else "%+.3f m" % e.offset_m,
            "-" if e.heading_rad is None else "%+.3f rad" % e.heading_rad,
            "" if not e.warnings else "  | " + "; ".join(e.warnings[:2])))
        rows.append({"image": p, "status": e.status, "confidence": round(e.confidence, 3),
                     "offset_m": "" if e.offset_m is None else round(e.offset_m, 4),
                     "heading_rad": "" if e.heading_rad is None else round(e.heading_rad, 4),
                     "centre_minus_image_centre_m": "" if e.centre_minus_image_centre_m is None else round(e.centre_minus_image_centre_m, 4)})
        if out:
            parent = os.path.basename(os.path.dirname(os.path.abspath(p)))
            cv2.imwrite(os.path.join(out, "%s_%s.png" % (parent, os.path.splitext(os.path.basename(p))[0])),
                        render_state_debug(img, e))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if out:
        print("debug images in", out)


if __name__ == "__main__":
    main()
