"""Ground-plane camera model and calibration fitting for the JetRacer camera (no hardware, no simulator).

Model (the simplest one that can use the known 0.26 m lane width): a pinhole camera over a flat floor, no roll,
small pitch, no lens distortion. For an image row y below the horizon row y_h:

    depth along the camera axis   D(y) = F / (y - y_h)                  F = focal_px * camera_height (m*px)
    lateral ground metres per px  s(y) = k / (y - y_h)                  k = effective camera height (m)
    a point at column x is        X = (x - x_axis(y)) * s(y)  metres to the RIGHT of the vehicle axis line

so the lane width W seen as w(y) pixels gives  w(y) = W (y - y_h) / k, which is linear in y. That is the part
the recorded frames CAN calibrate (y_h and k, `fit_width_model`). What they cannot give is F (needs one known
LONGITUDINAL distance) or the vehicle axis line in the image (needs a known pose). Nothing here has a default
numeric value: a parameter is either supplied with a status ('measured' / 'fitted' / 'assumed') or it is unset,
and anything that needs an unset parameter reports UNCALIBRATED instead of guessing.

Assumptions you should keep in mind: the wide-angle lens is NOT undistorted here; the floor is flat; roll is
zero. The fitted width model leaves ~12 % scatter in the recordings, so the model is only a first approximation.
"""
import json

import numpy as np

# parameter name -> meaning
PARAMS = {
    "horizon_row_px": "image row y_h of the horizon in the pinhole ground model",
    "camera_height_eff_m": "k: effective camera height (height / cos(pitch)), metres",
    "depth_scale_F": "F = focal length (px) x k: converts rows to forward distance, metre*pixel",
    "axis_x_ref_px": "image column of the vehicle's forward axis at row `axis_ref_row`",
    "axis_slope": "d(column)/d(row) of the vehicle's forward axis line in the image (0 = vertical)",
    "camera_forward_of_rear_axle_m": "distance from the rear-axle centre forward to the camera lens, metres",
}
STATUSES = ("measured", "fitted", "assumed")


class CalibrationError(ValueError):
    pass


class CameraCalibration(object):
    def __init__(self, lane_width_m=0.26, image_size=(320, 240), axis_ref_row=140):
        if lane_width_m <= 0:
            raise CalibrationError("lane_width_m must be > 0")
        self.lane_width_m = float(lane_width_m)
        self.image_size = tuple(image_size)       # (width, height)
        self.axis_ref_row = float(axis_ref_row)
        self.values = {k: None for k in PARAMS}
        self.status = {k: "unset" for k in PARAMS}
        self.notes = {}

    # ---- setting / validation -------------------------------------------------------------
    def set(self, name, value, status, note=""):
        if name not in PARAMS:
            raise CalibrationError("unknown calibration parameter %r" % name)
        if status not in STATUSES:
            raise CalibrationError("status must be one of %s" % (STATUSES,))
        value = float(value)
        if not np.isfinite(value):
            raise CalibrationError("%s must be finite" % name)
        if name in ("camera_height_eff_m", "depth_scale_F") and value <= 0:
            raise CalibrationError("%s must be > 0" % name)
        if name == "camera_forward_of_rear_axle_m" and not (0.0 <= value <= 1.0):
            raise CalibrationError("camera_forward_of_rear_axle_m must be within 0..1 m")
        if name == "horizon_row_px" and not (-1000 <= value < self.image_size[1]):
            raise CalibrationError("horizon_row_px is outside any plausible range")
        self.values[name] = value
        self.status[name] = status
        if note:
            self.notes[name] = note
        return self

    def get(self, name):
        return self.values[name]

    def missing_for_state(self, scale_source="measured_width"):
        """Names of parameters that must still be supplied before offset/heading in the simulator's
        convention can be computed."""
        need = ["horizon_row_px", "depth_scale_F", "axis_x_ref_px", "axis_slope", "camera_forward_of_rear_axle_m"]
        if scale_source == "model":
            need.append("camera_height_eff_m")
        return [n for n in need if self.values[n] is None]

    def assumed_parameters(self):
        return [n for n, s in self.status.items() if s == "assumed"]

    # ---- geometry ---------------------------------------------------------------------------
    def depth_m(self, y):
        """Forward distance (m) from the camera of the ground point seen at row(s) y."""
        y = np.asarray(y, float)
        den = y - self.values["horizon_row_px"]
        if np.any(den <= 0):
            raise CalibrationError("row at or above the horizon row: no ground point there")
        return self.values["depth_scale_F"] / den

    def lateral_scale_m_per_px(self, y):
        y = np.asarray(y, float)
        den = y - self.values["horizon_row_px"]
        if np.any(den <= 0):
            raise CalibrationError("row at or above the horizon row: no ground point there")
        return self.values["camera_height_eff_m"] / den

    def axis_column(self, y):
        return self.values["axis_x_ref_px"] + self.values["axis_slope"] * (np.asarray(y, float) - self.axis_ref_row)

    # ---- persistence ------------------------------------------------------------------------
    def to_dict(self):
        return {"lane_width_m": self.lane_width_m, "image_size": list(self.image_size), "axis_ref_row": self.axis_ref_row,
                "parameters": {k: {"value": self.values[k], "status": self.status[k], "meaning": PARAMS[k],
                                   "note": self.notes.get(k, "")} for k in PARAMS}}

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        c = cls(d["lane_width_m"], tuple(d["image_size"]), d.get("axis_ref_row", 140))
        for k, p in d["parameters"].items():
            if p["value"] is not None:
                c.set(k, p["value"], p["status"], p.get("note", ""))
        return c


# ---- fitting helpers (used by perception/calibrate.py once data exists) --------------------------
def fit_width_model(ys, widths, lane_width_m, clip_sigma=2.5):
    """Fit w(y) = a*y + b to measured lane widths (pixels) -> (horizon_row, k, rms_px, n_used).
    horizon_row = -b/a, k = lane_width_m / a. Two rounds of sigma clipping."""
    ys, widths = np.asarray(ys, float), np.asarray(widths, float)
    keep = np.ones(len(ys), bool)
    for _ in range(3):
        a, b = np.polyfit(ys[keep], widths[keep], 1)
        res = widths - (a * ys + b)
        s = res[keep].std()
        keep = np.abs(res) <= clip_sigma * s
    if a <= 0:
        raise CalibrationError("lane width does not grow toward the camera: cannot fit a ground model")
    return -b / a, lane_width_m / a, float(np.sqrt(np.mean(res[keep] ** 2))), int(keep.sum())


def fit_depth_model(rows, distances_m):
    """From >= 2 known forward distances (metres, measured from the lens) of tape strips seen at image `rows`:
    1/D = (y - y_h)/F is linear in y. Returns (F, y_h, rms_distance_error_m)."""
    rows, d = np.asarray(rows, float), np.asarray(distances_m, float)
    if len(rows) < 2 or np.any(d <= 0):
        raise CalibrationError("need at least two strips at positive distances")
    slope, icpt = np.polyfit(rows, 1.0 / d, 1)
    if slope <= 0:
        raise CalibrationError("strip distances do not shrink with row: wrong order or bad measurements")
    F = 1.0 / slope
    y_h = -icpt * F
    pred = F / (rows - y_h)
    return float(F), float(y_h), float(np.sqrt(np.mean((pred - d) ** 2)))


def fit_vehicle_axis(ys, centre_xs, axis_ref_row):
    """From frames where the car is known to be centred in the lane and aligned with it, the lane centre IS the
    vehicle's forward axis. Fit column = x0 + slope * (row - axis_ref_row) -> (x0, slope, rms_px)."""
    ys, xs = np.asarray(ys, float), np.asarray(centre_xs, float)
    if len(ys) < 3 or np.ptp(ys) < 20:
        raise CalibrationError("need lane-centre points over at least 20 rows")
    slope, icpt = np.polyfit(ys - axis_ref_row, xs, 1)
    pred = icpt + slope * (ys - axis_ref_row)
    return float(icpt), float(slope), float(np.sqrt(np.mean((pred - xs) ** 2)))
