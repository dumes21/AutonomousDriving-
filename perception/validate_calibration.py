"""Check a finished calibration against photos taken at KNOWN car positions. Offline; reads images, prints a table.

    python perception/validate_calibration.py CAL.json truth.csv [--csv results.csv]

truth.csv has one line per photo or group of photos (paths relative to the CSV's folder; patterns like L25_*.jpg are allowed and
every matching photo gets that truth; lines starting with # are ignored):

    image,offset_cm,yaw_deg
    val_left25.jpg,+2.5,0
    val_right25.jpg,-2.5,0
    val_yaw_left5.jpg,0,+5

Conventions (the simulator's, same as docs/CAMERA_TO_STATE.md):
  offset_cm : how far the REAR-AXLE centre is from the lane centre line. POSITIVE = car moved LEFT of the centre.
  yaw_deg   : angle between the car's forward axis and the lane direction. POSITIVE = car rotated LEFT (counter-clockwise from above).
Do NOT include the centred photos used for `fit-axis` as validation: they were used to build the calibration.

Prints truth, estimate, error per photo, then bias / rms per quantity and a sign check. The pass thresholds are the ones in
docs/CAMERA_TO_STATE.md (1 cm, 2 degrees); they are guidelines, change them with --tol-cm / --tol-deg.
Exit status 1 if any photo has no valid estimate or any check fails. Writes nothing unless --csv is given (new file only).
"""
import argparse
import csv
import glob
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boundary_detector as bd  # noqa: E402
from ground_plane import CameraCalibration  # noqa: E402
from state_estimator import StateEstimator  # noqa: E402


def read_truth(path):
    base = os.path.dirname(os.path.abspath(path))
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("image,"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                sys.exit("bad line in %s (want image,offset_cm,yaw_deg): %r" % (path, line))
            pat = parts[0] if os.path.isabs(parts[0]) else os.path.join(base, parts[0])
            matches = sorted(glob.glob(pat))                 # a file name or a pattern such as L25_*.jpg (every match counts)
            if not matches:
                sys.exit("no image matches %r (listed in %s)" % (pat, path))
            rows += [(m, float(parts[1]), float(parts[2])) for m in matches]
    if not rows:
        sys.exit("no rows in %s" % path)
    return rows


def evaluate(calib, truth, estimator=None):
    """-> list of dicts: image, status, true/est offset (cm) and yaw (deg), errors (est - true)."""
    est_ = estimator or StateEstimator(calib)
    out = []
    for path, off_cm, yaw_deg in truth:
        e = est_.estimate(bd.load_image(path))
        r = {"image": path, "status": e.status, "true_offset_cm": off_cm, "true_yaw_deg": yaw_deg,
             "est_offset_cm": None, "est_yaw_deg": None, "err_offset_cm": None, "err_yaw_deg": None,
             "warnings": "; ".join(e.warnings[:2])}
        if e.valid:
            r["est_offset_cm"] = 100.0 * e.offset_m
            r["est_yaw_deg"] = math.degrees(e.heading_rad)
            r["err_offset_cm"] = r["est_offset_cm"] - off_cm
            r["err_yaw_deg"] = r["est_yaw_deg"] - yaw_deg
        out.append(r)
    return out


def summarise(results, tol_cm=1.0, tol_deg=2.0, sign_min_cm=2.0, sign_min_deg=4.0):
    """-> (list of text lines, ok flag)."""
    lines, ok = [], True
    valid = [r for r in results if r["err_offset_cm"] is not None]
    if len(valid) < len(results):
        ok = False
        lines.append("FAIL: %d of %d photos have no valid estimate" % (len(results) - len(valid), len(results)))
    if not valid:
        return lines, False
    eo = np.array([r["err_offset_cm"] for r in valid])
    ey = np.array([r["err_yaw_deg"] for r in valid])
    lines.append("offset error: bias %+.2f cm, rms %.2f cm, worst %.2f cm   (guideline +-%.1f cm)" % (
        eo.mean(), math.sqrt((eo ** 2).mean()), np.abs(eo).max(), tol_cm))
    lines.append("yaw error   : bias %+.2f deg, rms %.2f deg, worst %.2f deg  (guideline +-%.1f deg)" % (
        ey.mean(), math.sqrt((ey ** 2).mean()), np.abs(ey).max(), tol_deg))
    if np.abs(eo).max() > tol_cm:
        ok = False
        lines.append("FAIL: offset error above %.1f cm" % tol_cm)
    if np.abs(ey).max() > tol_deg:
        ok = False
        lines.append("FAIL: yaw error above %.1f deg" % tol_deg)
    for name, tk, ek, mn in (("offset", "true_offset_cm", "est_offset_cm", sign_min_cm),
                             ("yaw", "true_yaw_deg", "est_yaw_deg", sign_min_deg)):
        wrong = [r for r in valid if abs(r[tk]) >= mn and r[tk] * r[ek] < 0]
        if wrong:
            ok = False
            lines.append("FAIL: %s has the WRONG SIGN on %d photo(s): %s (sign convention or axis calibration problem)" % (
                name, len(wrong), ", ".join(os.path.basename(r["image"]) for r in wrong)))
    pts = [(r["true_offset_cm"], r["est_offset_cm"]) for r in valid if r["true_yaw_deg"] == 0]
    if len(pts) >= 3 and np.ptp([p[0] for p in pts]) >= 3.0:
        slope, icpt = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
        lines.append("offset, zero-yaw photos only: estimate = %.2f x truth %+.2f cm (a perfect calibration gives 1.00 and 0.00)"
                     % (slope, icpt))
    if ok:
        lines.append("PASS (against the guideline thresholds). Still only as good as these %d photos." % len(valid))
    return lines, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("calibration")
    ap.add_argument("truth_csv")
    ap.add_argument("--csv", help="write the per-photo results here (must not exist)")
    ap.add_argument("--tol-cm", type=float, default=1.0)
    ap.add_argument("--tol-deg", type=float, default=2.0)
    a = ap.parse_args()
    if a.csv and os.path.exists(a.csv):
        sys.exit("%s already exists; nothing is overwritten" % a.csv)
    calib = CameraCalibration.load(a.calibration)
    missing = calib.missing_for_state("measured_width")
    if missing:
        sys.exit("calibration incomplete, cannot validate: missing %s" % ", ".join(missing))
    res = evaluate(calib, read_truth(a.truth_csv))
    fmt = "%-28s %-9s %9s %9s %8s | %8s %8s %8s"
    print(fmt % ("image", "status", "true cm", "est cm", "err cm", "true deg", "est deg", "err deg"))
    for r in res:
        f = lambda v, p="%+.2f": "-" if v is None else p % v  # noqa: E731
        print(fmt % (os.path.basename(r["image"])[:28], r["status"], f(r["true_offset_cm"]), f(r["est_offset_cm"]),
                     f(r["err_offset_cm"]), f(r["true_yaw_deg"], "%+.1f"), f(r["est_yaw_deg"], "%+.1f"), f(r["err_yaw_deg"], "%+.1f")))
        if r["warnings"] and r["status"] != "OK":
            print("    " + r["warnings"])
    lines, ok = summarise(res, a.tol_cm, a.tol_deg)
    print()
    print("\n".join(lines))
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(res[0]))
            w.writeheader()
            w.writerows(res)
        print("saved", a.csv)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
