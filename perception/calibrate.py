"""Build / update the camera calibration file step by step. Offline; reads images, writes a JSON file.

    # 1. What the existing recordings can give (needs only the 0.26 m lane width): horizon row and k
    python perception/calibrate.py fit-width data/2026*/ --out perception/calibration/initial.json

    # 2. Direct measurements with a ruler (no image needed)
    python perception/calibrate.py set perception/calibration/initial.json --out cal2.json \
        --param camera_forward_of_rear_axle_m --value 0.11 --status measured

    # 3. Tape strips at known distances from the lens -> focal/depth scale F (and a better horizon row)
    python perception/calibrate.py fit-depth cal2.json --rows 120 150 185 --distances 0.60 0.40 0.30 --out cal3.json

    # 4. Photo(s) with the car centred and aligned in the lane -> the vehicle axis line in the image
    python perception/calibrate.py fit-axis cal3.json centred_*.jpg --out cal4.json

    # 3a. Find the image rows of white tape strips in a photo (so you do not have to read pixel rows by hand)
    python perception/calibrate.py find-strips strips.jpg --expect 3 --annotate strips_marked.png

    # 5. Print a calibration file: every parameter with its value and status, and what is still missing
    python perception/calibrate.py show CAL.json

Every command writes a NEW file (it refuses to overwrite). Parameters never get default numbers: they are
fitted, measured, assumed (only if you say so with --status assumed) or unset.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boundary_detector as bd  # noqa: E402
from ground_plane import (CalibrationError, CameraCalibration, fit_depth_model,  # noqa: E402
                          fit_vehicle_axis, fit_width_model)


def _new_out(path):
    if os.path.exists(path):
        sys.exit("%s already exists; choose a new --out (nothing is overwritten)" % path)
    d = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(d):
        os.makedirs(d)


def cmd_fit_width(a):
    _new_out(a.out)
    files = bd.list_images(a.images) if not any(os.path.isdir(p) for p in a.images) else \
        sorted(sum([bd.list_images([p]) for p in a.images], []))
    ys, ws, n_valid = [], [], 0
    for p in files:
        det = bd.detect(bd.load_image(p), "lab")
        if det.valid and det.confidence >= a.min_confidence:
            n_valid += 1
            ys += [r["y"] for r in det.rows]
            ws += [r["width_px"] for r in det.rows]
    if n_valid < 20:
        sys.exit("only %d valid frames: not enough to fit" % n_valid)
    yh, k, rms, n_used = fit_width_model(ys, ws, a.lane_width)
    cal = CameraCalibration(a.lane_width)
    note = "fitted from %d valid frames (%d rows, %d after clipping); width residual rms %.1f px; flat ground, no distortion" % (
        n_valid, len(ys), n_used, rms)
    cal.set("horizon_row_px", yh, "fitted", note)
    cal.set("camera_height_eff_m", k, "fitted", note)
    cal.save(a.out)
    print("horizon_row_px = %.1f, camera_height_eff_m = %.3f (from lane width %.2f m)\n%s\nsaved %s" % (yh, k, a.lane_width, note, a.out))
    print("STILL UNSET (needed for offset/heading): %s" % ", ".join(cal.missing_for_state()))


def cmd_set(a):
    _new_out(a.out)
    cal = CameraCalibration.load(a.calibration)
    cal.set(a.param, a.value, a.status, a.note or "")
    cal.save(a.out)
    print("set %s = %s (%s); saved %s" % (a.param, a.value, a.status, a.out))


def cmd_fit_depth(a):
    _new_out(a.out)
    if len(a.rows) != len(a.distances):
        sys.exit("--rows has %d values but --distances has %d: give one distance per row, same order" % (len(a.rows), len(a.distances)))
    cal = CameraCalibration.load(a.calibration)
    F, yh, rms = fit_depth_model(a.rows, a.distances)
    note = "from %d tape strips at known distances; residual rms %.3f m" % (len(a.rows), rms)
    cal.set("depth_scale_F", F, "fitted", note)
    cal.set("horizon_row_px", yh, "fitted", note + " (replaces the width-model horizon)")
    cal.save(a.out)
    print("F = %.2f, horizon_row_px = %.1f, %s; saved %s" % (F, yh, note, a.out))
    if rms > 0.03:
        print("WARNING: residual %.3f m is large for tape measured to ~1 cm: re-check the tape distances / row readings" % rms)
    if len(a.rows) <= 2:
        print("NOTE: with 2 strips the fit has no redundancy (residual is always 0); use 3 or more")


def find_white_strips(bgr, min_run_px=40, tophat_min=35, kernel_rows=41, row_gap=2, min_row=50):
    """Image rows of horizontal bright (white-tape) strips. A strip row = a row with a horizontal run of at least
    `min_run_px` pixels that are brighter (Lab L) than the vertical grey opening of the image (kernel `kernel_rows`
    tall, i.e. taller than the tape) by `tophat_min`. Orange pixels are excluded; rows above `min_row` (room / horizon) are ignored. Returns a list of
    {"row": centre row (float), "y0", "y1", "run_px"} sorted from the top of the image (far) to the bottom (near)."""
    h, w = bgr.shape[:2]
    lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_rows))
    bright = cv2.subtract(lum, cv2.morphologyEx(lum, cv2.MORPH_OPEN, k)) >= tophat_min
    orange = bd.ld.orange_mask(bgr, bd.ld.LaneConfig(roi_top=0, roi_bottom_margin=0)) > 0
    bright &= ~orange
    hit = [0] * min_row
    for y in range(min_row, h):
        runs = [r for r in bd.ld._runs(bright[y].astype(np.uint8)) if r[1] - r[0] >= min_run_px]
        hit.append(max([r[1] - r[0] for r in runs]) if runs else 0)
    strips, cur = [], None
    for y, run in enumerate(hit):
        if run:
            if cur is not None and y - cur["y1"] <= row_gap + 1:
                cur["y1"] = y
                cur["w"].append((y, run))
            else:
                if cur:
                    strips.append(cur)
                cur = {"y0": y, "y1": y, "w": [(y, run)]}
    if cur:
        strips.append(cur)
    out = []
    for c in strips:
        ys = np.array([a for a, _ in c["w"]], float)
        rs = np.array([b for _, b in c["w"]], float)
        out.append({"row": float((ys * rs).sum() / rs.sum()), "y0": c["y0"], "y1": c["y1"], "run_px": int(rs.max())})
    return out


def cmd_find_strips(a):
    if a.annotate:
        _new_out(a.annotate)
    img = bd.load_image(a.image)
    if img.shape[:2] != (240, 320):
        print("WARNING: image is %dx%d, not 320x240 (the camera setup used everywhere else)" % (img.shape[1], img.shape[0]))
    found = find_white_strips(img, a.min_run, a.tophat_min, min_row=a.min_row)
    for i, f in enumerate(found):
        print("strip %d: row %.1f (rows %d-%d, widest run %d px)" % (i + 1, f["row"], f["y0"], f["y1"], f["run_px"]))
    if a.annotate:
        vis = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        for r in range(0, img.shape[0], 20):                                  # ruler: a tick + number every 20 rows
            cv2.line(vis, (0, r * 3), (14, r * 3), (255, 255, 255), 1)
            cv2.putText(vis, str(r), (16, r * 3 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        for i, f in enumerate(found):
            y = int(round(f["row"] * 3))
            cv2.line(vis, (0, y), (vis.shape[1] - 1, y), (0, 255, 0), 1)
            cv2.putText(vis, "strip %d  row %.1f" % (i + 1, f["row"]), (60, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(a.annotate, vis)
        print("annotated image (3x, row ruler on the left): %s" % a.annotate)
    if len(found) != a.expect:
        sys.exit("found %d strips, expected %d. Open the annotated image: if a strip is missing or an extra one is a "
                 "false detection, read the rows off the ruler and pass them to fit-depth by hand." % (len(found), a.expect))
    print("rows, far -> near (small row = far): %s" % " ".join("%.1f" % f["row"] for f in found))
    print("so, with tape at 0.60 / 0.45 / 0.30 m from the lens:  --rows %s --distances %s" % (
        " ".join("%.1f" % f["row"] for f in found), " ".join(["0.60", "0.45", "0.30"][:len(found)]) if len(found) == 3 else "<far..near>"))


def cmd_show(a):
    cal = CameraCalibration.load(a.calibration)
    for k in ("horizon_row_px", "camera_height_eff_m", "depth_scale_F", "axis_x_ref_px", "axis_slope",
              "camera_forward_of_rear_axle_m"):
        v = cal.get(k)
        print("%-32s %-12s %s" % (k, "unset" if v is None else "%.4f" % v, cal.status[k]))
    missing = cal.missing_for_state("measured_width")
    print("missing for offset/heading (default scale_source=measured_width): %s" % (", ".join(missing) or "nothing"))
    print("assumed parameters (blocks --enable-motors): %s" % (", ".join(cal.assumed_parameters()) or "none"))
    if missing:
        sys.exit(1)


def cmd_fit_axis(a):
    _new_out(a.out)
    cal = CameraCalibration.load(a.calibration)
    ys, xs, n = [], [], 0
    for p in bd.list_images(a.images):
        det = bd.detect(bd.load_image(p), "lab")
        if not det.valid:
            print("skipping %s (%s)" % (p, det.status))
            continue
        n += 1
        ys += [r["y"] for r in det.rows]
        xs += [r["centre_x"] for r in det.rows]
    x0, slope, rms = fit_vehicle_axis(ys, xs, cal.axis_ref_row)
    note = "from %d frames with the car centred and aligned; residual rms %.1f px" % (n, rms)
    cal.set("axis_x_ref_px", x0, "fitted", note)
    cal.set("axis_slope", slope, "fitted", note)
    cal.save(a.out)
    print("axis column at row %d = %.1f px, slope %.4f px/row; %s; saved %s" % (cal.axis_ref_row, x0, slope, note, a.out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit-width"); p.add_argument("images", nargs="+"); p.add_argument("--out", required=True)
    p.add_argument("--lane-width", type=float, default=0.26); p.add_argument("--min-confidence", type=float, default=0.5)
    p.set_defaults(fn=cmd_fit_width)
    p = sub.add_parser("set"); p.add_argument("calibration"); p.add_argument("--out", required=True)
    p.add_argument("--param", required=True); p.add_argument("--value", type=float, required=True)
    p.add_argument("--status", required=True, choices=["measured", "fitted", "assumed"]); p.add_argument("--note")
    p.set_defaults(fn=cmd_set)
    p = sub.add_parser("fit-depth"); p.add_argument("calibration"); p.add_argument("--out", required=True)
    p.add_argument("--rows", type=float, nargs="+", required=True)
    p.add_argument("--distances", type=float, nargs="+", required=True, help="metres from the lens, same order as --rows")
    p.set_defaults(fn=cmd_fit_depth)
    p = sub.add_parser("fit-axis"); p.add_argument("calibration"); p.add_argument("images", nargs="+")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_fit_axis)
    p = sub.add_parser("find-strips"); p.add_argument("image"); p.add_argument("--expect", type=int, default=3)
    p.add_argument("--annotate", help="write a 3x image with the detected rows and a row ruler (must not exist)")
    p.add_argument("--min-run", type=int, default=40, help="shortest bright horizontal run (px) that counts as a strip")
    p.add_argument("--tophat-min", type=int, default=35, help="brightness (Lab L) above the local floor, default 35")
    p.add_argument("--min-row", type=int, default=50, help="ignore image rows above this (room / horizon), default 50")
    p.set_defaults(fn=cmd_find_strips)
    p = sub.add_parser("show"); p.add_argument("calibration"); p.set_defaults(fn=cmd_show)
    a = ap.parse_args()
    try:
        a.fn(a)
    except CalibrationError as e:
        sys.exit("calibration error: %s" % e)


if __name__ == "__main__":
    main()
