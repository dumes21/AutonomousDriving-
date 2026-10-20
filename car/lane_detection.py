"""Orange lane-boundary detector for the JetRacer camera (OpenCV + numpy only).

Works on one BGR frame (the recorded 320x240 images). It only perceives; it never
touches the car. Python 3.6 compatible so it can later run on the Jetson.

    python car/lane_detection.py frame.jpg [more.jpg ...] [--out DIR] [--tophat-min 12]

How it works
------------
1. Colour: convert to Lab. Under the pink room lighting the orange tape is far
   easier to separate by Lab b* (yellow-blue axis) than by HSV hue (the tape's
   hue drifts to ~0-15 and its saturation is washed out). The absolute b* of
   the tape falls toward the image edges (vignetting), so a fixed threshold
   loses the right-hand line. Instead we take a white top-hat of b* (b* minus
   its grey opening with a kernel wider than the tape): narrow bright-in-b*
   structures are kept whatever the local floor level, and we require
   top-hat >= `tophat_min` and b* >= `lab_b_min` (a low absolute floor).
2. Region of interest: rows from `roi_top` down (above that is the room, which
   has warm colours too) and the lowest `roi_bottom_margin` rows (car bonnet).
3. Noise filter: morphological open/close, then drop connected blobs smaller
   than `min_blob_area`.
4. Sample rows: on each row in `sample_rows` take the horizontal runs of orange
   pixels. A row is usable only if it has exactly two runs, each narrower than
   `max_run_width`, not touching the image border (a clipped line would bias the
   centre) and separated by `min_lane_width_px`..`max_lane_width_px`. Left run =
   left boundary, right run = right boundary.
5. Lane centre = midpoint of the two runs on each usable row. A straight line
   is fitted through the centres (x as a function of row).

What is returned (LaneResult)
-----------------------------
status      "ok" | "one_line" | "no_line" | "ambiguous" | "insufficient"
valid       True only for status "ok": two boundaries seen on enough rows over a
            long enough span, and the fit is consistent. Anything else leaves
            `offset_norm`, `heading_slope` etc. as None: we do NOT guess a
            centre from a single line, because the lane width in pixels depends
            on distance and camera pose, which we have not calibrated.
confidence  0..1, a heuristic: (usable rows / 8, max 1) x fit quality. Not a probability.
offset_norm lane offset in IMAGE space, see below.
heading_slope  image-space direction of the lane centre line, see below.
curvature   second-order coefficient of the centre line (None if < 5 rows).

IMAGE-SPACE FEATURES ARE NOT METRES OR RADIANS
----------------------------------------------
offset_norm = (centre_x - image_centre_x) / (lane_width_px / 2), measured at
the reference row `ref_row`. Sign matches JetRacerEnv: + means the lane centre
is to the right of the camera, i.e. the car is LEFT of centre. It is a ratio, so
it is independent of pixel scale, but it assumes the camera sits on the car's
centre line, looks straight ahead, and ignores lens distortion and camera
height/tilt. It is therefore only a proxy for JetRacerEnv's
lane_offset / lane_half_width; it equals it only after calibration.

heading_slope = d(centre_x)/d(rows upward), in pixels per pixel. + means the
centre line moves right as it recedes (the track bends/points right of the
camera, i.e. the car points LEFT of the track, as JetRacerEnv's + heading_error).
It mixes true heading error, lateral offset (perspective) and track curvature,
and is not an angle. Do not divide by pi and feed it to a policy trained in the
simulator without calibrating it against known poses.
"""
import argparse
import os

import cv2
import numpy as np

STATUS_OK = "ok"
STATUS_ONE_LINE = "one_line"
STATUS_NO_LINE = "no_line"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_INSUFFICIENT = "insufficient"


class LaneConfig(object):
    """All tunables in one place. Defaults are for 320x240 frames."""

    def __init__(self, **kw):
        # colour (Lab, OpenCV 8-bit scale: 128 = neutral)
        self.lab_b_min = 128           # absolute floor on b*
        self.tophat_kernel = 61        # px, must be wider than the tape (~30 px near the camera)
        self.tophat_min = 12           # b* units above the local floor
        # region of interest
        self.roi_top = 75              # ignore rows above (room / horizon)
        self.roi_bottom_margin = 12    # ignore the lowest rows (car bonnet)
        # noise filtering
        self.open_kernel = 3
        self.close_kernel = 5
        self.min_blob_area = 40        # px
        # row sampling
        self.sample_rows = list(range(100, 228, 8))
        self.max_run_width = 70        # px; wider = a near-horizontal stretch of line, not a crossing
        self.min_run_width = 2
        self.border_margin = 2         # a run within this of the left/right edge is "clipped"
        self.min_lane_width_px = 40
        self.max_lane_width_px = 300
        # validity
        self.min_valid_rows = 3
        self.full_confidence_rows = 8  # usable rows for full coverage credit
        self.min_row_span = 30         # px between first and last usable row
        self.ref_row = 140             # offset is read here (must lie inside the usable span).
                                       # Near rows are often clipped by the image edge, so this is
                                       # a look-ahead row, not the row under the wheels.
        self.ref_row_tolerance = 10
        self.max_fit_rms_px = 6.0
        self.min_confidence = 0.35
        for k, v in kw.items():
            if not hasattr(self, k):
                raise TypeError("unknown LaneConfig option: %s" % k)
            setattr(self, k, v)


class LaneResult(object):
    def __init__(self):
        self.status = STATUS_NO_LINE
        self.valid = False
        self.confidence = 0.0
        self.offset_norm = None       # image-space, see module docstring
        self.heading_slope = None     # image-space, see module docstring
        self.curvature = None
        self.lane_width_px = None     # at ref_row
        self.centre_x_ref = None
        self.n_rows_used = 0
        self.n_rows_sampled = 0
        self.fit_rms_px = None
        self.line_pixels = 0          # orange pixels in the ROI after filtering
        self.rows = []                # per-row dicts: y, left, right, centre, width
        self.mask = None              # filtered orange mask (full frame, uint8 0/255)
        self.centre_fit = None        # (a, b, c): x = a*y^2 + b*y + c  or (0, b, c)
        self.visible_side = None      # for one_line: "left"/"right" by image half (ambiguous in corners)

    def as_dict(self):
        return {
            "status": self.status, "valid": int(self.valid), "confidence": round(self.confidence, 3),
            "offset_norm": _r(self.offset_norm), "heading_slope": _r(self.heading_slope),
            "curvature": _r(self.curvature, 6), "lane_width_px": _r(self.lane_width_px),
            "n_rows_used": self.n_rows_used, "n_rows_sampled": self.n_rows_sampled,
            "fit_rms_px": _r(self.fit_rms_px), "line_pixels": self.line_pixels,
        }


def _r(v, n=4):
    return "" if v is None else round(float(v), n)


def orange_mask(bgr, cfg):
    """Binary mask (0/255) of orange-tape pixels inside the ROI, noise-filtered."""
    h = bgr.shape[0]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    b = lab[:, :, 2]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.tophat_kernel, cfg.tophat_kernel))
    tophat = cv2.subtract(b, cv2.morphologyEx(b, cv2.MORPH_OPEN, k))
    mask = ((tophat >= cfg.tophat_min) & (b >= cfg.lab_b_min)).astype(np.uint8) * 255
    mask[:cfg.roi_top] = 0
    if cfg.roi_bottom_margin > 0:
        mask[h - cfg.roi_bottom_margin:] = 0
    if cfg.open_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((cfg.open_kernel,) * 2, np.uint8))
    if cfg.close_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((cfg.close_kernel,) * 2, np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= cfg.min_blob_area:
            keep[i] = 255
    return keep[labels]


def _runs(row):
    """(start, end_exclusive) of each run of non-zero pixels in a 1-D array."""
    padded = np.concatenate(([0], (row > 0).astype(np.int8), [0]))
    d = np.diff(padded)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def detect_lane(bgr, cfg=None):
    """Detect the two orange boundaries in one BGR frame. Returns a LaneResult."""
    cfg = cfg or LaneConfig()
    res = LaneResult()
    h, w = bgr.shape[:2]
    mask = orange_mask(bgr, cfg)
    res.mask = mask
    res.line_pixels = int(np.count_nonzero(mask))
    if res.line_pixels == 0:
        return res  # status no_line

    rows = []
    n_one = n_amb = 0
    for y in cfg.sample_rows:
        if y >= h:
            continue
        runs = [r for r in _runs(mask[y]) if r[1] - r[0] >= cfg.min_run_width]
        if not runs:
            continue
        # A run touching the border is clipped (line leaves the image): unusable for a centre.
        good = [r for r in runs if (r[1] - r[0]) <= cfg.max_run_width
                and r[0] > cfg.border_margin and r[1] < w - cfg.border_margin]
        if len(runs) == 2 and len(good) == 2:
            (l0, l1), (r0, r1) = good
            lx, rx = 0.5 * (l0 + l1), 0.5 * (r0 + r1)
            if cfg.min_lane_width_px <= rx - lx <= cfg.max_lane_width_px:
                rows.append({"y": y, "left": lx, "right": rx, "centre": 0.5 * (lx + rx), "width": rx - lx})
                continue
            n_amb += 1
        elif len(runs) == 1:
            n_one += 1
        else:
            n_amb += 1
    res.rows = rows
    res.n_rows_used = len(rows)
    res.n_rows_sampled = len([y for y in cfg.sample_rows if y < h])

    if len(rows) < cfg.min_valid_rows:
        res.confidence = 0.0
        if rows:
            res.status = STATUS_INSUFFICIENT      # two lines on only 1-2 rows
        elif n_amb > 0:
            res.status = STATUS_AMBIGUOUS         # runs seen, but clipped/merged/too many
        elif n_one > 0:
            res.status = STATUS_ONE_LINE          # exactly one line, on every row that has any
            xs = np.where(mask.any(axis=0))[0]
            res.visible_side = "left" if xs.mean() < w / 2 else "right"
        return res

    ys = np.array([r["y"] for r in rows], float)
    cs = np.array([r["centre"] for r in rows])
    ws = np.array([r["width"] for r in rows])
    span = ys.max() - ys.min()
    # Centre line fit: linear always, quadratic (for curvature) only with enough rows.
    b, c = np.polyfit(ys, cs, 1)
    res.centre_fit = (0.0, b, c)
    if len(rows) >= 5 and span >= cfg.min_row_span:
        a2, b2, c2 = np.polyfit(ys, cs, 2)
        res.curvature = float(a2)
        res.centre_fit = (a2, b2, c2)
    # Residual against the fit actually used (a bend is not a fit error).
    fa, fb, fc = res.centre_fit
    rms = float(np.sqrt(np.mean((cs - (fa * ys * ys + fb * ys + fc)) ** 2)))
    res.fit_rms_px = rms
    wb, wc = np.polyfit(ys, ws, 1)  # width should grow smoothly toward the camera
    wrms = float(np.sqrt(np.mean((ws - (wb * ys + wc)) ** 2)))

    coverage = min(1.0, len(rows) / float(cfg.full_confidence_rows))
    quality = float(np.exp(-rms / 6.0) * np.exp(-wrms / 10.0))
    res.confidence = float(coverage * quality)

    ref = cfg.ref_row
    if span < cfg.min_row_span or not (ys.min() - cfg.ref_row_tolerance <= ref <= ys.max() + cfg.ref_row_tolerance):
        res.status = STATUS_INSUFFICIENT  # we would have to extrapolate to the reference row
        return res
    if rms > cfg.max_fit_rms_px or res.confidence < cfg.min_confidence:
        res.status = STATUS_AMBIGUOUS
        return res

    centre_ref = res.centre_fit[0] * ref * ref + res.centre_fit[1] * ref + res.centre_fit[2]
    width_ref = wb * ref + wc
    if width_ref < cfg.min_lane_width_px:
        res.status = STATUS_AMBIGUOUS
        return res
    res.centre_x_ref = float(centre_ref)
    res.lane_width_px = float(width_ref)
    res.offset_norm = float((centre_ref - (w - 1) / 2.0) / (width_ref / 2.0))
    res.heading_slope = float(-b)   # rows increase downward, so "per row upward" = -b
    res.status = STATUS_OK
    res.valid = True
    return res


def annotate(bgr, res, cfg=None):
    """Debug image: orange pixels highlighted, usable row points, fitted centre, text."""
    cfg = cfg or LaneConfig()
    out = bgr.copy()
    h, w = out.shape[:2]
    if res.mask is not None:
        out[res.mask > 0] = (0.35 * out[res.mask > 0] + 0.65 * np.array([0, 255, 255])).astype(np.uint8)
    cv2.line(out, (0, cfg.roi_top), (w, cfg.roi_top), (255, 255, 255), 1)
    cv2.line(out, ((w - 1) // 2, h - 60), ((w - 1) // 2, h), (255, 255, 255), 1)   # image centre
    for r in res.rows:
        y = int(r["y"])
        cv2.circle(out, (int(r["left"]), y), 3, (255, 0, 0), -1)     # left boundary: blue
        cv2.circle(out, (int(r["right"]), y), 3, (0, 0, 255), -1)    # right boundary: red
        cv2.circle(out, (int(r["centre"]), y), 3, (0, 255, 0), -1)   # centre: green
    if res.centre_fit is not None and res.rows:
        a, b, c = res.centre_fit
        y0, y1 = int(min(r["y"] for r in res.rows)), int(max(r["y"] for r in res.rows))
        pts = np.array([[a * y * y + b * y + c, y] for y in range(y0, y1 + 1, 2)], np.int32)
        cv2.polylines(out, [pts], False, (0, 255, 0), 1)
    colour = (0, 200, 0) if res.valid else (0, 0, 255)
    off = "" if res.offset_norm is None else " off %+.2f hd %+.2f" % (res.offset_norm, res.heading_slope)
    cv2.putText(out, "%s c=%.2f%s" % (res.status, res.confidence, off), (4, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description="Run the orange-lane detector on image files.")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", help="folder for annotated debug images (default: just print)")
    ap.add_argument("--tophat-min", type=int,
                    help="b* top-hat threshold (default %d); lower = more sensitive" % LaneConfig().tophat_min)
    ap.add_argument("--roi-top", type=int, help="first image row to consider (default %d)" % LaneConfig().roi_top)
    args = ap.parse_args()

    kw = {}
    if args.tophat_min is not None:
        kw["tophat_min"] = args.tophat_min
    if args.roi_top is not None:
        kw["roi_top"] = args.roi_top
    cfg = LaneConfig(**kw)
    if args.out and not os.path.isdir(args.out):
        os.makedirs(args.out)
    for path in args.images:
        img = cv2.imread(path)
        if img is None:
            print("%s: cannot read" % path)
            continue
        res = detect_lane(img, cfg)
        d = res.as_dict()
        print("%s: %s valid=%d conf=%.2f offset_norm=%s heading_slope=%s rows=%d/%d  (image-space, uncalibrated)"
              % (path, res.status, res.valid, res.confidence, d["offset_norm"] or "-", d["heading_slope"] or "-",
                 res.n_rows_used, res.n_rows_sampled))
        if args.out:
            cv2.imwrite(os.path.join(args.out, os.path.basename(path)), annotate(img, res, cfg))


if __name__ == "__main__":
    main()
