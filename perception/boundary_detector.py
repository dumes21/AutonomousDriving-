"""Yellow/orange lane-boundary detector for saved JetRacer camera images (perception only).

Standalone: reads image files, writes debug images. It does not talk to the car, the simulator
or the DQN, and it reports positions in IMAGE PIXELS only.

    python perception/boundary_detector.py IMAGE_OR_DIR [...] [--method lab|hsv|both] [--out DIR] [--csv FILE]

Two colour methods are provided and can be compared on the same image:

lab  (default) Reuses car/lane_detection.py: Lab b* white top-hat + a low absolute floor.
hsv  HSV with a WRAPPED hue window and a saturation top-hat. Under the pink cast the tape's hue is split:
     ~63 % of tape pixels have hue 0-20 and ~37 % have hue 170-179 (the red side of pink), while the pink
     floor sits at hue 155-175. So a single narrow range such as H 10-30 misses most of the tape, and a wide
     range pulls in the floor. The saturation top-hat (S minus its local grey opening) keeps only narrow
     structures that are more saturated than their surroundings, whatever the lighting level.
     Defaults (OpenCV scale, H 0-179): hue 0-22 or 172-179, S >= 60, V >= 110, S top-hat >= 20.
     The numbers come from pixels the Lab method selects in the recorded frames, so they are tuned to this
     camera, track and lighting. They are NOT validated against hand-labelled data.

Both methods share the same row-based boundary logic (car/lane_detection.detect_lane): on each sampled image
row exactly two un-clipped orange runs are needed, the left one is the left boundary and the right one the right
boundary. Otherwise the result is INVALID, with no boundary or centre coordinates (nothing is fabricated).

Pixel quantities only: `image_centre_x_px` is simply the middle column of the image. The camera is not
assumed to sit on the vehicle's centre line, so the lane centre is NOT converted into a lateral offset.
`vehicle_centre_x_px` is None until it is calibrated.
"""
import argparse
import contextlib
import csv
import datetime
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "car"))
import lane_detection as ld  # noqa: E402  (existing, committed detector: Lab mask + row logic)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


class HSVConfig(object):
    """Tunable thresholds for the HSV method (OpenCV scale: H 0-179, S/V 0-255)."""

    def __init__(self, **kw):
        self.hue_low_max = 22        # warm window: hue 0..hue_low_max ...
        self.hue_high_min = 172      # ... or hue_high_min..179 (wraps around red)
        self.s_min = 60
        self.v_min = 110
        self.s_tophat_min = 20       # saturation above the local grey opening (0 disables the gate)
        self.tophat_kernel = 61      # px, wider than the tape (~30 px near the camera)
        for k, v in kw.items():
            if not hasattr(self, k):
                raise TypeError("unknown HSVConfig option: %s" % k)
            setattr(self, k, v)


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise IOError("cannot read image: %s" % path)
    return img


def hsv_mask(bgr, hsv_cfg, lane_cfg):
    """Binary (0/255) mask of orange tape, HSV method, with the same ROI/morphology/blob filter as the Lab method."""
    h_img = bgr.shape[0]
    h, s, v = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    m = ((h <= hsv_cfg.hue_low_max) | (h >= hsv_cfg.hue_high_min)) & (s >= hsv_cfg.s_min) & (v >= hsv_cfg.v_min)
    if hsv_cfg.s_tophat_min > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hsv_cfg.tophat_kernel, hsv_cfg.tophat_kernel))
        m &= cv2.subtract(s, cv2.morphologyEx(s, cv2.MORPH_OPEN, k)) >= hsv_cfg.s_tophat_min
    mask = m.astype(np.uint8) * 255
    mask[:lane_cfg.roi_top] = 0                                           # region of interest
    if lane_cfg.roi_bottom_margin > 0:
        mask[h_img - lane_cfg.roi_bottom_margin:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((lane_cfg.open_kernel,) * 2, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((lane_cfg.close_kernel,) * 2, np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= lane_cfg.min_blob_area:
            keep[i] = 255
    return keep[labels]


@contextlib.contextmanager
def _mask_override(fn):
    """Let car/lane_detection.detect_lane use another mask function, without editing that file."""
    original = ld.orange_mask
    ld.orange_mask = fn
    try:
        yield
    finally:
        ld.orange_mask = original


class BoundaryResult(object):
    def __init__(self):
        self.method = ""
        self.valid = False                  # True only when BOTH boundaries were found confidently
        self.status = ""                    # ok | one_line | ambiguous | insufficient | no_line
        self.confidence = 0.0               # 0..1 heuristic (usable rows x fit quality); not a probability
        self.warnings = []                  # human-readable reasons when not valid / unreliable
        self.image_width = 0
        self.image_height = 0
        self.image_centre_x_px = 0.0        # middle column of the image, NOT the vehicle centre
        self.vehicle_centre_x_px = None     # unknown until calibrated
        self.rows = []                      # valid only: dicts y, left_x, right_x, centre_x, width_px
        self.centre_fit = None              # valid only: (a, b, c) with x = a*y^2 + b*y + c
        self.centre_minus_image_centre_px = None   # valid only: lane centre x - image centre x at ref_row (pixels)
        self.lane_width_px = None           # valid only, at ref_row
        self.ref_row = None
        self.candidate_points = []          # INVALID frames: unassigned single-line points (x, y), debug only
        self.mask = None
        self.roi = None                     # (top_row, bottom_row_exclusive)


def detect(bgr, method="lab", lane_cfg=None, hsv_cfg=None):
    lane_cfg = lane_cfg or ld.LaneConfig()
    hsv_cfg = hsv_cfg or HSVConfig()
    if method == "lab":
        raw = ld.detect_lane(bgr, lane_cfg)
    elif method == "hsv":
        with _mask_override(lambda img, cfg: hsv_mask(img, hsv_cfg, cfg)):
            raw = ld.detect_lane(bgr, lane_cfg)
    else:
        raise ValueError("method must be 'lab' or 'hsv'")

    h, w = bgr.shape[:2]
    r = BoundaryResult()
    r.method, r.status, r.confidence = method, raw.status, raw.confidence
    r.image_width, r.image_height = w, h
    r.image_centre_x_px = (w - 1) / 2.0
    r.mask = raw.mask
    r.roi = (lane_cfg.roi_top, h - lane_cfg.roi_bottom_margin)
    r.valid = bool(raw.valid)

    if r.valid:
        r.rows = [{"y": q["y"], "left_x": q["left"], "right_x": q["right"], "centre_x": q["centre"],
                   "width_px": q["width"]} for q in raw.rows]
        r.centre_fit = raw.centre_fit
        r.ref_row = lane_cfg.ref_row
        r.lane_width_px = raw.lane_width_px
        r.centre_minus_image_centre_px = raw.centre_x_ref - r.image_centre_x_px
        if raw.confidence < 0.5:
            r.warnings.append("low confidence (%.2f): treat with caution" % raw.confidence)
        return r

    # INVALID: no boundary or centre coordinates are returned, only explicit reasons.
    reasons = {
        ld.STATUS_ONE_LINE: "only ONE boundary visible (%s side of the image); the other is out of view or hidden"
                            % (raw.visible_side or "unknown"),
        ld.STATUS_AMBIGUOUS: "boundaries ambiguous (merged, clipped by the image edge, or inconsistent fit)",
        ld.STATUS_INSUFFICIENT: "both boundaries seen on too few rows to measure a lane centre reliably",
        ld.STATUS_NO_LINE: "no boundary detected",
    }
    r.warnings.append("INVALID: " + reasons.get(raw.status, raw.status))
    if raw.mask is not None:
        for y in lane_cfg.sample_rows:
            if y >= h:
                continue
            runs = [a for a in ld._runs(raw.mask[y]) if a[1] - a[0] >= lane_cfg.min_run_width]
            if len(runs) == 1:
                r.candidate_points.append((0.5 * (runs[0][0] + runs[0][1]), y))
    return r


def render_debug(bgr, res, lane_cfg=None, scale=2):
    """Side-by-side debug image: overlay on the camera image | binary mask. Returns a BGR image."""
    lane_cfg = lane_cfg or ld.LaneConfig()
    h, w = bgr.shape[:2]
    big = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
    S = lambda v: int(round(v * scale))  # noqa: E731
    top, bot = res.roi

    # Region of interest: darken everything outside it, outline it.
    shade = big.copy()
    shade[:S(top)] = (shade[:S(top)] * 0.35).astype(np.uint8)
    shade[S(bot):] = (shade[S(bot):] * 0.35).astype(np.uint8)
    overlay = shade
    cv2.line(overlay, (0, S(top)), (w * scale, S(top)), (255, 255, 255), 1)
    cv2.line(overlay, (0, S(bot)), (w * scale, S(bot)), (255, 255, 255), 1)
    cv2.putText(overlay, "ROI (rows %d-%d)" % (top, bot), (4, S(top) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)

    # Detected orange pixels, tinted.
    if res.mask is not None:
        m = cv2.resize(res.mask, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST) > 0
        overlay[m] = (0.45 * overlay[m] + 0.55 * np.array([0, 255, 255])).astype(np.uint8)

    # Image centre column (not the vehicle centre).
    cx = S(res.image_centre_x_px)
    cv2.line(overlay, (cx, S(bot) - 40), (cx, S(bot)), (255, 255, 255), 1)
    cv2.putText(overlay, "image centre", (cx + 3, S(bot) - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                cv2.LINE_AA)

    if res.valid:
        for q in res.rows:
            y = S(q["y"])
            cv2.circle(overlay, (S(q["left_x"]), y), 4, (255, 0, 0), -1)       # left boundary: blue
            cv2.circle(overlay, (S(q["right_x"]), y), 4, (0, 0, 255), -1)      # right boundary: red
            cv2.circle(overlay, (S(q["centre_x"]), y), 3, (0, 255, 0), -1)     # lane centre: green
        a, b, c = res.centre_fit
        ys = [q["y"] for q in res.rows]
        pts = np.array([[S(a * y * y + b * y + c), S(y)] for y in range(int(min(ys)), int(max(ys)) + 1, 2)], np.int32)
        cv2.polylines(overlay, [pts], False, (0, 255, 0), 2)                   # lane centreline
        banner, colour = "VALID (%s): both boundaries, conf %.2f" % (res.method, res.confidence), (0, 150, 0)
    else:
        for (x, y) in res.candidate_points:                                    # unassigned, debug only
            cv2.circle(overlay, (S(x), S(y)), 4, (0, 165, 255), 1)
        banner, colour = "INVALID (%s): %s" % (res.method, res.status), (0, 0, 220)
    cv2.rectangle(overlay, (0, 0), (w * scale, 22), colour, -1)
    cv2.putText(overlay, banner, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # Warnings in a solid red strip at the bottom (white text), the pixel-only reminder below it.
    lines = list(res.warnings)
    strip = 18 * len(lines)
    if strip:
        y0 = h * scale - 18 - strip
        cv2.rectangle(overlay, (0, y0), (w * scale, h * scale - 18), (0, 0, 200), -1)
        for i, msg in enumerate(lines):
            cv2.putText(overlay, msg[:86], (6, y0 + 13 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                        cv2.LINE_AA)
    cv2.putText(overlay, "pixels only - not metres / radians (uncalibrated)", (6, h * scale - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    mask_img = np.zeros((h * scale, w * scale, 3), np.uint8)
    if res.mask is not None:
        m = cv2.resize(res.mask, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        mask_img[m > 0] = (255, 255, 255)
    cv2.line(mask_img, (0, S(top)), (w * scale, S(top)), (0, 200, 255), 1)
    cv2.line(mask_img, (0, S(bot)), (w * scale, S(bot)), (0, 200, 255), 1)
    cv2.putText(mask_img, "binary mask (%s)" % res.method, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1,
                cv2.LINE_AA)
    return np.hstack([overlay, mask_img])


def list_images(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for e in IMAGE_EXTS:
                out += glob.glob(os.path.join(p, "*" + e))
        else:
            out.append(p)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="image files and/or folders")
    ap.add_argument("--method", choices=["lab", "hsv", "both"], default="lab")
    ap.add_argument("--out", help="debug folder (default: logs/perception_debug/<timestamp>); must not exist yet")
    ap.add_argument("--no-images", action="store_true", help="only print/CSV the results, no debug images")
    ap.add_argument("--csv", help="write one row per image and method to this CSV (must not exist yet)")
    args = ap.parse_args()

    files = list_images(args.images)
    if not files:
        sys.exit("no images found")
    out = None
    if not args.no_images:
        out = args.out or os.path.join(ROOT, "logs", "perception_debug", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(out)                                   # raises if it already exists: never overwrites
    if args.csv and os.path.exists(args.csv):
        sys.exit("%s already exists" % args.csv)
    methods = ["lab", "hsv"] if args.method == "both" else [args.method]
    rows = []
    for path in files:
        img = load_image(path)
        for m in methods:
            res = detect(img, m)
            print("%s [%s]: %s conf=%.2f%s%s" % (
                path, m, "VALID" if res.valid else "INVALID(%s)" % res.status, res.confidence,
                "" if not res.valid else "  lane centre %.1f px vs image centre %.1f px at row %d, lane width %.0f px"
                % (res.centre_minus_image_centre_px + res.image_centre_x_px, res.image_centre_x_px, res.ref_row,
                   res.lane_width_px),
                "" if not res.warnings else "  WARN: " + "; ".join(res.warnings)))
            rows.append({"image": path, "method": m, "valid": int(res.valid), "status": res.status,
                         "confidence": round(res.confidence, 3),
                         "lane_centre_x_at_ref_row_px": "" if not res.valid else round(res.centre_minus_image_centre_px + res.image_centre_x_px, 1),
                         "image_centre_x_px": res.image_centre_x_px,
                         "lane_width_px_at_ref_row": "" if not res.valid else round(res.lane_width_px, 1),
                         "rows_used": len(res.rows)})
            if out:
                name = "%s_%s.png" % (os.path.splitext(os.path.basename(path))[0], m)
                parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
                cv2.imwrite(os.path.join(out, parent + "_" + name), render_debug(img, res))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if out:
        print("debug images in", out)


if __name__ == "__main__":
    main()
