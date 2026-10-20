"""Run the orange-lane detector over recorded sessions (offline; reads images only).

    python train/analyse_lane_data.py                    # all data/2*/ sessions
    python train/analyse_lane_data.py data/20261013_003714 --out logs/lane

Writes into --out (default logs/lane, git-ignored):
    lane_features.csv      one row per recorded frame (see COLUMNS)
    summary_by_session.csv success rates per session
    debug/<kind>/*.jpg     annotated frames: straight, corner, failure_<status>

Prints success rates by session, by how hard the human was steering (a proxy for
corners) and by the detected centre-line curvature, a threshold-sensitivity
check, frame-to-frame stability, and the correlation of the image-space
features with the recorded steering.

The features are IMAGE-SPACE proxies (see car/lane_detection.py). Nothing here
is a distance in metres or an angle in radians. Raw images are never modified.
"""
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "car"))
from lane_detection import LaneConfig, annotate, detect_lane  # noqa: E402
from session_labels import read_label_rows  # noqa: E402

CENTRE_ROWS = (110, 125, 140, 155)   # image rows at which the fitted lane centre is reported
COLUMNS = (["session", "image", "frame", "steering", "throttle", "timestamp", "in_clean",
            "status", "valid", "confidence", "offset_norm", "heading_slope", "curvature",
            "lane_width_px", "n_rows_used", "n_rows_sampled", "fit_rms_px", "line_pixels"]
           + ["centre_x_row%d" % r for r in CENTRE_ROWS])
STATUSES = ["ok", "one_line", "ambiguous", "insufficient", "no_line"]
TURN_STEER = 0.4          # |steering| at/above this = human is steering hard (corner proxy)
CORNER_CURV = 0.002       # |curvature| (px/px^2) at/above this = curved centre line


def load_session(session, cfg):
    """List of dicts: one per frame in labels.csv (discarded segments excluded)."""
    clean = set()
    cpath = os.path.join(session, "labels_clean.csv")
    if os.path.exists(cpath):
        with open(cpath, newline="") as f:
            clean = {r["filename"] for r in csv.DictReader(f)}
    out = []
    for r in read_label_rows(session):
        path = os.path.join(session, r["filename"])
        img = cv2.imread(path)
        if img is None:
            print("  cannot read", path)
            continue
        res = detect_lane(img, cfg)
        d = res.as_dict()
        row = {"session": os.path.basename(os.path.normpath(session)), "image": os.path.relpath(path, ROOT),
               "frame": int(os.path.splitext(r["filename"])[0]), "steering": float(r["steering"]),
               "throttle": float(r["throttle"]), "timestamp": float(r["timestamp"]),
               "in_clean": int(r["filename"] in clean)}
        row.update(d)
        for yr in CENTRE_ROWS:
            v = ""
            if res.rows and res.centre_fit is not None and res.valid:
                ys = [q["y"] for q in res.rows]
                if min(ys) - 5 <= yr <= max(ys) + 5:     # interpolate only, never extrapolate far
                    a, b, c = res.centre_fit
                    v = round(a * yr * yr + b * yr + c, 1)
            row["centre_x_row%d" % yr] = v
        out.append((row, img, res))
    return out


def pct(n, d):
    return 100.0 * n / d if d else float("nan")


def status_table(rows, title):
    print("\n%s" % title)
    print("%-22s %6s %7s %9s %10s %13s %8s %9s" % ("", "frames", "ok %", "one_line", "ambiguous", "insufficient",
                                                   "no_line", "mean conf"))
    groups = {}
    for r in rows:
        groups.setdefault(r[0], []).append(r[1])
    for name in sorted(groups):
        g = groups[name]
        n = len(g)
        c = {s: sum(1 for x in g if x["status"] == s) for s in STATUSES}
        print("%-22s %6d %7.1f %9.1f %10.1f %13.1f %8.1f %9.2f"
              % (name, n, pct(c["ok"], n), pct(c["one_line"], n), pct(c["ambiguous"], n),
                 pct(c["insufficient"], n), pct(c["no_line"], n), np.mean([x["confidence"] for x in g])))


def fnum(v):
    return float(v) if v != "" else float("nan")


def save_debug(items, out_dir, cfg, per_kind=8):
    """Annotated examples: straights, corners and failures."""
    ok = [(r, i, res) for r, i, res in items if r["valid"]]
    picks = {}
    # straights: valid and almost no curvature, the human not steering hard
    straight = [t for t in ok if t[2].curvature is not None and abs(t[2].curvature) < 0.0006
                and abs(t[0]["steering"]) < TURN_STEER]
    picks["straight"] = straight
    # corners: valid with strongly curved centre line, or valid while steering hard
    picks["corner"] = sorted([t for t in ok if t[2].curvature is not None and abs(t[2].curvature) >= CORNER_CURV],
                             key=lambda t: -abs(t[2].curvature))
    picks["corner_steering"] = [t for t in ok if abs(t[0]["steering"]) >= TURN_STEER]
    for s in STATUSES[1:]:
        picks["failure_" + s] = [t for t in items if t[0]["status"] == s]
    # corner failures specifically (human steering hard, detector not ok)
    picks["failure_while_turning"] = [t for t in items if not t[0]["valid"] and abs(t[0]["steering"]) >= TURN_STEER]
    for kind, lst in picks.items():
        if not lst:
            continue
        d = os.path.join(out_dir, "debug", kind)
        os.makedirs(d, exist_ok=True)
        idx = np.linspace(0, len(lst) - 1, min(per_kind, len(lst))).astype(int)
        for k in idx:
            r, img, res = lst[k]
            name = "%s_%06d.jpg" % (r["session"], r["frame"])
            cv2.imwrite(os.path.join(d, name), annotate(img, res, cfg))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", help="session folders (default: data/2*)")
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "lane"))
    ap.add_argument("--no-debug", action="store_true", help="skip the annotated images")
    ap.add_argument("--no-sweep", action="store_true", help="skip the threshold-sensitivity sweep")
    args = ap.parse_args()

    sessions = args.sessions or sorted(glob.glob(os.path.join(ROOT, "data", "2*")))
    sessions = [s for s in sessions if os.path.isdir(s)]
    if not sessions:
        sys.exit("no session folders found")
    os.makedirs(args.out, exist_ok=True)
    cfg = LaneConfig()

    items = []
    for s in sessions:
        print("processing", s)
        items += load_session(s, cfg)
    rows = [t[0] for t in items]
    print("%d frames in %d sessions" % (len(rows), len(sessions)))

    with open(os.path.join(args.out, "lane_features.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote", os.path.join(args.out, "lane_features.csv"))

    # --- success rates -----------------------------------------------------------------------
    by_sess = [(r["session"], r) for r in rows] + [("ALL", r) for r in rows]
    status_table(by_sess, "Detection status by session (all recorded frames)")
    status_table([(r["session"] + " (clean)", r) for r in rows if r["in_clean"]]
                 + [("ALL (clean)", r) for r in rows if r["in_clean"]],
                 "Same, only frames kept in labels_clean.csv")
    with open(os.path.join(args.out, "summary_by_session.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "frames"] + STATUSES + ["ok_percent", "mean_confidence"])
        for name in sorted({r["session"] for r in rows}) + ["ALL"]:
            g = [r for r in rows if name == "ALL" or r["session"] == name]
            w.writerow([name, len(g)] + [sum(1 for x in g if x["status"] == s) for s in STATUSES]
                       + ["%.1f" % pct(sum(1 for x in g if x["valid"]), len(g)),
                          "%.3f" % np.mean([x["confidence"] for x in g])])

    # --- corners ------------------------------------------------------------------------------
    turning = [r for r in rows if abs(r["steering"]) >= TURN_STEER]
    gentle = [r for r in rows if abs(r["steering"]) < TURN_STEER]
    status_table([("|steer| >= %.1f (turning)" % TURN_STEER, r) for r in turning]
                 + [("|steer| <  %.1f (gentle)" % TURN_STEER, r) for r in gentle],
                 "By how hard the human was steering (proxy for corners; the steering label is not a "
                 "ground-truth corner marker)")
    curved = [r for r in rows if r["valid"] and r["curvature"] != "" and abs(fnum(r["curvature"])) >= CORNER_CURV]
    print("\nAmong the %d valid frames, %d (%.0f%%) have a clearly curved centre line (|curvature| >= %g). "
          "Frames where the detector fails have no curvature, so this cannot give a corner failure rate."
          % (sum(r["valid"] for r in rows), len(curved), pct(len(curved), sum(r["valid"] for r in rows)), CORNER_CURV))

    # --- stability ---------------------------------------------------------------------------
    by_s = {}
    for r in rows:
        by_s.setdefault(r["session"], []).append(r)
    d_off, d_hd = [], []
    for g in by_s.values():
        g.sort(key=lambda r: r["timestamp"])
        for a, b in zip(g, g[1:]):
            if a["valid"] and b["valid"] and 0 < b["timestamp"] - a["timestamp"] < 0.3:
                d_off.append(abs(fnum(b["offset_norm"]) - fnum(a["offset_norm"])))
                d_hd.append(abs(fnum(b["heading_slope"]) - fnum(a["heading_slope"])))
    if d_off:
        print("\nFrame-to-frame change between consecutive valid frames (~8 Hz recording): "
              "|d offset_norm| median %.3f / p95 %.3f; |d heading_slope| median %.3f / p95 %.3f"
              % (np.median(d_off), np.percentile(d_off, 95), np.median(d_hd), np.percentile(d_hd, 95)))

    # --- relation to the human's steering -----------------------------------------------------
    ok = [r for r in rows if r["valid"]]
    if len(ok) > 10:
        st = np.array([r["steering"] for r in ok])
        for k in ("offset_norm", "heading_slope", "curvature"):
            v = np.array([fnum(r[k]) for r in ok])
            m = np.isfinite(v)
            if m.sum() > 10:
                print("corr(%s, recorded steering) over %d valid frames: %+.2f" % (k, m.sum(), np.corrcoef(v[m], st[m])[0, 1]))
        print("(Car steering: negative = left, positive = right. Features use the sim's sign: +offset = car left of "
              "centre, +heading = car points left of the track. Steering right to recover from either gives a "
              "POSITIVE correlation, but corners also drive it, so this is only a sanity check of the signs.)")
        off = np.array([r["offset_norm"] for r in ok], float)
        print("offset_norm over valid frames: mean %+.2f, p5 %+.2f, p95 %+.2f (sim: 0 = centred, +-1 = lane edge)"
              % (off.mean(), np.percentile(off, 5), np.percentile(off, 95)))

    # --- threshold sensitivity ---------------------------------------------------------------
    if not args.no_sweep:
        print("\nSensitivity of the ok-rate to the top-hat threshold (all frames):")
        imgs = [t[1] for t in items]
        for th in (8, 12, 16, 24):
            c = LaneConfig(tophat_min=th)
            n_ok = sum(1 for im in imgs if detect_lane(im, c).valid)
            print("  tophat_min=%2d: %.1f%% ok" % (th, pct(n_ok, len(imgs))))

    if not args.no_debug:
        save_debug(items, args.out, cfg)
        print("\nannotated examples in", os.path.join(args.out, "debug"))


if __name__ == "__main__":
    main()
