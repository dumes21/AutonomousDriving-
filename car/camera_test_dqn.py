"""Offline CALIBRATED camera -> DQN test: run the trained policy on recorded JetRacer frames. No motors, no camera access.

    python car/camera_test_dqn.py --calibration CAL.json frame1.jpg frame2.jpg ...
    python car/camera_test_dqn.py --calibration CAL.json data/2026*/*.jpg --out results/camera_dqn_calibrated [--categories cats.csv] [--sequence]

Observation = perception/state_estimator.py (StateEstimator + to_dqn_observation), the SAME pipeline car/drive_dqn.py uses:
    [lane_offset_m / 0.13, heading_error_rad / pi, last_steering]   last_steering = 1 - previous action (simulator convention)
with offset + = car LEFT of the centre line and heading + = car pointing LEFT of the track (the simulator's signs).
A --calibration file is REQUIRED and must be complete (perception/calibrate.py; docs/CALIBRATION_DAY.md). There is no fallback
to the old image-space mapping (offset_norm, heading_slope / pi): that was not a physical heading and has been removed.
Even with a complete calibration the numbers are UNVERIFIED until perception/validate_calibration.py has passed on known poses.

A frame is REJECTED (no observation, no action) when: the detector does not see both boundaries or its confidence is below
--min-confidence; the estimator reports INVALID_GEOMETRY / UNCALIBRATED; or the estimate falls outside the simulator's range
(offset beyond the lane half-width or heading beyond pi, which to_dqn_observation clips). Rejections reset previous steering to 0,
as in the pipeline. previous_steering is 0.0 for every image (unrelated stills); --sequence carries it between consecutive frames of a
genuine driving sequence.

--out DIR writes predictions.csv, summary.json and an annotated image per frame (refuses to overwrite; --no-annotated skips the images).
--categories CSV (columns image,category) adds a category column and a per-category breakdown.
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for sub in ("car", "perception", "inference"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from dqn_policy import ACTION_LABELS, load_policy  # noqa: E402
from ground_plane import CalibrationError, CameraCalibration  # noqa: E402
from lane_detection import LaneConfig, annotate, detect_lane  # noqa: E402
import state_estimator as se  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "experiments", "dqn", "20261020_151620_footprint_v1", "final.zip")
SIM_START_HEADING_NORM = 0.2 / np.pi      # the policy only ever started within +-0.2 rad (0.064 normalised)

ACTIONS = ACTION_LABELS                    # 0 LEFT, 1 STRAIGHT, 2 RIGHT (same table the drive_dqn pipeline uses)

COLUMNS = ["image", "category", "status", "detector_status", "accepted", "confidence", "est_offset_m", "est_heading_rad",
           "obs_offset", "obs_heading", "obs_prev_steering", "heading_obs_clipped", "action", "action_name",
           "rows_used", "curve_model", "extrapolation_m", "reject_reason", "warnings",
           "diag_image_offset_norm", "diag_image_heading_slope"]   # the last two are the old image-space values: DIAGNOSTIC ONLY


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+")
    p.add_argument("--calibration", required=True, help="camera calibration JSON from perception/calibrate.py (required, must be complete)")
    p.add_argument("--out", help="folder for predictions.csv, summary.json and annotated images (must not exist yet)")
    p.add_argument("--categories", help="CSV with columns image,category (image = path as given)")
    p.add_argument("--no-annotated", action="store_true", help="with --out: skip the annotated images (large image sets)")
    p.add_argument("--sequence", action="store_true", help="chronological frames: carry previous steering between frames")
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--model", default=MODEL_PATH, help="SB3 .zip (default footprint_v1) or the exported .npz")
    return p.parse_args()


def load_calibration(path):
    """Load and require a COMPLETE calibration. Exits with an explanation otherwise (never a silent fallback)."""
    if not os.path.exists(path):
        sys.exit("calibration file not found: %s\nA real calibration is required (docs/CALIBRATION_DAY.md)." % path)
    try:
        calib = CameraCalibration.load(path)
    except (CalibrationError, KeyError, ValueError) as e:
        sys.exit("calibration file %s is not valid: %s" % (path, e))
    missing = calib.missing_for_state("measured_width")
    if missing:
        sys.exit("calibration %s is INCOMPLETE, missing: %s\nRefusing to run: there is no fallback to the image-space mapping."
                 % (path, ", ".join(missing)))
    return calib


def observe(estimator, policy, image):
    """One frame -> dict of results, in the same order of checks as inference/pipeline.py CameraToActionPipeline.step():
    estimator status, observation vector, out-of-range clip, then the policy. Any rejection resets the policy's previous steering."""
    r = {"status": "", "accepted": False, "est_offset_m": None, "est_heading_rad": None, "obs": None, "heading_obs_clipped": None,
         "action": None, "reject_reason": "", "warnings": "", "curve_model": "", "extrapolation_m": None, "rows_used": 0,
         "confidence": 0.0, "detector_status": ""}
    try:
        est = estimator.estimate(image)
    except Exception as e:                                       # noqa: B902 - any failure rejects the frame
        policy.reset()
        r.update(status="ERROR", reject_reason="state estimation failed: %r" % (e,))
        return r
    det = est.detection
    r.update(confidence=est.confidence, detector_status=det.status if det is not None else "",
             rows_used=len(det.rows) if det is not None and det.valid else 0, est_offset_m=est.offset_m,
             est_heading_rad=est.heading_rad, curve_model=est.curve_model_used or "", extrapolation_m=est.extrapolation_m,
             warnings="; ".join(est.warnings))
    if est.status != se.OK:
        why = "; ".join(est.warnings[:2]) or "no valid state"
        policy.reset()
        r.update(status=est.status, reject_reason=why if why.startswith(est.status) else "%s: %s" % (est.status, why))
        return r
    obs = se.to_dqn_observation(est, policy.last_steering)
    if obs.vector is None:
        policy.reset()
        r.update(status="NO_OBSERVATION", reject_reason="no observation: " + "; ".join(obs.warnings))
        return r
    r["heading_obs_clipped"] = bool(obs.clipped)
    r["warnings"] = "; ".join(est.warnings + obs.warnings)
    if obs.clipped:
        policy.reset()
        r.update(status="OUT_OF_RANGE", reject_reason="estimate outside the simulator's range: " + "; ".join(obs.warnings))
        return r
    action = policy.act(obs.vector)
    r.update(status="OK", accepted=True, obs=obs.vector, action=action.index)
    return r


def annotate_with_action(image, row):
    out = annotate(image, detect_lane(image), LaneConfig())     # existing overlay (its on-image text is the old image-space diagnostic)
    out = cv2.resize(out, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    pad = np.zeros((72, out.shape[1], 3), np.uint8)
    if row["accepted"]:
        lines = ["DQN: %s   conf %.2f   status %s" % (row["action_name"], row["confidence"], row["status"]),
                 "est offset %+.1f cm  heading %+.1f deg  (sim signs: + = left)" % (100 * row["est_offset_m"], np.degrees(row["est_heading_rad"])),
                 "obs [%+.2f, %+.3f, %+.0f]   category: %s" % (row["obs_offset"], row["obs_heading"], row["obs_prev_steering"],
                                                               row["category"] or "-")]
        colour = (0, 255, 255)
    else:
        lines = ["REJECTED: %s" % row["reject_reason"], "no observation, no action", "category: %s" % (row["category"] or "-")]
        colour = (0, 0, 255)
    for i, t in enumerate(lines):
        cv2.putText(pad, t[:95], (6, 18 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return np.vstack([out, pad])


def summarise(rows, model_path, sequence, calib_path, calib):
    n = len(rows)
    acc = [r for r in rows if r["accepted"]]

    def stats(vals):
        v = np.array(vals, float)
        return None if not len(v) else {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}

    def breakdown(rs):
        a = [r for r in rs if r["accepted"]]
        c = {k: sum(1 for r in a if r["action_name"] == k) for k in ACTIONS.values()}
        return {"images": len(rs), "accepted": len(a), "action_counts": c}

    status = {}
    for r in rows:
        status[r["status"]] = status.get(r["status"], 0) + 1
    cats = sorted(set(r["category"] for r in rows if r["category"]))
    assumed = calib.assumed_parameters()
    return {
        "model": os.path.relpath(model_path, ROOT) if model_path.startswith(ROOT) else model_path,
        "calibration_file": calib_path,
        "calibration_parameter_status": dict(calib.status),
        "calibration_assumed_parameters": assumed,
        "observation_note": ("calibrated state estimator (perception/state_estimator.py); UNVERIFIED until "
                             "perception/validate_calibration.py passes on known poses"
                             + ("; CONTAINS ASSUMED PARAMETERS: " + ", ".join(assumed) if assumed else "")),
        "previous_steering": "carried over (--sequence), reset to 0 after any rejection" if sequence else "0.0 for every image",
        "images": n, "accepted_for_dqn": len(acc), "accepted_pct": 100.0 * len(acc) / max(1, n),
        "status_counts": status,
        "confidence_accepted": stats([r["confidence"] for r in acc]),
        "est_offset_m_accepted": stats([r["est_offset_m"] for r in acc]),
        "est_heading_rad_accepted": stats([r["est_heading_rad"] for r in acc]),
        "obs_offset_accepted": stats([r["obs_offset"] for r in acc]),
        "obs_heading_accepted": stats([r["obs_heading"] for r in acc]),
        "pct_accepted_heading_beyond_training_start_range": (
            100.0 * sum(1 for r in acc if abs(r["obs_heading"]) > SIM_START_HEADING_NORM) / max(1, len(acc))),
        "training_start_heading_norm_limit": float(SIM_START_HEADING_NORM),
        "action_counts": {k: sum(1 for r in acc if r["action_name"] == k) for k in ACTIONS.values()},
        "by_category": {c: breakdown([r for r in rows if r["category"] == c]) for c in cats},
    }


def main():
    args = parse_args()
    calib = load_calibration(args.calibration)
    if args.out:
        if os.path.exists(os.path.join(args.out, "predictions.csv")):
            sys.exit("%s already contains results; choose a new --out (nothing is overwritten)" % args.out)
        os.makedirs(os.path.join(args.out, "annotated") if not args.no_annotated else args.out, exist_ok=True)
    categories = {}
    if args.categories:
        with open(args.categories) as f:
            categories = {r["image"]: r["category"] for r in csv.DictReader(f)}

    policy = load_policy(args.model)                              # same DQNPolicy / numpy Q-network class drive_dqn.py uses
    estimator = se.StateEstimator(calib, min_confidence=args.min_confidence)
    print("model:", args.model)
    print("calibration:", args.calibration)
    if calib.assumed_parameters():
        print("WARNING: calibration contains ASSUMED (unmeasured) parameters: %s" % ", ".join(calib.assumed_parameters()))
    print("NOTE: estimates are UNVERIFIED until perception/validate_calibration.py has passed on known poses.")

    rows = []
    for image_path in args.images:
        image = cv2.imread(image_path)

        if image is None:
            print("Cannot read:", image_path)
            continue

        if not args.sequence:
            policy.reset()                                        # unrelated stills: previous steering 0.0

        r = observe(estimator, policy, image)
        row = {"image": image_path, "category": categories.get(image_path, ""), "status": r["status"],
               "detector_status": r["detector_status"], "accepted": r["accepted"], "confidence": round(r["confidence"], 3),
               "est_offset_m": r["est_offset_m"], "est_heading_rad": r["est_heading_rad"],
               "obs_offset": None, "obs_heading": None, "obs_prev_steering": None,
               "heading_obs_clipped": r["heading_obs_clipped"], "action": r["action"],
               "action_name": ACTIONS[r["action"]] if r["action"] is not None else "", "rows_used": r["rows_used"],
               "curve_model": r["curve_model"], "extrapolation_m": r["extrapolation_m"], "reject_reason": r["reject_reason"],
               "warnings": r["warnings"], "diag_image_offset_norm": None, "diag_image_heading_slope": None}
        raw = detect_lane(image)                                  # old image-space values, logged as diagnostics only
        row["diag_image_offset_norm"], row["diag_image_heading_slope"] = raw.offset_norm, raw.heading_slope
        if r["accepted"]:
            row.update({"obs_offset": float(r["obs"][0]), "obs_heading": float(r["obs"][1]), "obs_prev_steering": float(r["obs"][2])})
            print(image_path, "offset: %+.3f m" % r["est_offset_m"], "heading: %+.3f rad" % r["est_heading_rad"],
                  "conf:", round(r["confidence"], 2), "observation:", r["obs"], "action:", row["action_name"])
        else:
            print(image_path, "REJECTED:", r["reject_reason"])
        rows.append(row)
        if args.out and not args.no_annotated:
            name = "%s_%s" % (os.path.basename(os.path.dirname(os.path.abspath(image_path))), os.path.basename(image_path))
            cv2.imwrite(os.path.join(args.out, "annotated", name), annotate_with_action(image, row))

    if args.out:
        with open(os.path.join(args.out, "predictions.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if r[k] is None else (round(r[k], 4) if isinstance(r[k], float) else r[k])) for k in COLUMNS})
        summary = summarise(rows, args.model, args.sequence, args.calibration, calib)
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print("\nsaved predictions.csv, summary.json%s in" % ("" if args.no_annotated else " and annotated/"), args.out)
        print(json.dumps({k: summary[k] for k in ("images", "accepted_pct", "action_counts", "status_counts")}))


if __name__ == "__main__":
    main()
