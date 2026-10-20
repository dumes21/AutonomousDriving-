# Orange-lane detector: what it measures and how reliable it is

Code: `car/lane_detection.py` (detector, CLI), `train/analyse_lane_data.py` (offline analysis of `data/2*/`).
Outputs go to `logs/lane/` (git-ignored): `lane_features.csv`, `summary_by_session.csv`, `debug/<kind>/*.jpg`.

## The data
Four sessions, 849 frames (labels.csv; 756 also in labels_clean.csv). All 320x240 colour JPEGs, ~8 Hz, with
`filename, steering, throttle, timestamp` (+ `segment` in three sessions). No camera calibration or ground-truth pose
exists. The camera is a wide-angle (fisheye-like) lens, low and forward-facing; the bonnet shows in the lower corners.

## Method (short)
Lab colour space, not HSV: under the pink lighting the tape's hue/saturation are washed out, but its b* (yellow-blue)
is clearly higher than the floor's. The tape's b* also fades toward the image edges, so a fixed threshold lost the right-hand
line; the detector uses a white top-hat of b* (narrow bright structures) plus a low absolute floor. ROI = below row 75,
above the bonnet. Per sampled row it needs exactly two un-clipped orange runs; the lane centre is their midpoint and a
line/quadratic is fitted. Everything is tunable in `LaneConfig` (and `--tophat-min` on the CLI).

## Results (all recorded frames, default settings)

| session | frames | ok % | one line | ambiguous | insufficient |
|---|---|---|---|---|---|
| 20261013_001444 | 87 | 98.9 | 0 | 0 | 1.1 |
| 20261013_003714 | 465 | 93.3 | 0 | 1.9 | 4.7 |
| 20261013_005718 | 105 | 28.6 | 35.2 | 5.7 | 30.5 |
| 20261013_010649 | 192 | 78.1 | 16.7 | 2.6 | 2.6 |
| **all** | 849 | **82.4** | 8.1 | 2.4 | 7.1 |

* "ok" = two boundaries seen on enough rows over a long enough span, with a consistent fit. Every other status leaves the
  features empty (no guess from one line: its pixel distance to the lane centre depends on pose and distance, uncalibrated).
* **Corners.** When the human was steering hard (|steering| >= 0.4, a rough corner proxy) ok drops from 89.7 % to 73.6 %;
  most of the loss is "insufficient" (13.6 %) and "one line" (10.2 %). Inspecting the images: in a corner the inside line
  runs off the edge of the image, or touches the border and is discarded as clipped, so the detector has nothing to measure.
  The orange *pixels* are found well; what fails is having both boundaries in view.
* Session 005718 is the weakest (29 % ok): the footage is dominated by tight turns with one line out of frame.
* Threshold sensitivity: top-hat 8/12/16 gives 81/82/81 % ok, so it is not tuned to a knife edge, but 24 gives 0 %
  (the tape contrast in this lighting is only ~15-25 units): a change in lighting could break it entirely.
* Frames where the detector failed have no curvature, so a true "corner failure rate" cannot be computed without
  hand-labelled corner frames. The steering-based figure above is a proxy only.
* Speed: ~5 ms/frame on a laptop. Not measured on the Jetson Nano.

Not established: accuracy. There is no ground truth offset or heading, so we cannot say how *right* the numbers are,
only that they are stable (median frame-to-frame change 0.02 in offset_norm) and have sensible signs (they correlate
+0.58 / +0.48 with the human's recovery steering, corners included). Lighting, other tracks, other lap directions
and people/objects near the lines are untested. **This is a prototype that works on four recordings; it is not
evidence the car can drive from it.**

## Features vs JetRacerEnv's observation

| JetRacerEnv obs | definition (metres/radians, exact) | detector feature | what it really is |
|---|---|---|---|
| `lane_offset` | signed distance to the centre line / 0.20 m half-width, + = car LEFT | `offset_norm` | (lane-centre x - image centre x) / (half lane width in px), at row 140. Same sign. Ratio of pixels at one row. Assumes camera on the car's centre line, looking along it, no lens distortion. Yaw and track curvature also shift it. **Not** equal to metres/0.20. |
| `heading_error` / pi | car heading - track direction, + = pointing LEFT | `heading_slope` | slope of the lane centre line in the image (px per px, + = centre drifts right going forward). Same sign. Mixes heading, offset and curvature (corner frames give ~-0.8 vs ~-0.1 on straights). It is **not** an angle, so `/pi` is meaningless for it. |
| `last_steering` (+1 left, 0, -1 right) | the previous action | not from the image | known from our own command; remember the car's sign is the opposite (negative = left) and straight is about +0.35. |

Offsets: valid frames have offset_norm between about -0.4 and +0.5 (5th-95th percentile), while the sim's
`lane_offset` is +-1 at the lane edge. That is consistent with the car staying mid-lane, but the scales are not
known to match.

## Calibration needed before these can replace the sim's observations
1. Put the car at known lateral offsets (e.g. -0.1, 0, +0.1 m) and headings (e.g. -20, -10, 0, +10, +20 deg) on a straight,
   record frames, and fit `offset_norm`/`heading_slope` against truth. Or, better, fit a ground-plane homography (known tape
   spacing) and compute offset in metres and heading in radians from the undistorted line geometry.
2. Measure the real lane width (distance between tape lines) and compare with the sim's 0.40 m.
3. Decide what the policy does when the detector says "not ok" (hold last steering, slow down, steer toward the visible
   line). The sim never has missing observations; this has to be added to training (observation dropout).
4. Check on-car latency and Jetson run-time, and test under different lighting.
