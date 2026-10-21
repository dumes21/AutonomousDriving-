# Two-line-only perception pipeline (interim design)

Decision: the policy only ever sees a lane measurement when **both** orange boundaries are confidently detected.
No single-line recovery, no filling gaps. Code: `car/lane_detection.py` (unchanged detector),
`car/lane_observation.py` (observation + safe-stop gate), `tests/test_lane_observation.py`, `train/replay_gate.py`.
Nothing here touches motors, the simulator, or the recordings.

## Valid vs invalid
`observation_from_result()` returns a valid observation only if the detector status is `ok` (two un-clipped boundaries
on >= 3 rows over >= 30 px, consistent fit) **and** `confidence >= 0.5` (gate setting; the detector alone accepts 0.35).
Otherwise it is explicitly invalid: `valid=False`, a `status`/`reason` (`one_line`, `ambiguous`, `insufficient`,
`no_line`, `low_confidence`, `stale`), and `offset_norm`/`heading_slope` are `None`. `obs.vector()` on an invalid
observation raises `InvalidObservation`; there is no zero, no previous value and no single-line estimate to fall back on.
The confidence number is a heuristic (usable rows x fit quality), not a calibrated probability.

## Safe-stop rule (for the physical car)
The caller (the future driving script) owns the motors; the gate only decides. `SafeStopGate.update()` / `.check_stale()`
return `drive_allowed`. When it is False the caller must set **throttle 0** and must not run the policy.

| situation | result |
|---|---|
| gate just created / after any latch | **stopped**; driving needs `rearm()` |
| one invalid frame | throttle 0 for that frame; resumes on the next valid frame |
| invalid for >= 0.3 s, or 3 invalid frames in a row | **latched stop** |
| >= 50 % of frames invalid in the last 1 s (>= 4 frames) | **latched stop** (flicker rule) |
| no new frame (camera stalled), or none for 0.25 s | invalid; counts toward a latch; the old measurement is never reused |
| latched | stays stopped even when valid frames return; `rearm()` is an operator action and works only after 5 consecutive valid frames |

Throttle 0 is the stop: steering is left untouched (the car coasts to a halt), it does not command a new steering value.
Combine this with the existing safeguards in `car/drive_model.py` (hold-to-drive deadman, SIGTERM handler, throttle cap).
The gate has not been connected to any driving script and has not been run on the car.

## How often would it stop? (replay of the 4 recordings, human-driven, ~8 Hz)
`python train/replay_gate.py`, best case with instant re-arm:

| session | frames | drive allowed | latches | longest stop |
|---|---|---|---|---|
| 20261013_001444 | 87 | 88.5 % | 0 | 5 frames |
| 20261013_003714 | 465 | 77.4 % | 6 | 29 |
| 20261013_005718 | 105 | 11.4 % | 3 | 49 |
| 20261013_010649 | 192 | 63.5 % | 3 | 32 |
| **all** | 849 | **67.3 %** | **12** | |

So a two-line-only car would stop about every 8 s of recorded driving, almost always in corners, and in tight turns
(session 005718) it would barely move. The confidence gate of 0.5 costs ~6 points vs. the detector's own 0.35 (82.4 % -> 76.8 % valid frames).
Two practical levers, neither implemented: (a) a wider/higher camera view so both lines stay in frame through bends,
(b) accepting one-line recovery later. Also note these are human trajectories; a policy may behave differently.

## How the features differ from the simulator's observation
| simulator (`JetRacerEnv`) | OpenCV feature |
|---|---|
| `lane_offset` = exact signed distance of the car from the centre line / half-width (m / m) | `offset_norm` = (lane-centre x - image-centre x) / (half lane width in px) at image row 140. Sign matches (+ = car left). Pixel ratio, not metres; measured at a look-ahead row, so yaw leaks in (~depth x tan yaw); assumes camera on the car's centreline, flat floor, no lens distortion. Lane-width pixels differ by session (182-208 px at row 140), suggesting the camera pose changes between mountings. |
| `heading_error / pi` = exact angle between car heading and track direction | `heading_slope` = image-plane slope of the lane centre (px per px; + = centre drifts right going forward, matching + = car points left). Not an angle; mixes heading, offset and curvature (about -0.1 on straights, -0.8 in corners). `/pi` is meaningless for it. |
| `last_steering` = +1 left / 0 / -1 right | not perceived; known from our own command, with the car's opposite sign (negative = left) and straight about +0.35. |
| always available | available only on valid frames (see above) |

The simulator's lane in the working tree is 0.26 m wide (`lane_half_width 0.13`); `offset_norm x 0.13 m` is the
first-order metric offset under the flat-ground model.

## Minimum calibration to align them
1. **Rigidly fix the camera** and measure its height above the floor with a ruler (the width-vs-row fit implies ~0.125 m if
   the lane is 0.26 m). Re-do the checks below whenever the mount is touched.
2. **Confirm what 0.26 m is**: tape centre to tape centre (what the detector measures) or inner edge to inner edge, and the tape width.
3. **Focal length / depth scale** (needed only for heading in radians and true look-ahead distance): one photo of the straight
   section with two or three cross-strips of tape at known spacing (e.g. 0.30 / 0.60 / 0.90 m ahead), car centred and aligned.
4. **Validation images with measured poses** (straight section): lateral offsets -10, -5, +5, +10 cm at zero yaw; yaw -20, -10, +10, +20 deg at zero
   offset. Compare the computed offset/heading with the ruler/protractor values; fit a scale/bias if needed. If errors near the image edges
   exceed ~1 cm, take 10-15 checkerboard views for lens-distortion calibration.
5. Reference-point offset between the camera and the simulator's car reference point (measure with a ruler).
About 10-12 images in total, plus the measurements above.

Not decided here: the simulator's observation changes (validity flag, noise) and what the DQN does with it. The simulator is untouched.
