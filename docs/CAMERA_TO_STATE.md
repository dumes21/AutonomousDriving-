# Camera -> DQN state estimation (Step 2)

Code: `perception/state_estimator.py` (estimator + DQN normalisation + debug images), `perception/ground_plane.py`
(camera model, calibration object, fitting helpers), `perception/calibrate.py` (build a calibration file step by step),
`perception/calibration/initial_fit_from_recordings.json` (what the recordings can calibrate), `tests/test_state_estimator.py`.
Reuses the Lab detector in `perception/boundary_detector.py`. Nothing here touches the car, the simulator, training or the checkpoint.

## 1. What the simulator defines (read from `sim/track_env.py`, not assumed)
| quantity | definition |
|---|---|
| reference point | the car's (x, y) is the **rear-axle centre** (the bicycle model moves that point). Body is drawn/checked around it. |
| `lane_offset` | signed perpendicular distance from the rear axle to the track centre line, metres: `-dx*sin(t) + dy*cos(t)` with (dx, dy) = car minus nearest centre-line waypoint, t = track direction there. **+ = car is LEFT of the centre line** (the track is counter-clockwise, so left = inside). |
| `heading_error` | `car heading - track tangent` wrapped to [-pi, pi). **+ = car points LEFT of the track direction.** |
| `last_steering` | `float(1 - action)`: **+1 = left (action 0), 0 = straight, -1 = right**. The physical car's sign is the opposite (negative = left) and its straight is ~+0.35; converting is not done in this module. |
| observation | `[offset / 0.13, heading / pi, last_steering]`, each of the first two clipped to [-1, 1] (float32) |

## 2. What is known about the camera
* Frames: 320x240 colour JPEG (`jetcam CSICamera`, capture 1640x1232 scaled down; `car/camera_test.py`, `car/drive_model.py`).
* Not recorded anywhere in the repo: mounting height, pitch, lateral/forward position, focal length / FOV, lens distortion (the image is
  visibly wide-angle). Nothing was invented.
* From the recordings plus the 0.26 m lane width (`calibrate.py fit-width`, 651 valid two-line frames): effective camera height
  **k = 0.124 m** and horizon row **45** *under a flat-floor, no-distortion pinhole model*. Caveats found while fitting: the width-vs-row
  relation is not linear (median width 112, 157, 192, 226, 244, 258 px at rows ~107, 122, 137, 152, 167, 182: it saturates near the
  camera); near rows are only used when both lines are un-clipped (selection bias); per-row width scatters +-15 % between frames
  (yaw, bends, pitch); per-session fits differ (horizon 43-54, k 0.117-0.128). So these two numbers are **rough, unverified**.

**Are the existing frames + 0.26 m enough for a ground-plane calibration? No, not for offset and heading.** They give a per-row
ruler (lateral metres per pixel), but they cannot give: (a) the focal/depth scale F (needs one known *longitudinal* distance), (b) the
vehicle's forward axis in the image (needs a known pose), (c) the camera-to-rear-axle distance, (d) lens distortion. They are enough
for one diagnostic: the lane centre lies a median +0.4 cm (5-95 %: -5.4 .. +6.3 cm) from the image-centre column over all valid frames
(metres via the per-row lane width). That is **not** the vehicle offset.

## 3. What is implemented
1. `StateEstimator.estimate(bgr)` -> `StateEstimate` with `status` OK / UNCALIBRATED / INVALID_DETECTION / INVALID_GEOMETRY, `valid`,
   `calibrated`, `confidence`, `offset_m`, `heading_rad` (simulator conventions above), warnings, the debug diagnostics.
2. Geometry: each row's left/right boundary column -> lateral metres (scale = 0.26 m / measured lane width in px at that row, or the fitted
   model) relative to the **calibrated vehicle-axis line**, depth `F/(row - horizon)` plus the camera-to-rear-axle distance -> centre-line points in
   the vehicle frame (origin rear axle, x forward, y left) -> line (or parabola in bends) -> foot of the perpendicular from the rear axle: offset =
   signed distance, heading error = minus the tangent angle there.
3. Missing or unset calibration is never defaulted: with any required parameter unset the status is UNCALIBRATED and offset/heading are `None`.
   The image midpoint is never used as the vehicle centre. Parameters can carry the status `assumed`, which is reported on every estimate.
4. `to_dqn_observation(estimate, last_steering)` -> `[offset/0.13, heading/pi, last_steering]` float32, clipped, with warnings for clipping and for
   headings beyond the 0.2 rad the policy saw at episode starts. An invalid estimate gives `vector=None`, never zeros. `last_steering` must be given in the
   simulator convention.

## 4. Known limitations (measured on a SYNTHETIC camera, not the real one)
Only ~0.12 m of lane is visible, starting ~0.2 m ahead of the rear axle, so offset/heading at the car are extrapolated backwards:
| case (0.5 px column noise) | offset error | heading error |
|---|---|---|
| straight, linear fit | 1.4 mm rms | 0.005 rad rms |
| straight, parabola | 10 mm rms | 0.076 rad rms (noise amplified: not used on straights) |
| R = 0.72 m bend, linear fit | ~6 cm | **-0.42 rad bias** (lane direction 0.3 m ahead differs from the direction at the car) |
| R = 0.72 m bend, parabola | ~1.3 cm | **+0.11 rad bias** (even with perfect pixels) |
`curve_model='auto'` (default) switches to the parabola when the fitted curvature exceeds 1/m. Heading in bends is therefore unreliable
whatever the calibration; the unit tests pin this behaviour. A what-if run with *guessed* parameters (statuses 'assumed') moved the heading 5-95 %
range from -0.98..+0.54 to -0.42..+0.50 rad and shifted the offset median by up to 13 cm when the guessed focal length and camera position changed:
the outputs are meaningless until F and the camera position are measured. Perception noise is also not modelled in the simulator-trained policy.

## 5. Minimum calibration (when you have the car)
Mount the camera rigidly first and do not touch it afterwards. All files are written with `python perception/calibrate.py ...`
(each command writes a NEW file); start from `perception/calibration/initial_fit_from_recordings.json`.
1. **Ruler**: distance from the rear-axle centre forward to the camera lens -> `camera_forward_of_rear_axle_m` (`calibrate.py set ... --status measured`).
   Also note the camera's lateral position relative to the car centre line (should be ~0) and its height above the floor (cross-check for k = 0.124 m).
2. **Depth scale F + horizon**: on the straight, lay 3 strips of tape across the lane at 0.30, 0.45, 0.60 m (measure from the lens, along the floor) in front of the
   car; take one photo; read the image row of each strip (its centre); `calibrate.py fit-depth cal.json --rows R1 R2 R3 --distances 0.30 0.45 0.60 --out cal2.json`.
   This is the only way to get F from a known longitudinal distance. (Do it twice with a second car position to check.)
3. **Vehicle axis in the image**: put the car centred in the lane (rear AND front axle on the centre line, measure with a ruler) and parallel to it,
   photograph on a straight 3-5 times; `calibrate.py fit-axis cal2.json frames... --out cal3.json`. This gives the column of the car's forward axis at every row,
   replacing any "image centre = vehicle centre" assumption (it absorbs camera lateral offset and yaw).
4. **Sign check** (also the validation set, ~10 images on the straight): offsets of -5, -2.5, +2.5, +5 cm at zero yaw, then yaw of -10, -5, +5, +10 degrees at zero offset
   (protractor/marked lines). Run `python perception/state_estimator.py FRAMES --calibration cal3.json`. Expected: car moved LEFT -> offset **positive**;
   car rotated LEFT (counter-clockwise from above) -> heading **positive**. Report error per pose; fit a scale/bias if systematic.
5. If the error near the image edges exceeds ~1 cm / 2 degrees, add a lens-distortion calibration (checkerboard, 10-15 views) and undistort before detection.
6. Bends: take 3-4 images on a bend with the car centred (and +-3 cm) to measure the bend bias in section 4 on the real camera.

## 6. Ready for DQN inference?
**No.** The pipeline and its sign conventions are implemented and unit-tested on a synthetic camera, and the detector works on the real frames, but
(1) the real camera is UNCALIBRATED (F, vehicle axis and camera position unset, so every real frame is reported UNCALIBRATED), (2) nothing has been validated
against measured poses, (3) heading in bends has a structural bias of ~0.1 rad even with a perfect calibration, (4) the policy has only ever seen
noise-free observations (and is very sensitive to disturbance: ~0.8 cm body clearance), (5) 17 % of frames have no valid two-line detection and need the safe-stop gate.
