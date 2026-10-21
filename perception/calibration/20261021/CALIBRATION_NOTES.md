# Calibration 2026-10-21: record (NOT VALIDATED)

Status: **NOT VALIDATED. It FAILS the repo's 1.0 cm offset guideline on the two known-pose photos taken so far** (see 'Validation so far').
Fitted from one marker photo and one centred photo. Do not use for driving until `perception/validate_calibration.py` passes on a
complete set of separate known-pose photos (docs/CALIBRATION_DAY.md step 3c) and the remaining caveats below are resolved.

## Inputs
| item | value | source |
|---|---|---|
| lane width | 0.26 m | user measurement |
| camera-to-rear-axle (horizontal) | 0.16 m | user measurement, status `measured` |
| markers | 3 yellow strips across the lane at 0.40 / 0.50 / 0.60 m **ahead of the rear axle** | user placement; which part of the strip was measured (centre / near edge) is not recorded: centre assumed |
| `data/calibration_20261021/markers_40_50_60.jpg` | 320x240 | copied from ~/Desktop (original untouched) |
| `data/calibration_20261021/centred.jpg` | 320x240 | copied from ~/Desktop; the user's name was `centred_reference.jpg`, this file is ASSUMED to be it |
| `data/calibration_20261021/marker40.jpg` | 320x240 | single-marker test shot, not used |

## Procedure (repo tools, in order)
1. `calibrate.py set` camera_forward_of_rear_axle_m = 0.16 (measured) -> `step1_camera_forward.json`
2. Marker rows read from the image (yellow hue 24-45, S>=55, V>=100, connected components): far 122.2, middle 140.5, near 166.0.
   The repo model takes distances **from the lens**, so 0.60/0.50/0.40 m ahead of the rear axle became 0.44/0.34/0.24 m (minus 0.16).
   `calibrate.py fit-depth --rows 122.2 140.5 166.0 --distances 0.44 0.34 0.24` -> F = 22.96, horizon row = 71.1, residual rms 0.8 cm (3 points, 1 dof) -> `step2_depth.json`
3. `calibrate.py fit-axis centred.jpg` -> axis column 174.9 px at row 140, slope 0.0057 px/row, residual 0.5 px, from 1 frame -> `CAL.json` (identical to `step3_axis.json`)

## Parameter status in CAL.json
horizon_row_px 71.09 fitted | depth_scale_F 22.96 fitted | axis_x_ref_px 174.95 fitted | axis_slope 0.0057 fitted |
camera_forward_of_rear_axle_m 0.16 measured | camera_height_eff_m 0.124 fitted **from the older recordings, not this session; unused with the default `measured_width` scale**.
No parameter is `assumed`.

## Caveats found (see the chat report for numbers)
* The distance convention (ahead of the REAR AXLE, converted to the lens) is decisive: not converting gives F = 52, horizon 35.
* Axis from ONE photo: no placement repeatability. Lane centre sits 15 px (about 2.6 cm at row 140) right of the image centre.
* Depth model has no camera-pitch term. Rows 122-204 agree with an exact pitched-camera model within ~1.5 cm, but rows 100-116 (the detector's far rows)
  differ by 5-14 cm between the two models: the markers do not reach that far. A 4th marker at 0.80 m ahead of the rear axle would settle it.
* No lens distortion model. Near-marker row is uncertain by about +-2 px (the strip is dark in the middle); a strict colour threshold gives 170 instead of 166.

## Validation so far (2026-10-21; independent of the fit, CAL.json was NOT refitted)
Photos in `data/calibration_20261021/validation/` with `truth.csv`; run with `perception/validate_calibration.py`.
| photo | truth | estimate | error | verdict vs guideline (1.0 cm, 2 deg) |
|---|---|---|---|---|
| offset_left_2p5cm.jpg (conf 0.90) | +2.50 cm, 0 deg | +1.37 cm, +1.32 deg | -1.13 cm, +1.32 deg | offset FAIL, heading pass |
| offset_right_2p5cm.jpg (conf 0.57) | -2.50 cm, 0 deg | -1.36 cm, -0.63 deg | +1.14 cm, -0.63 deg | offset FAIL, heading pass |
Offset errors are equal and opposite (bias +0.01 cm): the estimates are about 0.55x the true offset in both directions, i.e. a scale error, not a
shifted zero. Row-by-row the lane's apparent lateral shift shrinks toward the camera (2.8 cm at 0.95 m ahead of the rear axle to 1.9 cm at 0.37 m),
which is consistent with lens distortion (not modelled) but is NOT proven. Only 2 of the documented validation poses exist (no +-5 cm, no yaw poses,
no bend, no re-placed repeats). Do not apply a correction factor from two photos. Next: +-5 cm poses, 2-3 re-placed centred photos, then a checkerboard
lens-distortion calibration if the scale error persists.
(The two generated validation CSVs and debug overlays were not committed: they contain absolute local paths / large images. Re-run the command to regenerate them.)
