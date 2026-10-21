# Calibration day checklist (camera only: no motors, no DQN)

Nothing below drives the car. The Jetson only runs the camera web page. All calibration maths runs on the LAPTOP.
Assumptions to change if wrong: repo on the Jetson is `~/seth-jetracer`, Jetson user is `jetson`, laptop repo is the folder below.

Laptop repo: `/Users/seth/reinforcement-learning/jetracer-at3-final/AutonomousDriving-`

## 0. Before leaving (laptop, 1 min)
```
cd /Users/seth/reinforcement-learning/jetracer-at3-final/AutonomousDriving-
venv/bin/python -m pytest -q tests/test_calibration_workflow.py     # expect: 6 passed
git status --short car perception tests docs                        # your changes are all uncommitted; commit/copy to the Jetson as you wish
```
Copy `car/camera_capture_web.py` (and `car/mjpeg_stream.py` if it is not there already) to the Jetson repo, e.g.
`scp car/camera_capture_web.py car/mjpeg_stream.py jetson@<car-ip>:~/seth-jetracer/car/`

Bring: ruler/tape measure, white tape, protractor or printed angle template, marker, a way to mark the floor point under the lens.

## 1. Jetson: start the camera page
```
ssh jetson@<car-ip>
cd ~/seth-jetracer
python3 --version                      # expect 3.6.x
python3 -c "import cv2, jetcam; print(cv2.__version__)"
ps aux | grep -E "teleop_record|drive_model|drive_dqn|camera_stream|camera_capture" | grep -v grep   # nothing should be listed
pkill -f camera_capture_web            # only if a stale copy is listed
hostname -I                            # the car's IP
python3 car/camera_capture_web.py --out data/calib
```
Laptop browser: `http://<car-ip>:8000/`. Type a label into the box, click Capture (or press space with the cursor outside the box).
The file is saved on the Jetson as `<label>_<time>.jpg` (320x240). Keep this window running for the whole session.
If it prints "could not open the camera": stop the other program; as a last resort `sudo systemctl restart nvargus-daemon`.

## 2. Measure with the ruler (write the numbers down)
* `R` = distance from the **rear-axle centre** forward to the **camera lens** (metres). Put it in step 5.
* Mark on the floor the point directly under the lens. All tape distances are measured **from that point, along the floor**, straight ahead.

## 3. Photos (label in the page, 2-3 photos each; re-place the car between repeats)
**a. Strips** (car on the straight, parallel to the lane, stationary): white tape across the lane, centre of each strip at
**0.30, 0.45, 0.60 m** from the lens floor point. Check on the live view that all 3 are visible. If the 0.30 m one is out of view, shift all three
by the same amount and use those distances in step 5. Label `strips` (3 photos).
**b. Centred** (remove the strips): rear AND front axle centres on the lane centre line (ruler), car parallel to the lane. Label `centred` (5 photos, re-place each time).
**c. Validation** (offset = rear-axle centre vs lane centre line; yaw about the rear-axle centre; + = LEFT / counter-clockwise from above):

| label | offset | yaw | label | offset | yaw |
|---|---|---|---|---|---|
| `C0` | 0 | 0 | `yL5` | 0 | +5 deg |
| `L25` | +2.5 cm (left) | 0 | `yR5` | 0 | -5 deg |
| `R25` | -2.5 cm (right) | 0 | `yL10` | 0 | +10 deg |
| `L50` | +5 cm | 0 | `yR10` | 0 | -10 deg |
| `R50` | -5 cm | 0 | `L25yL5` | +2.5 cm | +5 deg |
|  |  |  | `R25yR5` | -2.5 cm | -5 deg |

2 photos per label (re-place the car between them). Do NOT reuse the `centred` photos for validation.

## 4. Copy the photos to the laptop
```
cd /Users/seth/reinforcement-learning/jetracer-at3-final/AutonomousDriving-
scp -r jetson@<car-ip>:~/seth-jetracer/data/calib data/calib
ls data/calib | head -50
```
Then write `data/calib/truth.csv` (offset in cm, yaw in degrees, + = left):
```
cat > data/calib/truth.csv <<'EOF'
# image pattern,offset_cm,yaw_deg
C0_*.jpg,0,0
L25_*.jpg,+2.5,0
R25_*.jpg,-2.5,0
L50_*.jpg,+5,0
R50_*.jpg,-5,0
yL5_*.jpg,0,+5
yR5_*.jpg,0,-5
yL10_*.jpg,0,+10
yR10_*.jpg,0,-10
L25yL5_*.jpg,+2.5,+5
R25yR5_*.jpg,-2.5,-5
EOF
```

## 5. Build CAL.json (laptop; every command writes a NEW file, replace R and the file names)
```
cd /Users/seth/reinforcement-learning/jetracer-at3-final/AutonomousDriving-
D=perception/calibration/day1
venv/bin/python perception/calibrate.py set perception/calibration/initial_fit_from_recordings.json --out $D/c1.json \
    --param camera_forward_of_rear_axle_m --value R --status measured --note "ruler: rear axle centre to lens"
ls data/calib/strips_*.jpg
venv/bin/python perception/calibrate.py find-strips data/calib/strips_<TIME>.jpg --expect 3 --annotate $D/strips_marked.png
```
Open `$D/strips_marked.png` (green lines = detected rows; row ruler on the left) and confirm each green line sits on a white strip.
The command prints a ready-made `fit-depth` line; if it finds the wrong count, read the rows off the ruler and type them yourself
(rows are listed far -> near, i.e. 0.60, 0.45, 0.30 m):
```
venv/bin/python perception/calibrate.py fit-depth $D/c1.json --rows R_FAR R_MID R_NEAR --distances 0.60 0.45 0.30 --out $D/c2.json
```
Check the printout: residual rms should be a few mm; horizon_row_px should be roughly 40-55 (the recordings gave 45); a residual above 3 cm triggers a warning.
```
venv/bin/python perception/calibrate.py fit-axis $D/c2.json data/calib/centred_*.jpg --out $D/CAL.json
venv/bin/python perception/calibrate.py show $D/CAL.json       # must end with: missing ... nothing / assumed ... none
```
`fit-axis` prints "skipping ..." for any centred photo where both orange lines are not found: you need at least 3 used.

## 6. Validate
```
venv/bin/python perception/validate_calibration.py $D/CAL.json data/calib/truth.csv --csv $D/validation.csv
venv/bin/python perception/state_estimator.py data/calib/L25_*.jpg --calibration $D/CAL.json --out $D/debug_L25    # optional: look at one debug image
```
Read the table: signs first (left photos must show positive offset, left-yaw photos positive yaw), then bias and rms. Exit status 0 = PASS against
the 1 cm / 2 degree guideline. A systematic scale (the "estimate = a x truth + b" line) means a calibration problem, not noise.
Do not edit the numbers to make it pass: record the result, and if it fails, re-measure `R`, the strip distances and the centred placement first.

## What this does NOT establish
* Lens distortion (not modelled; wide-angle). Errors growing toward the image edges mean a checkerboard calibration is needed.
* Bends (heading bias ~0.1 rad structural). Only straights are validated here.
* Nothing here sets the steering map, throttle, or touches the DQN. `drive_dqn.py --enable-motors` still needs a measured steering map.
