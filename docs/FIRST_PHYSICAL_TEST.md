# First physical test: what is ready, what must still be done

Pipeline (all offline-tested, nothing has touched the car): camera frame -> `perception/boundary_detector.py` (Lab) ->
`perception/state_estimator.py` -> safety gate (`car/lane_observation.py`) -> DQN (`inference/dqn_policy.py`, numpy copy of
`experiments/dqn/20261020_151620_footprint_v1/final.zip`) -> proposed action -> optional adapter (`inference/vehicle_adapter.py`).
Runner: `car/drive_dqn.py` (DRY RUN unless `--enable-motors`). Everything below marked **MANDATORY** must be true before motors are enabled.

## What the software does today
* DQN: observation `[offset/0.13, heading/pi, last_steering]`; actions `0 = LEFT, 1 = STRAIGHT, 2 = RIGHT` (simulator: +0.4 rad, 0, -0.4 rad);
  `last_steering = 1 - action` (+1 left). Greedy. The `.npz` reproduces the SB3 checkpoint's actions exactly (3,000 random inputs checked).
  SB3 needs Python >= 3.10, so the car uses the `.npz` (numpy only). The checkpoint file is untouched.
* Any doubt = STOP (throttle 0): no valid two-boundary detection, camera not calibrated, stale/missing frame, estimate outside the simulator's
  range, gate not armed or latched, emergency stop, any exception. Missing values are never replaced by zeros. On every STOP the previous-steering
  memory resets to 0 (the simulator's reset state).
* Real recorded frames currently give **STOP on all 849 frames** (UNCALIBRATED), as intended.

## Sign mismatch you must handle (simulator vs car)
The simulator's LEFT is **positive** (+1 / +0.4 rad). The physical car turns left with **negative** commands (`-0.8` = strong left, `+0.6` = slight
right, about `+0.35` = straight, per the team's measurements, not verified by this code). Nothing converts automatically: the adapter needs a
`SteeringMap` file with the physical command for LEFT, STRAIGHT and RIGHT, each marked `measured`, with `LEFT < STRAIGHT < RIGHT`.
The car's steering is asymmetric, and the policy was trained with a symmetric +-0.4 rad, so choose LEFT and RIGHT values that give the **same turning
radius** (the simulator's full lock is `0.15 / tan(0.4) = 0.355 m`). Also note the policy is almost bang-bang (it rarely picks STRAIGHT): expect
the servo to flip between the two extremes every decision.

## A. Camera calibration (MANDATORY; details and commands in docs/CAMERA_TO_STATE.md)
1. Fix the camera rigidly. Measure camera height and the distance from the **rear-axle centre** forward to the lens. Record them.
2. Photo of the straight with 3 tape strips at known distances from the lens (e.g. 0.30 / 0.45 / 0.60 m): `calibrate.py fit-depth`.
3. Photos with the car centred and parallel in the lane: `calibrate.py fit-axis`. `calibrate.py set` for the ruler measurement (status `measured`).
4. Validation images at known poses on the straight (offset +-2.5, +-5 cm; yaw +-5, +-10 deg) and on a bend (centred, +-3 cm). Run
   `python perception/state_estimator.py FRAMES --calibration CAL.json`. **MANDATORY pass criteria (suggested, set your own before looking):**
   sign correct (car moved LEFT -> offset positive; car rotated LEFT -> heading positive); offset error <= 1.5 cm; heading error <= 0.05 rad on the straight.
   On the bend the estimator has a known bias (~0.1 rad); decide whether that is acceptable before any bend.
5. The calibration file must contain no `assumed` parameters.

## B. Steering and speed calibration (MANDATORY; wheels raised first)
1. With wheels off the ground, check that each candidate command turns the wheels the expected way and never reaches a stop/brownout (keep |steering| <= 0.8).
2. On the floor at the chosen test throttle, drive circles at each candidate steering value and measure the radius; pick LEFT/STRAIGHT/RIGHT values with
   matching radii (~0.36 m for full lock, if the car can do it) and a straight value that really goes straight. Write them in a JSON file:
   `{"steer_limit": <<=0.8>, "LEFT": {"value": <v>, "status": "measured"}, "STRAIGHT": {...}, "RIGHT": {...}}`.
3. Speed: minimum usable throttle (team finding ~0.10-0.12). Time the car over a measured 1.0 m on the straight at the throttle you will use. The simulator
   assumes **0.6 m/s** and a decision every **0.05 s**. If the real speed is much lower, the policy's timing is off: note the ratio and decide whether to
   accept it for a straight-only test (do not claim lap performance).
4. Loop rate: run the dry run (below) on the car and check the log timestamps: the simulator decides at 20 Hz.

## C. Dry run (motors cannot move; nothing is enabled)
Laptop, replaying recordings or new frames (expected today with the incomplete calibration: STOP everywhere):
`python car/drive_dqn.py --calibration perception/calibration/initial_fit_from_recordings.json --frames data/20261013_001444`
On the car, live camera, car on the track by hand, **still no motors**:
`python3 car/drive_dqn.py --calibration CAL.json --camera --max-steps 300`
Check: proposed LEFT/RIGHT agree with how you push the car (left of centre -> RIGHT, pointing left -> RIGHT); covering one boundary or the lens gives STOP within a
few frames and a latched stop that needs re-arming; unplug/stall the camera -> STOP. Read `logs/dqn_dryrun_*.csv`.

## D. First driving trial (**MANDATORY: A, B and C passed first**)
1. Wheels raised once more, then on a short straight at the lowest usable throttle, one person holding the stop key / car, plus a person ready to lift it.
2. Command (the only motor-enabled command): `python3 car/drive_dqn.py --calibration CAL.json --camera --enable-motors --steering-map STEER.json --throttle T`
   It refuses to start if the calibration is incomplete or has `assumed` values, the steering map is not fully `measured`, or `--throttle` is missing (or > 0.6).
   Type `YES`, press **A** once valid frames are shown, **hold W** to drive (0.5 s deadman), **X** = emergency stop. SIGTERM/SIGHUP/exit stop the car.
   This motor path has **never been run on hardware**: treat the first run as a test of the safety behaviour, not of driving quality.
3. Check the log (`logs/dqn_motors_*.csv`) after each run: status, offset/heading, action, any STOP reasons.
4. Only after several clean straight runs consider a bend; the heading estimate is biased there and the lane may leave the camera view (STOP is the expected outcome).

## Known blockers / risks (not solved here)
* Real camera is UNCALIBRATED; the offset/heading estimator is unvalidated on real poses; heading in bends is structurally biased (~0.1 rad).
* The policy has only seen noise-free observations and has ~0.8 cm body clearance; with action noise it crashed in the simulator 87-92 % of episodes.
* About 18-23 % of real frames have no valid two-line detection; in tight corners one boundary leaves the view.
* Speed / decision-rate mismatch with the simulator, asymmetric steering, bang-bang policy, throttle sign (`--invert-throttle`) and minimum throttle are unmeasured.
* Latency of camera + estimation is unknown (the estimator does about 5 ms on a laptop; not timed on the Jetson).
