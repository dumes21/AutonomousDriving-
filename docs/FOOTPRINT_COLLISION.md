# Footprint-aware off-track rule

## What was wrong
`JetRacerEnv` ended an episode only when the car's **reference point** (the rear-axle centre that the bicycle model moves)
was more than `lane_half_width` (0.13 m) from the centre line. The body was ignored, and it was drawn wrongly:
a 0.25 x 0.15 m rectangle *centred* on that point (real width 0.17 m; wrong reference point). A car 0.10 m off-centre
(the old DQN's mean |offset|) therefore had up to 0.185 m of lateral extent from the centre line, i.e. ~5.5 cm of the body over
the 0.13 m line, yet the episode counted as 0 % crashes.

## What it does now (`sim/track_env.py`, `Config`)
* `footprint_collision = True`: the episode ends if **any part of the body outline** is farther than `lane_half_width` from the centre
  line. The outline is `car_width` (0.17 m, measured) x body length, with corners and points along both long sides
  (a bend's inside edge can be touched mid-edge), oriented by the car's heading. `info["footprint_margin_m"]` is the clearance
  (negative = over the line).
* Body length = `car_rear_overhang` + `wheelbase` + `car_front_overhang`; the two overhangs are set **independently**
  (rear axle -> rear bumper, front axle -> front bumper). Both are **NOT MEASURED placeholders** (0.025 m each, i.e. 0.20 m
  long). The wheelbase is 0.15 m (measured, approx.).
* `footprint_collision = False` restores the old centre-only rule (used here to compare).
* `reset()` redraws the random start (same ranges) until the whole body starts inside the lane.
* Rendering draws the same footprint to scale, with the rear axle (white dot) and nose (yellow dot).
* Reward, observation (3 numbers), actions (3) and dynamics are unchanged. Only the wheelbase changed (0.17 -> 0.15 m, measured), which makes the steering a little tighter.

## Why this matters: the lane is tight
With a 0.17 m body in a 0.26 m lane the allowed centre error is +-4.5 cm on a straight, and the margin when perfectly
aligned on a bend centre line (0.72 m radius) is only +0.024 m (length 0.20), +0.018 m (length 0.25): the front outer corner swings
outward in a bend.

| car length (earlier test, equal overhangs) | wheelbase | scripted centre controller (20 seeds, footprint rule) |
|---|---|---|
| 0.18 / 0.20 | 0.17 | 4.52 laps, 0 % crashes |
| 0.22 | 0.17 | 0.42 laps, 100 % crashes |
| 0.25 | 0.17 | 0.09 laps, 100 % crashes |
| 0.20 / 0.22 | 0.15 | 4.53 laps, 0 % crashes |
| 0.24 | 0.15 | 1.33 laps, 95 % crashes |
| 0.25 | 0.15 | 0.19 laps, 100 % crashes |

The outcome flips sharply with body length: the simple 3-action controller survives when the aligned bend margin is >= ~0.022 m and fails at
<= ~0.020 m. **Measure the real car length and wheelbase before trusting any result.**

## Previously trained DQN (`experiments/dqn/20261020_144343_baseline`, trained under the old rule), 20 seeds, no retraining
| rule | laps | crashes | mean abs offset | mean duration |
|---|---|---|---|---|
| old (centre only) | 5.06 | 0 % | 0.100 m | 50 s |
| footprint (length 0.20) | 0.04 | **100 %** | 0.017 m | 0.4 s |

Same 100 % under length 0.20/0.25/0.30, wheelbase 0.15 or width 0.15 m. The policy lives on a line 10 cm inside the centre
line, which the new rule rejects within about half a second.

## Measurements still needed (checklist)
Measure on a flat table with a ruler; use the centre of each axle.
1. **Rear overhang**: rear axle centre -> rearmost point of the car (bumper, battery, anything that could cross the tape).
2. **Front overhang**: front axle centre -> foremost point (bumper, camera mount).
   Cross-check: rear overhang + wheelbase + front overhang = bumper-to-bumper length; measure that too.
3. **Wheelbase**: front axle centre -> rear axle centre (currently 0.15 m).
4. **Width**: widest point of the body including wheels (currently 0.17 m).
5. Track: is 0.26 m tape-centre to tape-centre or inner edge to inner edge? Tape width. Bend radius (0.72 m is an estimate), straight length (1.0 m).
6. Optional: full-lock turning circle radius (rear axle) on the floor; the sim assumes 0.15 / tan(0.4) = 0.355 m.

Validate once you have them:
`python sim/check_footprint.py --front-overhang F --rear-overhang R --wheelbase 0.15 --width 0.17`
(what-if, nothing edited), then put the numbers in `Config` and run `python -m pytest -q && python sim/test_env.py`.
