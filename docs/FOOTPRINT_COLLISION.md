# Footprint-aware off-track rule

## What was wrong
`JetRacerEnv` ended an episode only when the car's **reference point** (the rear-axle centre that the bicycle model moves)
was more than `lane_half_width` (0.13 m) from the centre line. The body was ignored, and it was drawn wrongly:
a 0.25 x 0.15 m rectangle *centred* on that point (real width 0.17 m; wrong reference point). A car 0.10 m off-centre
(the old DQN's mean |offset|) therefore had up to 0.185 m of lateral extent from the centre line, i.e. ~5.5 cm of the body over
the 0.13 m line, yet the episode counted as 0 % crashes.

## What it does now (`sim/track_env.py`, `Config`)
* `footprint_collision = True`: the episode ends if **any part of the body outline** is farther than `lane_half_width` from the centre
  line. The outline is `car_width` (0.17 m, measured) x `car_length`, with corners and points along both long sides
  (a bend's inside edge can be touched mid-edge), oriented by the car's heading. `info["footprint_margin_m"]` is the clearance
  (negative = over the line).
* `car_length = 0.20` m: **NOT MEASURED, an assumption.** `car_rear_overhang = None` -> equal front/rear overhang
  `(car_length - wheelbase) / 2`: also an assumption. Please measure bumper-to-bumper length and the rear axle position.
* `footprint_collision = False` restores the old centre-only rule (used here to compare).
* `reset()` redraws the random start (same ranges) until the whole body starts inside the lane.
* Rendering draws the same footprint to scale, with the rear axle (white dot) and nose (yellow dot).
* Reward, observation (3 numbers), actions (3) and dynamics are unchanged. The wheelbase is still 0.17 m (measured: ~0.15 m).

## Why this matters: the lane is tight
With a 0.17 m body in a 0.26 m lane the allowed centre error is +-4.5 cm on a straight, and the margin when perfectly
aligned on a bend centre line (0.72 m radius) is only +0.024 m (length 0.20), +0.018 m (length 0.25): the front outer corner swings
outward in a bend.

| car length | wheelbase | scripted centre controller (20 seeds, footprint rule) |
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
