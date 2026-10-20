# Simulator review and sim-to-real gaps

Reviewed `sim/track_env.py` (JetRacerEnv) on branch `feat-imitation-learning`.
Not yet measured on the car: everything marked **(measure)** needs hardware data.

## What the environment is

- Stadium track, 2.0 m straights, 0.8 m centre-line bend radius, counter-clockwise.
  Lane half-width 0.20 m. Episode ends when `|offset| > half-width`.
- Kinematic bicycle, wheelbase 0.17 m, **constant 0.6 m/s**, dt 0.05 s, max steer 0.4 rad.
- Actions: 0 left, 1 straight, 2 right. Observation `[lane_offset/0.2, heading_error/pi, last_steering]`,
  float32 in [-1, 1] (clipped).
- Reward: `ds/(v*dt) + 0.1*(1-|offset|/hw)` per step, -10 on leaving the lane.

## Validation result (no bugs found, no simulator changes made)

- `check_env` passes; observations stay inside the space; seeded reset is reproducible; Gymnasium 5-tuple API OK.
- Steering convention is internally consistent: action 0 gives +steer, heading grows (counter-clockwise = left),
  `+offset` = left of centre, `+heading_error` = pointing left, `last_steering +1` = left.
- Reward: ~+1 per step when progressing at full speed; crash -10.
- A hand-written P controller (`view_track.py --policy centre`) completes ~3.3 laps with 0/20 crashes, so the
  track is solvable. "Always straight" crashes 20/20 in about 0.1 laps.

## Sim vs real: things that will hurt transfer

1. **Steering sign flips.** The sim uses +left. The car uses negative = left, positive = right. The deploy wrapper
   must negate, and also apply the measured offset below. Easy to get wrong; test with the wheels off the ground first.
2. **"Straight" is not 0 on the car.** Physical straight is about +0.35, not 0. A sim action of "straight" must map to +0.35.
3. **Physical steering is asymmetric.** -0.8 is a strong left, +0.6 only a slight right. The sim is perfectly symmetric
   (+-0.4 rad) so the learned policy will assume left and right are equally strong. Right bends, which the car turns
   least well, would be where it fails first. **(measure)** turning radius at, e.g., -0.8, -0.4, +0.35, +0.6, +0.8.
4. **Command ceiling +-0.8.** Beyond it risks brownouts, so the sim's full-lock action must map to at most -0.8 on the left,
   and probably less than +0.8 on the right. With the car's real limits the sim's 0.4 rad lock (turn radius ~0.40 m)
   may be tighter than the car can manage on the 0.8 m bends. **(measure)**
5. **Speed.** The sim holds 0.6 m/s. The car's minimum throttle (~0.10-0.12) gives some real speed we have not measured,
   and it varies with battery. **(measure)** speed at the throttle we will use, then set `Config.speed` to it.
6. **No actuation lag or noise.** Sim steering changes instantly. Servo response, camera latency (~50-100 ms; one or two
   sim steps) and frame jitter are missing. Plan domain randomisation: steering gain/offset per episode, action
   delay of 1-3 steps, observation noise.
7. **Observation estimation from OpenCV.**
   - `lane_offset` and `heading_error` are measurable from the two orange lines if the camera is calibrated to a
     ground-plane (bird's-eye) view. Without that the numbers are only a rough proxy and will not match the sim's metres.
   - The sim's lane half-width (0.20 m) must equal half the real distance between the lines **(measure)**.
   - In bends only one line may be visible, and lines can be lost at the frame edge or in glare; the sim never has
     missing observations.
   - `last_steering` is exact in the sim; on the car it is our own command, so it is known (but not the servo's true angle).
8. **Heading obs scale.** `heading_error/pi` is normally within about +-0.1, so it is a tiny input next to the offset.
   Consider scaling by ~0.5 rad instead (note: changing it changes the trained policy's input, so do it before training).
9. **Partial observability.** The agent does not see upcoming curvature, so entering a bend is only inferred from
   offset/heading drift. A look-ahead feature from the camera (e.g. line angle further up the image) would help, but
   it must be producible by OpenCV on the car.

## Reward / termination behaviours to watch

- **Inside-line corner cutting** (observed earlier with DQN): progress is measured along the centre line, so hugging the
  inside of a bend earns more progress per metre. It stays inside the lane, so the crash penalty never discourages it.
  This also makes the policy sensitive to line-detection error on the real inside line.
- **No steering-change penalty.** Zig-zag between left and right costs almost nothing, but on the car this wears and
  stresses the servo and shakes the camera. Add a small penalty on `action != previous action` (or on |steer|).
- **Termination at exactly the lane edge** is a hard cliff on a 0.2 m half-width; the real boundary is a painted line the
  car may drive over briefly. Consider a slightly tighter or softer boundary after the real lane width is measured.
- **Random start** anywhere on the track with |offset| <= 0.06 m and |heading| <= 0.2 rad: fine for exploration but does
  not include recovery from large errors, which is what the real car will face.

## Do before training DQN

1. Measure on the car: turn radius for the left/right commands above, speed at the chosen throttle, real lane width.
2. Define the action-to-command table (3 actions -> e.g. left about -0.8 or less, straight +0.35, right about +0.6-0.8,
   from the measurements) and put the asymmetry into the sim (separate left/right steering limits, plus a
   straight-offset), instead of a symmetric +-0.4 rad.
3. Set `Config.speed` and, if needed, the lane width / bend radius to match the real track.
4. Add steering lag (1-2 steps) and small observation noise to the env; add a steering-change penalty.
5. Decide the heading scale, then train; record every run's config so the policy can be matched to its env.
6. Write the OpenCV lane estimator and check it produces `[offset, heading]` that agree with ground truth on recorded frames.
