# JetRacer lane following (43008 Reinforcement Learning)

We train a small car to drive around a track with DQN (Stable-Baselines3),
first in our own fast 2D simulator, then on a real JetRacer.
Current scope: **Level 1 only** (lane following).

## Layout

```
sim/track_env.py        the simulator (gymnasium env "JetRacerEnv")
sim/demo.py             watch the sim, or benchmark its speed
tests/test_track_env.py tests for the sim
car/drive_test.py       runs ON the JetRacer: steering sweep + short drive
car/camera_test.py      runs ON the JetRacer: camera fps check + save images
airc-rl-agent/          upstream reference code, we are NOT building on it
PROJECT_PLAN.md         the project plan
```

## Run the sim (laptop)

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python sim/demo.py               # pygame window, random actions
python sim/demo.py --benchmark   # 100k random steps, prints steps/sec
pytest                           # run the tests
```

The observation is only `[lane_offset, heading_error, last_steering]`, because
that is all the real car's camera pipeline will be able to give us later.

## Run the car scripts (on the Jetson, Python 3.6)

**Lift the wheels off the ground for the first run.**

```
python3 car/drive_test.py                       # throttle capped at 0.3
python3 car/drive_test.py --max-throttle 0.4    # refuses anything above 0.6
python3 car/camera_test.py --every 1 --max-frames 20   # JPEGs go to data/
python3 car/camera_test.py --show               # also show a window
```

These scripts only need numpy, opencv, `jetracer` and `jetcam`. They do not
import anything from `sim/`.
