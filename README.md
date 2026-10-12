# JetRacer lane following (43008 Reinforcement Learning)

We train a small car to drive around a track with DQN (Stable-Baselines3),
first in our own fast 2D simulator, then on a real JetRacer.
Current scope: **Level 1 only** (lane following).
Prior art: [masato-ka/airc-rl-agent](https://github.com/masato-ka/airc-rl-agent) (not used here).

## Layout

```
sim/track_env.py        the simulator (gymnasium env "JetRacerEnv")
sim/demo.py             watch the sim, or benchmark its speed
sim/train.py            train DQN, compare with baselines
tests/test_track_env.py tests for the sim
car/drive_test.py       runs ON the JetRacer: steering sweep + short drive
car/camera_test.py      runs ON the JetRacer: camera fps check + save images
car/teleop_record.py    runs ON the JetRacer: keyboard driving + recording training data
car/drive_model.py      runs ON the JetRacer: drive with a trained steering model
train/inspect_data.py   check a recorded session (stats, histogram, sample frames)
train/train_bc.py       train ResNet18 to copy the human's steering (imitation learning)
PROJECT_PLAN.md         the project plan
```

## Run the sim (laptop)

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python sim/demo.py               # pygame window, random actions
python sim/demo.py --benchmark   # 100k random steps, prints steps/sec
python sim/train.py --seed 0     # train DQN (300k steps, ~5 min), prints a results table
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

## Imitation learning (drive by hand, then copy the driver)

```
python3 car/teleop_record.py                 # ON the car, over SSH. A/D steer, W/S throttle, R record, Q quit
python train/inspect_data.py data/<session>  # laptop: copy data/<session> over first, check the plots in logs/
python train/train_bc.py data/<session> [more sessions] [--flip]   # saves model_log/bc/best.pt
python3 car/drive_model.py best.pt           # ON the car: copy best.pt over; wheels up for the first run, X stops
```

`train_bc.py --limit 200 --epochs 2` is a quick smoke test. The car scripts cap throttle (0.3 by default,
refuse > 0.6) and always set throttle to 0 on exit.
