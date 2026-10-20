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
sim/test_env.py         pre-flight check: SB3 check_env + 100 random steps
sim/view_track.py       look at the track/car (window, or a GIF when headless)
tests/test_track_env.py tests for the sim
car/drive_test.py       runs ON the JetRacer: steering sweep + short drive
car/camera_test.py      runs ON the JetRacer: camera fps check + save images
car/teleop_record.py    runs ON the JetRacer: keyboard driving + recording training data
car/drive_model.py      runs ON the JetRacer: drive with a trained steering model
car/camera_stream.py    runs ON the JetRacer: stream the camera only (no driving)
car/mjpeg_stream.py     camera stream used by the car scripts above
tools/view_camera.py    laptop: OpenCV window showing the car's camera stream
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
python sim/test_env.py           # validate the env before training
python sim/view_track.py --policy centre   # window; add --gif out.gif when headless
```

The observation is only `[lane_offset, heading_error, last_steering]`, because
that is all the real car's camera pipeline will be able to give us later.
Known sim-to-real gaps are listed in `docs/SIM_TO_REAL.md`.
For SageMaker use `requirements-sagemaker.txt` (headless OpenCV, no torch pin).

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
python3 car/teleop_record.py --invert-steering --invert-throttle   # ON the car, over SSH
python train/inspect_data.py data/<session>   # laptop: copy data/<session> over first, check the plots in logs/
python train/train_bc.py data/<session> [more sessions] [--flip]   # saves model_log/bc/best.pt
python3 car/drive_model.py best.pt --invert-steering --invert-throttle   # ON the car: copy best.pt over
```

**Our car's calibration:** steering gain **1.0**, offset **0.0**, steer limit **0.8**, plus
`--invert-steering` and/or `--invert-throttle` if needed (A should steer left, W should drive forward;
check on the stand with the wheels up). The scripts set gain and offset explicitly because the library's
built-in default gain is -0.65. These are the defaults, so normally you only add the invert flags:

```
python3 car/teleop_record.py --invert-steering --invert-throttle
python3 car/drive_model.py best.pt --invert-steering --invert-throttle
# explicit form: --steering-gain 1.0 --steering-offset 0.0 --steer-limit 0.8
```

Use the same flags for recording and driving, otherwise the model's steering comes out mirrored or scaled
differently. The steer limit clamps every steering command (recorded labels and model output) to +-0.8.

**Keys in `teleop_record.py`:** A/D nudge the steering target by 0.1 and it stays latched, C recentres,
W/S change throttle, SPACE stops, R toggles recording, Q quits (`--steer-mode hold` gives the old
"steer only while the key is held" behaviour).

**Safety:** throttle is capped (0.3 by default, refuses > 0.6, in both directions). There is a 0.5 s deadman:
with no key for 0.5 s throttle drops to 0 and you press W again (in `drive_model.py` you must HOLD W to drive).
Throttle and steering are zeroed on exit, Ctrl-C, SIGTERM and SIGHUP (e.g. the SSH connection dropping).
Set a short key-repeat delay on your laptop so the deadman does not trip before key repeat starts.
For the first run, lift the wheels off the ground. `X` stops `drive_model.py`.

`train_bc.py --limit 200 --epochs 2` is a quick smoke test.

**Watching the camera:** `teleop_record.py` and `drive_model.py` stream the camera on port 8080
(`--stream-port 0` turns it off). On the laptop, run `python tools/view_camera.py <car-ip>` and leave it
open: the window pops up when a car script starts and goes away when it stops. Q quits, S saves a
snapshot, C toggles the red line marking the part `train_bc.py` crops off. `car/camera_stream.py`
streams without driving. Find the car's IP with `hostname -I` on the car.
