"""Run the trained DQN through the camera pipeline: DRY RUN by default (no motors), Python 3.6 compatible.

    camera frame -> perception/boundary_detector (Lab) -> perception/state_estimator -> safety gate -> DQN -> proposed action

DRY RUN (nothing can move; no hardware import):
    python car/drive_dqn.py --calibration CAL.json --frames data/20261013_001444        # replay recorded frames
    python car/drive_dqn.py --calibration CAL.json --camera --max-steps 300             # live camera on the car, still no motors

MOTORS (only after docs/FIRST_PHYSICAL_TEST.md sections A-C are done; this path has NOT been run on hardware):
    python3 car/drive_dqn.py --calibration CAL.json --camera --enable-motors --steering-map STEER.json --throttle T
It refuses to start unless: the calibration is complete and has no 'assumed' parameters, the steering map has LEFT / STRAIGHT / RIGHT
all marked 'measured', and --throttle is given explicitly (<= 0.6, no default). Keys: A = arm (after enough valid frames),
HOLD W = drive (0.5 s deadman), X = emergency stop. Any invalid frame, stale camera, missing calibration or error = throttle 0;
persistent loss latches the stop until you press A again. SIGTERM / SIGHUP / exit all stop the car.

With an empty or incomplete calibration every frame is UNCALIBRATED -> STOP: that is the intended behaviour.
"""
import argparse
import csv
import glob
import math
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
for sub in ("inference", "perception", "car"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import cv2  # noqa: E402

from dqn_policy import load_policy  # noqa: E402
from ground_plane import CameraCalibration  # noqa: E402
from pipeline import CameraToActionPipeline  # noqa: E402
import state_estimator as se  # noqa: E402
import vehicle_adapter as va  # noqa: E402

DEFAULT_MODEL = os.path.join(ROOT, "inference", "dqn_footprint_v1.npz")
LOG_DIR = os.path.join(ROOT, "logs")
COLUMNS = ["t", "status", "label", "reason", "offset_m", "heading_rad", "obs_offset", "obs_heading", "obs_last_steering",
           "confidence", "armed", "estop"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help=".npz from inference/dqn_policy.py export (or an SB3 .zip on the laptop)")
    p.add_argument("--calibration", required=True, help="camera calibration JSON (perception/calibrate.py)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", help="folder of recorded images (replay; never touches hardware)")
    src.add_argument("--camera", action="store_true", help="live CSI camera (needs jetcam; still a dry run unless --enable-motors)")
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 = until the frames run out / Ctrl-C)")
    p.add_argument("--stale-after", type=float, default=0.25, help="seconds after which a frame counts as stale")
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--enable-motors", action="store_true", help="NOT a dry run. See the module docstring for the requirements")
    p.add_argument("--steering-map", help="JSON with LEFT/STRAIGHT/RIGHT physical commands, each marked measured")
    p.add_argument("--throttle", type=float, help="forward throttle (0, 0.6]; no default")
    p.add_argument("--invert-throttle", action="store_true")
    p.add_argument("--deadman", type=float, default=0.5)
    p.add_argument("--no-log", action="store_true")
    return p.parse_args()


def iter_frames(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
    if not files:
        sys.exit("no images in %s" % folder)
    for f in files:
        yield cv2.imread(f)


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt()


def check_motor_requirements(args, calib):
    """Everything that must be true before any motor command is possible. Exits with an explanation otherwise."""
    problems = []
    if not args.camera:
        problems.append("motors need --camera (live frames), not --frames")
    missing = calib.missing_for_state("measured_width")
    if missing:
        problems.append("camera calibration incomplete: %s" % ", ".join(missing))
    if calib.assumed_parameters():
        problems.append("calibration contains 'assumed' parameters: %s" % ", ".join(calib.assumed_parameters()))
    if not args.steering_map:
        problems.append("--steering-map is required (UNCALIBRATED steering is never guessed)")
    if args.throttle is None:
        problems.append("--throttle is required (no default)")
    steering_map = None
    if args.steering_map:
        try:
            steering_map = va.SteeringMap.load(args.steering_map)
        except Exception as e:                                  # noqa: B902
            problems.append("steering map rejected: %s" % e)
    if problems:
        sys.exit("MOTORS NOT ENABLED:\n  - " + "\n  - ".join(problems))
    return steering_map


def main():
    args = parse_args()
    calib = CameraCalibration.load(args.calibration)
    policy = load_policy(args.model)
    estimator = se.StateEstimator(calib, min_confidence=args.min_confidence)
    pipeline = CameraToActionPipeline(estimator, policy, stale_after_s=args.stale_after)

    adapter = va.DryRunAdapter()
    keys = None
    if args.enable_motors:
        steering_map = check_motor_requirements(args, calib)
        print("MOTOR MODE. Wheels must be OFF THE GROUND for the first run. Type YES to continue: ", end="")
        sys.stdout.flush()
        if sys.stdin.readline().strip() != "YES":
            sys.exit("not confirmed; nothing was started")
        adapter = va.JetRacerAdapter.from_hardware(steering_map, args.throttle, args.invert_throttle)
        keys = KeyReader()
        signal.signal(signal.SIGTERM, _raise_interrupt)    # the finally block below then stops the car
        signal.signal(signal.SIGHUP, _raise_interrupt)     # e.g. the SSH connection dropped
        print("Keys: A = arm, HOLD W = drive, X = emergency stop.")
    else:
        print("DRY RUN: no motor commands are possible in this mode.")
    print("Calibration status: %s" % ("complete" if not calib.missing_for_state("measured_width") else
                                      "INCOMPLETE (%s) -> every frame will be UNCALIBRATED -> STOP" %
                                      ", ".join(calib.missing_for_state("measured_width"))))

    log = None
    if not args.no_log:
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(LOG_DIR, "dqn_%s_%s.csv" % ("motors" if args.enable_motors else "dryrun", time.strftime("%Y%m%d_%H%M%S")))
        log_f = open(path, "w", newline="")
        log = csv.writer(log_f)
        log.writerow(COLUMNS)
        print("logging to %s" % path)

    camera = None
    if args.camera:
        from jetcam.csi_camera import CSICamera                 # only on the car
        camera = CSICamera(width=320, height=240, capture_width=1640, capture_height=1232, capture_fps=30)
        frames = None
    else:
        frames = iter_frames(args.frames)

    counts = {}
    t0 = time.time()
    last_w = 0.0
    step = 0
    try:
        while True:
            now = time.time() if camera is not None else step * 0.1            # replay: a 10 Hz clock
            if keys is not None:
                for k in keys.read():
                    if k in ("x", "EOF"):
                        pipeline.emergency_stop("operator pressed X")
                        print("\nEMERGENCY STOP")
                        return
                    if k == "a":
                        print("\narm: %s" % pipeline.arm())
                    if k == "w":
                        last_w = now
            frame = camera.read() if camera is not None else next(frames, None)
            if frame is None and camera is None:
                break
            cmd = pipeline.step(frame, now, frame_time=now)
            if not args.enable_motors:
                pipeline.arm()                     # dry run only: stands in for the operator's arm key
            adapter.apply(cmd, now, deadman_ok=(now - last_w <= args.deadman) if keys is not None else True)
            counts[cmd.label] = counts.get(cmd.label, 0) + 1
            est = cmd.estimate
            obs = cmd.observation
            tele = "" if est is None else "conf %.2f" % est.confidence          # live readout (print only; no logic)
            if est is not None and est.offset_m is not None:
                tele += "  offset %+6.1f cm  heading %+6.1f deg" % (100.0 * est.offset_m, math.degrees(est.heading_rad))
            print("%4d  %-8s %-17s %-46s %s" % (step, cmd.label, cmd.status, tele, cmd.reason[:60]))
            if log is not None:
                log.writerow(["%.3f" % (now - (t0 if camera is not None else 0.0)), cmd.status, cmd.label, cmd.reason[:200],
                              "" if est is None or est.offset_m is None else "%.4f" % est.offset_m,
                              "" if est is None or est.heading_rad is None else "%.4f" % est.heading_rad,
                              "" if obs is None else "%.4f" % obs[0], "" if obs is None else "%.4f" % obs[1],
                              "" if obs is None else "%+.0f" % obs[2], "" if est is None else "%.2f" % est.confidence,
                              int(not pipeline.gate.latched), int(pipeline.estop)])
                log_f.flush()
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        adapter.stop() if args.enable_motors else None
        if keys is not None:
            keys.restore()
        print("\nsummary of proposed commands: %s" % counts)
        print("hardware commands issued: %d" % getattr(adapter, "hardware_commands_issued", 0))
        sys.stdout.flush()
        if camera is not None:
            os._exit(0)                                         # skip the camera release that can hang (as in drive_model.py)


class KeyReader(object):
    """Non-blocking keyboard (cbreak mode), motor mode only. Same idea as car/drive_model.py."""

    def __init__(self):
        import select, termios, tty  # noqa: E401
        self._select, self._termios = select, termios
        if not sys.stdin.isatty():
            sys.exit("motor mode needs a real terminal for the A / W / X keys")
        self._old = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def read(self):
        out = []
        while self._select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == "":
                out.append("EOF")
                break
            out.append(ch.lower())
        return out

    def restore(self):
        self._termios.tcsetattr(sys.stdin, self._termios.TCSADRAIN, self._old)


if __name__ == "__main__":
    main()
