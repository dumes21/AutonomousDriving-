"""Drive the JetRacer with a trained steering model (run ON the car, Python 3.6).

camera -> crop/resize -> ResNet18 -> steering, at a fixed, capped throttle.

DEADMAN: the car only gets throttle while you HOLD the W key (key repeat). If
no W arrives for 0.5 s throttle drops to 0 (steering keeps following the model)
and you must hold W again. Press X (or Ctrl-C) to stop completely. The car also
stops if the camera or model fails, and on SIGTERM / SIGHUP (e.g. the SSH
connection drops). This script needs a real terminal for the W key.

    python3 car/drive_model.py model_log/bc/best.pt --invert-steering --invert-throttle
    python3 car/drive_model.py best.pt --throttle 0.25 --smooth 0.6

The camera is streamed on port 8080 while this runs: on the laptop,
`python tools/view_camera.py <car-ip>` shows it. --stream-port 0 turns it off.

Tip: set a short key-repeat delay on your laptop, otherwise the deadman can
trip between the first W press and the repeat starting.

FIRST RUN: put the car on a stand with the wheels off the ground and check
that the steering turns towards what the camera sees.
"""
import argparse
import os
import select
import signal
import sys
import termios
import time
import tty

try:
    import cv2
    import numpy as np
    import torch
    import torchvision
except ImportError as e:
    print("ERROR: missing library ({}). Needs opencv, numpy, torch, torchvision.".format(e))
    sys.exit(1)

try:
    from jetracer.nvidia_racecar import NvidiaRacecar
    from jetcam.csi_camera import CSICamera
except ImportError as e:
    print("ERROR: could not import jetracer/jetcam ({}). Run this on the JetRacer.".format(e))
    sys.exit(1)

from mjpeg_stream import start_stream

HARD_LIMIT = 0.6  # the script refuses to go above this


def parse_args():
    p = argparse.ArgumentParser(description="Drive with a trained steering model")
    p.add_argument("model", help="checkpoint saved by train/train_bc.py (best.pt)")
    p.add_argument("--throttle", type=float, default=0.25,
                   help="constant throttle (default 0.25, refuses > {})".format(HARD_LIMIT))
    p.add_argument("--smooth", type=float, default=0.5,
                   help="weight of the NEW prediction, 0..1 (1 = no smoothing, default 0.5)")
    p.add_argument("--invert-steering", action="store_true", help="flip the sign of the model output")
    p.add_argument("--invert-throttle", action="store_true",
                   help="send the NEGATED throttle to the car (use if forward needs a negative value)")
    p.add_argument("--deadman", type=float, default=0.5,
                   help="seconds without a W key press before throttle is set to 0 (default 0.5)")
    p.add_argument("--steering-gain", type=float, default=1.0,
                   help="car.steering_gain (default 1.0; the library default of -0.65 is NOT used)")
    p.add_argument("--steering-offset", type=float, default=0.0, help="car.steering_offset (default 0.0)")
    p.add_argument("--steer-limit", type=float, default=0.8,
                   help="clamp every steering command to +-this value (default 0.8)")
    p.add_argument("--throttle-gain", type=float, default=None, help="leave unset to keep the library default")
    p.add_argument("--capture-width", type=int, default=1640)
    p.add_argument("--capture-height", type=int, default=1232)
    p.add_argument("--stream-port", type=int, default=8080,
                   help="stream the camera on this port for tools/view_camera.py (0 = off, default 8080)")
    args = p.parse_args()
    if args.throttle <= 0 or args.throttle > HARD_LIMIT:
        p.error("--throttle must be in (0, {}], got {}".format(HARD_LIMIT, args.throttle))
    if not 0.0 < args.smooth <= 1.0:
        p.error("--smooth must be in (0, 1]")
    if args.deadman <= 0:
        p.error("--deadman must be > 0")
    if not 0.0 < args.steer_limit <= 1.0:
        p.error("--steer-limit must be in (0, 1]")
    return args


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu")
    model = torchvision.models.resnet18()  # same layout as in training; weights come from the file
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def preprocess(bgr, ckpt, device):
    """Same steps as train/train_bc.py: crop the top, resize, BGR->RGB, normalise."""
    img = bgr[int(bgr.shape[0] * ckpt["crop"]):]
    img = cv2.resize(img, tuple(ckpt["size"]), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array(ckpt["mean"], dtype=np.float32)) / np.array(ckpt["std"], dtype=np.float32)
    x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return x.to(device)


def read_keys():
    """Return the (lower-case) keys typed since the last call. Non-blocking; needs cbreak mode."""
    keys = []
    while select.select([sys.stdin], [], [], 0)[0]:
        ch = sys.stdin.read(1)
        if ch == "":  # EOF: the terminal went away
            keys.append("EOF")
            break
        keys.append(ch.lower())
    return keys


def throttle_for_car(forward, cap, invert):
    """forward is 0..cap (positive = forward). Returns the value sent to car.throttle.
    The cap applies to the absolute value, so nothing exceeds it in either direction."""
    value = max(0.0, min(cap, forward))
    return -value if (invert and value > 0) else value  # avoid sending -0.0


def stop_car(car):
    """Throttle and steering to 0. Tries twice in case the first write fails."""
    for _ in range(2):
        try:
            car.throttle = 0.0
            car.steering = 0.0
            return
        except Exception as e:
            print("WARNING: failed to stop the car: {}".format(e))


def main():
    args = parse_args()
    if not sys.stdin.isatty():
        print("ERROR: run this from a real terminal (e.g. SSH). The W-key deadman needs keyboard input.")
        sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model on {} ...".format(device))
    model, ckpt = load_model(args.model, device)
    print("Model: epoch {}, val MAE {:.4f}".format(ckpt.get("epoch"), ckpt.get("val_mae", float("nan"))))

    print("=" * 60)
    print("FIRST RUN: LIFT THE WHEELS OFF THE GROUND.")
    print("HOLD W to drive (deadman: {} s without W = throttle 0). X or Ctrl-C stops.".format(args.deadman))
    print("Starting in 5 s.")
    print("=" * 60)
    time.sleep(5)

    car = NvidiaRacecar()
    camera = None
    streamer = None
    old_tty = termios.tcgetattr(sys.stdin)

    def stop_and_exit(signum, frame):
        stop_car(car)  # zero everything first, then leave (the finally block below also runs)
        sys.exit(1)

    signal.signal(signal.SIGTERM, stop_and_exit)
    signal.signal(signal.SIGHUP, stop_and_exit)  # e.g. the SSH connection dropped
    try:
        car.throttle = 0.0
        # Set explicitly: the library default gain is -0.65, which is not what our car is calibrated for.
        car.steering_gain = args.steering_gain
        car.steering_offset = args.steering_offset
        if args.throttle_gain is not None:
            car.throttle_gain = args.throttle_gain
        print("In use: steering_gain={} steering_offset={} throttle_gain={}".format(
            car.steering_gain, car.steering_offset, car.throttle_gain))
        print("Throttle {} (hard limit {}), steer limit {}, invert_throttle={}, invert_steering={}".format(
            args.throttle, HARD_LIMIT, args.steer_limit, args.invert_throttle, args.invert_steering))

        camera = CSICamera(width=320, height=240,
                           capture_width=args.capture_width, capture_height=args.capture_height,
                           capture_fps=30)
        streamer = start_stream(args.stream_port)
        # Warm-up: the first inference is slow (GPU init). Do it before the wheels turn.
        with torch.no_grad():
            model(preprocess(camera.read(), ckpt, device))
        tty.setcbreak(sys.stdin.fileno())  # read keys without waiting for Enter

        print("Ready. Hold W to drive, X to stop.")
        sign = -1.0 if args.invert_steering else 1.0
        steering = 0.0
        last_w_time = 0.0  # when W last arrived (deadman)
        frames, t_fps = 0, time.time()
        while True:
            for key in read_keys():
                if key == "x" or key == "EOF":
                    print("\r\nStopping ({}).".format("X pressed" if key == "x" else "terminal closed"))
                    return
                if key == "w":
                    last_w_time = time.time()

            image = camera.read()
            if streamer is not None:
                streamer.publish(image)
            with torch.no_grad():
                raw = float(model(preprocess(image, ckpt, device)).item())
            if raw != raw:  # NaN check
                raise RuntimeError("model output is NaN")
            raw = max(-args.steer_limit, min(args.steer_limit, sign * raw))
            steering = args.smooth * raw + (1.0 - args.smooth) * steering  # exponential smoothing
            steering = max(-args.steer_limit, min(args.steer_limit, steering))
            car.steering = steering

            # Deadman: throttle only while W keeps repeating.
            go = time.time() - last_w_time <= args.deadman
            car.throttle = throttle_for_car(args.throttle if go else 0.0, args.throttle, args.invert_throttle)

            frames += 1
            if time.time() - t_fps >= 1.0:
                print("\rfps {:5.1f}   steering {:+.2f}   {}".format(
                    frames / (time.time() - t_fps), steering, "DRIVING" if go else "throttle 0 - hold W"))
                frames, t_fps = 0, time.time()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        stop_car(car)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
        if streamer is not None:
            streamer.stop()
        if camera is not None:
            try:
                camera.cap.release()
            except Exception:
                pass
        print("Throttle set to 0.")


if __name__ == "__main__":
    main()
