"""Drive the JetRacer with a trained steering model (run ON the car, Python 3.6).

camera -> crop/resize -> ResNet18 -> steering, at a fixed, capped throttle.
Press X (or Ctrl-C) to stop. The car also stops if the camera or model fails.

    python3 car/drive_model.py model_log/bc/best.pt
    python3 car/drive_model.py best.pt --throttle 0.25 --smooth 0.6

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

HARD_LIMIT = 0.6  # the script refuses to go above this


def parse_args():
    p = argparse.ArgumentParser(description="Drive with a trained steering model")
    p.add_argument("model", help="checkpoint saved by train/train_bc.py (best.pt)")
    p.add_argument("--throttle", type=float, default=0.25,
                   help="constant throttle (default 0.25, refuses > {})".format(HARD_LIMIT))
    p.add_argument("--smooth", type=float, default=0.5,
                   help="weight of the NEW prediction, 0..1 (1 = no smoothing, default 0.5)")
    p.add_argument("--invert-steering", action="store_true", help="flip the sign of the model output")
    p.add_argument("--steering-gain", type=float, default=None, help="leave unset to keep the library default")
    p.add_argument("--steering-offset", type=float, default=None, help="leave unset to keep the library default")
    p.add_argument("--throttle-gain", type=float, default=None, help="leave unset to keep the library default")
    p.add_argument("--capture-width", type=int, default=1640)
    p.add_argument("--capture-height", type=int, default=1232)
    args = p.parse_args()
    if args.throttle <= 0 or args.throttle > HARD_LIMIT:
        p.error("--throttle must be in (0, {}], got {}".format(HARD_LIMIT, args.throttle))
    if not 0.0 < args.smooth <= 1.0:
        p.error("--smooth must be in (0, 1]")
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


def key_pressed(char):
    """True if `char` was typed (non-blocking). Needs the terminal in cbreak mode."""
    while select.select([sys.stdin], [], [], 0)[0]:
        if sys.stdin.read(1).lower() == char:
            return True
    return False


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model on {} ...".format(device))
    model, ckpt = load_model(args.model, device)
    print("Model: epoch {}, val MAE {:.4f}".format(ckpt.get("epoch"), ckpt.get("val_mae", float("nan"))))

    print("=" * 60)
    print("FIRST RUN: LIFT THE WHEELS OFF THE GROUND.")
    print("Press X (or Ctrl-C) at any time to stop. Starting in 5 s.")
    print("=" * 60)
    time.sleep(5)

    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(1))  # so finally still runs
    car = NvidiaRacecar()
    camera = None
    use_keys = sys.stdin.isatty()
    old_tty = termios.tcgetattr(sys.stdin) if use_keys else None
    try:
        car.throttle = 0.0
        if args.steering_gain is not None:
            car.steering_gain = args.steering_gain
        if args.steering_offset is not None:
            car.steering_offset = args.steering_offset
        if args.throttle_gain is not None:
            car.throttle_gain = args.throttle_gain
        print("In use: steering_gain={} steering_offset={} throttle_gain={}".format(
            car.steering_gain, car.steering_offset, car.throttle_gain))

        camera = CSICamera(width=320, height=240,
                           capture_width=args.capture_width, capture_height=args.capture_height,
                           capture_fps=30)
        # Warm-up: the first inference is slow (GPU init). Do it before the wheels turn.
        with torch.no_grad():
            model(preprocess(camera.read(), ckpt, device))
        if use_keys:
            tty.setcbreak(sys.stdin.fileno())  # read keys without waiting for Enter
        else:
            print("(stdin is not a terminal: the X key is disabled, use Ctrl-C)")

        print("Driving at throttle {}. Press X to stop.".format(args.throttle))
        sign = -1.0 if args.invert_steering else 1.0
        steering = 0.0
        frames, t_fps = 0, time.time()
        car.throttle = args.throttle
        while True:
            if use_keys and key_pressed("x"):
                print("\r\nX pressed, stopping.")
                break
            image = camera.read()
            with torch.no_grad():
                raw = float(model(preprocess(image, ckpt, device)).item())
            if raw != raw:  # NaN check
                raise RuntimeError("model output is NaN")
            raw = max(-1.0, min(1.0, sign * raw))
            steering = args.smooth * raw + (1.0 - args.smooth) * steering  # exponential smoothing
            car.steering = steering

            frames += 1
            if time.time() - t_fps >= 1.0:
                print("\rfps {:5.1f}   steering {:+.2f}   ".format(frames / (time.time() - t_fps), steering))
                frames, t_fps = 0, time.time()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        for _ in range(2):  # try twice in case the first write fails
            try:
                car.throttle = 0.0
                car.steering = 0.0
                break
            except Exception as e:
                print("WARNING: failed to stop the car: {}".format(e))
        if old_tty is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
        if camera is not None:
            try:
                camera.cap.release()
            except Exception:
                pass
        print("Throttle set to 0.")


if __name__ == "__main__":
    main()
