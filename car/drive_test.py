"""Hardware bring-up for the JetRacer (run ON the car, Python 3.6).

Sweeps the steering -1 -> +1 -> -1, then drives forward briefly at a low,
capped throttle. Throttle is always set back to 0 when the script ends,
including on Ctrl-C or an error.

    python3 car/drive_test.py
    python3 car/drive_test.py --max-throttle 0.4 --steering-gain -0.65
"""
import argparse
import signal
import sys
import time

try:
    from jetracer.nvidia_racecar import NvidiaRacecar
except ImportError as e:
    print("ERROR: could not import jetracer ({}). Run this on the JetRacer.".format(e))
    sys.exit(1)

HARD_LIMIT = 0.6  # the script refuses to go above this, whatever the flag says


def parse_args():
    p = argparse.ArgumentParser(description="JetRacer steering sweep + short drive")
    p.add_argument("--max-throttle", type=float, default=0.3,
                   help="throttle used for the drive (default 0.3, refuses > {})".format(HARD_LIMIT))
    p.add_argument("--drive-seconds", type=float, default=1.5, help="how long to drive forward")
    p.add_argument("--sweep-seconds", type=float, default=4.0, help="time for -1 -> +1 -> -1")
    p.add_argument("--steering-gain", type=float, default=None, help="leave unset to keep the library default")
    p.add_argument("--steering-offset", type=float, default=None, help="leave unset to keep the library default")
    p.add_argument("--throttle-gain", type=float, default=None, help="leave unset to keep the library default")
    args = p.parse_args()
    if args.max_throttle <= 0 or args.max_throttle > HARD_LIMIT:
        p.error("--max-throttle must be in (0, {}], got {}".format(HARD_LIMIT, args.max_throttle))
    return args


def sweep_steering(car, seconds):
    """Move steering -1 -> +1 -> -1 in small steps, printing as we go."""
    n = 40
    # 0..n..0 triangle wave mapped to -1..+1..-1
    values = [-1.0 + 2.0 * i / (n / 2.0) for i in range(n // 2 + 1)]
    values += values[-2::-1]
    for v in values:
        car.steering = v
        print("  steering = {:+.2f}".format(v))
        time.sleep(seconds / len(values))
    car.steering = 0.0


def main():
    args = parse_args()

    print("=" * 60)
    print("WARNING: LIFT THE WHEELS OFF THE GROUND for the first run!")
    print("The car will move its wheels. Press Ctrl-C now to abort.")
    print("=" * 60)
    time.sleep(5)

    # Turn SIGTERM into a normal exit so the finally block still runs.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(1))

    car = NvidiaRacecar()
    try:
        car.throttle = 0.0
        # Only touch settings the user asked for; otherwise keep library defaults.
        if args.steering_gain is not None:
            car.steering_gain = args.steering_gain
        if args.steering_offset is not None:
            car.steering_offset = args.steering_offset
        if args.throttle_gain is not None:
            car.throttle_gain = args.throttle_gain
        print("In use: steering_gain={} steering_offset={} throttle_gain={}".format(
            car.steering_gain, car.steering_offset, car.throttle_gain))
        print("Throttle for the drive: {} (hard limit {})".format(args.max_throttle, HARD_LIMIT))

        print("Sweeping steering...")
        sweep_steering(car, args.sweep_seconds)

        print("Driving forward for {:.1f} s at throttle {}...".format(args.drive_seconds, args.max_throttle))
        car.throttle = args.max_throttle
        time.sleep(args.drive_seconds)
        car.throttle = 0.0
        print("Done.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        # Always stop, even if the line above fails; try twice in case the first write errors.
        for _ in range(2):
            try:
                car.throttle = 0.0
                car.steering = 0.0
                break
            except Exception as e:
                print("WARNING: failed to stop the car: {}".format(e))
        print("Throttle set to 0.")


if __name__ == "__main__":
    main()
