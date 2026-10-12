"""Drive the JetRacer from the keyboard over SSH and record training data
(run ON the car, Python 3.6).

Keys:   A / D  steer left / right (steering ramps, it does not jump)
        W / S  more / less throttle (small steps, capped)
        SPACE  stop (throttle and steering to 0)
        R      start / stop recording
        Q      quit

While recording AND throttle > 0, a JPEG is saved about every 0.1 s into
data/<session timestamp>/ and a row (filename, steering, throttle, timestamp)
is appended to labels.csv in that folder.

    python3 car/teleop_record.py
    python3 car/teleop_record.py --max-throttle 0.35

Note: terminals do not report key RELEASES. A/D steer while the key is held
(key repeat keeps refreshing it) and steering ramps back to centre shortly
after you let go. Key repeat has a start-up delay, so for a long turn tap
and hold A/D; --hold makes the "held" window longer.
"""
import argparse
import csv
import curses
import os
import signal
import sys
import time

try:
    import cv2
except ImportError as e:
    print("ERROR: could not import opencv ({}).".format(e))
    sys.exit(1)

try:
    from jetracer.nvidia_racecar import NvidiaRacecar
except ImportError as e:
    print("ERROR: could not import jetracer ({}). Run this on the JetRacer.".format(e))
    sys.exit(1)

try:
    from jetcam.csi_camera import CSICamera
except ImportError as e:
    print("ERROR: could not import jetcam ({}). Run this on the JetRacer.".format(e))
    sys.exit(1)

HARD_LIMIT = 0.6  # the script refuses to go above this
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
THROTTLE_STEP = 0.05
SAVE_EVERY = 0.1  # seconds between saved frames


def parse_args():
    p = argparse.ArgumentParser(description="Keyboard teleop + data recorder for the JetRacer")
    p.add_argument("--max-throttle", type=float, default=0.3,
                   help="throttle cap (default 0.3, refuses > {})".format(HARD_LIMIT))
    p.add_argument("--steer-rate", type=float, default=2.5,
                   help="steering change per second towards the target (default 2.5 = full lock in 0.4 s)")
    p.add_argument("--hold", type=float, default=0.3,
                   help="seconds a steering key counts as held after its last key press")
    p.add_argument("--invert-steering", action="store_true",
                   help="swap the meaning of A and D if the car steers the wrong way")
    p.add_argument("--out", default=DATA_DIR, help="folder that session folders go into (default <repo>/data)")
    p.add_argument("--steering-gain", type=float, default=None, help="leave unset to keep the library default")
    p.add_argument("--steering-offset", type=float, default=None, help="leave unset to keep the library default")
    p.add_argument("--throttle-gain", type=float, default=None, help="leave unset to keep the library default")
    p.add_argument("--capture-width", type=int, default=1640)
    p.add_argument("--capture-height", type=int, default=1232)
    args = p.parse_args()
    if args.max_throttle <= 0 or args.max_throttle > HARD_LIMIT:
        p.error("--max-throttle must be in (0, {}], got {}".format(HARD_LIMIT, args.max_throttle))
    return args


class Recorder(object):
    """Owns the session folder and labels.csv. Created the first time recording starts."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.session_dir = None
        self.csv_file = None
        self.writer = None
        self.count = 0

    def start(self):
        if self.session_dir is None:
            name = time.strftime("%Y%m%d_%H%M%S")
            self.session_dir = os.path.join(self.out_dir, name)
            os.makedirs(self.session_dir)
            self.csv_file = open(os.path.join(self.session_dir, "labels.csv"), "w", newline="")
            self.writer = csv.writer(self.csv_file)
            self.writer.writerow(["filename", "steering", "throttle", "timestamp"])

    def save(self, image, steering, throttle):
        now = time.time()
        name = "{:06d}.jpg".format(self.count)
        if not cv2.imwrite(os.path.join(self.session_dir, name), image):
            raise RuntimeError("could not write image (disk full?)")
        self.writer.writerow([name, "{:.4f}".format(steering), "{:.4f}".format(throttle), "{:.3f}".format(now)])
        self.csv_file.flush()  # keep the csv usable even if the script is killed
        self.count += 1

    def close(self):
        if self.csv_file is not None:
            self.csv_file.close()


def run(stdscr, args, car, camera, recorder):
    try:
        curses.curs_set(0)  # hide the cursor (not every terminal supports this)
    except curses.error:
        pass
    stdscr.nodelay(True)  # getch() returns -1 instead of waiting

    steering, throttle = 0.0, 0.0     # what is sent to the car
    target = 0.0                       # where steering is heading
    left_until, right_until = 0.0, 0.0  # time until which A / D count as "held"
    recording = False
    last_save = 0.0
    last_key = ""
    t_prev = time.time()
    sign = -1.0 if args.invert_steering else 1.0

    while True:
        # camera.read() blocks until the next frame (~30 fps), so it paces this loop too.
        image = camera.read()
        now = time.time()
        dt = now - t_prev
        t_prev = now

        # --- read every key waiting in the buffer ---
        while True:
            ch = stdscr.getch()
            if ch == -1:
                break
            key = chr(ch).lower() if 0 <= ch < 256 else ""
            if key == "a":
                left_until = now + args.hold
                last_key = "A"
            elif key == "d":
                right_until = now + args.hold
                last_key = "D"
            elif key == "w":
                throttle = min(args.max_throttle, throttle + THROTTLE_STEP)
                last_key = "W"
            elif key == "s":
                throttle = max(0.0, throttle - THROTTLE_STEP)
                last_key = "S"
            elif key == " ":
                throttle, steering, target = 0.0, 0.0, 0.0
                left_until = right_until = 0.0
                last_key = "SPACE"
            elif key == "r":
                recording = not recording
                if recording:
                    recorder.start()
                last_key = "R"
            elif key == "q":
                return

        # --- steering: ramp towards the target ---
        if now < left_until and now >= right_until:
            target = -1.0 * sign
        elif now < right_until and now >= left_until:
            target = 1.0 * sign
        else:
            target = 0.0  # nothing held, or both held: go straight
        max_move = args.steer_rate * dt
        steering += max(-max_move, min(max_move, target - steering))
        steering = max(-1.0, min(1.0, steering))

        throttle = max(0.0, min(args.max_throttle, throttle))
        car.steering = steering
        car.throttle = throttle

        # --- record ---
        if recording and throttle > 0 and now - last_save >= SAVE_EVERY:
            last_save = now
            recorder.save(image, steering, throttle)

        # --- show state ---
        held = ("A " if now < left_until else "- ") + ("D" if now < right_until else "-")
        lines = [
            "JetRacer teleop    A/D steer  W/S throttle  SPACE stop  R record  Q quit",
            "",
            "keys held:  {}      last key: {}".format(held, last_key),
            "steering:   {:+.2f}  (target {:+.2f})".format(steering, target),
            "throttle:   {:.2f}   (cap {:.2f})".format(throttle, args.max_throttle),
            "recording:  {}".format("ON  <<<" if recording else "off"),
            "frames:     {}".format(recorder.count),
            "session:    {}".format(recorder.session_dir or "(starts when you press R)"),
            "loop:       {:.0f} Hz".format(1.0 / dt if dt > 0 else 0.0),
        ]
        stdscr.erase()
        for i, line in enumerate(lines):
            try:
                stdscr.addstr(i, 0, line)
            except curses.error:
                pass  # terminal too small, skip the line
        stdscr.refresh()


def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(1))  # so finally still runs

    car = NvidiaRacecar()
    camera = None
    recorder = Recorder(args.out)
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
        print("Throttle cap {} (hard limit {}). Starting camera...".format(args.max_throttle, HARD_LIMIT))
        camera = CSICamera(width=320, height=240,
                           capture_width=args.capture_width, capture_height=args.capture_height,
                           capture_fps=30)
        curses.wrapper(run, args, car, camera, recorder)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        for _ in range(2):  # try twice in case the first write fails
            try:
                car.throttle = 0.0
                car.steering = 0.0
                break
            except Exception as e:
                print("WARNING: failed to stop the car: {}".format(e))
        if camera is not None:
            try:
                camera.cap.release()
            except Exception:
                pass
        recorder.close()
        print("Throttle set to 0.")
        if recorder.session_dir is not None:
            print("Saved {} frames to {}".format(recorder.count, recorder.session_dir))


if __name__ == "__main__":
    main()
