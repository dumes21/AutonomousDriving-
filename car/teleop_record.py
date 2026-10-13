"""Drive the JetRacer from the keyboard over SSH and record training data
(run ON the car, Python 3.6).

Keys:   A / D  steer left / right: each press nudges the steering target by 0.1
               and it STAYS there (latched). The real steering ramps smoothly to it.
        C      recentre steering (target 0)
        W / S  more / less throttle (small steps, capped)
        SPACE  stop (throttle and steering to 0)
        R      start / stop recording
        Q      quit

DEADMAN: if no key arrives for 0.5 s, throttle drops to 0 (steering is left as
it is) and you must press W again. So hold W (key repeat) or keep tapping keys
while driving. On SIGTERM / SIGHUP (e.g. the SSH connection drops) throttle
and steering are zeroed before the script exits.

While recording AND throttle > 0, a JPEG is saved about every 0.1 s into
data/<session timestamp>/ and a row (filename, steering, throttle, timestamp)
is appended to labels.csv in that folder. The recorded throttle is the
forward amount (0..cap), whatever --invert-throttle is.

    python3 car/teleop_record.py --invert-steering --invert-throttle
    python3 car/teleop_record.py --max-throttle 0.35 --steer-mode hold

Tips: terminals send no key RELEASES, and only the most recently pressed key
repeats. Set a short key-repeat delay on your laptop, otherwise the 0.5 s
deadman can trip between the first W press and the repeat starting.
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
STEER_NUDGE = 0.1  # latched mode: change of the steering target per A/D press
SAVE_EVERY = 0.1   # seconds between saved frames


def parse_args():
    p = argparse.ArgumentParser(description="Keyboard teleop + data recorder for the JetRacer")
    p.add_argument("--max-throttle", type=float, default=0.3,
                   help="throttle cap (default 0.3, refuses > {})".format(HARD_LIMIT))
    p.add_argument("--invert-throttle", action="store_true",
                   help="send the NEGATED throttle to the car (use if W drives the car backwards)")
    p.add_argument("--invert-steering", action="store_true",
                   help="swap the meaning of A and D if the car steers the wrong way")
    p.add_argument("--steer-mode", choices=["latched", "hold"], default="latched",
                   help="latched (default): A/D nudge the target and it stays. "
                        "hold: steer only while A/D repeat, then return to centre")
    p.add_argument("--steer-rate", type=float, default=2.5,
                   help="steering change per second towards the target (default 2.5 = full lock in 0.4 s)")
    p.add_argument("--hold", type=float, default=0.3,
                   help="hold mode only: seconds a steering key counts as held after its last press")
    p.add_argument("--deadman", type=float, default=0.5,
                   help="seconds without any key before throttle is set to 0 (default 0.5)")
    p.add_argument("--out", default=DATA_DIR, help="folder that session folders go into (default <repo>/data)")
    p.add_argument("--steering-gain", type=float, default=None, help="leave unset to keep the library default")
    p.add_argument("--steering-offset", type=float, default=None, help="leave unset to keep the library default")
    p.add_argument("--throttle-gain", type=float, default=None, help="leave unset to keep the library default")
    p.add_argument("--capture-width", type=int, default=1640)
    p.add_argument("--capture-height", type=int, default=1232)
    args = p.parse_args()
    if args.max_throttle <= 0 or args.max_throttle > HARD_LIMIT:
        p.error("--max-throttle must be in (0, {}], got {}".format(HARD_LIMIT, args.max_throttle))
    if args.deadman <= 0:
        p.error("--deadman must be > 0")
    return args


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

    steering = 0.0                      # what is sent to the car
    target = 0.0                        # where steering is heading
    forward = 0.0                       # throttle setpoint, 0..cap (positive = forward)
    left_until, right_until = 0.0, 0.0  # hold mode: time until which A / D count as held
    last_key_time = 0.0                 # for the deadman
    recording = False
    last_save = 0.0
    last_key = ""
    t_prev = time.time()
    sign = -1.0 if args.invert_steering else 1.0  # A = left = -1 on the car, unless inverted
    latched = args.steer_mode == "latched"

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
            if ch == curses.KEY_RESIZE:
                continue  # terminal resized: not a key press
            last_key_time = now  # ANY key keeps the deadman alive
            key = chr(ch).lower() if 0 <= ch < 256 else ""
            if key == "a":
                if latched:
                    target = round(max(-1.0, min(1.0, target - sign * STEER_NUDGE)), 2)
                else:
                    left_until = now + args.hold
                last_key = "A"
            elif key == "d":
                if latched:
                    target = round(max(-1.0, min(1.0, target + sign * STEER_NUDGE)), 2)
                else:
                    right_until = now + args.hold
                last_key = "D"
            elif key == "c":
                target = 0.0
                last_key = "C"
            elif key == "w":
                forward = min(args.max_throttle, forward + THROTTLE_STEP)
                last_key = "W"
            elif key == "s":
                forward = max(0.0, forward - THROTTLE_STEP)
                last_key = "S"
            elif key == " ":
                forward, steering, target = 0.0, 0.0, 0.0
                left_until = right_until = 0.0
                last_key = "SPACE"
            elif key == "r":
                recording = not recording
                if recording:
                    recorder.start()
                last_key = "R"
            elif key == "q":
                return

        # --- deadman: no key for a while -> throttle 0 (steering stays), press W to start again ---
        deadman_left = args.deadman - (now - last_key_time)
        if deadman_left <= 0:
            forward = 0.0

        # --- steering: ramp towards the target ---
        if not latched:
            if now < left_until and now >= right_until:
                target = -1.0 * sign
            elif now < right_until and now >= left_until:
                target = 1.0 * sign
            else:
                target = 0.0  # nothing held, or both held: go straight
        max_move = args.steer_rate * dt
        steering += max(-max_move, min(max_move, target - steering))
        steering = max(-1.0, min(1.0, steering))

        forward = max(0.0, min(args.max_throttle, forward))
        car.steering = steering
        car.throttle = throttle_for_car(forward, args.max_throttle, args.invert_throttle)

        # --- record ---
        if recording and forward > 0 and now - last_save >= SAVE_EVERY:
            last_save = now
            recorder.save(image, steering, forward)

        # --- show state ---
        if forward > 0:
            deadman_text = "ok ({:.1f} s left)".format(deadman_left)
        else:
            deadman_text = "throttle 0 - press W" if deadman_left <= 0 else "ok"
        lines = [
            "JetRacer teleop   A/D steer  C centre  W/S throttle  SPACE stop  R record  Q quit",
            "",
            "last key:   {}".format(last_key),
            "steering:   actual {:+.2f}   target {:+.2f}   ({})".format(steering, target, args.steer_mode),
            "throttle:   {:.2f} forward   (cap {:.2f}, sent {:+.2f})".format(
                forward, args.max_throttle, throttle_for_car(forward, args.max_throttle, args.invert_throttle)),
            "deadman:    {}".format(deadman_text),
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

    car = NvidiaRacecar()
    camera = None
    recorder = Recorder(args.out)

    def stop_and_exit(signum, frame):
        stop_car(car)  # zero everything first, then leave (the finally block below also runs)
        sys.exit(1)

    signal.signal(signal.SIGTERM, stop_and_exit)
    signal.signal(signal.SIGHUP, stop_and_exit)  # e.g. the SSH connection dropped
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
        print("Throttle cap {} (hard limit {}), invert_throttle={}, invert_steering={}, steer mode {}".format(
            args.max_throttle, HARD_LIMIT, args.invert_throttle, args.invert_steering, args.steer_mode))
        print("Starting camera...")
        camera = CSICamera(width=320, height=240,
                           capture_width=args.capture_width, capture_height=args.capture_height,
                           capture_fps=30)
        curses.wrapper(run, args, car, camera, recorder)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        stop_car(car)
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
