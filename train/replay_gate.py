"""Replay recorded sessions through the two-line-only safe-stop gate (offline, images only).

    python train/replay_gate.py [session ...]

Answers: if the car had been driven only on two confident boundaries, how often
would it have been allowed to drive, how often would the stop have latched, and
how long would it stay stopped? The operator re-arm is simulated as instant once
the gate allows it (best case), so real downtime would be longer.
"""
import glob
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "car"))
from lane_detection import detect_lane  # noqa: E402
from lane_observation import SafeStopGate  # noqa: E402
from session_labels import read_label_rows  # noqa: E402


def replay(session):
    rows = sorted(read_label_rows(session), key=lambda r: float(r["timestamp"]))
    gate = SafeStopGate()
    n = drive = latches = 0
    runs, run = [], 0           # lengths (frames) of consecutive not-allowed stretches
    was_latched = True
    for r in rows:
        img = cv2.imread(os.path.join(session, r["filename"]))
        t = float(r["timestamp"])
        d = gate.update(detect_lane(img), t)
        if d.latched and not was_latched:
            latches += 1
        was_latched = d.latched
        if d.latched:
            gate.rearm()                         # best case: operator re-arms as soon as allowed
            was_latched = gate.latched
        n += 1
        if d.drive_allowed:
            drive += 1
            if run:
                runs.append(run)
            run = 0
        else:
            run += 1
    if run:
        runs.append(run)
    return n, drive, latches, runs


def main():
    sessions = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "data", "2*")))
    print("%-18s %7s %12s %8s %22s" % ("session", "frames", "drive allowed", "latches", "stop stretches (frames)"))
    tot = [0, 0, 0]
    for s in sessions:
        n, drive, latches, runs = replay(s)
        tot = [tot[0] + n, tot[1] + drive, tot[2] + latches]
        print("%-18s %7d %11.1f%% %8d   median %s, longest %s"
              % (os.path.basename(s), n, 100.0 * drive / n, latches,
                 sorted(runs)[len(runs) // 2] if runs else 0, max(runs) if runs else 0))
    print("%-18s %7d %11.1f%% %8d" % ("ALL", tot[0], 100.0 * tot[1] / tot[0], tot[2]))
    print("(~8 Hz recordings: 1 frame ~ 0.13 s. Best case: instant operator re-arm; on a real run every latch "
          "needs a deliberate operator action.)")


if __name__ == "__main__":
    main()
