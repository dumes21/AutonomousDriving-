"""Check that a recorded session looks sane before training on it.

    python train/inspect_data.py data/20261012_141500

Prints counts and label statistics, and saves two PNGs into logs/:
a steering histogram and a grid of sample frames with their labels.
"""
import argparse
import csv
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def read_labels(session):
    """Return a list of (image_path, steering, throttle, timestamp), sorted by time."""
    rows = []
    with open(os.path.join(session, "labels.csv"), newline="") as f:
        for r in csv.DictReader(f):
            rows.append((os.path.join(session, r["filename"]), float(r["steering"]),
                         float(r["throttle"]), float(r["timestamp"])))
    rows.sort(key=lambda r: r[3])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="session folder containing labels.csv and the JPEGs")
    parser.add_argument("--samples", type=int, default=16, help="frames in the sample grid")
    parser.add_argument("--out", default=os.path.join(ROOT, "logs"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    name = os.path.basename(os.path.normpath(args.session))

    rows = read_labels(args.session)
    missing = [r[0] for r in rows if not os.path.isfile(r[0])]
    rows = [r for r in rows if os.path.isfile(r[0])]
    if not rows:
        raise SystemExit("No usable frames found in " + args.session)
    steer = np.array([r[1] for r in rows])
    thr = np.array([r[2] for r in rows])
    t = np.array([r[3] for r in rows])

    print("session:        ", args.session)
    print("frames:         ", len(rows), "(%d listed in csv but missing on disk)" % len(missing))
    print("duration:        %.1f s, about %.1f frames/s" % (t[-1] - t[0], len(rows) / max(t[-1] - t[0], 1e-6)))
    print("steering:        mean %+.3f  std %.3f  min %+.2f  max %+.2f" % (steer.mean(), steer.std(), steer.min(), steer.max()))
    print("throttle:        mean %.3f  min %.2f  max %.2f" % (thr.mean(), thr.min(), thr.max()))
    print("near-straight:   %.0f%% of frames have |steering| < 0.1" % (100 * np.mean(np.abs(steer) < 0.1)))
    print("left / right:    %d frames steering left (<-0.1), %d right (>0.1)" % (
        np.sum(steer < -0.1), np.sum(steer > 0.1)))
    gaps = np.diff(t)
    print("time gaps > 1 s: %d (pauses in recording)" % np.sum(gaps > 1.0))
    img = cv2.imread(rows[0][0])
    print("image size:      %dx%d" % (img.shape[1], img.shape[0]))

    plt.figure(figsize=(6, 3.5))
    plt.hist(steer, bins=41, range=(-1, 1))
    plt.xlabel("steering")
    plt.ylabel("frames")
    plt.title("Steering histogram: %s (n=%d)" % (name, len(rows)))
    plt.tight_layout()
    hist_path = os.path.join(args.out, "steering_hist_%s.png" % name)
    plt.savefig(hist_path, dpi=120)
    plt.close()

    # Grid of frames spread evenly across the session
    n = min(args.samples, len(rows))
    picks = np.linspace(0, len(rows) - 1, n).astype(int)
    cols = 4
    n_rows = int(np.ceil(n / float(cols)))
    fig, axes = plt.subplots(n_rows, cols, figsize=(3 * cols, 2.4 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), picks):
        path, s, th, _ = rows[i]
        ax.imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
        ax.set_title("#%d steer %+.2f thr %.2f" % (i, s, th), fontsize=8)
    plt.tight_layout()
    grid_path = os.path.join(args.out, "samples_%s.png" % name)
    plt.savefig(grid_path, dpi=100)
    plt.close()
    print("saved:          ", hist_path)
    print("saved:          ", grid_path)


if __name__ == "__main__":
    main()
