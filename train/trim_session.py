"""Drop the end of every recorded segment, where the car has usually left the track.

    python train/trim_session.py data/20261013_003714 data/20261013_005718
    python train/trim_session.py data/*/ --seconds 1.5

For each session, the last --seconds of every segment are removed and the rest is written
to labels_clean.csv next to labels.csv. labels.csv, labels_discarded.csv and the images are
never modified. Discarded segments are left out. A session recorded before segments existed
(no "segment" column) is treated as one segment.
"""
import argparse
import csv
import os
from collections import OrderedDict

from session_labels import read_label_rows


def trim_rows(rows, seconds):
    """Keep the rows of each segment whose timestamp is more than `seconds` before the
    segment's last timestamp. Returns the kept rows in time order."""
    segments = OrderedDict()
    for r in sorted(rows, key=lambda r: float(r["timestamp"])):
        segments.setdefault(r.get("segment"), []).append(r)  # no column -> one segment (None)
    kept = []
    for seg_rows in segments.values():
        end = float(seg_rows[-1]["timestamp"])
        kept += [r for r in seg_rows if float(r["timestamp"]) < end - seconds]
    kept.sort(key=lambda r: float(r["timestamp"]))
    return kept


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sessions", nargs="+", help="session folders containing labels.csv")
    p.add_argument("--seconds", type=float, default=1.5, help="seconds to drop at the end of each segment")
    p.add_argument("--out-name", default="labels_clean.csv")
    args = p.parse_args()
    if args.out_name in ("labels.csv", "labels_discarded.csv"):
        p.error("refusing to overwrite " + args.out_name)

    total_before = total_after = 0
    for session in args.sessions:
        session = os.path.normpath(session)
        if os.path.basename(session) in ("discarded", "trimmed"):
            continue
        src = os.path.join(session, "labels.csv")
        if not os.path.isfile(src):
            print("%s: no labels.csv, skipped" % session)
            continue
        with open(src, newline="") as f:
            fields = csv.DictReader(f).fieldnames
        rows = read_label_rows(session)
        kept = trim_rows(rows, args.seconds)
        n_seg = len({r.get("segment") for r in rows})
        with open(os.path.join(session, args.out_name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(kept)
        print("%s: %d segment(s), %d frames -> %d after trimming (-%d)" % (
            session, n_seg, len(rows), len(kept), len(rows) - len(kept)))
        total_before += len(rows)
        total_after += len(kept)
    print("total: %d frames -> %d" % (total_before, total_after))


if __name__ == "__main__":
    main()
