"""Read a recorded session's labels.csv, leaving out discarded segments.

car/teleop_record.py already removes a discarded segment's rows from labels.csv, but
any segment listed in labels_discarded.csv is skipped here as well, in case the car
script was killed half-way through discarding. Sessions recorded before segments
existed (no "segment" column) load unchanged.
"""
import csv
import os


def discarded_segments(session):
    path = os.path.join(session, "labels_discarded.csv")
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {r["segment"] for r in csv.DictReader(f) if r.get("segment")}


def read_label_rows(session, labels="labels.csv"):
    """Rows (dicts) of the session's labels file, without discarded segments."""
    skip = discarded_segments(session)
    with open(os.path.join(session, labels), newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("segment") not in skip]  # no column -> None -> kept
