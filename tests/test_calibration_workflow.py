"""End-to-end rehearsal of the calibration-day workflow on SYNTHETIC images (no camera, no hardware).

A fake camera (numbers below are test fixtures, NOT the real JetRacer camera) renders orange lane tape, white strips
across the lane and a car at known poses. The real command-line tools are then run exactly as in the checklist:
set -> find-strips -> fit-depth -> fit-axis -> show -> validate_calibration, and the known fixture values must come back.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CAL = os.path.join(ROOT, "perception", "calibrate.py")
VAL = os.path.join(ROOT, "perception", "validate_calibration.py")
INITIAL = os.path.join(ROOT, "perception", "calibration", "initial_fit_from_recordings.json")

Y_H, K, F, W = 45.0, 0.124, 18.6, 0.26         # fixture: horizon row, k, F (f = F/K = 150 px), lane width
AX0, AXS, L_CAM = 163.0, 0.02, 0.11             # fixture vehicle axis (column at row 140, px/row), camera-to-rear-axle
FLOOR, TAPE, WHITE = (176, 150, 178), (100, 125, 178), (235, 232, 240)


def render(offset=0.0, heading=0.0, strips=(), tape_w=0.025, strip_t=0.02):
    """Camera image of a straight lane seen by a car `offset` m LEFT of the centre, pointing `heading` rad LEFT."""
    img = np.full((240, 320, 3), FLOOR, np.uint8)
    img[:int(Y_H) + 3] = (60, 60, 60)
    phi = -heading
    for D in strips:                              # white tape across the lane, D metres ahead of the lens
        y0, y1 = Y_H + F / (D + strip_t / 2), Y_H + F / (D - strip_t / 2)
        for y in range(int(math.ceil(y0)), int(y1) + 1):
            s = K / (y - Y_H)
            axis = AX0 + AXS * (y - 140)
            img[y, max(0, int(axis - 0.17 / s)):min(320, int(axis + 0.17 / s))] = WHITE
    for y in range(int(Y_H) + 4, 240):
        s = K / (y - Y_H)
        x_f = L_CAM + F / (y - Y_H)
        axis = AX0 + AXS * (y - 140)
        yc = -offset / math.cos(phi) + math.tan(phi) * x_f
        for lat in (yc + 0.5 * W / math.cos(phi), yc - 0.5 * W / math.cos(phi)):
            xc = axis - lat / s
            half = 0.5 * tape_w / s / math.cos(phi)
            img[y, max(0, int(round(xc - half))):max(0, int(round(xc + half)))] = TAPE
    return cv2.GaussianBlur(img, (3, 3), 0)


def run(args, ok=True):
    p = subprocess.run([sys.executable] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if ok:
        assert p.returncode == 0, p.stdout
    return p


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    d = tmp_path_factory.mktemp("calday")
    cv2.imwrite(str(d / "strips.jpg"), render(strips=(0.60, 0.45, 0.30)))
    for i in range(3):
        cv2.imwrite(str(d / ("centred_%d.jpg" % i)), render(0.0, 0.0))
    truth = ["# image,offset_cm,yaw_deg"]
    for name, off, yaw in (("val_L25", 2.5, 0), ("val_R25", -2.5, 0), ("val_L50", 5, 0), ("val_R50", -5, 0),
                           ("val_yawL5", 0, 5), ("val_yawR5", 0, -5), ("val_yawL10", 0, 10), ("val_yawR10", 0, -10)):
        cv2.imwrite(str(d / (name + "_20261022_101500_000.jpg")), render(off / 100.0, math.radians(yaw)))
        truth.append("%s*.jpg,%+.1f,%+.1f" % (name, off, yaw))
    (d / "truth.csv").write_text("\n".join(truth) + "\n")
    return d


def test_full_workflow_recovers_the_fixture_camera_and_passes_validation(session):
    d = str(session)
    c1, c2, cal = (os.path.join(d, n) for n in ("c1.json", "c2.json", "CAL.json"))
    run([CAL, "set", INITIAL, "--out", c1, "--param", "camera_forward_of_rear_axle_m", "--value", "0.11", "--status", "measured"])
    p = run([CAL, "find-strips", os.path.join(d, "strips.jpg"), "--expect", "3", "--annotate", os.path.join(d, "strips_marked.png")])
    assert os.path.exists(os.path.join(d, "strips_marked.png"))
    rows = [l for l in p.stdout.splitlines() if l.startswith("rows, far -> near")][0].split(":")[1].split()
    assert len(rows) == 3 and [float(r) for r in rows] == sorted(float(r) for r in rows)
    p = run([CAL, "fit-depth", c1, "--rows"] + rows + ["--distances", "0.60", "0.45", "0.30", "--out", c2])
    run([CAL, "fit-axis", c2] + [os.path.join(d, "centred_%d.jpg" % i) for i in range(3)] + ["--out", cal])
    p = run([CAL, "show", cal])
    assert "nothing" in p.stdout and "none" in p.stdout
    from_json = __import__("json").load(open(cal))["parameters"]
    assert from_json["depth_scale_F"]["value"] == pytest.approx(F, rel=0.06)
    assert from_json["horizon_row_px"]["value"] == pytest.approx(Y_H, abs=3.0)
    assert from_json["axis_x_ref_px"]["value"] == pytest.approx(AX0, abs=1.5)
    assert from_json["axis_slope"]["value"] == pytest.approx(AXS, abs=0.01)
    assert from_json["camera_forward_of_rear_axle_m"]["status"] == "measured"
    p = run([VAL, cal, os.path.join(d, "truth.csv"), "--csv", os.path.join(d, "results.csv")], ok=False)
    assert p.returncode == 0, p.stdout          # all errors within 1 cm / 2 deg on a perfect pinhole camera
    assert "PASS" in p.stdout


def test_tools_refuse_to_overwrite(session):
    d = str(session)
    p = run([CAL, "set", INITIAL, "--out", os.path.join(d, "c1.json"), "--param", "axis_slope", "--value", "0", "--status", "measured"], ok=False)
    assert p.returncode != 0 and "already exists" in p.stdout
    p = run([VAL, os.path.join(d, "CAL.json"), os.path.join(d, "truth.csv"), "--csv", os.path.join(d, "results.csv")], ok=False)
    assert p.returncode != 0 and "already exists" in p.stdout


def test_validation_catches_a_wrong_sign_and_a_wrong_axis(session):
    """A mirrored truth file (left/right swapped) must FAIL with a sign message, not pass quietly."""
    d = session
    lines = ["# mirrored on purpose"]
    for l in (d / "truth.csv").read_text().splitlines()[1:]:
        n, o, y = l.split(",")
        lines.append("%s,%s,%s" % (n, -float(o), -float(y)))
    (d / "truth_mirrored.csv").write_text("\n".join(lines) + "\n")
    p = run([VAL, str(d / "CAL.json"), str(d / "truth_mirrored.csv")], ok=False)
    assert p.returncode == 1 and "WRONG SIGN" in p.stdout


def test_incomplete_calibration_cannot_be_validated(session):
    p = run([VAL, INITIAL, os.path.join(str(session), "truth.csv")], ok=False)
    assert p.returncode != 0 and "incomplete" in p.stdout


def test_fit_depth_rejects_mismatched_lengths_and_find_strips_reports_wrong_counts(session):
    d = str(session)
    p = run([CAL, "fit-depth", os.path.join(d, "c1.json"), "--rows", "70", "90", "110", "--distances", "0.6", "0.3", "--out",
             os.path.join(d, "bad.json")], ok=False)
    assert p.returncode != 0 and "one distance per row" in p.stdout and not os.path.exists(os.path.join(d, "bad.json"))
    p = run([CAL, "find-strips", os.path.join(d, "centred_0.jpg"), "--expect", "3"], ok=False)     # no strips in this image
    assert p.returncode != 0 and "found 0 strips" in p.stdout


def test_capture_page_labels_files_safely_and_never_overwrites(tmp_path, monkeypatch):
    """The capture web tool's FrameStore (imported with a fake jetcam): label -> file name, sanitised, no overwrite."""
    fake = tmp_path / "fake" / "jetcam"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("")
    (fake / "csi_camera.py").write_text("class CSICamera(object):\n    pass\n")
    monkeypatch.syspath_prepend(str(tmp_path / "fake"))
    monkeypatch.syspath_prepend(os.path.join(ROOT, "car"))
    import importlib
    for m in ("jetcam", "jetcam.csi_camera", "camera_capture_web", "mjpeg_stream"):
        monkeypatch.delitem(sys.modules, m, raising=False)      # restored afterwards: other tests assert jetcam is never imported
    cw = importlib.import_module("camera_capture_web")
    store = cw.FrameStore(str(tmp_path / "out"))
    with pytest.raises(RuntimeError):
        store.save("L25")                                  # no frame yet
    store.update(np.zeros((240, 320, 3), np.uint8))
    a = store.save("L25")
    b = store.save("../../etc/passwd L25")                 # path characters are stripped
    c = store.save("")
    assert os.path.basename(a).startswith("L25_") and os.path.dirname(b) == str(tmp_path / "out")
    assert "passwd" in os.path.basename(b) and "/" not in os.path.basename(b) and ".." not in os.path.basename(b)
    assert os.path.basename(c).startswith("capture_")
    assert len({a, b, c}) == 3 and cv2.imread(a).shape == (240, 320, 3)
    for m in ("jetcam", "jetcam.csi_camera", "camera_capture_web", "mjpeg_stream"):
        sys.modules.pop(m, None)                                # do not leave the fake camera module behind
