import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "perception"))
import boundary_detector as B  # noqa: E402

FLOOR = (176, 150, 178)      # pinkish floor (BGR), like the recorded frames
TAPE = (100, 125, 178)       # orange tape under the pink cast (BGR)


def synthetic(left=True, right=True):
    img = np.full((240, 320, 3), FLOOR, np.uint8)
    img[:75] = (60, 60, 60)
    if left:
        cv2.line(img, (110, 100), (30, 225), TAPE, 14)
    if right:
        cv2.line(img, (215, 100), (290, 225), TAPE, 14)
    return cv2.GaussianBlur(img, (3, 3), 0)


@pytest.mark.parametrize("method", ["lab", "hsv"])
def test_two_lines_give_a_valid_result_with_centre_between_the_boundaries(method):
    res = B.detect(synthetic(), method)
    assert res.valid and res.status == "ok"
    assert res.vehicle_centre_x_px is None                 # never assumed
    for q in res.rows:
        assert q["left_x"] < q["centre_x"] < q["right_x"]
    assert res.rows[0]["left_x"] < res.image_centre_x_px < res.rows[0]["right_x"]


@pytest.mark.parametrize("method", ["lab", "hsv"])
@pytest.mark.parametrize("left,right", [(True, False), (False, True), (False, False)])
def test_missing_boundary_gives_invalid_result_without_coordinates(method, left, right):
    res = B.detect(synthetic(left, right), method)
    assert not res.valid
    assert res.rows == [] and res.centre_fit is None
    assert res.centre_minus_image_centre_px is None and res.lane_width_px is None
    assert res.warnings and res.warnings[0].startswith("INVALID")


def test_debug_image_is_produced_for_valid_and_invalid_results(tmp_path):
    for img in (synthetic(), synthetic(True, False)):
        out = B.render_debug(img, B.detect(img, "lab"))
        assert out.shape[0] == 480 and out.shape[1] == 1280


def test_hsv_hue_window_wraps_around_red():
    # tape pixels whose hue is on the red side of pink (172..179) must be accepted, not only hue 0..22
    cfg = B.HSVConfig(s_tophat_min=0)
    img = np.zeros((240, 320, 3), np.uint8)
    img[100:140, 100:140] = cv2.cvtColor(np.uint8([[[176, 120, 190]]]), cv2.COLOR_HSV2BGR)
    m = B.hsv_mask(img, cfg, B.ld.LaneConfig())
    assert m[120, 120] > 0
