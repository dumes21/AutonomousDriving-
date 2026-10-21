import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "car"))
from lane_detection import LaneResult, STATUS_OK, STATUS_ONE_LINE  # noqa: E402
from lane_observation import (InvalidObservation, SafeStopGate,  # noqa: E402
                              observation_from_result)


def good(conf=0.8):
    r = LaneResult()
    r.status, r.valid, r.confidence = STATUS_OK, True, conf
    r.offset_norm, r.heading_slope = 0.1, -0.2
    return r


def bad(status=STATUS_ONE_LINE):
    r = LaneResult()
    r.status = status
    return r


def armed_gate(**kw):
    g = SafeStopGate(**kw)
    t = 0.0
    for _ in range(g.rearm_valid_frames):
        g.update(good(), t)
        t += 0.1
    assert g.rearm()
    return g, t


def test_invalid_observation_has_no_numbers_and_never_zero():
    obs = observation_from_result(bad(), 0.0)
    assert not obs.valid and obs.offset_norm is None and obs.heading_slope is None
    with pytest.raises(InvalidObservation):
        obs.vector()


def test_valid_only_for_two_confident_lines():
    assert observation_from_result(good(0.8), 0.0).valid
    low = observation_from_result(good(0.4), 0.0, min_confidence=0.5)
    assert not low.valid and low.status == "low_confidence"
    for status in ("one_line", "ambiguous", "insufficient", "no_line"):
        assert not observation_from_result(bad(status), 0.0).valid


def test_starts_stopped_until_armed():
    g = SafeStopGate()
    assert not g.update(good(), 0.0).drive_allowed
    assert not g.rearm()                       # not enough valid frames yet
    g, t = armed_gate()
    assert g.update(good(), t).drive_allowed


def test_single_invalid_frame_stops_that_frame_without_latching():
    g, t = armed_gate()
    d = g.update(bad(), t)
    assert not d.drive_allowed and not d.latched and d.observation.offset_norm is None
    assert g.update(good(), t + 0.1).drive_allowed     # resumes on a fresh valid frame


def test_persistent_invalid_latches_and_needs_operator_rearm():
    g, t = armed_gate()
    for i in range(3):
        d = g.update(bad(), t + 0.05 * i)
    assert d.latched and not d.drive_allowed
    for i in range(20):                                # valid frames alone do not resume driving
        d = g.update(good(), t + 1.0 + 0.05 * i)
        assert not d.drive_allowed
    assert g.rearm()
    assert g.update(good(), t + 3.0).drive_allowed


def test_rearm_refused_without_consecutive_valid_frames():
    g, t = armed_gate()
    for i in range(3):
        g.update(bad(), t + 0.05 * i)
    g.update(good(), t + 1.0)
    assert not g.rearm()
    assert g.latched


def test_slow_invalid_period_latches_by_time():
    g, t = armed_gate()
    g.update(bad(), t)
    d = g.update(bad(), t + 0.35)                      # only 2 frames, but 0.35 s
    assert d.latched


def test_flickering_frames_latch():
    g, t = armed_gate()
    latched = False
    for i in range(12):
        d = g.update(good() if i % 2 else bad(), t + 0.1 * i)
        latched = latched or d.latched
    assert latched


def test_stale_camera_stops_driving():
    g, t = armed_gate()
    assert not g.check_stale(t + 0.1).drive_allowed    # no new frame: never reuse the old measurement
    d = g.check_stale(t + 0.5)
    assert not d.drive_allowed and d.observation.status == "stale"


def test_real_recorded_frame_gives_valid_observation():
    cv2 = pytest.importorskip("cv2")
    from lane_detection import detect_lane
    path = os.path.join(ROOT, "data", "20261013_001444", "000000.jpg")
    if not os.path.exists(path):
        pytest.skip("recording not present")
    obs = observation_from_result(detect_lane(cv2.imread(path)), 0.0)
    assert obs.valid and len(obs.vector()) == 2
