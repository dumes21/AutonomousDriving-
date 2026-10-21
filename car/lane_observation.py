"""Two-line-only lane observation + safe-stop gate (no hardware access, Python 3.6 compatible).

Sits between car/lane_detection.py and a policy. Rules:
  * A measurement exists ONLY when the detector saw BOTH boundaries and is
    confident (LaneResult.valid and confidence >= min_confidence).
  * Otherwise the observation is explicitly INVALID. It carries no numbers, and
    asking for its vector raises InvalidObservation. Missing values are never
    replaced by 0 (0 would mean "perfectly centred"), never held over from an
    older frame, and never filled in from a single visible line.
  * Whenever the observation is invalid the gate says drive_allowed = False,
    and the caller must command throttle 0 for that frame. If invalid frames
    persist, the gate LATCHES the stop: driving stays disallowed until an
    operator calls rearm() AND several consecutive valid frames have been seen.

Usage in a control loop (the caller owns the motors):

    gate = SafeStopGate()
    while True:
        frame = camera.read()
        if frame is None:                      # camera stalled / failed
            decision = gate.check_stale(time.time())
        else:
            decision = gate.update(detect_lane(frame), time.time())
        if not decision.drive_allowed:
            car.throttle = 0.0                 # safe stop. Do not run the policy.
            continue
        offset, heading = decision.observation.vector()
        ...policy(offset, heading, last_steering)...

The offset/heading numbers are IMAGE-SPACE proxies (see docs/TWO_LINE_PIPELINE.md);
they must be calibrated before a simulator-trained policy can use them.
"""
import collections

from lane_detection import STATUS_OK


class InvalidObservation(Exception):
    """Raised when numbers are requested from an invalid observation."""


class LaneObservation(object):
    """Immutable-by-convention result of one frame."""

    def __init__(self, valid, status, reason, confidence, offset_norm, heading_slope, timestamp):
        self.valid = valid
        self.status = status              # detector status, or "stale" / "low_confidence"
        self.reason = reason              # human-readable, why invalid ("" when valid)
        self.confidence = confidence
        self.offset_norm = offset_norm    # None unless valid
        self.heading_slope = heading_slope  # None unless valid
        self.timestamp = timestamp

    def vector(self):
        """(offset_norm, heading_slope). Raises InvalidObservation if not valid."""
        if not self.valid:
            raise InvalidObservation(self.reason)
        return (self.offset_norm, self.heading_slope)

    @classmethod
    def invalid(cls, status, reason, timestamp, confidence=0.0):
        return cls(False, status, reason, confidence, None, None, timestamp)


def observation_from_result(res, now, min_confidence=0.5):
    """LaneResult -> LaneObservation. Valid only for two confident boundaries."""
    if res.valid and res.status == STATUS_OK and res.offset_norm is not None \
            and res.heading_slope is not None:
        if res.confidence >= min_confidence:
            return LaneObservation(True, STATUS_OK, "", res.confidence, res.offset_norm,
                                   res.heading_slope, now)
        return LaneObservation.invalid(
            "low_confidence", "confidence %.2f < %.2f" % (res.confidence, min_confidence), now, res.confidence)
    return LaneObservation.invalid(res.status, "boundaries not both detected (%s)" % res.status, now,
                                   res.confidence)


class Decision(object):
    def __init__(self, drive_allowed, observation, latched, reason):
        self.drive_allowed = drive_allowed
        self.observation = observation    # LaneObservation (valid iff drive_allowed)
        self.latched = latched            # True: stop persists until rearm()
        self.reason = reason              # why driving is not allowed ("" when allowed)


class SafeStopGate(object):
    """Turns a stream of observations into drive / stop decisions."""

    def __init__(self, min_confidence=0.5, latch_after_s=0.3, latch_after_frames=3,
                 window_s=1.0, window_invalid_fraction=0.5, window_min_frames=4,
                 rearm_valid_frames=5, stale_after_s=0.25):
        self.min_confidence = min_confidence
        self.latch_after_s = latch_after_s              # continuous invalid time that latches
        self.latch_after_frames = latch_after_frames    # ... or this many invalid frames in a row
        self.window_s = window_s                        # flicker rule: too many invalid in the window
        self.window_invalid_fraction = window_invalid_fraction
        self.window_min_frames = window_min_frames
        self.rearm_valid_frames = rearm_valid_frames    # consecutive valid frames needed to rearm
        self.stale_after_s = stale_after_s              # no frame for this long = invalid
        self.latched = True                              # start stopped: the operator must arm explicitly
        self.latch_reason = "not armed"
        self._invalid_since = None
        self._invalid_run = 0
        self._valid_run = 0
        self._recent = collections.deque()               # (time, valid)
        self._last_frame_time = None

    # --- inputs -----------------------------------------------------------------------------
    def update(self, lane_result, now):
        """Feed one detector result. Returns a Decision."""
        obs = observation_from_result(lane_result, now, self.min_confidence)
        self._last_frame_time = now
        return self._decide(obs, now)

    def check_stale(self, now):
        """Call when no frame arrived (camera returned nothing / loop is late)."""
        if self._last_frame_time is not None and now - self._last_frame_time <= self.stale_after_s:
            # Not stale yet, but we have no NEW measurement: do not drive on the old one.
            obs = LaneObservation.invalid("stale", "no new frame", now)
            return Decision(False, obs, self.latched, "no new frame")
        obs = LaneObservation.invalid("stale", "no frame for %.2f s" % self.stale_after_s, now)
        return self._decide(obs, now)

    def rearm(self):
        """Operator action. Succeeds only if enough consecutive valid frames were seen since
        the stop; otherwise the stop stays latched. Returns True if rearmed."""
        if self.latched and self._valid_run >= self.rearm_valid_frames:
            self.latched = False
            self.latch_reason = ""
            self._recent.clear()          # judge the new run on new frames only
            self._invalid_since = None
            self._invalid_run = 0
            return True
        return not self.latched

    def force_latch(self, reason):
        """Operator / system stop (e.g. emergency stop): latch AND forget earlier good frames, so that re-arming
        needs a fresh run of valid frames recorded after this call."""
        self._latch(reason)
        self._valid_run = 0
        self._invalid_run = 0
        self._invalid_since = None
        self._recent.clear()

    # --- internals --------------------------------------------------------------------------
    def _latch(self, reason):
        self.latched = True
        self.latch_reason = reason

    def _decide(self, obs, now):
        self._recent.append((now, obs.valid))
        while self._recent and now - self._recent[0][0] > self.window_s:
            self._recent.popleft()

        if obs.valid:
            self._valid_run += 1
            self._invalid_run = 0
            self._invalid_since = None
        else:
            self._valid_run = 0
            self._invalid_run += 1
            if self._invalid_since is None:
                self._invalid_since = now
            if not self.latched:
                if now - self._invalid_since >= self.latch_after_s:
                    self._latch("invalid for %.2f s" % (now - self._invalid_since))
                elif self._invalid_run >= self.latch_after_frames:
                    self._latch("%d invalid frames in a row" % self._invalid_run)

        if not self.latched and len(self._recent) >= self.window_min_frames:
            bad = sum(1 for _, v in self._recent if not v)
            if bad >= self.window_invalid_fraction * len(self._recent):
                self._latch("%d of last %d frames invalid" % (bad, len(self._recent)))

        if self.latched:
            return Decision(False, obs, True, self.latch_reason)
        if not obs.valid:
            return Decision(False, obs, False, obs.reason)   # transient: stop this frame
        return Decision(True, obs, False, "")
