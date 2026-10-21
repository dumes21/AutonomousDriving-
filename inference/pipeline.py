"""Camera frame -> detector -> state estimator -> safety gate -> DQN -> PROPOSED action (or STOP). No hardware.

The pipeline never moves anything: it returns a Command, which a vehicle adapter (inference/vehicle_adapter.py) may
apply. Every doubt resolves to STOP, never to a default observation:

  * emergency stop engaged                       -> STOP
  * frame missing, or older than stale_after_s   -> STOP
  * boundaries not both found / low confidence   -> STOP  (no zero or previous observation is substituted)
  * camera calibration incomplete (UNCALIBRATED) -> STOP
  * estimate out of the simulator's range        -> STOP  (offset beyond the lane half-width means "over the line")
  * safety gate not armed or latched             -> STOP  (car/lane_observation.SafeStopGate: latches on persistent loss)
  * any exception in estimation or inference     -> STOP

On every STOP the policy's previous-steering memory is reset to 0 (straight), matching the simulator's reset state,
so driving resumes from a neutral state. The diagnostic warnings of the estimator are carried in Command.warnings.
Python 3.6 compatible.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
for sub in ("car", "perception", "inference"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from dqn_policy import ObservationError  # noqa: E402
from lane_detection import STATUS_OK, LaneResult  # noqa: E402
from lane_observation import SafeStopGate  # noqa: E402
import state_estimator as se  # noqa: E402


class Command(object):
    def __init__(self, stop, reason, status, action=None, observation=None, warnings=None, estimate=None):
        self.stop = stop                      # True: do not drive (throttle 0)
        self.reason = reason                  # why, "" when proposing an action
        self.status = status                  # OK, UNCALIBRATED, INVALID_DETECTION, STALE, ESTOP, ...
        self.action = action                  # dqn_policy.Action or None
        self.observation = observation        # the 3-vector given to the DQN, or None
        self.warnings = list(warnings or [])
        self.estimate = estimate              # StateEstimate or None

    @property
    def label(self):
        return "STOP" if self.stop else self.action.label


def _lane_result_for_gate(est):
    """Adapt a StateEstimate to the LaneResult the existing SafeStopGate expects."""
    r = LaneResult()
    r.valid = bool(est.valid)
    r.status = STATUS_OK if est.valid else str(est.status).lower()
    r.confidence = est.confidence
    r.offset_norm, r.heading_slope = est.offset_m, est.heading_rad
    return r


class CameraToActionPipeline(object):
    def __init__(self, estimator, policy, gate=None, stale_after_s=0.25):
        self.estimator = estimator
        self.policy = policy
        self.gate = gate or SafeStopGate(stale_after_s=stale_after_s)
        self.stale_after_s = stale_after_s
        self.estop = False
        self.estop_reason = ""

    # ---- operator controls ------------------------------------------------------------------
    def emergency_stop(self, reason="manual emergency stop"):
        self.estop, self.estop_reason = True, reason
        self.gate.force_latch(reason)
        self.policy.reset()

    def clear_emergency_stop(self):
        """Explicit operator action. The gate stays latched until arm() succeeds."""
        self.estop, self.estop_reason = False, ""

    def arm(self):
        """Operator re-arm after a latched stop: succeeds only after enough consecutive valid frames."""
        if self.estop:
            return False
        ok = self.gate.rearm()
        if ok:
            self.policy.reset()
        return ok

    # ---- one control step -------------------------------------------------------------------
    def _stop(self, reason, status, warnings=None, estimate=None):
        self.policy.reset()
        return Command(True, reason, status, warnings=warnings, estimate=estimate)

    def step(self, frame, now, frame_time=None):
        if self.estop:
            return self._stop(self.estop_reason or "emergency stop", "ESTOP")
        if frame is None:
            d = self.gate.check_stale(now)
            return self._stop("no frame: %s" % (d.reason or "camera stalled"), "STALE")
        if frame_time is not None and now - frame_time > self.stale_after_s:
            self.gate.check_stale(now + self.stale_after_s)         # counts toward the persistent-loss latch
            return self._stop("frame is %.2f s old (> %.2f s)" % (now - frame_time, self.stale_after_s), "STALE")
        try:
            est = self.estimator.estimate(frame)
        except Exception as e:                                      # noqa: B902 - any failure must stop the car
            self.gate.check_stale(now + self.stale_after_s)
            return self._stop("state estimation failed: %r" % (e,), "ERROR")
        decision = self.gate.update(_lane_result_for_gate(est), now)
        if est.status != se.OK:
            why = "; ".join(w for w in est.warnings[:2]) or "no valid state"
            return self._stop(why if why.startswith(est.status) else "%s: %s" % (est.status, why),
                              est.status, est.warnings, est)
        if not decision.drive_allowed:
            return self._stop("safety gate: %s" % decision.reason, "GATE", est.warnings, est)
        try:
            obs = se.to_dqn_observation(est, self.policy.last_steering)
            if obs.vector is None:
                return self._stop("no observation: " + "; ".join(obs.warnings), "NO_OBSERVATION", obs.warnings, est)
            if obs.clipped:
                return self._stop("estimate outside the simulator's range: " + "; ".join(obs.warnings),
                                  "OUT_OF_RANGE", obs.warnings, est)
            action = self.policy.act(obs.vector)
        except (ObservationError, Exception) as e:                  # noqa: B902
            return self._stop("inference failed: %r" % (e,), "ERROR", est.warnings, est)
        return Command(False, "", "OK", action=action, observation=obs.vector, warnings=est.warnings + obs.warnings,
                       estimate=est)
