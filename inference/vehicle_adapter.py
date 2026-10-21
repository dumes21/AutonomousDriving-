"""Turn a pipeline Command into vehicle commands. INACTIVE by default: nothing here is enabled or calibrated.

Existing hardware interface in this repo (car/drive_model.py, car/teleop_record.py, car/drive_test.py):
  * jetracer.nvidia_racecar.NvidiaRacecar: `car.steering` (about -1..1) and `car.throttle`, with `car.steering_gain`
    (library default -0.65! the scripts set 1.0) and `car.steering_offset` (scripts set 0.0).
  * Measured by the team (not verified by this code): steering command -0.8 = strong LEFT, +0.6 = only a slight RIGHT,
    about +0.35 = straight, |steering| > ~0.8 risks brownouts; minimum usable forward throttle ~0.10-0.12.
  * The scripts cap throttle at HARD_LIMIT = 0.6, use a 0.5 s deadman (hold W), and stop = throttle 0 and steering 0
    (written twice), also on SIGTERM / SIGHUP.

SIGN MISMATCH to be aware of: the simulator's action 0 (LEFT) is steering +1 / +0.4 rad, while the physical car turns left
with NEGATIVE command values. The adapter therefore takes an explicit SteeringMap {LEFT, STRAIGHT, RIGHT} -> physical command
and never derives one from the simulator. Because the physical steering is asymmetric (and the policy was trained with a
symmetric +-0.4 rad), the three values must be chosen so that LEFT and RIGHT give the same turning radius; that needs the
steering calibration in docs/FIRST_PHYSICAL_TEST.md. Until then SteeringMap cannot be built.
Python 3.6 compatible; `jetracer` is imported only inside JetRacerAdapter.from_hardware().
"""
import json

HARD_LIMIT = 0.6        # same as car/drive_model.py: never exceed this throttle magnitude
LABELS = ("LEFT", "STRAIGHT", "RIGHT")


class CalibrationMissing(Exception):
    pass


class SteeringMap(object):
    """Physical steering command for each action. All three values, each marked 'measured', are required."""

    def __init__(self, left, straight, right, steer_limit, statuses=None):
        statuses = statuses or {}
        for name in LABELS:
            if statuses.get(name) != "measured":
                raise CalibrationMissing("steering for %s is not marked 'measured' (status %r): check it on the car with "
                                         "the wheels raised first" % (name, statuses.get(name, "UNCALIBRATED")))
        if steer_limit is None or not (0 < steer_limit <= 0.8):
            raise CalibrationMissing("steer_limit must be given and within (0, 0.8]")
        vals = [float(left), float(straight), float(right)]
        if any(abs(v) > steer_limit for v in vals):
            raise CalibrationMissing("steering values %s exceed the steer limit %.2f" % (vals, steer_limit))
        if not (vals[0] < vals[1] < vals[2]):
            raise CalibrationMissing("expected LEFT < STRAIGHT < RIGHT (negative = left on this car), got %s" % vals)
        self.values = {"LEFT": vals[0], "STRAIGHT": vals[1], "RIGHT": vals[2]}
        self.steer_limit = steer_limit

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["LEFT"]["value"], d["STRAIGHT"]["value"], d["RIGHT"]["value"], d.get("steer_limit"),
                   {k: d[k].get("status") for k in LABELS})

    def command_for(self, label):
        return self.values[label]


class DryRunAdapter(object):
    """Records what WOULD be sent. Issues no hardware command."""

    def __init__(self):
        self.log = []
        self.hardware_commands_issued = 0

    def apply(self, command, now, deadman_ok=True):
        self.log.append((now, command.label, bool(command.stop or not deadman_ok)))

    def stop(self):
        self.log.append((None, "STOP", True))


class JetRacerAdapter(object):
    """Applies commands to a NvidiaRacecar-like object. Refuses to exist without a measured SteeringMap and an explicit
    throttle, and drops throttle to 0 on any STOP, missing deadman, or exception. Never used by the offline tests with real hardware."""

    def __init__(self, car, steering_map, throttle, invert_throttle=False, enabled=False):
        if not enabled:
            raise CalibrationMissing("motors are not enabled (pass enabled=True deliberately)")
        if steering_map is None:
            raise CalibrationMissing("no steering map")
        if throttle is None or not (0.0 < throttle <= HARD_LIMIT):
            raise CalibrationMissing("throttle must be given explicitly and be within (0, %.1f]" % HARD_LIMIT)
        self.car = car
        self.map = steering_map
        self.throttle = float(throttle)
        self.invert_throttle = invert_throttle
        self.hardware_commands_issued = 0

    @classmethod
    def from_hardware(cls, steering_map, throttle, invert_throttle=False, steering_gain=1.0, steering_offset=0.0):
        from jetracer.nvidia_racecar import NvidiaRacecar      # only here: needs the real car
        car = NvidiaRacecar()
        car.throttle = 0.0
        car.steering_gain = steering_gain                      # explicit, as in car/drive_model.py
        car.steering_offset = steering_offset
        return cls(car, steering_map, throttle, invert_throttle, enabled=True)

    def _write(self, throttle, steering):
        self.car.throttle = throttle
        self.car.steering = steering
        self.hardware_commands_issued += 1

    def apply(self, command, now=None, deadman_ok=True):
        try:
            if command.stop or not deadman_ok:
                self.car.throttle = 0.0                        # steering left as it is; the car coasts to a stop
                self.hardware_commands_issued += 1
                return
            self.car.steering = self.map.command_for(command.action.label)
            value = self.throttle
            self.car.throttle = -value if self.invert_throttle else value
            self.hardware_commands_issued += 2
        except Exception:                                      # noqa: B902 - any failure: stop and re-raise
            self.stop()
            raise

    def stop(self):
        """Throttle and steering to 0, written twice (same as car/drive_model.stop_car)."""
        for _ in range(2):
            try:
                self.car.throttle = 0.0
                self.car.steering = 0.0
                self.hardware_commands_issued += 2
                return
            except Exception:                                  # noqa: B902
                pass
