import time
from jetracer.nvidia_racecar import NvidiaRacecar
 
car = NvidiaRacecar()
car.throttle = 0.0
car.steering_gain = 1.0
car.steering_offset = 0.0
 
commands = [
    0.0,
    -0.2, -0.4, -0.6, -0.8,
    0.0,
    0.2, 0.4, 0.6, 0.8,
    0.0
]
 
try:
    for value in commands:
        print(f"Steering command: {value:+.2f}")
        car.steering = value
        time.sleep(2)
finally:
    car.steering = 0.0
    car.throttle = 0.0