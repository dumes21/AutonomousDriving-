import time
from jetracer.nvidia_racecar import NvidiaRacecar

car = NvidiaRacecar()
car.throttle_gain = 0.5   # limit max speed

try:
    print("Steering left")
    car.steering = -0.5
    time.sleep(1)

    print("Steering right")
    car.steering = 0.5
    time.sleep(1)

    print("Steering centre")
    car.steering = 0.0
    time.sleep(1)

    print("Throttle forward (gentle)")
    car.throttle = 0.3
    time.sleep(2)

    print("Stop")
    car.throttle = 0.0
    time.sleep(1)

    print("Throttle reverse (gentle)")
    car.throttle = -0.3
    time.sleep(2)
finally:
    car.throttle = 0.0
    car.steering = 0.0
    print("Stopped")
