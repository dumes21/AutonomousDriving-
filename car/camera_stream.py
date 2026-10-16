"""Stream the JetRacer camera over HTTP so you can watch it from the laptop
(run ON the car, Python 3.6).

Serves an MJPEG stream at http://<car-ip>:<port>/ . Watch it with
tools/view_camera.py on the laptop, or just open the URL in a browser.
Find the car's IP with `hostname -I`.

    python3 car/camera_stream.py
    python3 car/camera_stream.py --port 8080 --quality 70

Only one program can use the camera at a time. teleop_record.py and
drive_model.py stream the camera themselves, so you only need this script
for looking at the camera without driving.
"""
import argparse
import sys
import time

try:
    from jetcam.csi_camera import CSICamera
except ImportError as e:
    print("ERROR: could not import jetcam ({}). Run this on the JetRacer.".format(e))
    sys.exit(1)

from mjpeg_stream import MjpegStreamer


def parse_args():
    p = argparse.ArgumentParser(description="MJPEG stream of the JetRacer camera")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100 (default 80)")
    p.add_argument("--capture-width", type=int, default=1640)
    p.add_argument("--capture-height", type=int, default=1232)
    return p.parse_args()


def main():
    args = parse_args()
    camera = CSICamera(width=320, height=240,
                       capture_width=args.capture_width, capture_height=args.capture_height,
                       capture_fps=30)
    streamer = MjpegStreamer(args.port, args.quality)
    print("Streaming on http://<car-ip>:{}/  (car IP: run `hostname -I`). Ctrl-C to stop.".format(args.port))

    frames, t_fps = 0, time.time()
    try:
        while True:
            streamer.publish(camera.read())  # read() blocks until the next frame (~30 fps)
            frames += 1
            now = time.time()
            if now - t_fps >= 5.0:
                print("camera fps: {:.1f}   viewers: {}".format(frames / (now - t_fps), streamer.clients))
                frames, t_fps = 0, now
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        streamer.stop()
        try:
            camera.cap.release()
        except Exception:
            pass


if __name__ == "__main__":
    main()
