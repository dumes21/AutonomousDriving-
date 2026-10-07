"""Camera check for the JetRacer (run ON the car, Python 3.6).

Opens the CSI camera (320x240, 30 fps) through jetcam, prints the measured
fps, and saves timestamped JPEGs to data/ every --every seconds, up to
--max-frames images. Works headless; use --show to open a window.

    python3 car/camera_test.py
    python3 car/camera_test.py --every 0.5 --max-frames 100 --show
"""
import argparse
import os
import sys
import time

try:
    import cv2
except ImportError as e:
    print("ERROR: could not import opencv ({}).".format(e))
    sys.exit(1)

try:
    from jetcam.csi_camera import CSICamera
except ImportError as e:
    print("ERROR: jetcam is not installed or not importable ({}).".format(e))
    print("Run this on the JetRacer, where jetcam is installed.")
    sys.exit(1)

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def parse_args():
    p = argparse.ArgumentParser(description="JetRacer CSI camera fps check + image saver")
    p.add_argument("--every", type=float, default=1.0, help="seconds between saved images (default 1)")
    p.add_argument("--max-frames", type=int, default=30, help="stop after saving this many images")
    p.add_argument("--out", default=DEFAULT_OUT, help="folder for JPEGs (default: <repo>/data)")
    p.add_argument("--show", action="store_true", help="also show a window (needs a display)")
    # Sensor mode the camera captures in; gstreamer scales it down to 320x240.
    p.add_argument("--capture-width", type=int, default=1640)
    p.add_argument("--capture-height", type=int, default=1232)
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    camera = CSICamera(width=320, height=240,
                       capture_width=args.capture_width, capture_height=args.capture_height,
                       capture_fps=30)
    print("Camera open. Saving to {} every {} s (max {} images). Ctrl-C to stop.".format(
        os.path.abspath(args.out), args.every, args.max_frames))

    saved = 0
    frames = 0
    t_fps = time.time()
    t_save = 0.0  # 0 so the first frame is saved straight away
    try:
        while saved < args.max_frames:
            image = camera.read()  # BGR numpy array, 240x320x3
            frames += 1
            now = time.time()

            if now - t_fps >= 1.0:
                print("fps: {:.1f}  (image shape {})".format(frames / (now - t_fps), image.shape))
                frames = 0
                t_fps = now

            if now - t_save >= args.every:
                t_save = now
                name = time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + "_{:03d}.jpg".format(int(now * 1000) % 1000)
                cv2.imwrite(os.path.join(args.out, name), image)
                saved += 1
                print("saved {} ({}/{})".format(name, saved, args.max_frames))

            if args.show:
                cv2.imshow("camera", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        try:
            camera.cap.release()
        except Exception:
            pass
        if args.show:
            cv2.destroyAllWindows()
        print("Saved {} image(s).".format(saved))


if __name__ == "__main__":
    main()
