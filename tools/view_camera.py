"""OpenCV window showing the JetRacer camera (run on the LAPTOP).

teleop_record.py, drive_model.py and camera_stream.py all stream the camera.
Start this once and leave it running: it waits for the car, the window pops
up when one of those scripts starts, and it goes back to waiting when it stops.

    python tools/view_camera.py 192.168.1.42
    python tools/view_camera.py 192.168.1.42 --port 8080 --scale 3

Keys:  Q / ESC  quit
       S        save a snapshot to logs/snapshots/
       C        toggle the crop line (what train_bc.py cuts off the top)
"""
import argparse
import os
import time
import urllib.request

import cv2
import numpy as np

SNAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "snapshots")


def parse_args():
    p = argparse.ArgumentParser(description="View the JetRacer camera stream")
    p.add_argument("host", help="car IP address or hostname")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--scale", type=float, default=2.0, help="window size multiplier (default 2)")
    p.add_argument("--crop", type=float, default=0.35, help="crop fraction to mark (same as train_bc.py)")
    return p.parse_args()


def mjpeg_frames(url):
    """Yield decoded BGR frames from an MJPEG stream by finding JPEG start/end markers."""
    stream = urllib.request.urlopen(url, timeout=5)
    buf = b""
    while True:
        chunk = stream.read(4096)
        if not chunk:
            raise ConnectionError("stream closed by the car")
        buf += chunk
        start = buf.find(b"\xff\xd8")
        end = buf.find(b"\xff\xd9", start + 2)
        if start != -1 and end != -1:
            jpg, buf = buf[start:end + 2], buf[end + 2:]
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                yield img


WINDOW = "JetRacer camera"


def show_stream(url, args, state):
    """Show frames until the stream ends (returns False) or the user quits (returns True)."""
    frames, fps, t_fps = 0, 0.0, time.time()
    for img in mjpeg_frames(url):
        frames += 1
        now = time.time()
        if now - t_fps >= 1.0:
            fps, frames, t_fps = frames / (now - t_fps), 0, now

        view = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
        if state["crop"]:
            y = int(view.shape[0] * args.crop)
            cv2.line(view, (0, y), (view.shape[1], y), (0, 0, 255), 1)
        cv2.putText(view, "{:.0f} fps".format(fps), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.imshow(WINDOW, view)
        state["open"] = True

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return True
        if key == ord("c"):
            state["crop"] = not state["crop"]
        if key == ord("s"):
            os.makedirs(SNAP_DIR, exist_ok=True)
            name = os.path.join(SNAP_DIR, time.strftime("%Y%m%d_%H%M%S") + ".jpg")
            cv2.imwrite(name, img)  # the raw 320x240 frame, no overlay
            print("saved", name)
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return True  # window closed with the X button
    return False


def main():
    args = parse_args()
    url = "http://{}:{}/".format(args.host, args.port)
    state = {"crop": True, "open": False}
    print("Waiting for the car's camera at {} (Ctrl-C to quit) ...".format(url))
    try:
        while True:
            try:
                if show_stream(url, args, state):
                    break
            except OSError:
                pass  # car script not running yet, or it just stopped
            if state["open"]:
                cv2.destroyWindow(WINDOW)
                state["open"] = False
                print("Stream stopped. Waiting for the car again ...")
            cv2.waitKey(1)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
