"""Browser camera viewer with a Capture button (run ON the car, Python 3.6). Camera only: no motors, no DQN.

    python3 car/camera_capture_web.py
    python3 car/camera_capture_web.py --port 8000 --stream-port 8080 --out data/captures

Then open  http://<car-ip>:8000/  on the laptop (car IP: `hostname -I` on the Jetson). You see the live camera and a
Capture button (or press the space bar); each click saves the current 320x240 frame as a JPEG in --out on the Jetson.
Type a label in the box first (e.g. L25) and the file is named L25_<time>.jpg instead of capture_<time>.jpg.
Copy them to the laptop with e.g.  scp jetson@<car-ip>:~/seth-jetracer/data/captures/*.jpg .

Reuses car/mjpeg_stream.py for the live view (served on --stream-port) and the same CSICamera settings as
car/camera_stream.py. Only one program can use the camera at a time: stop teleop_record.py / drive_model.py first.
Files are never overwritten. There is no authentication: use it on a trusted network only.
"""
import argparse
import http.server
import json
import os
import re
import socketserver
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse

try:
    import cv2
    from jetcam.csi_camera import CSICamera
except ImportError as e:
    print("ERROR: could not import cv2/jetcam ({}). Run this on the JetRacer.".format(e))
    sys.exit(1)

from mjpeg_stream import MjpegStreamer

PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>JetRacer camera</title>
<style>
 body{font-family:sans-serif;background:#111;color:#eee;margin:0;padding:16px;text-align:center}
 img{max-width:100%;border:1px solid #444;background:#000}
 input{font-size:16px;padding:6px;border-radius:6px;border:1px solid #555;background:#222;color:#eee}
 button{font-size:20px;padding:12px 28px;margin:12px;border-radius:8px;border:0;background:#2a7;color:#fff;cursor:pointer}
 button:disabled{background:#555}
 #msg{min-height:1.5em;color:#9d9}
 #err{color:#f88}
</style></head><body>
<h2>JetRacer camera</h2>
<img id="cam" alt="live camera (waiting for stream...)">
<div><input id="label" placeholder="optional label, e.g. L25" maxlength="30" size="26"></div>
<div><button id="cap">Capture (space)</button></div>
<div id="msg"></div><div id="err"></div>
<div>Saved this session: <span id="n">0</span></div>
<script>
 var host = location.hostname, n = 0, busy = false;
 document.getElementById('cam').src = 'http://' + host + ':__STREAM_PORT__/';
 function capture() {
   if (busy) return; busy = true;
   var b = document.getElementById('cap'); b.disabled = true;
   fetch('/capture?label=' + encodeURIComponent(document.getElementById('label').value), {method: 'POST'}).then(function(r){ return r.json().then(function(j){ return [r.ok, j]; }); })
   .then(function(x){
     if (x[0]) { n++; document.getElementById('n').textContent = n;
       document.getElementById('msg').textContent = 'saved ' + x[1].path + ' (' + x[1].bytes + ' bytes)';
       document.getElementById('err').textContent = ''; }
     else { document.getElementById('err').textContent = 'capture failed: ' + x[1].error; }
   }).catch(function(e){ document.getElementById('err').textContent = 'capture failed: ' + e; })
   .then(function(){ busy = false; b.disabled = false; });
 }
 document.getElementById('cap').onclick = capture;
 document.addEventListener('keydown', function(e){ if (e.code === 'Space' && e.target.tagName !== 'INPUT') { e.preventDefault(); capture(); } });
</script></body></html>
"""


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class FrameStore(object):
    """Holds the newest camera frame and saves it on request."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.lock = threading.Lock()
        self.frame = None
        self.saved = 0

    def update(self, frame):
        with self.lock:
            self.frame = frame

    def save(self, label=""):
        label = re.sub(r"[^A-Za-z0-9_-]", "", label or "")[:30]          # becomes part of the file name
        prefix = label if label else "capture"
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
        if frame is None:
            raise RuntimeError("no camera frame yet")
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int((time.time() % 1) * 1000)
        path = os.path.join(self.out_dir, "{}_{}_{:03d}.jpg".format(prefix, stamp, millis))
        k = 1
        while os.path.exists(path):                       # never overwrite
            path = os.path.join(self.out_dir, "{}_{}_{:03d}_{}.jpg".format(prefix, stamp, millis, k))
            k += 1
        if not cv2.imwrite(path, frame):
            raise RuntimeError("could not write " + path)
        self.saved += 1
        return path


def make_handler(store, stream_port):
    page = PAGE.replace(b"__STREAM_PORT__", str(stream_port).encode())

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):                                # capture only via POST (a stray GET can't write files)
            url = urlparse(self.path)
            if url.path != "/capture":
                self._send(404, b"not found", "text/plain")
                return
            try:
                path = store.save(parse_qs(url.query).get("label", [""])[0])
                body = {"path": path, "bytes": os.path.getsize(path)}
                self._send(200, json.dumps(body).encode(), "application/json")
                print("saved {}".format(path))
            except Exception as e:                        # noqa: B902
                self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")

        def log_message(self, fmt, *args):
            pass

    return Handler


def parse_args():
    p = argparse.ArgumentParser(description="Browser camera viewer + capture button (camera only)")
    p.add_argument("--port", type=int, default=8000, help="web page port (default 8000)")
    p.add_argument("--stream-port", type=int, default=8080, help="live video port (default 8080)")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "captures"),
                   help="folder on the Jetson for saved images (default data/captures)")
    p.add_argument("--quality", type=int, default=80, help="live view JPEG quality")
    p.add_argument("--capture-width", type=int, default=1640)
    p.add_argument("--capture-height", type=int, default=1232)
    return p.parse_args()


def main():
    args = parse_args()
    store = FrameStore(args.out)
    # Bind both ports BEFORE opening the camera: a leftover copy of this script (or any other program) holding a port
    # then fails with a clear message instead of a traceback with the camera already open.
    try:
        server = _Server(("0.0.0.0", args.port), make_handler(store, args.stream_port))
    except OSError as e:
        sys.exit("ERROR: web port {} is not available ({}). Is another copy still running? Try `pkill -f camera_capture_web` "
                 "or use --port.".format(args.port, e))
    try:
        streamer = MjpegStreamer(args.stream_port, args.quality)
    except OSError as e:
        server.server_close()
        sys.exit("ERROR: stream port {} is not available ({}). Is another copy (or camera_stream.py) still running? "
                 "Try `pkill -f camera_capture_web` or use --stream-port.".format(args.stream_port, e))
    try:
        camera = CSICamera(width=320, height=240, capture_width=args.capture_width,
                           capture_height=args.capture_height, capture_fps=30)
    except Exception as e:                                # noqa: B902 - camera busy / not connected
        streamer.stop()
        server.server_close()
        sys.exit("ERROR: could not open the camera ({}). Another program (teleop_record.py, drive_model.py, "
                 "camera_stream.py, a stale copy of this script) may be using it.".format(e))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("Open http://<car-ip>:{}/ on the laptop (car IP: `hostname -I`). Images go to {}. Ctrl-C to stop.".format(
        args.port, os.path.abspath(args.out)))
    try:
        while True:
            frame = camera.read()                         # blocks until the next frame (~30 fps)
            store.update(frame)
            streamer.publish(frame)
    except KeyboardInterrupt:
        print("\nStopped. {} image(s) saved.".format(store.saved))
    finally:
        server.shutdown()
        streamer.stop()
        try:
            camera.cap.release()
        except Exception:
            pass


if __name__ == "__main__":
    main()
