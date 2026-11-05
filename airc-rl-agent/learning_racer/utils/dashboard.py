"""Live "JetRacer RushHour3" dashboard served by `racer train`, `racer demo` and demo_control.py.

Open http://<car-ip>:<port>/ in any browser. Shows the camera next to the VAE reconstruction,
the reconstruction loss, and the steering/throttle the agent sends. Standard library + Pillow
only (Python 3.6 on the Jetson). Nothing is decoded or JPEG-encoded unless a browser is
connected, and errors here never stop training or driving.
"""
import http.server
import io
import json
import socketserver
import threading
import time

from PIL import Image

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JetRacer RushHour3</title>
<style>
:root { --bg:#0f1115; --panel:#181b22; --line:#2a2f3a; --text:#e8eaf0; --muted:#8b93a7;
        --accent:#ff8a2a; --ok:#3ecf8e; --warn:#f2c94c; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
header { display:flex; align-items:center; justify-content:center; gap:14px; padding:18px 16px 6px; }
h1 { margin:0; font-size:clamp(24px,4vw,40px); letter-spacing:.5px; }
h1 span { color:var(--accent); }
#mode { font-size:13px; font-weight:600; padding:4px 10px; border-radius:999px;
        border:1px solid var(--line); color:var(--muted); }
#mode.live { color:var(--ok); border-color:var(--ok); }
main { max-width:1400px; margin:0 auto; padding:12px 16px 24px; }
.hero { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px; position:relative; }
.panel h2 { margin:0 0 10px; font-size:14px; font-weight:600; color:var(--muted);
            text-transform:uppercase; letter-spacing:1px; }
.view { aspect-ratio:2/1; background:#000; border-radius:8px; overflow:hidden; position:relative; }
.view img { width:100%; height:100%; object-fit:contain; display:block; }
.paused { position:absolute; inset:0; display:none; align-items:center; justify-content:center;
          background:rgba(15,17,21,.72); color:var(--warn); font-weight:600; text-align:center; padding:12px; }
body.stale .paused { display:flex; }
.stats { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-top:16px; }
.stat .value { font-size:clamp(26px,3.4vw,40px); font-weight:700; font-variant-numeric:tabular-nums; }
.stat .unit { font-size:14px; color:var(--muted); margin-left:6px; font-weight:500; }
.bar { position:relative; height:14px; margin-top:12px; background:#0b0d11;
       border:1px solid var(--line); border-radius:7px; overflow:hidden; }
.bar .fill { position:absolute; top:0; bottom:0; background:var(--accent); }
.bar .mid { position:absolute; left:50%; top:-2px; bottom:-2px; width:2px; background:var(--muted); }
.ends { display:flex; justify-content:space-between; font-size:12px; color:var(--muted); margin-top:4px; }
@media (max-width:760px) { .hero, .stats { grid-template-columns:1fr; } }
</style></head>
<body class="stale">
<header><h1>JetRacer <span>RushHour3</span></h1><div id="mode">connecting</div></header>
<main>
  <section class="hero">
    <div class="panel"><h2>Camera (what the VAE sees)</h2>
      <div class="view"><img id="cam" src="/camera.mjpg" alt="camera">
        <div class="paused">Waiting for the car&hellip;</div></div></div>
    <div class="panel"><h2>VAE reconstruction</h2>
      <div class="view"><img id="rec" src="/recon.mjpg" alt="reconstruction">
        <div class="paused">Waiting for the car&hellip;</div></div></div>
  </section>
  <section class="stats">
    <div class="panel stat"><h2>Reconstruction loss</h2><div class="value" id="loss">&ndash;</div></div>
    <div class="panel stat"><h2>Steering</h2><div class="value" id="steer">&ndash;</div>
      <div class="bar"><div class="fill" id="steerFill"></div><div class="mid"></div></div>
      <div class="ends"><span>left</span><span>right</span></div></div>
    <div class="panel stat"><h2>Throttle</h2><div class="value" id="thr">&ndash;</div>
      <div class="bar"><div class="fill" id="thrFill" style="left:0"></div></div>
      <div class="ends"><span>0</span><span id="thrMax">max</span></div></div>
  </section>
</main>
<script>
var $ = function (id) { return document.getElementById(id); };
function fmt(x, d) { return (x === null || x === undefined) ? "\\u2013" : x.toFixed(d); }
function reconnect(img, url) { img.onerror = function () {
  setTimeout(function () { img.src = url + "?t=" + Date.now(); }, 1500); }; }
reconnect($("cam"), "/camera.mjpg"); reconnect($("rec"), "/recon.mjpg");
function poll() {
  fetch("/state.json", {cache: "no-store"}).then(function (r) { return r.json(); }).then(function (s) {
    var live = s.t > 0 && (s.now - s.t) < 1.5;
    document.body.classList.toggle("stale", !live);
    $("mode").textContent = s.mode + (live ? " \\u00b7 live" : " \\u00b7 paused");
    $("mode").classList.toggle("live", live);
    $("loss").textContent = s.loss === null ? "\\u2013" : Math.round(s.loss).toLocaleString();
    if (s.steering !== null) {
      var m = s.steer_max || 1, f = Math.max(-1, Math.min(1, s.steering / m)) * 50;
      $("steer").innerHTML = fmt(s.steering, 2) + '<span class="unit">' +
        (Math.abs(s.steering) < 0.05 ? "straight" : (s.steering < 0 ? "left" : "right")) + "</span>";
      $("steerFill").style.left = (f < 0 ? 50 + f : 50) + "%"; $("steerFill").style.width = Math.abs(f) + "%";
    }
    if (s.throttle !== null) {
      var t = Math.abs(s.throttle), tm = s.throttle_max || 1;
      $("thr").innerHTML = fmt(t, 2) + '<span class="unit">forward</span>';
      $("thrFill").style.width = Math.min(100, 100 * t / tm) + "%"; $("thrMax").textContent = fmt(tm, 2);
    }
  }).catch(function () {
    document.body.classList.add("stale"); $("mode").textContent = "disconnected"; $("mode").classList.remove("live");
  }).then(function () { setTimeout(poll, 200); });
}
poll();
</script></body></html>
"""


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Dashboard(object):
    STREAMS = ("camera", "recon")

    def __init__(self, port, mode="", steer_max=1.0, throttle_max=1.0, max_fps=10.0, quality=80):
        self.min_dt = 1.0 / max_fps
        self.quality = quality
        self.cond = threading.Condition()
        self.jpeg = dict((name, None) for name in self.STREAMS)
        self.seq = 0
        self.state = {"mode": mode, "t": 0.0, "loss": None, "steering": None, "throttle": None,
                      "steer_max": abs(steer_max), "throttle_max": abs(throttle_max)}
        self.viewers = 0         # open MJPEG streams
        self.last_poll = 0.0     # time of the last /state.json request
        self.last_publish = 0.0
        self.running = True
        dash = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    self._send("text/html; charset=utf-8", PAGE.encode("utf-8"))
                elif path == "/state.json":
                    dash.last_poll = time.time()
                    with dash.cond:
                        state = dict(dash.state)
                    state["now"] = time.time()
                    self._send("application/json", json.dumps(state).encode("utf-8"))
                elif path in ("/camera.mjpg", "/recon.mjpg"):
                    dash._stream(self, path[1:-len(".mjpg")])
                else:
                    self.send_error(404)

            def _send(self, content_type, body):
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt, *args):
                pass  # keep the training terminal quiet

        self.server = _Server(("0.0.0.0", port), Handler)  # raises OSError if the port is taken
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def active(self):
        """True while someone is watching (an open stream, or the page polled recently)."""
        return self.viewers > 0 or time.time() - self.last_poll < 2.0

    def publish(self, camera_rgb, recon_rgb, loss, steering, throttle):
        """camera_rgb / recon_rgb: HxWx3 uint8 RGB arrays. Rate-limited to max_fps."""
        now = time.time()
        if now - self.last_publish < self.min_dt:
            return
        self.last_publish = now
        frames = {"camera": self._encode(camera_rgb), "recon": self._encode(recon_rgb)}
        with self.cond:
            self.jpeg.update(frames)
            self.seq += 1
            self.state.update(t=now, loss=float(loss), steering=float(steering), throttle=float(throttle))
            self.cond.notify_all()

    def _encode(self, rgb):
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, "JPEG", quality=self.quality)
        return buf.getvalue()

    def _stream(self, handler, name):
        handler.send_response(200)
        handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        with self.cond:
            self.viewers += 1
        seen = -1
        try:
            while self.running:
                with self.cond:
                    self.cond.wait_for(lambda: self.seq != seen or not self.running, timeout=1.0)
                    if self.seq == seen or self.jpeg[name] is None:
                        continue
                    seen, jpeg = self.seq, self.jpeg[name]
                handler.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                handler.wfile.write("Content-Length: {}\r\n\r\n".format(len(jpeg)).encode())
                handler.wfile.write(jpeg)
                handler.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # browser closed or reloaded
        finally:
            with self.cond:
                self.viewers -= 1

    def stop(self):
        self.running = False
        with self.cond:
            self.cond.notify_all()
        self.server.shutdown()
        self.server.server_close()


def start_dashboard(port, mode="", steer_max=1.0, throttle_max=1.0):
    """Start the dashboard, or return None with a warning (port 0, or port in use).
    The dashboard is only for watching, so failing to start must never stop the car."""
    if not port or port <= 0:
        return None
    try:
        dashboard = Dashboard(port, mode, steer_max, throttle_max)
    except OSError as e:
        print("WARNING: dashboard not started on port {} ({}). Carrying on without it.".format(port, e))
        return None
    print("Dashboard: http://<car-ip>:{}/  (JetRacer RushHour3, {})".format(port, mode))
    return dashboard
