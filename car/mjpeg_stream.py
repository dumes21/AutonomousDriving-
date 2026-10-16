"""Tiny MJPEG-over-HTTP server shared by the car scripts (Python 3.6).

    streamer = MjpegStreamer(8080)   # starts serving in a background thread
    streamer.publish(image)          # call once per camera frame
    streamer.stop()

Watch with tools/view_camera.py on the laptop, or open http://<car-ip>:8080/
in a browser. Frames are only JPEG-encoded while someone is watching, so it
costs nothing when no viewer is connected.
"""
import http.server
import socketserver
import threading

import cv2


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class MjpegStreamer(object):
    def __init__(self, port, quality=80):
        self.params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        self.jpeg = None
        self.cond = threading.Condition()
        self.clients = 0
        self.running = True
        streamer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                with streamer.cond:
                    streamer.clients += 1
                try:
                    while streamer.running:
                        with streamer.cond:
                            streamer.cond.wait(1.0)
                            jpeg = streamer.jpeg
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write("Content-Length: {}\r\n\r\n".format(len(jpeg)).encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass  # viewer closed
                finally:
                    with streamer.cond:
                        streamer.clients -= 1

            def log_message(self, fmt, *args):
                pass  # keep the terminal quiet

        self.server = _Server(("0.0.0.0", port), Handler)  # raises OSError if the port is taken
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def publish(self, image):
        if self.clients == 0:
            return
        ok, buf = cv2.imencode(".jpg", image, self.params)
        if ok:
            with self.cond:
                self.jpeg = buf.tobytes()
                self.cond.notify_all()

    def stop(self):
        self.running = False
        self.server.shutdown()
        self.server.server_close()


def start_stream(port):
    """Start a streamer, or return None with a warning if it can't (e.g. port in use).
    The stream is only for watching, so failing to start it must never stop the car script."""
    if port <= 0:
        return None
    try:
        streamer = MjpegStreamer(port)
        print("Camera stream: http://<car-ip>:{}/  (laptop: python tools/view_camera.py <car-ip>)".format(port))
        return streamer
    except OSError as e:
        print("WARNING: camera stream not started on port {} ({}). Carrying on without it.".format(port, e))
        return None
