"""
test_camera.py -- quick check that the camera + HTTP video stream work.

No YOLO, no detection: just grab frames from the camera and serve them as an
MJPEG stream. Use this to confirm your camera and network are working before
running the full lab (partF.py).

    python test_camera.py

Then on your laptop (same MWireless network as the Pi), open
    http://<pi-ip>:8000/          (get <pi-ip> from `hostname -I` on the Pi)
Press Ctrl-C in this terminal to stop.
"""

import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2

STREAM_PORT = 8000
latest_jpeg = None                  # newest frame, set by the main loop

# ---- MJPEG HTTP server (runs in a background thread) ------------------------
class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        while True:
            if latest_jpeg is not None:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                self.wfile.write(latest_jpeg)
                self.wfile.write(b"\r\n")
            time.sleep(0.05)

    def log_message(self, *args):
        pass                        # silence per-request HTTP logging

threading.Thread(
    target=lambda: HTTPServer(("0.0.0.0", STREAM_PORT), MJPEGHandler).serve_forever(),
    daemon=True,
).start()
print(f"[INFO] streaming camera on port {STREAM_PORT}. On your laptop, open")
print(f"[INFO]   http://<pi-ip>:{STREAM_PORT}/   (get <pi-ip> from `hostname -I`)")

# ---- Open the camera --------------------------------------------------------
print("[INFO] opening camera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
time.sleep(2.0)

# ---- Grab frames and publish them to the stream ----------------------------
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[INFO] camera read failed -- is the camera connected?")
            break
        ok_jpg, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok_jpg:
            latest_jpeg = buf.tobytes()
except KeyboardInterrupt:
    pass

print("\n [INFO] Exiting -- cleanup \n")
cap.release()
