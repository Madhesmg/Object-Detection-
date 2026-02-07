"""MJPEG stream server for annotated video (ready for API consumption)."""

import threading
import time
import cv2
from flask import Flask, Response
import config as cfg

app = Flask(__name__)

# Shared: latest frame (BGR) to stream
_current_frame = None
_frame_lock = threading.Lock()


def update_frame(frame):
    """Set the latest frame to stream (called from pipeline)."""
    global _current_frame
    with _frame_lock:
        _current_frame = frame.copy() if frame is not None else None


def _generate_frames():
    """Generator yielding MJPEG frames."""
    global _current_frame
    while True:
        with _frame_lock:
            frame = _current_frame
        if frame is not None:
            _, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, cfg.MJPEG_QUALITY]
            )
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
        else:
            # Placeholder black frame
            import numpy as np
            black = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", black)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
        time.sleep(1 / 30)


@app.route("/video")
def video_feed():
    """MJPEG stream endpoint for API clients."""
    return Response(
        _generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/")
def index():
    return """
    <html>
    <head><title>Object Detection Stream</title></head>
    <body>
    <h1>Annotated Video Stream</h1>
    <img src="/video" alt="Video stream" style="max-width:100%;" />
    </body>
    </html>
    """


def run_server(host=None, port=None):
    host = host or cfg.MJPEG_HOST
    port = port or cfg.MJPEG_PORT
    app.run(host=host, port=port, threaded=True, use_reloader=False)


def start_background_server(host=None, port=None):
    t = threading.Thread(target=run_server, args=(host, port), daemon=True)
    t.start()
    return t
