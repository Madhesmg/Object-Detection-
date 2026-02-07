"""MJPEG stream server for annotated video (API)."""
import threading
import time
import cv2
from flask import Flask, Response
import config as cfg

app = Flask(__name__)
_current_frame = None
_frame_lock = threading.Lock()


def update_frame(frame):
    global _current_frame
    with _frame_lock:
        _current_frame = frame.copy() if frame is not None else None


def _generate_frames():
    global _current_frame
    while True:
        with _frame_lock:
            frame = _current_frame
        if frame is not None:
            quality = getattr(cfg, "MJPEG_QUALITY", 85)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        else:
            import numpy as np
            black = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", black)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(1 / 30)


@app.route("/video")
def video_feed():
    return Response(_generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


def start_background_server(host=None, port=None):
    import threading
    h = host or cfg.MJPEG_HOST
    p = port or cfg.MJPEG_PORT
    t = threading.Thread(target=lambda: app.run(host=h, port=p, threaded=True, use_reloader=False), daemon=True)
    t.start()
    return t
