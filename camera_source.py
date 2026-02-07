"""Camera and video input abstraction for webcam, IP camera, file, or URL."""

import cv2
from typing import Optional, Tuple


def open_source(source=None, camera_index: int = 0) -> cv2.VideoCapture:
    """Open video capture from camera index, file path, or URL (e.g. RTSP)."""
    if source is not None and source != "":
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source.")
    return cap


def read_frame(cap: cv2.VideoCapture) -> Tuple[bool, Optional[any]]:
    """Read one frame; returns (success, frame)."""
    ok, frame = cap.read()
    return ok, frame


def get_properties(cap: cv2.VideoCapture) -> dict:
    """Get width, height, fps if available."""
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return {"width": w, "height": h, "fps": fps}


def release(cap: cv2.VideoCapture) -> None:
    cap.release()
