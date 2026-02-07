"""Configuration for object detection, tracking, and counting."""
from pathlib import Path

# Camera / input: None = use webcam (CAMERA_INDEX); or set to video path / RTSP URL
INPUT_SOURCE = None  # None = webcam | "video.mp4" | "rtsp://ip:554/stream"
CAMERA_INDEX = 0     # 0 = default webcam, 1 = second camera

# YOLO: use local or fallback to parent project folder
_THIS_DIR = Path(__file__).resolve().parent
YOLO_MODEL = str(_THIS_DIR / "yolo11l.pt")
if not Path(YOLO_MODEL).exists():
    YOLO_MODEL = str(_THIS_DIR.parent / "Object counting and tracking" / "yolo11l.pt")
CONFIDENCE_THRESH = 0.5
IOU_THRESH = 0.5
CLASSES_TO_DETECT = [0, 1, 2, 3, 5, 6, 7]  # person, bicycle, car, motorcycle, bus, train, truck

# Default counting line (x1, y1, x2, y2)
DEFAULT_LINE = (690, 430, 1130, 430)

# MJPEG stream API
MJPEG_HOST = "0.0.0.0"
MJPEG_PORT = 9999
MJPEG_QUALITY = 85
