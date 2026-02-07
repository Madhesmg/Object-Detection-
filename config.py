"""Configuration for object detection, tracking, and counting system."""

# Camera / input
CAMERA_INDEX = 0  # 0 = default webcam; use RTSP/URL for IP camera, e.g. "rtsp://user:pass@ip:554/stream"
INPUT_SOURCE = None  # None = use CAMERA_INDEX; or set to video file path / URL

# YOLOv8
YOLO_MODEL = "yolov8n.pt"  # n=nano, s=small, m=medium, l=large, x=extra
CONFIDENCE_THRESH = 0.5
IOU_THRESH = 0.5
CLASSES_TO_DETECT = None  # None = all COCO classes; or [0, 1, 2, 3, 5, 7] for person, car, etc.

# Tracking (Ultralytics built-in: ByteTrack / BoT-SORT)
TRACKER_CONFIG = "bytetrack.yaml"

# Line crossing
# Line defined as (x1, y1, x2, y2) in pixel coords; can be overridden in GUI
DEFAULT_LINE = (0, 0, 640, 0)  # placeholder; user defines in GUI

# MJPEG stream API
MJPEG_HOST = "0.0.0.0"
MJPEG_PORT = 8765
MJPEG_QUALITY = 85

# Export
EXPORT_DIR = "exports"
HISTORY_CSV = "count_history.csv"
