"""Processing pipeline: camera -> YOLOv8 detect+track -> line crossing -> annotated frame + counts."""

import cv2
import threading
import time
from typing import Optional, Tuple, Callable, List, Any
from camera_source import open_source, read_frame, get_properties, release
from detector import DetectorTracker
from line_crossing import Line, LineCrossingCounter
import stream_server
import config

# COCO class names (YOLOv8 default)
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork",
    43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
    48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog",
    53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet", 62: "tv",
    63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
    73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}


class Pipeline:
    """Runs capture + detection + tracking + line crossing; exposes latest frame and counts."""

    def __init__(
        self,
        input_source=None,
        camera_index: int = 0,
        line: Optional[Line] = None,
    ):
        self.input_source = input_source if input_source is not None else config.INPUT_SOURCE
        self.camera_index = camera_index if self.input_source is None else 0
        self.line = line or Line(*config.DEFAULT_LINE)
        self.detector = DetectorTracker()
        self.counter = LineCrossingCounter(self.line, COCO_NAMES)
        self._cap = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_frame = None
        self._current_counts = {}
        self._current_detections = []
        self._lock = threading.Lock()
        self._on_count_callback: Optional[Callable[[List], None]] = None

    def set_line(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.line = Line(x1, y1, x2, y2)
        self.counter.set_line(self.line)

    def set_line_object(self, line: Line) -> None:
        self.line = line
        self.counter.set_line(line)

    def on_count(self, callback: Callable[[List], None]) -> None:
        self._on_count_callback = callback

    def start(self) -> None:
        if self._running:
            return
        self._cap = open_source(self.input_source, self.camera_index)
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap:
            release(self._cap)
            self._cap = None

    def _run_loop(self) -> None:
        while self._running and self._cap:
            ok, frame = read_frame(self._cap)
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            annotated, detections = self.detector.run(frame, persist=True)
            events = self.counter.update(detections)
            if events and self._on_count_callback:
                self._on_count_callback(events)
            # Draw counting line
            cv2.line(
                annotated,
                (int(self.line.x1), int(self.line.y1)),
                (int(self.line.x2), int(self.line.y2)),
                (0, 255, 0),
                2,
            )
            # Draw counts on frame
            counts = self.counter.get_counts_dict()
            y_offset = 30
            for cls_id, cnt in sorted(counts.items()):
                name = COCO_NAMES.get(cls_id, str(cls_id))
                cv2.putText(
                    annotated,
                    f"{name}: {cnt}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                y_offset += 25
            with self._lock:
                self._current_frame = annotated
                self._current_counts = dict(counts)
                self._current_detections = detections
            stream_server.update_frame(annotated)
            time.sleep(0.001)

    def get_latest(self) -> Tuple[Optional[Any], dict]:
        """Returns (frame, counts_dict)."""
        with self._lock:
            f = self._current_frame.copy() if self._current_frame is not None else None
            c = dict(self._current_counts)
        return f, c

    def get_counts(self) -> dict:
        return self.counter.get_counts_dict()

    def get_total_count(self) -> int:
        return self.counter.get_total()

    def reset_counts(self) -> None:
        self.counter.reset()

    def get_properties(self) -> dict:
        if self._cap is None:
            return {}
        return get_properties(self._cap)
