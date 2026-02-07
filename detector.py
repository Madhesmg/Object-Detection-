"""YOLOv8 object detection and tracking."""

import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO
import config


class DetectorTracker:
    def __init__(
        self,
        model_path: str = None,
        conf: float = None,
        iou: float = None,
        classes: Optional[List[int]] = None,
    ):
        self.model_path = model_path or config.YOLO_MODEL
        self.conf = conf if conf is not None else config.CONFIDENCE_THRESH
        self.iou = iou if iou is not None else config.IOU_THRESH
        self.classes = classes if classes is not None else config.CLASSES_TO_DETECT
        self.model = YOLO(self.model_path)

    def run(self, frame: np.ndarray, persist: bool = True) -> Tuple[np.ndarray, List[dict]]:
        results = self.model.track(
            frame, persist=persist, conf=self.conf, iou=self.iou,
            classes=self.classes, verbose=False,
        )
        annotated = results[0].plot()
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                cls_id = int(boxes.cls[i].cpu().numpy())
                conf = float(boxes.conf[i].cpu().numpy())
                tid = int(boxes.id[i].cpu().numpy()) if boxes.id is not None else None
                detections.append({"xyxy": xyxy, "track_id": tid, "class_id": cls_id, "conf": conf})
        return annotated, detections
