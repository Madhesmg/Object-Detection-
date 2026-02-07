"""Line definition and line-crossing logic for counting objects."""

import numpy as np
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class Line:
    """Line segment (x1, y1) -> (x2, y2) in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    direction: str = "both"  # "positive", "negative", "both"

    def side(self, x: float, y: float) -> float:
        """Signed distance from point (x,y) to line; >0 one side, <0 other."""
        # Vector along line
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        # Vector from (x1,y1) to (x,y)
        px = x - self.x1
        py = y - self.y1
        return dx * py - dy * px


class LineCrossingCounter:
    """Tracks which object IDs have crossed the line and updates counts per class."""

    def __init__(self, line: Line, class_names: Optional[dict] = None):
        self.line = line
        self.class_names = class_names or {}
        self._crossed_ids: set = set()
        self._last_side: dict = {}  # track_id -> last side (-1 or +1)
        self.counts: dict = {}  # class_id -> count

    def set_line(self, line: Line) -> None:
        self.line = line
        self._crossed_ids.clear()
        self._last_side.clear()

    def _center(self, xyxy: np.ndarray) -> Tuple[float, float]:
        x1, y1, x2, y2 = xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(
        self, detections: List[dict]
    ) -> List[dict]:
        """
        Update counts based on current detections. Returns list of crossing events
        in this frame: [{"track_id", "class_id", "direction"}, ...].
        """
        events = []
        for d in detections:
            tid = d.get("track_id")
            if tid is None:
                continue
            cls_id = d.get("class_id", 0)
            xyxy = d.get("xyxy")
            if xyxy is None:
                continue
            cx, cy = self._center(np.array(xyxy))
            side = self.line.side(cx, cy)
            if side > 0:
                current_side = 1
            elif side < 0:
                current_side = -1
            else:
                continue

            key = tid
            if key in self._crossed_ids:
                self._last_side[key] = current_side
                continue
            last = self._last_side.get(key)
            self._last_side[key] = current_side
            if last is not None and last != current_side:
                self._crossed_ids.add(key)
                self.counts[cls_id] = self.counts.get(cls_id, 0) + 1
                direction = "positive" if current_side > last else "negative"
                if self.line.direction == "both" or self.line.direction == direction:
                    events.append({
                        "track_id": tid,
                        "class_id": cls_id,
                        "direction": direction,
                    })
        return events

    def get_total(self) -> int:
        return sum(self.counts.values())

    def get_counts_dict(self) -> dict:
        return dict(self.counts)

    def reset(self) -> None:
        self._crossed_ids.clear()
        self._last_side.clear()
        self.counts.clear()
