"""Data flow for object counts: storage, history, export."""
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import csv
import json
import threading
from typing import Optional


@dataclass
class CountRecord:
    timestamp: str
    class_name: str
    total_so_far: int


class CountsStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._history: list[CountRecord] = []
        self._totals: dict[str, int] = {}

    def add_count(self, class_name: str) -> None:
        with self._lock:
            self._totals[class_name] = self._totals.get(class_name, 0) + 1
            self._history.append(CountRecord(
                timestamp=datetime.now().isoformat(),
                class_name=class_name,
                total_so_far=self._totals[class_name],
            ))

    def get_totals(self) -> dict[str, int]:
        with self._lock:
            return dict(self._totals)

    def get_history(self) -> list[CountRecord]:
        with self._lock:
            return list(self._history)

    def export_csv(self, path: str | Path) -> None:
        path = Path(path)
        with self._lock:
            rows = [asdict(r) for r in self._history]
        if not rows:
            path.write_text("timestamp,class_name,total_so_far\n", encoding="utf-8")
        else:
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["timestamp", "class_name", "total_so_far"])
                w.writeheader()
                w.writerows(rows)

    def export_json(self, path: str | Path) -> None:
        path = Path(path)
        with self._lock:
            data = {"totals": dict(self._totals), "history": [asdict(r) for r in self._history]}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def clear(self) -> None:
        with self._lock:
            self._history.clear()
            self._totals.clear()
