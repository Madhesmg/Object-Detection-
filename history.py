"""History and export for count data."""

import os
import csv
from datetime import datetime
from typing import List, Dict
import config

# Standard columns for history CSV (consistent across rows)
HISTORY_COLUMNS = [
    "timestamp",
    "total",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
]


def ensure_export_dir():
    os.makedirs(config.EXPORT_DIR, exist_ok=True)


def append_history_row(counts: Dict[int, int], total: int, class_names: dict = None):
    """Append one row to history CSV (timestamp, total, class counts)."""
    ensure_export_dir()
    path = os.path.join(config.EXPORT_DIR, config.HISTORY_CSV)
    class_names = class_names or {}
    # Map class_id -> name; use only names we have in HISTORY_COLUMNS
    name_to_id = {v: k for k, v in class_names.items()}
    row = {"timestamp": datetime.now().isoformat(), "total": total}
    for col in HISTORY_COLUMNS:
        if col in ("timestamp", "total"):
            continue
        cls_id = name_to_id.get(col)
        row[col] = counts.get(cls_id, 0) if cls_id is not None else 0
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def export_counts_csv(counts: Dict[int, int], class_names: dict, filepath: str = None) -> str:
    """Export current counts to a CSV file. Returns path used."""
    ensure_export_dir()
    if filepath is None:
        filepath = os.path.join(
            config.EXPORT_DIR,
            f"counts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
    total = sum(counts.values())
    rows = [
        {"class_id": k, "class_name": class_names.get(k, str(k)), "count": v}
        for k, v in sorted(counts.items())
    ]
    rows.append({"class_id": "", "class_name": "TOTAL", "count": total})
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "count"])
        writer.writeheader()
        writer.writerows(rows)
    return filepath


def read_history_csv(limit: int = 1000) -> List[dict]:
    """Read recent history rows from CSV."""
    path = os.path.join(config.EXPORT_DIR, config.HISTORY_CSV)
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            if len(rows) >= limit:
                break
    return list(reversed(rows))
