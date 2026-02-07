"""
Day-based storage: every 24h uses a separate folder (Monday, Tuesday, ...) with date.
Videos and counts stored by day, downloadable.
"""
from datetime import datetime
from pathlib import Path

STORAGE_ROOT = Path(__file__).resolve().parent / "storage"


def get_day_name() -> str:
    """Current weekday: Monday, Tuesday, ..., Sunday."""
    return datetime.now().strftime("%A")


def get_day_date_str() -> str:
    """e.g. Monday_2026-02-07"""
    return f"{get_day_name()}_{datetime.now().strftime('%Y-%m-%d')}"


def get_day_folder() -> Path:
    """Path to today's folder (e.g. storage/Saturday_2026-02-07). Creates if needed."""
    folder = STORAGE_ROOT / get_day_date_str()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def get_default_video_path() -> Path:
    """Default path: storage/<Day>_<date>/annotated_YYYY-MM-DD_HH-MM-SS.mp4"""
    return get_day_folder() / datetime.now().strftime("annotated_%Y-%m-%d_%H-%M-%S.mp4")


def get_default_counts_csv_path() -> Path:
    return get_day_folder() / datetime.now().strftime("counts_%Y-%m-%d_%H-%M.csv")


def list_day_folders() -> list[Path]:
    """List all day folders (e.g. Monday_2026-02-07) in storage, newest first."""
    if not STORAGE_ROOT.exists():
        return []
    folders = [p for p in STORAGE_ROOT.iterdir() if p.is_dir()]
    folders.sort(key=lambda p: p.name, reverse=True)
    return folders


def list_videos_in_folder(folder: Path) -> list[Path]:
    """List MP4 videos in a day folder."""
    if not folder.exists():
        return []
    return sorted(folder.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
