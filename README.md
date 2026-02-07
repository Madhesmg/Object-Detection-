# Object Detection, Tracking & Counting (End-to-End)

Real-time object detection (YOLOv8), tracking (ByteTrack), and line-crossing count with PyQt GUI and MJPEG stream API.

## Architecture

- **Input:** Webcam, IP camera (RTSP/URL), or video file
- **YOLOv8:** Detects Person, Car, Bike, Bus, Truck, etc. (COCO classes)
- **Tracking:** Assigns stable IDs to objects (Ultralytics built-in)
- **Line crossing:** Counts objects crossing a user-defined line
- **Output:** Annotated video in PyQt GUI + MJPEG stream for API
- **History & Export:** Log counts, export CSV

## Setup

```bash
cd "d:\RAACTS-Project\object detection"
pip install -r requirements.txt
```

First run will download the YOLOv8 model (e.g. `yolov8n.pt`).

## Run

```bash
python main.py
```

- **Start** – begin camera capture, detection, tracking, and counting
- **Define line** – set the counting line (x1, y1, x2, y2) in pixels
- **Reset counts** – clear counters
- **Export counts** – save current counts to CSV
- **History** – view recent count history from CSV

## MJPEG stream (API)

While the app is running, the annotated video is available at:

- **Stream URL:** `http://127.0.0.1:8765/video`
- **Web page:** `http://127.0.0.1:8765/`

Use these in other apps or browsers.

## Configuration

Edit `config.py`:

- `CAMERA_INDEX` – default webcam (0)
- `INPUT_SOURCE` – set to RTSP URL, file path, or IP camera URL
- `YOLO_MODEL` – `yolov8n.pt` (nano) to `yolov8x.pt` (largest)
- `MJPEG_PORT` – port for stream server (default 8765)
- `EXPORT_DIR` – folder for exports and history CSV

## Exports

- **exports/count_history.csv** – appended on line-crossing events
- **Export counts** button – saves current per-class counts to a CSV file of your choice

## Requirements

- Python 3.8+
- Webcam or other video source
- See `requirements.txt` for packages
