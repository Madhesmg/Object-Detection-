"""
End-to-end object detection, tracking, and counting.

Flow: Camera -> YOLOv8 detection + tracking -> line crossing logic
      -> Annotated video (PyQt GUI + MJPEG stream API)
      -> History & Export counts
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

import config
from pipeline import Pipeline
from stream_server import start_background_server
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_UseHighDpiScaleFactorRounding, True)

    pipeline = Pipeline(
        input_source=config.INPUT_SOURCE,
        camera_index=config.CAMERA_INDEX,
    )

    # Start MJPEG stream server (for API)
    start_background_server(config.MJPEG_HOST, config.MJPEG_PORT)
    print(f"MJPEG stream: http://127.0.0.1:{config.MJPEG_PORT}/video")

    window = MainWindow(pipeline)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
