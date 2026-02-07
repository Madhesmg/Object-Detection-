"""
Main entry: live video -> YOLO + tracking -> line crossing -> PyQt GUI.
Left: live video. Right: counts, history, export, download (by day).
"""
import sys

# Import torch BEFORE PyQt to avoid WinError 1114 (DLL init failed) on Windows
import torch  # noqa: E402

from counts_store import CountsStore
from stream_server import start_background_server
from gui import MainWindow
import config
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt


def main():
    counts_store = CountsStore()
    start_background_server(config.MJPEG_HOST, config.MJPEG_PORT)
    print(f"MJPEG stream: http://127.0.0.1:{config.MJPEG_PORT}/video")

    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_UseHighDpiScaleFactorRounding, True)
    window = MainWindow(counts_store)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
