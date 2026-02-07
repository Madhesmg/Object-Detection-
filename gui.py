"""
PyQt GUI: Left = live video, Right = counts + history + controls + download.
Video stored by day (Monday_2026-02-07), downloadable.
"""
import sys
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
    QDialog,
    QFormLayout,
    QScrollArea,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

import cv2

import config
from pipeline import Pipeline, COCO_NAMES
from counts_store import CountsStore
from storage_paths import (
    get_default_video_path,
    get_default_counts_csv_path,
    get_default_counts_json_path,
    list_day_folders,
    list_videos_in_folder,
    STORAGE_ROOT,
)
import stream_server


class DefineLineDialog(QDialog):
    def __init__(self, parent=None, x1=690, y1=430, x2=1130, y2=430):
        super().__init__(parent)
        self.setWindowTitle("Define counting line")
        layout = QFormLayout(self)
        self.spin_x1 = QSpinBox()
        self.spin_x1.setRange(0, 9999)
        self.spin_x1.setValue(int(x1))
        self.spin_y1 = QSpinBox()
        self.spin_y1.setRange(0, 9999)
        self.spin_y1.setValue(int(y1))
        self.spin_x2 = QSpinBox()
        self.spin_x2.setRange(0, 9999)
        self.spin_x2.setValue(int(x2))
        self.spin_y2 = QSpinBox()
        self.spin_y2.setRange(0, 9999)
        self.spin_y2.setValue(int(y2))
        layout.addRow("Point 1 (x1, y1):", None)
        h1 = QHBoxLayout()
        h1.addWidget(self.spin_x1)
        h1.addWidget(self.spin_y1)
        layout.addRow(h1)
        layout.addRow("Point 2 (x2, y2):", None)
        h2 = QHBoxLayout()
        h2.addWidget(self.spin_x2)
        h2.addWidget(self.spin_y2)
        layout.addRow(h2)
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        layout.addRow(ok_btn)

    def get_line(self):
        return self.spin_x1.value(), self.spin_y1.value(), self.spin_x2.value(), self.spin_y2.value()


class MainWindow(QMainWindow):
    def __init__(self, counts_store: CountsStore):
        super().__init__()
        self.counts_store = counts_store
        self.pipeline = Pipeline(
            input_source=config.INPUT_SOURCE,
            camera_index=config.CAMERA_INDEX,
        )
        self.pipeline.on_count(self._on_count_event)
        self.setWindowTitle("Object Detection & Counting - Live")
        self.setMinimumSize(1200, 700)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Horizontal splitter: Left = video, Right = data
        splitter = QSplitter(Qt.Horizontal)

        # Left: Video
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background-color: #1e1e1e; color: #888; font-size: 14px;")
        self.video_label.setText("Click Start to show live video")
        video_layout.addWidget(self.video_label, stretch=1)
        splitter.addWidget(video_widget)

        # Right: Counts + controls + history + download
        right_widget = QWidget()
        right_widget.setMinimumWidth(350)
        right_layout = QVBoxLayout(right_widget)

        # Controls
        controls = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setEnabled(False)
        self.btn_line = QPushButton("Define line")
        self.btn_line.clicked.connect(self._on_define_line)
        self.btn_reset = QPushButton("Reset counts")
        self.btn_reset.clicked.connect(self._on_reset)
        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_stop)
        controls.addWidget(self.btn_line)
        controls.addWidget(self.btn_reset)
        right_layout.addLayout(controls)

        self.chk_record = QCheckBox("Record video (saved to day folder: Monday_2026-02-07)")
        right_layout.addWidget(self.chk_record)

        # Current counts
        counts_grp = QGroupBox("Current counts")
        counts_layout = QVBoxLayout(counts_grp)
        self.counts_label = QLabel("—")
        self.counts_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        counts_layout.addWidget(self.counts_label)
        right_layout.addWidget(counts_grp)

        # History table
        hist_grp = QGroupBox("Count history")
        hist_layout = QVBoxLayout(hist_grp)
        self.history_table = QTableWidget(0, 3)
        self.history_table.setHorizontalHeaderLabels(["Time", "Class", "Total so far"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        hist_layout.addWidget(self.history_table)
        right_layout.addWidget(hist_grp)

        # Export counts
        exp_layout = QHBoxLayout()
        self.btn_export_csv = QPushButton("Export counts (CSV)")
        self.btn_export_csv.clicked.connect(self._export_csv)
        self.btn_export_json = QPushButton("Export counts (JSON)")
        self.btn_export_json.clicked.connect(self._export_json)
        exp_layout.addWidget(self.btn_export_csv)
        exp_layout.addWidget(self.btn_export_json)
        right_layout.addLayout(exp_layout)

        # Download section: by day (Monday_2026-02-07)
        dl_grp = QGroupBox("Download videos (by day)")
        dl_layout = QVBoxLayout(dl_grp)
        dl_layout.addWidget(QLabel("Day folders (Monday_2026-02-07):"))
        self.day_list = QListWidget()
        self.day_list.itemClicked.connect(self._on_day_selected)
        dl_layout.addWidget(self.day_list)
        dl_layout.addWidget(QLabel("Videos in selected day:"))
        self.video_list = QListWidget()
        self.video_list.itemDoubleClicked.connect(self._on_video_download)
        dl_layout.addWidget(self.video_list)
        self.btn_refresh_days = QPushButton("Refresh list")
        self.btn_refresh_days.clicked.connect(self._refresh_day_list)
        dl_layout.addWidget(self.btn_refresh_days)
        right_layout.addWidget(dl_grp)

        splitter.addWidget(right_widget)
        splitter.setSizes([700, 450])
        main_layout.addWidget(splitter)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_display)
        self._history_timer = QTimer(self)
        self._history_timer.timeout.connect(self._refresh_history)
        self._history_timer.start(1000)
        self._refresh_day_list()

    def _on_count_event(self, events):
        for e in events:
            cls_id = e.get("class_id")
            name = COCO_NAMES.get(cls_id, str(cls_id))
            self.counts_store.add_count(name)

    def _on_start(self):
        if not Path(config.YOLO_MODEL).exists():
            QMessageBox.warning(self, "Missing model", f"Model not found: {config.YOLO_MODEL}")
            return
        self.counts_store.clear()
        record_path = None
        if self.chk_record.isChecked():
            record_path = str(get_default_video_path())
            self.pipeline.set_record_path(record_path)
        else:
            self.pipeline.set_record_path(None)
        self.pipeline.start()
        self._timer.start(30)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.video_label.setText("")

    def _on_stop(self):
        self._timer.stop()
        self.pipeline.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.video_label.setText("Stopped")
        if self.chk_record.isChecked():
            QMessageBox.information(self, "Recording", "Video saved to day folder.")

    def _on_define_line(self):
        line = self.pipeline.line
        dlg = DefineLineDialog(self, line.x1, line.y1, line.x2, line.y2)
        if dlg.exec_() == QDialog.Accepted:
            x1, y1, x2, y2 = dlg.get_line()
            self.pipeline.set_line(float(x1), float(y1), float(x2), float(y2))

    def _on_reset(self):
        self.pipeline.reset_counts()
        self.counts_store.clear()
        self._refresh_history()

    def _update_display(self):
        frame, _ = self.pipeline.get_latest()
        if frame is not None:
            h, w = frame.shape[:2]
            bytes_per_line = 3 * w
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            else:
                qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            scaled = pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled)
        counts = self.pipeline.get_counts()
        total = self.pipeline.get_total_count()
        parts = [f"{COCO_NAMES.get(k, k)}: {v}" for k, v in sorted(counts.items())]
        self.counts_label.setText(f"Total: {total}  |  " + "  |  ".join(parts) if parts else f"Total: {total}")

    def _refresh_history(self):
        history = self.counts_store.get_history()
        self.history_table.setRowCount(len(history))
        for i, r in enumerate(history):
            self.history_table.setItem(i, 0, QTableWidgetItem(r.timestamp))
            self.history_table.setItem(i, 1, QTableWidgetItem(r.class_name))
            self.history_table.setItem(i, 2, QTableWidgetItem(str(r.total_so_far)))
        if history:
            self.history_table.scrollToBottom()

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export counts (CSV)", str(get_default_counts_csv_path()), "CSV (*.csv)"
        )
        if path:
            try:
                self.counts_store.export_csv(path)
                QMessageBox.information(self, "Export", f"Saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Export failed", str(e))

    def _export_json(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export counts (JSON)", str(get_default_counts_json_path()), "JSON (*.json)"
        )
        if path:
            try:
                self.counts_store.export_json(path)
                QMessageBox.information(self, "Export", f"Saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Export failed", str(e))

    def _refresh_day_list(self):
        self.day_list.clear()
        for folder in list_day_folders():
            self.day_list.addItem(QListWidgetItem(folder.name))
        self.video_list.clear()

    def _on_day_selected(self, item=None):
        if item is None:
            return
        day_name = item.text()
        folder = STORAGE_ROOT / day_name
        self.video_list.clear()
        for v in list_videos_in_folder(folder):
            self.video_list.addItem(QListWidgetItem(v.name))

    def _on_video_download(self, item):
        day_item = self.day_list.currentItem()
        if not day_item:
            return
        day_name = day_item.text()
        video_name = item.text()
        src = STORAGE_ROOT / day_name / video_name
        if not src.exists():
            QMessageBox.warning(self, "Error", f"File not found: {src}")
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save video as", str(src), "MP4 (*.mp4)"
        )
        if dest:
            import shutil
            try:
                shutil.copy2(str(src), dest)
                QMessageBox.information(self, "Download", f"Saved to {dest}")
            except Exception as e:
                QMessageBox.critical(self, "Download failed", str(e))

    def closeEvent(self, event):
        self._timer.stop()
        self._history_timer.stop()
        self.pipeline.stop()
        event.accept()
