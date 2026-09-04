"""
empty_camera_window.py (đặt ở thư mục gốc dự án)

Giai đoạn A - kiểm tra khung xương: mở cửa sổ, đọc camera, hiển thị hình
trực tiếp lên màn hình. KHÔNG có logic gesture ở đây - chỉ chứng minh
pipeline camera -> UI chạy được trước khi ghép vision/ + gesture_engine/
+ world/ + rendering/ thật.

Chạy: python empty_camera_window.py
"""

from __future__ import annotations

import sys

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow


class EmptyCameraWindow(QMainWindow):
    def __init__(self, camera_index: int = 0) -> None:
        super().__init__()
        self.setWindowTitle("AR Gesture Input - Camera Preview")

        self.label = QLabel(alignment=Qt.AlignCenter)
        self.setCentralWidget(self.label)
        self.resize(960, 540)

        self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Không mở được camera index={camera_index}")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(33)  # ~30 fps

    def _update_frame(self) -> None:
        ok, frame = self.capture.read()
        if not ok:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(image).scaled(
            self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.capture.release()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = EmptyCameraWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()