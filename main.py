"""
main.py (đặt ở thư mục gốc dự án)

Giai đoạn E - ghép full pipeline Layer 1 -> 2 -> 3 -> 4 chạy realtime lần
đầu tiên. Khác với empty_camera_window.py (Giai đoạn A - chỉ test khung
xương camera + UI, CHƯA có logic gesture nào), đây là entrypoint demo thật
của app: tay điều khiển object trôi nổi trong world-space, vẽ qua phối cảnh
giả (rendering/projection.py).

main.py là nơi DUY NHẤT trong dự án biết về cả 4 lớp cùng lúc - đúng nguyên
tắc kiến trúc "không lớp nào biết chi tiết implementation của lớp khác":
vision/, gesture_engine/, world/, rendering/ chỉ giao tiếp qua data
class/Protocol ở core/interfaces.py, còn việc NỐI chúng lại theo đúng thứ tự
là trách nhiệm của file này, không thuộc về bất kỳ package nào ở trên.

Vòng lặp mỗi frame (QTimer ~30fps):
  1. Đọc 1 frame camera (cv2.VideoCapture) - KHÔNG lật ảnh trước khi đưa vào
     HandTracker (mirrored_view=True mặc định đã tự bù nhãn trái/phải cho
     đúng ảnh gốc chưa lật - xem docstring HandTracker.__init__).
  2. vision.HandTracker.read_frame() -> list[GestureEvent] "thô" (0-2 event,
     1/tay phát hiện được, gesture_type=UNKNOWN).
  3. Mỗi event thô -> gesture_engine.GestureEngine.process() -> list event
     ĐÃ phân loại (luôn có 1 trạng thái nắm/xòe + 0..n event rời rạc).
  4. Mỗi event đã phân loại -> world.WorldState.handle_event() - cập nhật
     hit-test/grab-move/push trigger trên danh sách WorldAnchor.
  5. rendering.OverlayWindow.render(world_state.get_anchors()) - vẽ lại toàn
     bộ scene theo phối cảnh giả, occlusion đúng theo depth.

Tay biến mất khỏi khung hình đủ lâu -> gọi gesture_engine.reset_hand() để
rotation/push-pull không tính delta "nhảy" qua khoảng trống lúc tay vắng mặt
rồi quay lại (xem docstring GestureEngine.reset_hand). Ngưỡng số frame dùng
khớp với HandTracker.reset_after_missing_frames mặc định để 2 lớp nhất quán
với nhau - nếu đổi 1 bên, nhớ đổi bên còn lại.

Chạy: python main.py
"""

from __future__ import annotations

import sys

import cv2
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from core.gesture_event import HandLabel
from core.world_anchor import WorldAnchor
from gesture_engine import GestureEngine
from rendering.overlay_window import OverlayWindow
from rendering.projection import CameraConfig
from vision.hand_tracker import HandTracker
from world.interactions import OpenLinkAction, ShowInfoAction
from world.world_state import WorldState

_CAMERA_INDEX = 0
_FRAME_INTERVAL_MS = 33  # ~30 fps

# Phải khớp HandTracker.reset_after_missing_frames mặc định (vision/hand_tracker.py).
_RESET_AFTER_MISSING_FRAMES = 10


def _demo_anchors() -> list[WorldAnchor]:
    """Vài object demo trôi nổi ở 3 độ sâu khác nhau - đủ để thử phối cảnh
    (to/nhỏ theo depth), grab-move (cả 3 trục) và push (trigger action) ngay
    lần chạy full pipeline đầu tiên, không cần chỉnh sửa gì thêm."""
    return [
        WorldAnchor(
            position_3d=(0.3, 0.4, 0.15),
            metadata={
                "label": "Info",
                "color": (80, 160, 255),
                "action": ShowInfoAction(text="Xin chào từ world-space!"),
            },
        ),
        WorldAnchor(
            position_3d=(0.7, 0.4, 0.5),
            metadata={
                "label": "Docs",
                "color": (255, 170, 60),
                "action": OpenLinkAction(url="https://docs.python.org"),
            },
        ),
        WorldAnchor(
            position_3d=(0.5, 0.7, 0.85),
            metadata={"label": "Far", "color": (120, 220, 120)},
        ),
    ]


class GesturePipeline:
    """Ghép 4 lớp lại thành 1 vòng lặp realtime chạy trên QTimer."""

    def __init__(self, camera_config: CameraConfig, camera_index: int = _CAMERA_INDEX) -> None:
        self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Không mở được camera index={camera_index}")

        self.hand_tracker = HandTracker()
        self.gesture_engine = GestureEngine()
        self.world_state = WorldState()
        for anchor in _demo_anchors():
            self.world_state.add_anchor(anchor)

        self.overlay = OverlayWindow(camera_config)
        self._missing_streak: dict[HandLabel, int] = {HandLabel.LEFT: 0, HandLabel.RIGHT: 0}
        self._timer: QTimer | None = None

    def start(self) -> None:
        self.overlay.showFullScreen()
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(_FRAME_INTERVAL_MS)

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()
        self.hand_tracker.close()
        self.capture.release()

    def _tick(self) -> None:
        ok, frame = self.capture.read()
        if not ok:
            return  # mất frame thoáng qua - không crash, chờ frame kế tiếp

        raw_events = self.hand_tracker.read_frame(frame)
        self._decay_missing_hands({event.hand for event in raw_events})

        for raw_event in raw_events:
            for classified in self.gesture_engine.process(raw_event):
                self.world_state.handle_event(classified)

        self.overlay.render(self.world_state.get_anchors())

    def _decay_missing_hands(self, seen: set[HandLabel]) -> None:
        for hand in (HandLabel.LEFT, HandLabel.RIGHT):
            if hand in seen:
                self._missing_streak[hand] = 0
                continue
            self._missing_streak[hand] += 1
            if self._missing_streak[hand] == _RESET_AFTER_MISSING_FRAMES:
                self.gesture_engine.reset_hand(hand)


def main() -> None:
    app = QApplication(sys.argv)

    screen_geometry = app.primaryScreen().geometry()
    camera_config = CameraConfig(screen_width=screen_geometry.width(), screen_height=screen_geometry.height())

    pipeline = GesturePipeline(camera_config)
    pipeline.start()
    app.aboutToQuit.connect(pipeline.stop)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()