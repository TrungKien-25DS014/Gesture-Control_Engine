"""
rendering/overlay_window.py

Layer 4 - Rendering (Giai đoạn E): implement Protocol Renderer khai báo ở
core/interfaces.py. Cửa sổ overlay trong suốt, luôn nổi trên cùng, click-through
(không chặn chuột/tương tác với app khác bên dưới) - phủ toàn màn hình để mô
phỏng cảm giác "object trôi nổi trong không gian" của AR thật, dù demo chạy
trên webcam + màn hình phẳng.

Ranh giới lớp: OverlayWindow CHỈ đọc list[WorldAnchor] qua render() để vẽ,
không bao giờ chỉnh sửa WorldAnchor hay biết gì về GestureEvent/gesture_engine/
world logic (đúng docstring core/world_anchor.py và world/world_state.py -
"rendering/ chỉ đọc get_anchors() để vẽ, không bao giờ ghi ngược lại").

Phần tính toán phối cảnh (project()) và phần chuẩn bị danh sách vẽ đã sắp
đúng thứ tự (occlusion theo depth) nằm ở rendering/projection.py và
rendering/scene.py - đều là hàm thuần, test được không cần Qt. File này CHỈ
còn phần vẽ Qt thật (QPainter trong paintEvent), không thể test bằng
dữ liệu giả lập thuần túy như 3 lớp kia vì cần 1 QApplication/QWidget thật.

Khi port sang AR hardware thật (kính AR, Vision Pro...), CHỈ file này (và
rendering/scene.py phần bán kính pixel/màu) bị thay bằng renderer 3D thật
của thiết bị - core/, gesture_engine/, world/ giữ nguyên, đúng README "Design
Philosophy" đã viết ở Giai đoạn A.

Cập nhật Giai đoạn F - Polish tương tác:
  - Animation mượt: mọi anchor đi qua SceneAnimator (rendering/animation.py)
    trước khi build_render_items(), nên vị trí/scale vẽ ra LUÔN đuổi theo
    (không nhảy cứng) giá trị thật từ world/. dt tự đo bằng time.monotonic()
    giữa 2 lần render() liên tiếp - OverlayWindow là nơi duy nhất "biết
    thời gian trôi qua bao lâu", world/ và gesture_engine/ không quan tâm
    khái niệm dt của tầng hiển thị.
  - Glow feedback: touch_states (từ world/world_state.touch_states(), qua
    Protocol WorldLogic đã cập nhật ở core/interfaces.py) được đưa vào
    animator để làm mượt rồi vẽ thành vòng sáng quanh object TOUCHED/GRABBED
    - "đổi màu/glow khi depth tay khớp depth object" đúng ghi chú kế hoạch.
  - Pulse khi push: trigger_pulse() được main.py gọi khi
    world_state.pop_pending_push() trả về 1 anchor_id - object "nảy" nhẹ
    rồi tắt dần, không phải instant thay đổi kích thước.
"""

from __future__ import annotations

import math
import time

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QRadialGradient
from PySide6.QtWidgets import QWidget

from core.touch_state import TouchState
from core.world_anchor import WorldAnchor
from rendering.animation import DEFAULT_PUSH_PULSE_AMOUNT, SceneAnimator
from rendering.projection import CameraConfig
from rendering.scene import RenderItem, build_render_items

# Shadow/blur nhẹ thể hiện chiều sâu (kế hoạch Giai đoạn E) - lệch xuống dưới
# 1 chút để trông giống bóng đổ dưới object thay vì viền quanh object.
_SHADOW_OFFSET_PX = 6.0
_SHADOW_SPREAD = 1.3           # bán kính gradient bóng = radius_px * hệ số này
_SHADOW_MAX_ALPHA = 90         # 0-255, độ đậm bóng lúc object gần camera nhất

# Glow (Giai đoạn F): vòng sáng vẽ NGOÀI viền object, càng glow lớn càng
# rộng/đậm. Màu trắng ngả vàng nhẹ - trung tính, không lẫn với màu riêng của
# từng object (anchor.metadata["color"]).
_GLOW_COLOR = (255, 245, 200)
_GLOW_SPREAD = 1.6              # bán kính gradient glow = radius_px * hệ số này lúc glow=1.0
_GLOW_MAX_ALPHA = 150           # 0-255, độ đậm glow lúc glow=1.0 (GRABBED)


class OverlayWindow(QWidget):
    """Implement Protocol Renderer (core/interfaces.py). 1 instance = 1 cửa
    sổ overlay phủ toàn màn hình cho 1 phiên chạy pipeline."""

    def __init__(self, camera: CameraConfig, base_radius_px: float = 40.0, parent=None) -> None:
        super().__init__(parent)
        self._camera = camera
        self._base_radius_px = base_radius_px
        self._anchors: list[WorldAnchor] = []
        self._touch_states: dict[str, TouchState] = {}
        self._animator = SceneAnimator()
        self._last_render_time: float | None = None

        self.setWindowTitle("AR Gesture Overlay")
        # Nền trong suốt hoàn toàn - chỉ object vẽ ra mới hiện, phần còn lại
        # của cửa sổ để lộ desktop/app thật bên dưới.
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        # Click-through: overlay không được chặn chuột/tương tác thật của
        # người dùng với app khác - overlay chỉ để HIỂN THỊ, mọi tương tác
        # thật đi qua gesture (webcam), không qua cửa sổ này.
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool  # không hiện icon riêng ở taskbar, đúng cảm giác "overlay"
        )
        self.resize(camera.screen_width, camera.screen_height)

    # ---- Protocol Renderer (core/interfaces.py) ----

    def render(self, anchors: list[WorldAnchor], touch_states: dict[str, TouchState] | None = None) -> None:
        """Không gọi QPainter trực tiếp ở đây - chỉ lưu lại danh sách anchor
        mới nhất rồi yêu cầu Qt vẽ lại qua update() -> paintEvent(), đúng
        vòng đời vẽ chuẩn của Qt (vẽ ngoài paintEvent là lỗi).

        touch_states mặc định None -> coi như {} (không object nào touched/
        grabbed) - giữ tương thích lời gọi cũ từ trước Giai đoạn F."""
        self._anchors = anchors
        self._touch_states = touch_states if touch_states is not None else {}
        self.update()

    def trigger_pulse(self, anchor_id: str, amount: float = DEFAULT_PUSH_PULSE_AMOUNT) -> None:
        """Giai đoạn F: main.py gọi khi world_state.pop_pending_push() trả
        về 1 anchor_id (push vừa trúng object) - xem SceneAnimator.trigger_pulse."""
        self._animator.trigger_pulse(anchor_id, amount)

    # ---- Qt paint ----

    def paintEvent(self, event) -> None:  # noqa: N802 - tên bắt buộc theo Qt
        dt = self._consume_dt()
        self._animator.update(self._anchors, self._touch_states, dt)
        smoothed_anchors = self._animator.smoothed_anchors(self._anchors)
        glow_by_id = {anchor.id: self._animator.glow_of(anchor.id) for anchor in self._anchors}

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        items = build_render_items(smoothed_anchors, self._camera, self._base_radius_px, glow_by_id)

        # Vẽ theo thứ tự: bóng (tất cả) -> glow (tất cả) -> object (tất cả).
        # Glow nằm GIỮA bóng và object (không phải trên cùng) để không che
        # mất viền/nhãn của chính object đang phát sáng.
        for item in items:
            self._draw_shadow(painter, item)
        for item in items:
            self._draw_glow(painter, item)
        for item in items:
            self._draw_anchor(painter, item)

        painter.end()

    def _consume_dt(self) -> float:
        """dt (giây) kể từ lần paintEvent() trước - dùng cho SceneAnimator.
        Frame đầu tiên (chưa có mốc trước đó) trả 0.0, animator sẽ giữ
        nguyên current = target đã khởi tạo, không animate "từ hư không"."""
        now = time.monotonic()
        if self._last_render_time is None:
            self._last_render_time = now
            return 0.0
        dt = now - self._last_render_time
        self._last_render_time = now
        return dt

    def _draw_glow(self, painter: QPainter, item: RenderItem) -> None:
        if item.glow <= 0.0:
            return

        center = QPointF(item.screen_x, item.screen_y)
        radius = item.radius_px * _GLOW_SPREAD
        alpha = int(_GLOW_MAX_ALPHA * item.glow * item.alpha)
        if alpha <= 0:
            return

        r, g, b = _GLOW_COLOR
        gradient = QRadialGradient(center, radius)
        gradient.setColorAt(0.0, QColor(r, g, b, 0))       # trong suốt ở tâm - glow là VIỀN quanh object
        gradient.setColorAt(0.75, QColor(r, g, b, alpha))
        gradient.setColorAt(1.0, QColor(r, g, b, 0))

        painter.setPen(Qt.NoPen)
        painter.setBrush(gradient)
        painter.drawEllipse(center, radius, radius)

    def _draw_shadow(self, painter: QPainter, item: RenderItem) -> None:
        shadow_alpha = int(_SHADOW_MAX_ALPHA * item.alpha)
        if shadow_alpha <= 0:
            return

        center = QPointF(item.screen_x, item.screen_y + _SHADOW_OFFSET_PX)
        radius = item.radius_px * _SHADOW_SPREAD
        # Gradient tỏa tròn mờ dần ra biên - rẻ hơn nhiều so với áp
        # QGraphicsBlurEffect thật lên từng hình vẽ trong paintEvent, mà vẫn
        # tạo cảm giác "mờ" đúng yêu cầu kế hoạch Giai đoạn E.
        gradient = QRadialGradient(center, radius)
        gradient.setColorAt(0.0, QColor(0, 0, 0, shadow_alpha))
        gradient.setColorAt(1.0, QColor(0, 0, 0, 0))

        painter.setPen(Qt.NoPen)
        painter.setBrush(gradient)
        painter.drawEllipse(center, radius, radius)

    def _draw_anchor(self, painter: QPainter, item: RenderItem) -> None:
        r, g, b = item.color
        alpha_255 = max(0, min(255, int(255 * item.alpha)))
        center = QPointF(item.screen_x, item.screen_y)

        # Giai đoạn F (tùy chọn) - object demo xoay 3D: bóp bề ngang theo
        # cos(yaw) để giả lập đang xoay quanh trục dọc (giống 1 đĩa/đồng xu
        # xoay: nhìn thẳng -> tròn, nhìn cạnh -> dẹt). Sàn tối thiểu 30% bán
        # kính để object không biến mất hoàn toàn lúc "nhìn ngang cạnh".
        # yaw=0 (mọi anchor khác, mặc định) -> cos=1 -> hình tròn như cũ,
        # không ảnh hưởng gì đến object không xoay.
        width_scale = max(0.3, abs(math.cos(item.yaw)))
        radius_x = item.radius_px * width_scale
        radius_y = item.radius_px

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(r, g, b, alpha_255))
        painter.drawEllipse(center, radius_x, radius_y)

        if not item.label:
            return
        painter.setPen(QColor(255, 255, 255, alpha_255))
        painter.setFont(QFont("Arial", max(8, int(item.radius_px * 0.35))))
        rect = QRectF(
            item.screen_x - item.radius_px,
            item.screen_y - item.radius_px,
            item.radius_px * 2,
            item.radius_px * 2,
        )
        painter.drawText(rect, Qt.AlignCenter, item.label)