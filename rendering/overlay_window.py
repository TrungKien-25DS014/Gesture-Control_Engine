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
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QRadialGradient
from PySide6.QtWidgets import QWidget

from core.world_anchor import WorldAnchor
from rendering.projection import CameraConfig
from rendering.scene import RenderItem, build_render_items

# Shadow/blur nhẹ thể hiện chiều sâu (kế hoạch Giai đoạn E) - lệch xuống dưới
# 1 chút để trông giống bóng đổ dưới object thay vì viền quanh object.
_SHADOW_OFFSET_PX = 6.0
_SHADOW_SPREAD = 1.3           # bán kính gradient bóng = radius_px * hệ số này
_SHADOW_MAX_ALPHA = 90         # 0-255, độ đậm bóng lúc object gần camera nhất


class OverlayWindow(QWidget):
    """Implement Protocol Renderer (core/interfaces.py). 1 instance = 1 cửa
    sổ overlay phủ toàn màn hình cho 1 phiên chạy pipeline."""

    def __init__(self, camera: CameraConfig, base_radius_px: float = 40.0, parent=None) -> None:
        super().__init__(parent)
        self._camera = camera
        self._base_radius_px = base_radius_px
        self._anchors: list[WorldAnchor] = []

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

    def render(self, anchors: list[WorldAnchor]) -> None:
        """Không gọi QPainter trực tiếp ở đây - chỉ lưu lại danh sách anchor
        mới nhất rồi yêu cầu Qt vẽ lại qua update() -> paintEvent(), đúng
        vòng đời vẽ chuẩn của Qt (vẽ ngoài paintEvent là lỗi)."""
        self._anchors = anchors
        self.update()

    # ---- Qt paint ----

    def paintEvent(self, event) -> None:  # noqa: N802 - tên bắt buộc theo Qt
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        items = build_render_items(self._anchors, self._camera, self._base_radius_px)

        # Vẽ bóng trước cho TẤT CẢ object rồi mới vẽ object - tránh bóng của
        # object gần đè lên thân của object xa (đã được vẽ đúng thứ tự
        # depth ở build_render_items, nhưng bóng không nên chen vào giữa).
        for item in items:
            self._draw_shadow(painter, item)
        for item in items:
            self._draw_anchor(painter, item)

        painter.end()

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

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(r, g, b, alpha_255))
        painter.drawEllipse(center, item.radius_px, item.radius_px)

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