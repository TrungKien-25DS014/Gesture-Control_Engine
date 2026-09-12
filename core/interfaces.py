"""
core/interfaces.py

Ranh giới giao tiếp giữa các package (vision, gesture_engine, world, rendering).
Mỗi package chỉ được phép phụ thuộc vào các Protocol/data class ở core/,
không được import trực tiếp implementation của package khác.

vision/          -> GestureEvent (raw) -> gesture_engine/
gesture_engine/  -> GestureEvent (đã phân loại) -> world/
world/           -> list[WorldAnchor] -> rendering/
"""

from __future__ import annotations

from typing import Protocol

from core.gesture_event import GestureEvent
from core.touch_state import TouchState
from core.world_anchor import WorldAnchor


class VisionSource(Protocol):
    """
    vision/: nhận 1 frame ảnh, trả về GestureEvent "raw" (chưa phân loại
    gesture) cho mỗi tay phát hiện được trong frame đó.

    Cập nhật Giai đoạn B: đổi từ `GestureEvent | None` (giả định 1 tay)
    sang `list[GestureEvent]` vì cần detect đồng thời 2 tay. Giai đoạn A
    chưa hiện thực logic gesture nên chưa chốt cứng số lượng tay - đây là
    thay đổi hợp lệ trong phạm vi Protocol, không phá vỡ ranh giới giữa
    các lớp (world/ và gesture_engine/ vẫn chỉ biết đến GestureEvent).
    """

    def read_frame(self, frame_bgr) -> list[GestureEvent]: ...


class GestureEngine(Protocol):
    """
    gesture_engine/: nhận 1 GestureEvent "thô" (1 tay, 1 frame, gesture_type
    UNKNOWN, mang raw_landmarks) từ vision/, sinh ra GestureEvent đã phân
    loại (pinch/push/rotate/...).

    Cập nhật Giai đoạn C: process() ban đầu khai báo trả về 1 GestureEvent
    duy nhất. Thực tế 1 frame của 1 tay có thể sinh nhiều phân loại cùng
    lúc (vd: đang HAND_OPEN vừa lúc ROTATE, hoặc PINCH_START trùng frame
    với PUSH) - không có state machine đơn nào gộp hết mà không mất thông
    tin. Đổi sang `list[GestureEvent]` (0 hoặc nhiều phần tử: luôn có đúng
    1 event trạng thái nắm/xòe + 0..n event rời rạc). Tiền lệ giống hệt
    VisionSource.read_frame() đã đổi ở Giai đoạn B vì lý do tương tự - đây
    là thay đổi hợp lệ trong phạm vi Protocol, không phá vỡ ranh giới giữa
    các lớp (world/ vẫn chỉ biết đến GestureEvent, không biết gì về cách
    gesture_engine/ tính toán ra chúng).
    """

    def process(self, raw_event: GestureEvent) -> list[GestureEvent]: ...


class WorldLogic(Protocol):
    """world/: nhận GestureEvent, cập nhật danh sách WorldAnchor.

    Cập nhật Giai đoạn F: thêm touch_states() - world/ là nơi DUY NHẤT biết
    tay nào đang chạm/grab anchor nào (qua hit-test 3D đã có sẵn từ Giai
    đoạn D), nên feedback trực quan "glow khi depth tay khớp object" phải
    đọc từ đây, không phải rendering/ tự đoán lại hit-test lần 2 (tránh 2
    nơi tính cùng 1 logic, dễ lệch nhau). rendering/ (Giai đoạn E/F) chỉ đọc
    kết quả, không tự tính - đúng ranh giới 4 lớp xuyên suốt dự án.
    """

    def handle_event(self, event: GestureEvent) -> None: ...
    def get_anchors(self) -> list[WorldAnchor]: ...
    def touch_states(self) -> dict[str, TouchState]: ...


class Renderer(Protocol):
    """rendering/: nhận danh sách WorldAnchor, vẽ lên màn hình.

    Cập nhật Giai đoạn F: touch_states mặc định {} (dict rỗng = không object
    nào touched/grabbed) để renderer cũ (nếu có) không bắt buộc phải đổi
    chữ ký ngay - phù hợp nguyên tắc "port sang AR chỉ đổi rendering/".
    """

    def render(self, anchors: list[WorldAnchor], touch_states: dict[str, TouchState] = ...) -> None: ...