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
    """gesture_engine/: nhận raw landmark, sinh GestureEvent có ý nghĩa (pinch/push/...)."""

    def process(self, raw_event: GestureEvent) -> GestureEvent: ...


class WorldLogic(Protocol):
    """world/: nhận GestureEvent, cập nhật danh sách WorldAnchor."""

    def handle_event(self, event: GestureEvent) -> None: ...
    def get_anchors(self) -> list[WorldAnchor]: ...


class Renderer(Protocol):
    """rendering/: nhận danh sách WorldAnchor, vẽ lên màn hình."""

    def render(self, anchors: list[WorldAnchor]) -> None: ...