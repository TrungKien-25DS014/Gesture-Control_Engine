"""
core/gesture_event.py

GestureEvent: output của gesture_engine/, input của world/.

Khác với launcher 2D thông thường, event luôn mang position_3d (x, y, depth)
thay vì pos=(x, y). depth ở giai đoạn A/B là ước lượng tương đối, không phải
khoảng cách mét tuyệt đối - độ chính xác tuyệt đối không quan trọng bằng
việc ổn định và đúng chiều tăng/giảm.

Loại gesture cụ thể (pinch, push, pull, rotation...) được hiện thực đầy đủ
trong gesture_engine/state_machine.py và gesture_engine/depth_gestures.py.
Ở đây chỉ định nghĩa khung dữ liệu chung.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class HandLabel(Enum):
    LEFT = auto()
    RIGHT = auto()


class GestureType(Enum):
    """Giá trị thật được bổ sung khi hiện thực gesture_engine/."""
    UNKNOWN = auto()


@dataclass(frozen=True)
class GestureEvent:
    hand: HandLabel
    x: float                       # tọa độ ngang, normalized 0-1
    y: float                       # tọa độ dọc, normalized 0-1
    depth: float                   # ước lượng độ sâu tương đối, 0-1 (0 = gần camera nhất)
    timestamp: float               # giây, time.monotonic()
    gesture_type: GestureType = GestureType.UNKNOWN
    confidence: float = 1.0
    raw_landmarks: Optional[tuple] = None   # giữ landmark gốc để debug/test

    @property
    def position_3d(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.depth)