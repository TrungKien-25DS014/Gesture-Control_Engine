"""
core/world_anchor.py

WorldAnchor: đơn vị dữ liệu trung tâm của hệ thống.

Mọi object trong không gian ảo là một WorldAnchor. Layer world (world_state.py)
là nơi duy nhất sở hữu và chỉnh sửa WorldAnchor; rendering chỉ đọc để vẽ,
không bao giờ ghi ngược lại.

Hệ tọa độ: gốc (0,0,0) đặt tại vị trí camera, đơn vị mét.
Trục z dương = càng xa camera (đi vào màn hình).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any
import uuid

Vector3 = tuple[float, float, float]


@dataclass
class WorldAnchor:
    position_3d: Vector3
    rotation: Vector3 = (0.0, 0.0, 0.0)      # euler (pitch, yaw, roll) - radian
    scale: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def distance_to(self, point: Vector3) -> float:
        dx = self.position_3d[0] - point[0]
        dy = self.position_3d[1] - point[1]
        dz = self.position_3d[2] - point[2]
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def translated(self, delta: Vector3) -> "WorldAnchor":
        """Trả về bản sao đã dịch chuyển - dùng cho grab-and-move ở world_state.py."""
        new_pos = tuple(p + d for p, d in zip(self.position_3d, delta))
        return replace(self, position_3d=new_pos)