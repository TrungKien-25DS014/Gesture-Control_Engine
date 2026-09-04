"""
rendering/projection.py

Fake perspective projection: quy tắc duy nhất rendering/ dùng để biến
position_3d (world-space, từ core.WorldAnchor) thành pixel trên màn hình phẳng.

Đây KHÔNG phải camera 3D thật (không ma trận view/projection chuẩn OpenGL) -
chỉ là xấp xỉ đơn giản đủ để tạo cảm giác chiều sâu: vật ở depth lớn hơn thì
nhỏ hơn, mờ hơn.

Khi port sang AR hardware thật, TOÀN BỘ package rendering/ bị thay bằng
renderer 3D thật của thiết bị. core/, gesture_engine/, world/ giữ nguyên.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraConfig:
    screen_width: int
    screen_height: int
    focal_length: float = 1.0     # càng lớn -> hiệu ứng phối cảnh càng yếu
    min_depth: float = 0.0        # depth ước lượng gần camera nhất
    max_depth: float = 1.0        # depth ước lượng xa camera nhất


@dataclass(frozen=True)
class ProjectedPoint:
    screen_x: float
    screen_y: float
    scale: float         # 1.0 = kích thước gốc, <1 = nhỏ dần theo depth
    alpha: float          # 1.0 = full opacity, <1 = mờ dần theo depth


def project(position_3d: tuple[float, float, float], camera: CameraConfig) -> ProjectedPoint:
    x, y, depth = position_3d
    depth_norm = _clamp(depth, camera.min_depth, camera.max_depth)

    # depth càng lớn -> chia cho số càng lớn -> scale càng nhỏ
    scale = camera.focal_length / (camera.focal_length + depth_norm)

    center_x, center_y = camera.screen_width / 2, camera.screen_height / 2
    screen_x = center_x + (x - 0.5) * camera.screen_width * scale
    screen_y = center_y + (y - 0.5) * camera.screen_height * scale

    alpha = 1.0 - 0.6 * depth_norm  # vật xa mờ đi tối đa 60%

    return ProjectedPoint(screen_x=screen_x, screen_y=screen_y, scale=scale, alpha=max(alpha, 0.1))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))