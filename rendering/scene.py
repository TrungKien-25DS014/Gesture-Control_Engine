"""
rendering/scene.py

Layer 4 - Rendering (Giai đoạn E): lớp trung gian thuần Python (không Qt)
giữa world/ (list[WorldAnchor]) và overlay_window.py (vẽ Qt thật).

Tách riêng khỏi overlay_window.py vì 2 lý do:
  1. Đúng nguyên tắc dự án "test/log độc lập từng lớp bằng dữ liệu giả lập,
     không cần camera/hiển thị thật" - build_render_items() là hàm thuần
     (input WorldAnchor giả lập -> output RenderItem), test được không cần
     QApplication/màn hình thật, khác hẳn overlay_window.py phải có Qt event
     loop mới paintEvent() được.
  2. rendering/projection.py chỉ định nghĩa quy tắc phối cảnh cho 1 điểm
     (project()) - file này áp quy tắc đó cho CẢ danh sách anchor, cộng thêm
     phần overlay_window.py cần nhưng projection.py không nên biết: bán kính
     vẽ theo pixel, màu/nhãn lấy từ anchor.metadata, và THỨ TỰ vẽ (object xa
     vẽ trước, object gần vẽ sau -> object gần che khuất object xa, đúng
     cảm giác chiều sâu thật).

Cập nhật Giai đoạn F: RenderItem có thêm field `glow` (0-1) - cường độ hiệu
ứng "chạm/grab" cần vẽ (xem core/touch_state.py). Hàm này CHỈ gắn giá trị
glow đã được tính sẵn từ bên ngoài (`glow_by_id`) vào đúng RenderItem, không
tự tính touch/grab lại - việc "ai đang chạm ai" là của world/ (Giai đoạn D),
việc "làm mượt glow qua thời gian" là của rendering/animation.py (Giai đoạn
F); build_render_items() ở tầng thấp hơn cả hai, chỉ lắp ráp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.world_anchor import WorldAnchor
from rendering.projection import CameraConfig, project

# Màu mặc định khi anchor không gắn metadata["color"] - xám nhạt trung tính.
DEFAULT_COLOR = (200, 200, 200)


@dataclass(frozen=True)
class RenderItem:
    anchor_id: str
    screen_x: float
    screen_y: float
    radius_px: float   # bán kính vẽ trên màn hình - đã nhân cả scale phối cảnh lẫn anchor.scale
    alpha: float        # 0-1, giống ProjectedPoint.alpha (rendering/projection.py)
    depth: float         # depth gốc (0-1) - giữ lại để test thứ tự vẽ, không dùng để vẽ trực tiếp
    color: tuple[int, int, int]
    label: str
    glow: float = 0.0    # Giai đoạn F: 0 = không hiệu ứng, 1 = chạm/grab mạnh nhất (xem module docstring)
    yaw: float = 0.0     # Giai đoạn F (tùy chọn): góc xoay quanh trục dọc (radian) - anchor.rotation[1],
                          # overlay_window.py dùng để vẽ hiệu ứng "đĩa xoay" cho object demo (metadata["spin"])


def build_render_items(
    anchors: list[WorldAnchor],
    camera: CameraConfig,
    base_radius_px: float = 40.0,
    glow_by_id: Optional[dict[str, float]] = None,
) -> list[RenderItem]:
    """
    Project toàn bộ anchor sang tọa độ màn hình, rồi sắp xếp theo depth
    GIẢM DẦN (xa nhất trước, gần nhất sau cùng) - overlay_window.py chỉ cần
    vẽ đúng thứ tự list trả về là tự động có occlusion đúng (object gần che
    object xa), không cần tự tính z-order riêng.

    `glow_by_id`: map anchor_id -> cường độ glow (0-1), thường lấy từ
    SceneAnimator.glow_of() sau khi đã làm mượt (rendering/animation.py).
    Anchor không có trong map -> glow=0.0. Bỏ trống hoàn toàn (None) vẫn
    hoạt động bình thường, glow=0 cho mọi item - giữ tương thích ngược cho
    test/caller cũ chưa quan tâm hiệu ứng Giai đoạn F.
    """
    glow_by_id = glow_by_id or {}
    items = [_to_render_item(anchor, camera, base_radius_px, glow_by_id) for anchor in anchors]
    items.sort(key=lambda item: item.depth, reverse=True)
    return items


def _to_render_item(
    anchor: WorldAnchor,
    camera: CameraConfig,
    base_radius_px: float,
    glow_by_id: dict[str, float],
) -> RenderItem:
    projected = project(anchor.position_3d, camera)
    depth = anchor.position_3d[2]

    return RenderItem(
        anchor_id=anchor.id,
        screen_x=projected.screen_x,
        screen_y=projected.screen_y,
        radius_px=base_radius_px * projected.scale * anchor.scale,
        alpha=projected.alpha,
        depth=depth,
        color=tuple(anchor.metadata.get("color", DEFAULT_COLOR)),
        label=str(anchor.metadata.get("label", "")),
        glow=glow_by_id.get(anchor.id, 0.0),
        yaw=anchor.rotation[1],
    )