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
"""

from __future__ import annotations

from dataclasses import dataclass

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


def build_render_items(
    anchors: list[WorldAnchor],
    camera: CameraConfig,
    base_radius_px: float = 40.0,
) -> list[RenderItem]:
    """
    Project toàn bộ anchor sang tọa độ màn hình, rồi sắp xếp theo depth
    GIẢM DẦN (xa nhất trước, gần nhất sau cùng) - overlay_window.py chỉ cần
    vẽ đúng thứ tự list trả về là tự động có occlusion đúng (object gần che
    object xa), không cần tự tính z-order riêng.
    """
    items = [_to_render_item(anchor, camera, base_radius_px) for anchor in anchors]
    items.sort(key=lambda item: item.depth, reverse=True)
    return items


def _to_render_item(anchor: WorldAnchor, camera: CameraConfig, base_radius_px: float) -> RenderItem:
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
    )