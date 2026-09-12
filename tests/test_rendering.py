"""
tests/test_rendering.py

Layer 4 - Rendering (Giai đoạn E): test bằng WorldAnchor/CameraConfig giả
lập đúng nguyên tắc dự án - KHÔNG cần camera thật, KHÔNG cần màn hình thật
cho phần logic phối cảnh (rendering/projection.py, rendering/scene.py đều là
hàm thuần).

DoD Giai đoạn E cần verify được bằng test tự động (phần còn lại - "cảm giác
di chuyển tự nhiên" khi cầm tay thật - chỉ verify được thủ công, đúng bản
chất demo tương tác):
  1. Tay tiến/lùi (depth đổi) -> object to/nhỏ ĐÚNG chiều phối cảnh (xa hơn
     -> scale nhỏ hơn, mờ hơn).
  2. Object ở các depth khác nhau được vẽ ĐÚNG THỨ TỰ (xa vẽ trước, gần vẽ
     sau) để occlusion tự nhiên - đây là phần world_state.py grab-move dịch
     chuyển object theo cả 3 trục (kể cả depth) mà rendering/ phải phản ánh
     đúng khi vẽ lại.
  3. OverlayWindow (Qt thật) không crash khi render() với danh sách anchor
     rỗng hoặc có anchor - smoke test tối thiểu cho phần không thể test bằng
     dữ liệu giả lập thuần túy (paintEvent cần QApplication thật).
"""

from __future__ import annotations

import pytest

from core.touch_state import TouchState
from core.world_anchor import WorldAnchor
from rendering.projection import CameraConfig, project
from rendering.scene import DEFAULT_COLOR, build_render_items

# ---------------------------------------------------------------------------
# rendering/projection.py - phối cảnh giả cho 1 điểm
# ---------------------------------------------------------------------------


class TestProject:
    def test_farther_depth_is_smaller_and_more_transparent(self) -> None:
        camera = CameraConfig(screen_width=1000, screen_height=800)

        near = project((0.5, 0.5, 0.0), camera)
        far = project((0.5, 0.5, 1.0), camera)

        assert far.scale < near.scale
        assert far.alpha < near.alpha

    def test_scale_and_alpha_change_monotonically_with_depth(self) -> None:
        """Tay tiến/lùi liên tục (Giai đoạn E DoD) -> scale/alpha phải đổi
        ĐÚNG CHIỀU đơn điệu theo depth, không nhảy loạn giữa các mức."""
        camera = CameraConfig(screen_width=1000, screen_height=800)
        depths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        points = [project((0.5, 0.5, d), camera) for d in depths]

        scales = [p.scale for p in points]
        alphas = [p.alpha for p in points]
        assert scales == sorted(scales, reverse=True)
        assert alphas == sorted(alphas, reverse=True)

    def test_center_point_projects_to_screen_center(self) -> None:
        camera = CameraConfig(screen_width=1000, screen_height=800)
        point = project((0.5, 0.5, 0.0), camera)

        assert point.screen_x == pytest.approx(500)
        assert point.screen_y == pytest.approx(400)

    def test_depth_is_clamped_to_camera_range(self) -> None:
        camera = CameraConfig(screen_width=1000, screen_height=800, min_depth=0.0, max_depth=1.0)

        beyond_far = project((0.5, 0.5, 5.0), camera)
        at_far = project((0.5, 0.5, 1.0), camera)

        assert beyond_far.scale == pytest.approx(at_far.scale)
        assert beyond_far.alpha == pytest.approx(at_far.alpha)

    def test_alpha_never_drops_below_floor(self) -> None:
        camera = CameraConfig(screen_width=1000, screen_height=800)
        point = project((0.5, 0.5, 1.0), camera)

        assert point.alpha >= 0.1


# ---------------------------------------------------------------------------
# rendering/scene.py - build_render_items: chuẩn bị danh sách vẽ cho world/
# ---------------------------------------------------------------------------


class TestBuildRenderItems:
    def test_empty_world_produces_no_items(self) -> None:
        camera = CameraConfig(screen_width=1000, screen_height=800)
        assert build_render_items([], camera) == []

    def test_items_sorted_farthest_first_for_correct_occlusion(self) -> None:
        """world/world_state.py grab-move có thể đổi depth của anchor bất kỳ
        lúc nào - rendering/ phải LUÔN vẽ xa trước gần sau, không được giả
        định thứ tự cố định theo lúc tạo anchor."""
        camera = CameraConfig(screen_width=1000, screen_height=800)
        near = WorldAnchor(position_3d=(0.5, 0.5, 0.1))
        mid = WorldAnchor(position_3d=(0.5, 0.5, 0.5))
        far = WorldAnchor(position_3d=(0.5, 0.5, 0.9))

        # Cố ý truyền vào không theo thứ tự depth để chắc chắn build_render_items
        # tự sắp xếp lại, không dựa vào thứ tự input.
        items = build_render_items([mid, far, near], camera)

        assert [item.anchor_id for item in items] == [far.id, mid.id, near.id]

    def test_anchor_scale_multiplies_perspective_scale(self) -> None:
        """anchor.scale (world/) và scale phối cảnh theo depth (rendering/)
        phải NHÂN với nhau, không phải cái này ghi đè cái kia."""
        camera = CameraConfig(screen_width=1000, screen_height=800)
        small = WorldAnchor(position_3d=(0.5, 0.5, 0.3), scale=1.0)
        big = WorldAnchor(position_3d=(0.5, 0.5, 0.3), scale=2.0)

        items = {item.anchor_id: item for item in build_render_items([small, big], camera)}

        assert items[big.id].radius_px == pytest.approx(items[small.id].radius_px * 2.0)

    def test_missing_metadata_falls_back_to_defaults(self) -> None:
        camera = CameraConfig(screen_width=1000, screen_height=800)
        anchor = WorldAnchor(position_3d=(0.5, 0.5, 0.3))  # không có metadata color/label

        item = build_render_items([anchor], camera)[0]

        assert item.color == DEFAULT_COLOR
        assert item.label == ""

    def test_metadata_color_and_label_are_used(self) -> None:
        camera = CameraConfig(screen_width=1000, screen_height=800)
        anchor = WorldAnchor(
            position_3d=(0.5, 0.5, 0.3),
            metadata={"color": (10, 20, 30), "label": "Docs"},
        )

        item = build_render_items([anchor], camera)[0]

        assert item.color == (10, 20, 30)
        assert item.label == "Docs"

    def test_grab_move_along_depth_axis_updates_rendered_scale(self) -> None:
        """Mô phỏng đúng DoD Giai đoạn E: object bị grab-move ra xa hơn theo
        depth (trục thứ 3, không chỉ ngang/dọc) -> lần render kế tiếp phải
        vẽ nhỏ hơn hẳn lần trước, phản ánh đúng chuyển động world logic đã
        áp dụng (world/world_state.py._apply_grab_move)."""
        camera = CameraConfig(screen_width=1000, screen_height=800)
        anchor = WorldAnchor(position_3d=(0.5, 0.5, 0.1))

        before = build_render_items([anchor], camera)[0]

        moved_further_away = WorldAnchor(
            id=anchor.id, position_3d=(0.5, 0.5, 0.9), metadata=anchor.metadata
        )
        after = build_render_items([moved_further_away], camera)[0]

        assert after.radius_px < before.radius_px
        assert after.alpha < before.alpha


    def test_glow_defaults_to_zero_without_glow_by_id(self) -> None:
        camera = CameraConfig(screen_width=1000, screen_height=800)
        anchor = WorldAnchor(position_3d=(0.5, 0.5, 0.3))

        item = build_render_items([anchor], camera)[0]

        assert item.glow == pytest.approx(0.0)

    def test_glow_by_id_is_applied_to_matching_anchor_only(self) -> None:
        camera = CameraConfig(screen_width=1000, screen_height=800)
        glowing = WorldAnchor(id="glow-me", position_3d=(0.5, 0.5, 0.3))
        plain = WorldAnchor(id="plain", position_3d=(0.2, 0.2, 0.3))

        items = {
            item.anchor_id: item
            for item in build_render_items([glowing, plain], camera, glow_by_id={"glow-me": 0.8})
        }

        assert items["glow-me"].glow == pytest.approx(0.8)
        assert items["plain"].glow == pytest.approx(0.0)

    def test_yaw_defaults_to_zero_and_reflects_anchor_rotation(self) -> None:
        camera = CameraConfig(screen_width=1000, screen_height=800)
        still = WorldAnchor(position_3d=(0.5, 0.5, 0.3))
        spinning = WorldAnchor(position_3d=(0.5, 0.5, 0.3), rotation=(0.0, 1.57, 0.0))

        still_item = build_render_items([still], camera)[0]
        spinning_item = build_render_items([spinning], camera)[0]

        assert still_item.yaw == pytest.approx(0.0)
        assert spinning_item.yaw == pytest.approx(1.57)


# ---------------------------------------------------------------------------
# rendering/overlay_window.py - smoke test tối thiểu cần Qt thật
# ---------------------------------------------------------------------------


class TestOverlayWindowSmoke:
    """Không thể test bằng dữ liệu giả lập thuần túy (paintEvent cần
    QApplication thật) - chỉ verify render() không crash, khác hẳn phần
    logic thuần ở trên đã test đầy đủ bằng WorldAnchor giả lập."""

    def test_render_and_repaint_does_not_crash(self, qt_app) -> None:
        from rendering.overlay_window import OverlayWindow

        camera = CameraConfig(screen_width=400, screen_height=300)
        window = OverlayWindow(camera)

        window.render([])
        window.render(
            [
                WorldAnchor(position_3d=(0.5, 0.5, 0.2), metadata={"label": "A"}),
                WorldAnchor(position_3d=(0.2, 0.8, 0.7)),
            ]
        )
        window.repaint()  # ép paintEvent() chạy ngay, không chờ event loop

        window.close()

    def test_render_with_touch_states_and_pulse_does_not_crash(self, qt_app) -> None:
        """Giai đoạn F: touch_states (glow) và trigger_pulse() phải chạy
        được qua nhiều lần paintEvent() liên tiếp (animator + dt thật) mà
        không crash - đây là phần Qt thật không test bằng dữ liệu giả lập
        thuần túy được (khác build_render_items() đã test riêng ở trên)."""
        from rendering.overlay_window import OverlayWindow

        camera = CameraConfig(screen_width=400, screen_height=300)
        window = OverlayWindow(camera)
        anchor = WorldAnchor(id="a1", position_3d=(0.5, 0.5, 0.2), metadata={"label": "A"})

        window.render([anchor], {"a1": TouchState.TOUCHED})
        window.repaint()

        window.trigger_pulse("a1")
        window.render([anchor], {"a1": TouchState.GRABBED})
        window.repaint()

        # Anchor biến mất khỏi world (bị remove_anchor) - animator phải dọn
        # state nội bộ mà không crash ở frame kế tiếp.
        window.render([], {})
        window.repaint()

        window.close()