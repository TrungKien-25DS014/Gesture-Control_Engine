"""
tests/test_animation.py

Giai đoạn F - Polish tương tác: test rendering/animation.py bằng cách tự
truyền dt qua nhiều "frame" giả lập - đúng nguyên tắc dự án, không cần
Qt/camera thật (SceneAnimator thuần Python).

DoD Giai đoạn F cần verify được bằng test tự động (phần "cảm giác tự
nhiên" khi cầm tay thật chỉ verify được thủ công, đúng bản chất demo):
  1. Animation mượt: vị trí hiển thị đuổi dần theo target, không nhảy tức
     thời sang giá trị mới ngay frame đầu tiên sau khi target đổi.
  2. Anchor mới xuất hiện lần đầu -> hiển thị đúng ngay vị trí thật, không
     "bay" từ đâu đó vào.
  3. Glow đuổi theo đúng target theo TouchState (NONE/TOUCHED/GRABBED).
  4. Pulse: trigger_pulse() làm giá trị nhảy lên ngay, rồi tự tắt dần về 0
     qua các frame sau, không đứng yên ở giá trị pulse mãi.
  5. Anchor bị xóa khỏi world (không còn trong list truyền vào update())
     không làm animator phình to mãi (dọn state cũ).
"""

from __future__ import annotations

import pytest

from core.touch_state import TouchState
from core.world_anchor import WorldAnchor
from rendering.animation import SceneAnimator


class TestPositionSmoothing:
    def test_new_anchor_starts_exactly_at_its_target(self) -> None:
        animator = SceneAnimator()
        anchor = WorldAnchor(position_3d=(0.3, 0.4, 0.5))

        animator.update([anchor], touch_states={}, dt=0.033)

        smoothed = animator.smoothed_anchor(anchor)
        assert smoothed.position_3d == pytest.approx(anchor.position_3d)

    def test_moving_target_is_approached_gradually_not_instantly(self) -> None:
        animator = SceneAnimator()
        anchor = WorldAnchor(id="a1", position_3d=(0.0, 0.0, 0.0))
        animator.update([anchor], touch_states={}, dt=0.033)  # khởi tạo tại vị trí gốc

        moved = WorldAnchor(id="a1", position_3d=(1.0, 0.0, 0.0))
        animator.update([moved], touch_states={}, dt=0.033)

        smoothed_x = animator.smoothed_anchor(moved).position_3d[0]
        # Chưa tới target ngay lập tức (không nhảy cứng)...
        assert 0.0 < smoothed_x < 1.0

    def test_position_converges_to_target_after_many_frames(self) -> None:
        animator = SceneAnimator()
        anchor = WorldAnchor(id="a1", position_3d=(0.0, 0.0, 0.0))
        animator.update([anchor], touch_states={}, dt=0.033)

        moved = WorldAnchor(id="a1", position_3d=(1.0, 0.5, 0.2))
        for _ in range(200):  # rất nhiều frame -> đủ thời gian đuổi kịp target
            animator.update([moved], touch_states={}, dt=0.033)

        smoothed = animator.smoothed_anchor(moved)
        assert smoothed.position_3d == pytest.approx(moved.position_3d, abs=1e-3)

    def test_zero_dt_does_not_change_current_value(self) -> None:
        """Frame lỗi/trùng timestamp (dt<=0) không được làm giá trị nhảy
        hay raise lỗi chia cho 0."""
        animator = SceneAnimator()
        anchor = WorldAnchor(id="a1", position_3d=(0.0, 0.0, 0.0))
        animator.update([anchor], touch_states={}, dt=0.033)

        moved = WorldAnchor(id="a1", position_3d=(1.0, 0.0, 0.0))
        animator.update([moved], touch_states={}, dt=0.0)

        assert animator.smoothed_anchor(moved).position_3d[0] == pytest.approx(0.0)


class TestGlowSmoothing:
    def test_glow_rises_toward_touched_target(self) -> None:
        animator = SceneAnimator()
        anchor = WorldAnchor(id="a1", position_3d=(0.5, 0.5, 0.5))
        animator.update([anchor], touch_states={}, dt=0.033)
        assert animator.glow_of("a1") == pytest.approx(0.0)

        for _ in range(100):
            animator.update([anchor], touch_states={"a1": TouchState.TOUCHED}, dt=0.033)

        assert animator.glow_of("a1") > 0.3  # đã đuổi lên đáng kể so với 0

    def test_grabbed_glow_is_stronger_than_touched(self) -> None:
        animator = SceneAnimator()
        anchor = WorldAnchor(id="a1", position_3d=(0.5, 0.5, 0.5))

        for _ in range(200):
            animator.update([anchor], touch_states={"a1": TouchState.TOUCHED}, dt=0.033)
        touched_glow = animator.glow_of("a1")

        for _ in range(200):
            animator.update([anchor], touch_states={"a1": TouchState.GRABBED}, dt=0.033)
        grabbed_glow = animator.glow_of("a1")

        assert grabbed_glow > touched_glow

    def test_untouched_anchor_has_no_glow(self) -> None:
        animator = SceneAnimator()
        anchor = WorldAnchor(id="a1", position_3d=(0.5, 0.5, 0.5))
        animator.update([anchor], touch_states={}, dt=0.033)

        assert animator.glow_of("a1") == pytest.approx(0.0)
        assert animator.glow_of("does-not-exist") == pytest.approx(0.0)


class TestPushPulse:
    def test_trigger_pulse_bumps_scale_then_decays_back_down(self) -> None:
        animator = SceneAnimator()
        anchor = WorldAnchor(id="a1", position_3d=(0.5, 0.5, 0.5), scale=1.0)
        animator.update([anchor], touch_states={}, dt=0.033)

        animator.trigger_pulse("a1", amount=0.2)
        just_after = animator.smoothed_anchor(anchor).scale
        assert just_after > 1.0  # nảy to lên ngay

        for _ in range(60):
            animator.update([anchor], touch_states={}, dt=0.033)
        long_after = animator.smoothed_anchor(anchor).scale

        assert long_after < just_after
        assert long_after == pytest.approx(1.0, abs=1e-2)  # tắt hẳn về scale gốc

    def test_pulse_on_unknown_anchor_does_not_raise(self) -> None:
        animator = SceneAnimator()
        animator.trigger_pulse("never-seen-before")  # không crash


class TestCleanup:
    def test_removed_anchor_is_forgotten(self) -> None:
        animator = SceneAnimator()
        anchor = WorldAnchor(id="a1", position_3d=(0.5, 0.5, 0.5))
        animator.update([anchor], touch_states={}, dt=0.033)
        assert "a1" in animator._anims  # noqa: SLF001 - kiểm tra dọn state nội bộ

        animator.update([], touch_states={}, dt=0.033)  # anchor "a1" biến mất khỏi world

        assert "a1" not in animator._anims  # noqa: SLF001

    def test_smoothed_anchor_for_unknown_anchor_returns_original(self) -> None:
        animator = SceneAnimator()
        anchor = WorldAnchor(position_3d=(0.1, 0.2, 0.3))

        # Chưa từng update() -> smoothed_anchor phải trả về nguyên bản, không raise.
        assert animator.smoothed_anchor(anchor) == anchor