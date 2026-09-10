"""
tests/test_world_logic.py

Layer 3 - World Logic (Giai đoạn D): test bằng chuỗi GestureEvent giả lập
đúng nguyên tắc dự án - KHÔNG cần camera thật, KHÔNG cần raw_landmarks
(world/ chỉ nhận event đã phân loại từ gesture_engine/, xem docstring
world/world_state.py).

DoD Giai đoạn D cần verify:
  1. Hit-testing 3D: object được chọn đúng theo vị trí 3D thật (không phải
     góc trên vòng tròn 2D).
  2. Grab-and-move: pinch giữ + di chuyển tay -> object di chuyển đúng theo
     cả 3 trục (x, y, depth).
  3. Push trigger đúng action gắn vào anchor đang chạm.
  4. Toàn bộ chạy được chỉ với GestureEvent giả lập, không cần camera thật.

Ngoài ra test luôn world/interactions.py và world/launcher.py (các loại
WorldAction cụ thể) - mock hết side-effect thật (subprocess, webbrowser) để
test không thực sự mở app/browser/link nào.
"""

from __future__ import annotations

from typing import Optional

import pytest

from core.gesture_event import GestureEvent, GestureType, HandLabel
from core.world_anchor import WorldAnchor
from world.interactions import ActionResult, OpenLinkAction, RunScriptAction, ShowInfoAction
from world.launcher import LaunchAppAction
from world.world_state import WorldState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event(
    hand: HandLabel,
    position: tuple[float, float, float],
    gesture_type: GestureType = GestureType.HAND_OPEN,
    timestamp: float = 0.0,
) -> GestureEvent:
    """Dựng 1 GestureEvent đã phân loại trực tiếp (đúng input thật của
    world/ - không có raw_landmarks, xem core/interfaces.py WorldLogic)."""
    x, y, depth = position
    return GestureEvent(hand=hand, x=x, y=y, depth=depth, timestamp=timestamp, gesture_type=gesture_type)


class _FakeAction:
    """Test double cho WorldAction - đếm số lần execute() được gọi, không
    có side effect thật (khác OpenLinkAction/RunScriptAction/LaunchAppAction
    thật sự mở link/app)."""

    def __init__(self, message: str = "fake action") -> None:
        self.call_count = 0
        self._message = message

    def execute(self) -> ActionResult:
        self.call_count += 1
        return ActionResult(success=True, message=self._message)


# ---------------------------------------------------------------------------
# Hit-testing 3D
# ---------------------------------------------------------------------------


class TestHitTesting:
    def test_touches_nearest_anchor_within_radius(self) -> None:
        world = WorldState()
        near = world.add_anchor(WorldAnchor(position_3d=(0.5, 0.5, 0.3)))
        far = world.add_anchor(WorldAnchor(position_3d=(0.9, 0.9, 0.9)))

        world.handle_event(_event(HandLabel.RIGHT, (0.52, 0.51, 0.3)))

        assert world.touched_anchor_id(HandLabel.RIGHT) == near.id
        assert world.touched_anchor_id(HandLabel.RIGHT) != far.id

    def test_no_touch_when_outside_every_radius(self) -> None:
        world = WorldState()
        world.add_anchor(WorldAnchor(position_3d=(0.5, 0.5, 0.5)))

        world.handle_event(_event(HandLabel.RIGHT, (0.1, 0.1, 0.1)))

        assert world.touched_anchor_id(HandLabel.RIGHT) is None

    def test_hit_test_uses_real_3d_distance_not_2d_angle(self) -> None:
        """2 anchor trùng x/y nhưng khác depth - tay ở đúng depth nào thì
        chạm đúng anchor đó. Nếu hit-test chỉ dùng góc 2D (bỏ qua depth)
        thì test này sẽ sai vì 2 anchor "trùng vị trí" trên mặt phẳng x/y."""
        world = WorldState()
        near_hand = world.add_anchor(WorldAnchor(position_3d=(0.5, 0.5, 0.1)))
        far_from_hand = world.add_anchor(WorldAnchor(position_3d=(0.5, 0.5, 0.9)))

        world.handle_event(_event(HandLabel.LEFT, (0.5, 0.5, 0.1)))

        assert world.touched_anchor_id(HandLabel.LEFT) == near_hand.id
        assert world.touched_anchor_id(HandLabel.LEFT) != far_from_hand.id

    def test_larger_scale_gives_larger_hit_radius(self) -> None:
        world = WorldState()
        big = world.add_anchor(WorldAnchor(position_3d=(0.5, 0.5, 0.5), scale=3.0))

        # Đủ xa để trượt bán kính mặc định (scale=1.0) nhưng vẫn trong bán
        # kính đã nhân scale=3.0.
        world.handle_event(_event(HandLabel.RIGHT, (0.5 + 0.2, 0.5, 0.5)))

        assert world.touched_anchor_id(HandLabel.RIGHT) == big.id

    def test_picks_nearest_when_multiple_anchors_overlap_radius(self) -> None:
        world = WorldState()
        closer = world.add_anchor(WorldAnchor(position_3d=(0.5, 0.5, 0.5)))
        farther = world.add_anchor(WorldAnchor(position_3d=(0.55, 0.5, 0.5)))

        world.handle_event(_event(HandLabel.RIGHT, (0.51, 0.5, 0.5)))

        assert world.touched_anchor_id(HandLabel.RIGHT) == closer.id
        assert world.touched_anchor_id(HandLabel.RIGHT) != farther.id


# ---------------------------------------------------------------------------
# Grab-and-move
# ---------------------------------------------------------------------------


class TestGrabAndMove:
    def test_pinch_start_while_touching_begins_grab(self) -> None:
        world = WorldState()
        anchor = world.add_anchor(WorldAnchor(position_3d=(0.5, 0.5, 0.5)))

        world.handle_event(_event(HandLabel.RIGHT, (0.5, 0.5, 0.5), GestureType.PINCH_START))

        assert world.is_grabbing(HandLabel.RIGHT)
        assert world.get_anchor(anchor.id).position_3d == (0.5, 0.5, 0.5)  # chưa di chuyển gì

    def test_pinch_start_without_touching_anything_does_not_grab(self) -> None:
        world = WorldState()
        world.add_anchor(WorldAnchor(position_3d=(0.5, 0.5, 0.5)))

        world.handle_event(_event(HandLabel.RIGHT, (0.0, 0.0, 0.0), GestureType.PINCH_START))

        assert not world.is_grabbing(HandLabel.RIGHT)

    def test_grab_moves_anchor_on_all_three_axes(self) -> None:
        world = WorldState()
        anchor = world.add_anchor(WorldAnchor(position_3d=(0.5, 0.5, 0.5)))

        world.handle_event(_event(HandLabel.RIGHT, (0.5, 0.5, 0.5), GestureType.PINCH_START))
        # Tay di chuyển dần trên cả 3 trục trong lúc vẫn đang giữ pinch.
        world.handle_event(_event(HandLabel.RIGHT, (0.55, 0.5, 0.5), GestureType.HAND_CLOSED))
        world.handle_event(_event(HandLabel.RIGHT, (0.55, 0.6, 0.5), GestureType.HAND_CLOSED))
        world.handle_event(_event(HandLabel.RIGHT, (0.55, 0.6, 0.7), GestureType.HAND_CLOSED))

        moved = world.get_anchor(anchor.id)
        assert moved.position_3d == pytest.approx((0.55, 0.6, 0.7))

    def test_pinch_end_stops_following_hand(self) -> None:
        world = WorldState()
        anchor = world.add_anchor(WorldAnchor(position_3d=(0.5, 0.5, 0.5)))

        world.handle_event(_event(HandLabel.RIGHT, (0.5, 0.5, 0.5), GestureType.PINCH_START))
        world.handle_event(_event(HandLabel.RIGHT, (0.6, 0.5, 0.5), GestureType.PINCH_END))
        position_after_release = world.get_anchor(anchor.id).position_3d

        # Tay tiếp tục di chuyển sau khi đã xòe ra (PINCH_END) - anchor
        # không được bám theo nữa.
        world.handle_event(_event(HandLabel.RIGHT, (0.9, 0.9, 0.9), GestureType.HAND_OPEN))

        assert not world.is_grabbing(HandLabel.RIGHT)
        assert world.get_anchor(anchor.id).position_3d == position_after_release

    def test_two_hands_grab_independently(self) -> None:
        world = WorldState()
        anchor_a = world.add_anchor(WorldAnchor(position_3d=(0.2, 0.2, 0.2)))
        anchor_b = world.add_anchor(WorldAnchor(position_3d=(0.8, 0.8, 0.8)))

        world.handle_event(_event(HandLabel.LEFT, (0.2, 0.2, 0.2), GestureType.PINCH_START))
        world.handle_event(_event(HandLabel.RIGHT, (0.8, 0.8, 0.8), GestureType.PINCH_START))

        world.handle_event(_event(HandLabel.LEFT, (0.25, 0.2, 0.2), GestureType.HAND_CLOSED))

        # Tay trái di chuyển không được ảnh hưởng tới anchor của tay phải.
        assert world.get_anchor(anchor_a.id).position_3d == pytest.approx((0.25, 0.2, 0.2))
        assert world.get_anchor(anchor_b.id).position_3d == pytest.approx((0.8, 0.8, 0.8))

    def test_grabbing_one_anchor_does_not_move_untouched_anchor(self) -> None:
        world = WorldState()
        grabbed = world.add_anchor(WorldAnchor(position_3d=(0.5, 0.5, 0.5)))
        bystander = world.add_anchor(WorldAnchor(position_3d=(0.9, 0.9, 0.9)))

        world.handle_event(_event(HandLabel.RIGHT, (0.5, 0.5, 0.5), GestureType.PINCH_START))
        world.handle_event(_event(HandLabel.RIGHT, (0.6, 0.6, 0.6), GestureType.HAND_CLOSED))

        assert world.get_anchor(bystander.id).position_3d == (0.9, 0.9, 0.9)
        assert world.get_anchor(grabbed.id).position_3d != (0.5, 0.5, 0.5)

    def test_pull_while_grabbing_cancels_and_reverts_position(self) -> None:
        world = WorldState()
        anchor = world.add_anchor(WorldAnchor(position_3d=(0.5, 0.5, 0.5)))

        world.handle_event(_event(HandLabel.RIGHT, (0.5, 0.5, 0.5), GestureType.PINCH_START))
        world.handle_event(_event(HandLabel.RIGHT, (0.7, 0.7, 0.7), GestureType.HAND_CLOSED))
        assert world.get_anchor(anchor.id).position_3d != (0.5, 0.5, 0.5)

        world.handle_event(_event(HandLabel.RIGHT, (0.7, 0.7, 0.7), GestureType.PULL))

        assert world.get_anchor(anchor.id).position_3d == (0.5, 0.5, 0.5)
        assert not world.is_grabbing(HandLabel.RIGHT)

    def test_pull_without_grabbing_is_a_no_op(self) -> None:
        world = WorldState()
        anchor = world.add_anchor(WorldAnchor(position_3d=(0.5, 0.5, 0.5)))

        world.handle_event(_event(HandLabel.RIGHT, (0.5, 0.5, 0.5), GestureType.PULL))

        assert world.get_anchor(anchor.id).position_3d == (0.5, 0.5, 0.5)
        assert not world.is_grabbing(HandLabel.RIGHT)


# ---------------------------------------------------------------------------
# Push để kích hoạt
# ---------------------------------------------------------------------------


class TestPushActivation:
    def test_push_while_touching_executes_anchor_action(self) -> None:
        world = WorldState()
        action = _FakeAction("hello")
        world.add_anchor(WorldAnchor(position_3d=(0.5, 0.5, 0.5), metadata={"action": action}))

        world.handle_event(_event(HandLabel.RIGHT, (0.5, 0.5, 0.5), GestureType.PUSH))

        assert action.call_count == 1
        assert world.last_action_result == ActionResult(success=True, message="hello")

    def test_push_without_touching_anything_does_nothing(self) -> None:
        world = WorldState()
        action = _FakeAction()
        world.add_anchor(WorldAnchor(position_3d=(0.5, 0.5, 0.5), metadata={"action": action}))

        world.handle_event(_event(HandLabel.RIGHT, (0.0, 0.0, 0.0), GestureType.PUSH))

        assert action.call_count == 0
        assert world.last_action_result is None

    def test_push_on_anchor_without_action_does_not_crash(self) -> None:
        world = WorldState()
        world.add_anchor(WorldAnchor(position_3d=(0.5, 0.5, 0.5)))  # không gắn action

        world.handle_event(_event(HandLabel.RIGHT, (0.5, 0.5, 0.5), GestureType.PUSH))  # không raise

        assert world.last_action_result is None

    def test_push_triggers_only_the_touched_anchor(self) -> None:
        world = WorldState()
        touched_action = _FakeAction()
        other_action = _FakeAction()
        world.add_anchor(WorldAnchor(position_3d=(0.5, 0.5, 0.5), metadata={"action": touched_action}))
        world.add_anchor(WorldAnchor(position_3d=(0.9, 0.9, 0.9), metadata={"action": other_action}))

        world.handle_event(_event(HandLabel.RIGHT, (0.5, 0.5, 0.5), GestureType.PUSH))

        assert touched_action.call_count == 1
        assert other_action.call_count == 0


# ---------------------------------------------------------------------------
# Quản lý anchor
# ---------------------------------------------------------------------------


class TestAnchorManagement:
    def test_add_and_get_anchors(self) -> None:
        world = WorldState()
        a = world.add_anchor(WorldAnchor(position_3d=(0.1, 0.1, 0.1)))
        b = world.add_anchor(WorldAnchor(position_3d=(0.2, 0.2, 0.2)))

        assert {anchor.id for anchor in world.get_anchors()} == {a.id, b.id}

    def test_remove_anchor_clears_dangling_grab(self) -> None:
        world = WorldState()
        anchor = world.add_anchor(WorldAnchor(position_3d=(0.5, 0.5, 0.5)))
        world.handle_event(_event(HandLabel.RIGHT, (0.5, 0.5, 0.5), GestureType.PINCH_START))
        assert world.is_grabbing(HandLabel.RIGHT)

        world.remove_anchor(anchor.id)

        assert not world.is_grabbing(HandLabel.RIGHT)
        assert world.get_anchor(anchor.id) is None


# ---------------------------------------------------------------------------
# world/interactions.py - các loại WorldAction cụ thể (mock side effect thật)
# ---------------------------------------------------------------------------


class TestInteractionActions:
    def test_open_link_action_calls_webbrowser(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr("webbrowser.open", lambda url: calls.append(url) or True)

        result = OpenLinkAction(url="https://example.com").execute()

        assert calls == ["https://example.com"]
        assert result.success is True

    def test_open_link_action_failure_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(url: str) -> bool:
            raise RuntimeError("không có browser")

        monkeypatch.setattr("webbrowser.open", _boom)

        result = OpenLinkAction(url="https://example.com").execute()  # không raise

        assert result.success is False

    def test_run_script_action_calls_subprocess_popen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, list[str]] = {}

        class _FakePopen:
            def __init__(self, command: list[str]) -> None:
                captured["command"] = command

        monkeypatch.setattr("subprocess.Popen", _FakePopen)

        result = RunScriptAction(command=("echo", "hi")).execute()

        assert captured["command"] == ["echo", "hi"]
        assert result.success is True

    def test_show_info_action_calls_callback_with_text(self) -> None:
        received: list[str] = []

        result = ShowInfoAction(text="xin chào", on_show=received.append).execute()

        assert received == ["xin chào"]
        assert result.success is True


class TestLaunchAppAction:
    def test_launch_app_builds_macos_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        captured: dict[str, list[str]] = {}
        monkeypatch.setattr("subprocess.Popen", lambda cmd: captured.setdefault("command", cmd))

        LaunchAppAction(app_name="Calculator").execute()

        assert captured["command"] == ["open", "-a", "Calculator"]

    def test_launch_app_builds_windows_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("platform.system", lambda: "Windows")
        captured: dict[str, list[str]] = {}
        monkeypatch.setattr("subprocess.Popen", lambda cmd: captured.setdefault("command", cmd))

        LaunchAppAction(app_name="notepad.exe").execute()

        assert captured["command"] == ["cmd", "/c", "start", "", "notepad.exe"]

    def test_launch_app_builds_linux_command_with_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("platform.system", lambda: "Linux")
        captured: dict[str, list[str]] = {}
        monkeypatch.setattr("subprocess.Popen", lambda cmd: captured.setdefault("command", cmd))

        LaunchAppAction(app_name="gedit", args=("file.txt",)).execute()

        assert captured["command"] == ["gedit", "file.txt"]

    def test_launch_app_failure_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("platform.system", lambda: "Linux")

        def _boom(cmd: list[str]) -> None:
            raise FileNotFoundError("app không tồn tại")

        monkeypatch.setattr("subprocess.Popen", _boom)

        result = LaunchAppAction(app_name="app-khong-ton-tai").execute()  # không raise

        assert result.success is False


# ---------------------------------------------------------------------------
# End-to-end nhỏ: chuỗi event giống demo thật (grab rồi push object khác)
# ---------------------------------------------------------------------------


def test_end_to_end_grab_then_push_different_anchor() -> None:
    """Mô phỏng đúng kịch bản DoD: chuỗi GestureEvent giả lập -> object được
    chọn đúng theo vị trí 3D, di chuyển đúng theo grab, push trigger đúng
    action - toàn bộ không cần camera thật."""
    world = WorldState()
    movable = world.add_anchor(WorldAnchor(position_3d=(0.3, 0.3, 0.3)))
    button_action = _FakeAction("action đã chạy")
    button = world.add_anchor(WorldAnchor(position_3d=(0.8, 0.2, 0.4), metadata={"action": button_action}))

    # Tay phải chạm và grab object di chuyển được, kéo nó ra xa.
    world.handle_event(_event(HandLabel.RIGHT, (0.3, 0.3, 0.3), GestureType.PINCH_START))
    world.handle_event(_event(HandLabel.RIGHT, (0.5, 0.4, 0.3), GestureType.HAND_CLOSED))
    world.handle_event(_event(HandLabel.RIGHT, (0.5, 0.4, 0.3), GestureType.PINCH_END))

    assert world.get_anchor(movable.id).position_3d == pytest.approx((0.5, 0.4, 0.3))

    # Tay trái push đúng nút - không đụng tới object vừa di chuyển.
    world.handle_event(_event(HandLabel.LEFT, (0.8, 0.2, 0.4), GestureType.PUSH))

    assert button_action.call_count == 1
    assert world.get_anchor(button.id).position_3d == (0.8, 0.2, 0.4)  # nút không di chuyển


if __name__ == "__main__":
    raise SystemExit(
        "File này chứa pytest tests, không có demo camera thật (Giai đoạn D không cần "
        "camera - xem DoD). Chạy `pytest tests/test_world_logic.py` để test tự động."
    )