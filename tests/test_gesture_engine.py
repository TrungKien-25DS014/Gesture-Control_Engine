"""
tests/test_gesture_engine.py

Layer 2 - Gesture Engine: test theo đúng nguyên tắc dự án - test độc lập
bằng landmark giả lập trước, không cần camera thật. Bao gồm case giả lập
depth thay đổi (tay tiến/lùi) như kế hoạch Giai đoạn C yêu cầu.

Gồm 2 phần:
  1. Automated tests (pytest, chạy bằng `pytest tests/test_gesture_engine.py`)
     - landmark/depth giả lập hoàn toàn, chạy được trong CI.
  2. Live demo script (`python tests/test_gesture_engine.py --live`) - mở
     webcam thật, ghép HandTracker (Layer 1) + GestureEngine (Layer 2),
     log gesture đã phân loại real-time. Đây LÀ deliverable của Giai đoạn C
     (DoD: "Demo Layer 1+2 (chỉ log)"), không phải phần phụ.
"""

from __future__ import annotations

import math
import sys
import time

import pytest

from core.gesture_event import GestureEvent, GestureType, HandLabel
from gesture_engine import GestureEngine
from gesture_engine.depth_gestures import PinchDetector, PushPullDetector, RotationDetector
from gesture_engine.state_machine import HandPose, HandPoseStateMachine, openness_score

# Landmark index theo chuẩn MediaPipe Hands (giống các file khác trong dự án).
WRIST = 0
MIDDLE_FINGER_MCP = 9
THUMB_TIP = 4
INDEX_TIP = 8

_FINGER_MCP_IDX = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
_FINGER_TIP_IDX = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
_FINGER_Y_OFFSET = {"index": 0.0, "middle": 0.0, "ring": 0.02, "pinky": 0.05}


def _hand_landmarks(
    wrist: tuple = (0.5, 0.9),
    middle_mcp: tuple = (0.5, 0.6),
    open_fingers: bool = True,
    thumb_tip: tuple | None = None,
) -> tuple:
    """
    21 landmark giả lập theo thứ tự chuẩn MediaPipe Hands.

    open_fingers=True  -> 4 ngón (trừ cái) duỗi thẳng ra xa cổ tay (xòe).
    open_fingers=False -> đầu ngón co sát gốc ngón (nắm đấm).
    thumb_tip mặc định đặt xa ngón trỏ (không pinch) trừ khi truyền vào cụ
    thể một vị trí gần ngón trỏ để giả lập pinch.
    """
    points = [(0.5, 0.5, 0.0)] * 21
    points[WRIST] = (*wrist, 0.0)

    for name, mcp_idx in _FINGER_MCP_IDX.items():
        mcp_pos = (middle_mcp[0], middle_mcp[1] + _FINGER_Y_OFFSET[name])
        points[mcp_idx] = (*mcp_pos, 0.0)
        tip_idx = _FINGER_TIP_IDX[name]
        if open_fingers:
            # Đầu ngón xa cổ tay hơn nhiều so với MCP -> duỗi thẳng.
            points[tip_idx] = (mcp_pos[0], mcp_pos[1] - 0.25, 0.0)
        else:
            # Đầu ngón gần MCP (thậm chí ngang MCP) -> cong lại.
            points[tip_idx] = (mcp_pos[0], mcp_pos[1] + 0.01, 0.0)

    default_thumb = (wrist[0] - 0.15, middle_mcp[1] + 0.05)
    points[THUMB_TIP] = (*(thumb_tip or default_thumb), 0.0)
    return tuple(points)


def _partial_open_landmarks(num_extended: int) -> tuple:
    """Landmark với đúng `num_extended` trong 4 ngón (trừ cái) duỗi thẳng -
    dùng để test vùng OPENING giữa 2 ngưỡng."""
    points = list(_hand_landmarks(open_fingers=False))
    names = list(_FINGER_MCP_IDX.keys())
    for name in names[:num_extended]:
        mcp_idx = _FINGER_MCP_IDX[name]
        tip_idx = _FINGER_TIP_IDX[name]
        mcp_pos = points[mcp_idx]
        points[tip_idx] = (mcp_pos[0], mcp_pos[1] - 0.25, 0.0)
    return tuple(points)


def _raw_event(
    hand: HandLabel = HandLabel.RIGHT,
    landmarks: tuple | None = None,
    depth: float = 0.5,
    timestamp: float | None = None,
) -> GestureEvent:
    """GestureEvent "thô" giống dạng vision/hand_tracker.py trả về - dùng để
    đưa vào GestureEngine.process()."""
    return GestureEvent(
        hand=hand,
        x=0.5,
        y=0.5,
        depth=depth,
        timestamp=timestamp if timestamp is not None else time.monotonic(),
        raw_landmarks=landmarks if landmarks is not None else _hand_landmarks(),
    )


# ---------------------------------------------------------------------------
# openness_score / HandPoseStateMachine
# ---------------------------------------------------------------------------

class TestOpennessScore:
    def test_fully_open_hand_scores_one(self):
        assert openness_score(_hand_landmarks(open_fingers=True)) == 1.0

    def test_fully_closed_hand_scores_zero(self):
        assert openness_score(_hand_landmarks(open_fingers=False)) == 0.0

    def test_partial_open_scores_proportionally(self):
        assert openness_score(_partial_open_landmarks(2)) == 0.5


class TestHandPoseStateMachine:
    def test_starts_closed(self):
        assert HandPoseStateMachine().state is HandPose.CLOSED

    def test_transitions_through_opening_to_open(self):
        machine = HandPoseStateMachine()
        assert machine.update(_hand_landmarks(open_fingers=False)) is HandPose.CLOSED
        assert machine.update(_partial_open_landmarks(2)) is HandPose.OPENING
        assert machine.update(_hand_landmarks(open_fingers=True)) is HandPose.OPEN

    def test_closed_can_jump_directly_to_open(self):
        """Chuyển động nhanh giữa 2 frame liên tiếp -> cho phép bỏ qua OPENING,
        phản ánh đúng thực tế (không có frame nào ở giữa để quan sát)."""
        machine = HandPoseStateMachine()
        assert machine.update(_hand_landmarks(open_fingers=True)) is HandPose.OPEN

    def test_hysteresis_prevents_flicker_at_boundary(self):
        """Score dao động nhẹ quanh ngưỡng OPEN (0.75) khi đã ở OPEN -
        không được rớt thẳng về CLOSED, chỉ chuyển OPENING."""
        machine = HandPoseStateMachine()
        machine.update(_hand_landmarks(open_fingers=True))
        assert machine.state is HandPose.OPEN

        result = machine.update(_partial_open_landmarks(3))  # score = 0.75, đúng biên OPEN_THRESHOLD
        assert result is HandPose.OPEN

        result = machine.update(_partial_open_landmarks(2))  # score = 0.5, dưới OPEN nhưng trên CLOSED
        assert result is HandPose.OPENING

    def test_stable_over_twenty_repeated_cycles(self):
        """DoD Giai đoạn C: ổn định qua ít nhất 20 lần lặp nắm->xòe."""
        machine = HandPoseStateMachine()
        closed = _hand_landmarks(open_fingers=False)
        open_ = _hand_landmarks(open_fingers=True)

        for _ in range(20):
            assert machine.update(closed) is HandPose.CLOSED
            assert machine.update(open_) is HandPose.OPEN

    def test_reset_returns_to_closed(self):
        machine = HandPoseStateMachine()
        machine.update(_hand_landmarks(open_fingers=True))
        assert machine.state is HandPose.OPEN

        machine.reset()
        assert machine.state is HandPose.CLOSED


# ---------------------------------------------------------------------------
# PinchDetector
# ---------------------------------------------------------------------------

class TestPinchDetector:
    _NOT_PINCHING = _hand_landmarks()
    _PINCHING = _hand_landmarks(thumb_tip=(0.49, 0.36))  # sát ngón trỏ (index tip mặc định ~ (0.5, 0.35))

    def test_no_event_while_ratio_stays_above_threshold(self):
        detector = PinchDetector()
        for _ in range(5):
            assert detector.update(self._NOT_PINCHING) is None

    def test_start_fires_after_debounce_frames(self):
        detector = PinchDetector()
        for _ in range(PinchDetector.DEBOUNCE_FRAMES - 1):
            assert detector.update(self._PINCHING) is None
        assert detector.update(self._PINCHING) == "start"

    def test_end_fires_after_release_debounced(self):
        detector = PinchDetector()
        for _ in range(PinchDetector.DEBOUNCE_FRAMES):
            detector.update(self._PINCHING)

        for _ in range(PinchDetector.DEBOUNCE_FRAMES - 1):
            assert detector.update(self._NOT_PINCHING) is None
        assert detector.update(self._NOT_PINCHING) == "end"

    def test_single_frame_noise_does_not_trigger(self):
        """1 frame nhiễu thoáng qua ngưỡng rồi quay lại ngay -> không trigger."""
        detector = PinchDetector()
        detector.update(self._PINCHING)  # 1 frame ứng viên "start", chưa đủ debounce
        result = detector.update(self._NOT_PINCHING)  # quay lại ngay -> hủy ứng viên
        assert result is None
        assert detector._is_pinching is False

    def test_stable_over_twenty_repeated_cycles(self):
        detector = PinchDetector()
        for _ in range(20):
            for _ in range(PinchDetector.DEBOUNCE_FRAMES - 1):
                detector.update(self._PINCHING)
            assert detector.update(self._PINCHING) == "start"

            for _ in range(PinchDetector.DEBOUNCE_FRAMES - 1):
                detector.update(self._NOT_PINCHING)
            assert detector.update(self._NOT_PINCHING) == "end"


# ---------------------------------------------------------------------------
# RotationDetector
# ---------------------------------------------------------------------------

def _landmarks_at_angle(angle_rad: float, radius: float = 0.3, wrist: tuple = (0.5, 0.5)) -> tuple:
    middle_mcp = (
        wrist[0] + radius * math.cos(angle_rad),
        wrist[1] + radius * math.sin(angle_rad),
    )
    return _hand_landmarks(wrist=wrist, middle_mcp=middle_mcp)


class TestRotationDetector:
    def test_first_frame_returns_none(self):
        detector = RotationDetector()
        assert detector.update(_landmarks_at_angle(0.0)) is None

    def test_reports_signed_delta_between_frames(self):
        detector = RotationDetector()
        detector.update(_landmarks_at_angle(0.0))
        delta = detector.update(_landmarks_at_angle(0.5))
        assert delta == pytest.approx(0.5, abs=1e-6)

    def test_reports_negative_delta_for_opposite_direction(self):
        detector = RotationDetector()
        detector.update(_landmarks_at_angle(0.5))
        delta = detector.update(_landmarks_at_angle(0.0))
        assert delta == pytest.approx(-0.5, abs=1e-6)

    def test_handles_wraparound_at_pi_boundary(self):
        detector = RotationDetector()
        detector.update(_landmarks_at_angle(math.pi - 0.1))
        delta = detector.update(_landmarks_at_angle(-math.pi + 0.1))
        # Đi từ (pi - 0.1) sang (-pi + 0.1) theo chiều ngắn nhất = +0.2, KHÔNG
        # phải gần -2pi+0.2 (nếu không chuẩn hóa wraparound sẽ tính sai giá trị này).
        assert delta == pytest.approx(0.2, abs=1e-6)

    def test_below_noise_threshold_returns_none(self):
        detector = RotationDetector()
        detector.update(_landmarks_at_angle(0.0))
        assert detector.update(_landmarks_at_angle(0.01)) is None

    def test_reset_forgets_previous_angle(self):
        detector = RotationDetector()
        detector.update(_landmarks_at_angle(0.0))
        detector.reset()
        assert detector.update(_landmarks_at_angle(0.5)) is None


# ---------------------------------------------------------------------------
# PushPullDetector
# ---------------------------------------------------------------------------

class TestPushPullDetector:
    def test_no_trigger_while_window_filling(self):
        detector = PushPullDetector()
        for _ in range(PushPullDetector.WINDOW_SIZE - 1):
            assert detector.update(0.5) is None

    def test_decreasing_depth_triggers_push(self):
        """Tay tiến về camera -> depth giảm dần -> PUSH."""
        detector = PushPullDetector()
        depths = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        results = [detector.update(d) for d in depths]
        assert results == [None] * 5 + ["push"]

    def test_increasing_depth_triggers_pull(self):
        """Tay lùi ra xa -> depth tăng dần -> PULL."""
        detector = PushPullDetector()
        depths = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        results = [detector.update(d) for d in depths]
        assert results == [None] * 5 + ["pull"]

    def test_small_fluctuation_does_not_trigger(self):
        """Rung tay tự nhiên, biên độ nhỏ hơn ngưỡng -> không trigger gì."""
        detector = PushPullDetector()
        depths = [0.5, 0.52, 0.49, 0.51, 0.50, 0.53]
        results = [detector.update(d) for d in depths]
        assert all(r is None for r in results)

    def test_cooldown_prevents_immediate_retrigger(self):
        detector = PushPullDetector()
        for d in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
            detector.update(d)
        # Trong lúc cooldown, tiếp tục giảm depth mạnh vẫn không được trigger lại ngay.
        for _ in range(PushPullDetector.COOLDOWN_FRAMES):
            assert detector.update(0.1) is None

    def test_stable_over_twenty_repeated_cycles(self):
        detector = PushPullDetector()
        push_seq = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        # Đủ frame trung tính để cooldown hết hẳn và cửa sổ "quên" chuỗi cũ
        # trước khi bắt đầu chu kỳ push tiếp theo.
        neutral_seq = [0.8] * (PushPullDetector.COOLDOWN_FRAMES + PushPullDetector.WINDOW_SIZE)

        triggered = 0
        for _ in range(20):
            for d in push_seq:
                if detector.update(d) == "push":
                    triggered += 1
            for d in neutral_seq:
                detector.update(d)

        assert triggered == 20


# ---------------------------------------------------------------------------
# GestureEngine (facade ghép state machine + depth gestures)
# ---------------------------------------------------------------------------

class TestGestureEngine:
    def test_every_frame_yields_exactly_one_pose_event(self):
        engine = GestureEngine()
        events = engine.process(_raw_event(landmarks=_hand_landmarks(open_fingers=False)))

        pose_events = [e for e in events if e.gesture_type in (GestureType.HAND_CLOSED, GestureType.HAND_OPENING, GestureType.HAND_OPEN)]
        assert len(pose_events) == 1
        assert pose_events[0].gesture_type is GestureType.HAND_CLOSED

    def test_missing_raw_landmarks_returns_empty_list(self):
        engine = GestureEngine()
        raw = GestureEvent(hand=HandLabel.LEFT, x=0.5, y=0.5, depth=0.5, timestamp=time.monotonic())
        assert engine.process(raw) == []

    def test_left_and_right_hand_state_do_not_interfere(self):
        engine = GestureEngine()
        engine.process(_raw_event(hand=HandLabel.LEFT, landmarks=_hand_landmarks(open_fingers=True)))
        events = engine.process(_raw_event(hand=HandLabel.RIGHT, landmarks=_hand_landmarks(open_fingers=False)))

        pose = next(e for e in events if e.hand is HandLabel.RIGHT)
        assert pose.gesture_type is GestureType.HAND_CLOSED

    def test_pinch_start_and_end_surface_through_process(self):
        engine = GestureEngine()
        pinching = _hand_landmarks(thumb_tip=(0.49, 0.36))
        not_pinching = _hand_landmarks()

        last_events = []
        for _ in range(PinchDetector.DEBOUNCE_FRAMES):
            last_events = engine.process(_raw_event(landmarks=pinching))
        assert any(e.gesture_type is GestureType.PINCH_START for e in last_events)

        for _ in range(PinchDetector.DEBOUNCE_FRAMES):
            last_events = engine.process(_raw_event(landmarks=not_pinching))
        assert any(e.gesture_type is GestureType.PINCH_END for e in last_events)

    def test_rotation_event_carries_signed_delta_in_value(self):
        engine = GestureEngine()
        engine.process(_raw_event(landmarks=_landmarks_at_angle(0.0)))
        events = engine.process(_raw_event(landmarks=_landmarks_at_angle(0.5)))

        rotate_events = [e for e in events if e.gesture_type is GestureType.ROTATE]
        assert len(rotate_events) == 1
        assert rotate_events[0].value == pytest.approx(0.5, abs=1e-6)

    def test_push_and_pull_events_surface_with_correct_direction(self):
        engine = GestureEngine()
        landmarks = _hand_landmarks()

        push_events = []
        for depth in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
            push_events = engine.process(_raw_event(landmarks=landmarks, depth=depth))
        assert any(e.gesture_type is GestureType.PUSH for e in push_events)

        engine.reset_hand(HandLabel.RIGHT)

        pull_events = []
        for depth in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            pull_events = engine.process(_raw_event(landmarks=landmarks, depth=depth))
        assert any(e.gesture_type is GestureType.PULL for e in pull_events)

    def test_reset_hand_clears_only_that_hand(self):
        engine = GestureEngine()
        engine.process(_raw_event(hand=HandLabel.LEFT, landmarks=_hand_landmarks(open_fingers=True)))
        engine.process(_raw_event(hand=HandLabel.RIGHT, landmarks=_hand_landmarks(open_fingers=True)))

        engine.reset_hand(HandLabel.LEFT)

        left_state = engine._per_hand[HandLabel.LEFT].pose_state_machine.state
        right_state = engine._per_hand[HandLabel.RIGHT].pose_state_machine.state
        assert left_state is HandPose.CLOSED
        assert right_state is HandPose.OPEN

    def test_stable_full_pipeline_over_twenty_cycles(self):
        """DoD Giai đoạn C: nắm->xòe, pinch start/end, push/pull, rotation -
        tất cả log đúng, ổn định qua ít nhất 20 lần lặp, chạy qua đúng
        GestureEngine.process() (không gọi thẳng detector nội bộ)."""
        engine = GestureEngine()
        closed = _hand_landmarks(open_fingers=False)
        open_ = _hand_landmarks(open_fingers=True)

        for _ in range(20):
            events = engine.process(_raw_event(landmarks=closed))
            assert any(e.gesture_type is GestureType.HAND_CLOSED for e in events)

            events = engine.process(_raw_event(landmarks=open_))
            assert any(e.gesture_type is GestureType.HAND_OPEN for e in events)


# ---------------------------------------------------------------------------
# Live demo script - dùng để tự tay verify DoD Giai đoạn C (cần camera thật)
# ---------------------------------------------------------------------------

def _run_live_demo(camera_index: int = 0) -> None:
    import cv2

    from core.gesture_event import HandLabel as _HandLabel
    from vision.hand_tracker import HandTracker

    # Giống test_vision.py: KHÔNG cv2.flip() frame ở đây, chỉ in log ra
    # console. HandTracker nhận frame gốc và tự bù nhãn tay nội bộ.
    tracker = HandTracker(mirrored_view=True)
    engine = GestureEngine()
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Không mở được camera index={camera_index}")

    seen_hands: set = set()

    print("Thử: nắm -> xòe, pinch (cái+trỏ), xoay cổ tay, đẩy/kéo tay theo chiều sâu.")
    print("Lặp lại nhiều lần (>=20) để tự verify DoD Giai đoạn C. Ctrl+C để dừng.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue

            raw_events = tracker.read_frame(frame)
            current_hands = {e.hand for e in raw_events}

            # Tay vừa biến mất khỏi khung hình -> reset state gesture engine
            # cho tay đó, tránh delta giả khi tay quay lại (xem GestureEngine.reset_hand).
            for hand in seen_hands - current_hands:
                engine.reset_hand(hand)
            seen_hands = current_hands

            for raw_event in raw_events:
                for event in engine.process(raw_event):
                    if event.gesture_type.name in ("HAND_CLOSED", "HAND_OPENING", "HAND_OPEN"):
                        continue  # trạng thái thường trực - bỏ qua khi log để đỡ rối, log riêng dưới đây
                    extra = f" value={event.value:.3f}" if event.value is not None else ""
                    print(f"[{event.hand.name}] {event.gesture_type.name}{extra}")

                pose_gesture = engine._per_hand[raw_event.hand].pose_state_machine.gesture_type()
                print(f"[{raw_event.hand.name}] pose={pose_gesture.name}", end="\r")

            time.sleep(0.03)
    except KeyboardInterrupt:
        print("\nDừng.")
    finally:
        capture.release()
        tracker.close()


if __name__ == "__main__":
    if "--live" in sys.argv:
        _run_live_demo()
    else:
        raise SystemExit(
            "File này chứa pytest tests. Chạy `pytest tests/test_gesture_engine.py` để test tự động,\n"
            "hoặc `python tests/test_gesture_engine.py --live` để demo camera thật (verify DoD Giai đoạn C)."
        )