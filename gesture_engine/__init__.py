"""
gesture_engine/__init__.py

Layer 2 - Gesture Engine (Giai đoạn C): facade công khai ghép state machine
nắm/xòe (state_machine.py) + các gesture rời rạc pinch/rotation/push/pull
(depth_gestures.py) thành 1 GestureEngine implement đúng Protocol khai báo
ở core/interfaces.py (đã cập nhật process() -> list[GestureEvent], xem
comment trong file đó).

Đây là điểm DUY NHẤT trong package này đọc raw_landmarks - world/ ở Giai
đoạn D chỉ nhận GestureEvent đã phân loại, không bao giờ thấy landmark gốc.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from core.gesture_event import GestureEvent, GestureType, HandLabel
from gesture_engine.depth_gestures import PinchDetector, PushPullDetector, RotationDetector
from gesture_engine.state_machine import HandPose, HandPoseStateMachine

__all__ = [
    "GestureEngine",
    "HandPose",
    "HandPoseStateMachine",
    "PinchDetector",
    "RotationDetector",
    "PushPullDetector",
]


class _PerHandState:
    """Gộp toàn bộ state riêng cho 1 tay - không chia sẻ giữa 2 tay, giống
    quy ước HandTracker ở Layer 1 (mỗi tay có state làm mượt riêng)."""

    def __init__(self) -> None:
        self.pose_state_machine = HandPoseStateMachine()
        self.pinch_detector = PinchDetector()
        self.rotation_detector = RotationDetector()
        self.push_pull_detector = PushPullDetector()

    def reset(self) -> None:
        self.pose_state_machine.reset()
        self.pinch_detector.reset()
        self.rotation_detector.reset()
        self.push_pull_detector.reset()


class GestureEngine:
    """Implement Protocol GestureEngine ở core/interfaces.py.

    1 instance = 1 phiên xử lý, giữ state riêng biệt theo từng HandLabel -
    tay trái/phải không ảnh hưởng lẫn nhau (đúng ghi chú kế hoạch "gắn nhãn
    tay rõ ràng: tay nào trigger menu, tay nào pinch/push")."""

    def __init__(self) -> None:
        self._per_hand: dict[HandLabel, _PerHandState] = {
            HandLabel.LEFT: _PerHandState(),
            HandLabel.RIGHT: _PerHandState(),
        }

    def process(self, raw_event: GestureEvent) -> list[GestureEvent]:
        """
        raw_event: GestureEvent "thô" từ vision/ (gesture_type=UNKNOWN),
        BẮT BUỘC phải mang raw_landmarks - đây là input duy nhất cần
        landmark gốc trong toàn bộ pipeline.

        Trả về list GestureEvent đã phân loại phát sinh từ đúng frame này:
        luôn có đúng 1 event trạng thái nắm/xòe (HAND_CLOSED/OPENING/OPEN)
        + 0..n event rời rạc (PINCH_START/END, ROTATE, PUSH, PULL) tùy frame
        đó có xảy ra gì hay không.
        """
        if raw_event.raw_landmarks is None:
            # An toàn: raw_event thiếu landmark (vd: bị dựng thủ công sai) ->
            # không thể phân loại gì, trả list rỗng thay vì raise, nhất quán
            # với nguyên tắc "không raise vì lý do dữ liệu tay thiếu" ở
            # vision/hand_tracker.py.
            return []

        state = self._per_hand[raw_event.hand]
        landmarks = raw_event.raw_landmarks
        events: list[GestureEvent] = []

        state.pose_state_machine.update(landmarks)
        events.append(self._classified(raw_event, state.pose_state_machine.gesture_type()))

        pinch_result = state.pinch_detector.update(landmarks)
        if pinch_result == "start":
            events.append(self._classified(raw_event, GestureType.PINCH_START))
        elif pinch_result == "end":
            events.append(self._classified(raw_event, GestureType.PINCH_END))

        rotation_delta = state.rotation_detector.update(landmarks)
        if rotation_delta is not None:
            events.append(self._classified(raw_event, GestureType.ROTATE, value=rotation_delta))

        push_pull_result = state.push_pull_detector.update(raw_event.depth)
        if push_pull_result == "push":
            events.append(self._classified(raw_event, GestureType.PUSH))
        elif push_pull_result == "pull":
            events.append(self._classified(raw_event, GestureType.PULL))

        return events

    def reset_hand(self, hand: HandLabel) -> None:
        """Gọi khi HandTracker báo 1 tay biến mất khỏi khung hình lâu (xem
        HandTracker._decay_missing_hands) - tránh rotation/push-pull tính
        delta xuyên qua khoảng trống lúc tay không xuất hiện, gây ra delta
        giả không phản ánh chuyển động thật khi tay quay lại."""
        self._per_hand[hand].reset()

    @staticmethod
    def _classified(
        raw_event: GestureEvent, gesture_type: GestureType, value: Optional[float] = None
    ) -> GestureEvent:
        return dataclasses.replace(raw_event, gesture_type=gesture_type, value=value)