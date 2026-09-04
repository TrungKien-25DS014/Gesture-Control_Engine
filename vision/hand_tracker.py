"""
vision/hand_tracker.py

Layer 1 - Vision (Giai đoạn B).

Bọc MediaPipe Hands: nhận 1 frame BGR (từ cv2.VideoCapture, ví dụ trong
empty_camera_window.py / main.py), detect tối đa 2 tay cùng lúc, và trả về
danh sách GestureEvent "raw" - chưa mang gesture_type cụ thể (pinch/push/...),
việc phân loại gesture thật sự thuộc về gesture_engine/ ở Giai đoạn C.

Lưu ý cập nhật kiến trúc so với core/interfaces.py bản đầu (Giai đoạn A):
VisionSource.read_frame() ban đầu khai báo trả về `GestureEvent | None`
(1 tay). Giai đoạn B yêu cầu detect 2 tay cùng lúc nên đổi sang
`list[GestureEvent]` (0, 1 hoặc 2 phần tử). Đã cập nhật Protocol tương ứng
trong core/interfaces.py - đây là thay đổi hợp lệ vì Giai đoạn A tự nhận
"chưa cần logic gesture", chưa chốt cứng số lượng tay.

An toàn: không bao giờ raise exception vì tay biến mất khỏi khung hình -
trường hợp đó chỉ đơn giản không có event nào cho tay đó trong list trả về.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import mediapipe as mp

from core.gesture_event import GestureEvent, HandLabel
from vision.depth_estimator import DepthEstimator

# Dùng gốc ngón giữa (landmark 9) làm tâm quy chiếu x/y của cả bàn tay thay
# vì cổ tay (landmark 0) - ổn định hơn khi bàn tay xoay/nghiêng nhẹ, và
# cũng là 1 trong 2 điểm dùng để ước lượng depth nên nhất quán.
_ANCHOR_LANDMARK = 9

_NUM_LANDMARKS = 21


class HandTracker:
    """
    1 instance = 1 phiên tracking (giữ state làm mượt riêng theo từng tay).
    Không tự mở/đọc camera - nhận frame từ bên ngoài để không phụ thuộc vào
    cách app cụ thể mở camera (tách biệt UI khỏi vision, đúng nguyên tắc
    kiến trúc 4 lớp độc lập).
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        smoothing_alpha: float = 0.4,
        mirrored_view: bool = True,
        reset_after_missing_frames: int = 10,
    ) -> None:
        """
        smoothing_alpha: hệ số exponential moving average, 0 < alpha <= 1.
            alpha nhỏ -> mượt hơn nhưng trễ hơn; alpha=1 -> tắt làm mượt.

        mirrored_view: QUAN TRỌNG - chỉ đặt True khi frame_bgr đưa vào
            read_frame() là frame GỐC, CHƯA bị lật ngang (đúng những gì
            cv2.VideoCapture.read() trả về mặc định, không qua cv2.flip()).

            Lý do: MediaPipe được huấn luyện với giả định ảnh đầu vào đã bị
            lật ngang kiểu gương soi (selfie-view), nên khi đưa vào ảnh GỐC
            chưa lật, nhãn Left/Right nó trả về bị NGƯỢC so với tay thật -
            cần tự đảo lại (mirrored_view=True làm việc này).

            Nếu bạn ĐÃ tự cv2.flip(frame, 1) TRƯỚC khi gọi read_frame(),
            hãy đặt mirrored_view=False, vì lúc đó MediaPipe đã nhận đúng
            loại ảnh nó mong đợi, nhãn trả về đã đúng sẵn - đảo thêm 1 lần
            nữa sẽ làm SAI NGƯỢC (đây chính là bug: lật ảnh + đảo nhãn cùng
            lúc = 2 lần bù triệt tiêu nhau thành sai hoàn toàn).
        """
        if not 0 < smoothing_alpha <= 1:
            raise ValueError("smoothing_alpha phải trong (0, 1]")

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._depth_estimators: dict[HandLabel, DepthEstimator] = {
            HandLabel.LEFT: DepthEstimator(),
            HandLabel.RIGHT: DepthEstimator(),
        }
        self._smoothing_alpha = smoothing_alpha
        self._mirrored_view = mirrored_view
        self._reset_after_missing_frames = reset_after_missing_frames

        self._smoothed: dict[HandLabel, tuple[float, float, float]] = {}
        self._missing_streak: dict[HandLabel, int] = {}

    def read_frame(self, frame_bgr) -> list[GestureEvent]:
        """
        frame_bgr: numpy.ndarray ảnh BGR, đúng format trả về bởi
        cv2.VideoCapture.read(). Trả về list rỗng nếu không thấy tay nào -
        KHÔNG bao giờ raise vì lý do mất tay khỏi khung hình.
        """
        frame_rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB, tránh phụ thuộc cv2 chỉ để đảo kênh màu
        results = self._hands.process(frame_rgb)

        now = time.monotonic()
        events: list[GestureEvent] = []
        seen: set[HandLabel] = set()

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = self._resolve_hand_label(handedness)
                if label is None or label in seen:
                    # Bỏ qua nếu không xác định được tay, hoặc MediaPipe trả
                    # trùng nhãn cho 2 detection (hiếm, nhưng phải an toàn).
                    continue
                seen.add(label)

                points = tuple(
                    (lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark
                )
                if len(points) != _NUM_LANDMARKS:
                    continue

                raw_x, raw_y = points[_ANCHOR_LANDMARK][0], points[_ANCHOR_LANDMARK][1]
                raw_depth = self._depth_estimators[label].estimate(points)

                x, y, depth = self._smooth(label, raw_x, raw_y, raw_depth)
                self._missing_streak[label] = 0

                events.append(
                    GestureEvent(
                        hand=label,
                        x=x,
                        y=y,
                        depth=depth,
                        timestamp=now,
                        confidence=float(handedness.classification[0].score),
                        raw_landmarks=points,
                    )
                )

        self._decay_missing_hands(seen)
        return events

    def close(self) -> None:
        """Giải phóng tài nguyên MediaPipe - gọi khi đóng app."""
        self._hands.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- internal ---------------------------------------------------------

    def _resolve_hand_label(self, handedness) -> Optional[HandLabel]:
        if not handedness.classification:
            return None
        raw_label = handedness.classification[0].label  # "Left" hoặc "Right"
        is_left = raw_label == "Left"
        if self._mirrored_view:
            is_left = not is_left
        return HandLabel.LEFT if is_left else HandLabel.RIGHT

    def _smooth(
        self, label: HandLabel, x: float, y: float, depth: float
    ) -> tuple[float, float, float]:
        previous = self._smoothed.get(label)
        if previous is None:
            smoothed = (x, y, depth)
        else:
            a = self._smoothing_alpha
            smoothed = (
                a * x + (1 - a) * previous[0],
                a * y + (1 - a) * previous[1],
                a * depth + (1 - a) * previous[2],
            )
        self._smoothed[label] = smoothed
        return smoothed

    def _decay_missing_hands(self, seen: set[HandLabel]) -> None:
        """
        Tay không xuất hiện N frame liên tiếp -> xóa hẳn smoothing state.
        Tránh trường hợp tay biến mất 1-2 frame do detection nhiễu rồi quay
        lại: giữ state 1 chút cho mượt. Nhưng biến mất thật sự (rút tay ra
        khỏi khung hình) thì phải quên vị trí cũ, không thì lần tới tay xuất
        hiện lại ở vị trí khác sẽ bị "kéo lê" từ vị trí cũ do EMA.
        """
        for label in list(self._smoothed):
            if label in seen:
                continue
            streak = self._missing_streak.get(label, 0) + 1
            self._missing_streak[label] = streak
            if streak >= self._reset_after_missing_frames:
                self._smoothed.pop(label, None)
                self._missing_streak.pop(label, None)