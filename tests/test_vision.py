"""
tests/test_vision.py

Layer 1 - Vision: test theo đúng nguyên tắc dự án - test độc lập bằng dữ
liệu giả lập trước, không cần camera thật.

Gồm 2 phần:
  1. Automated tests (pytest, chạy bằng `pytest tests/test_vision.py`) -
     dùng landmark giả lập, không đụng camera/mediapipe detection thật nên
     chạy được trong CI.
  2. Live demo script (`python tests/test_vision.py --live`) - mở webcam
     thật, in (hand, x, y, depth) real-time để tự tay verify DoD Giai đoạn
     B: đưa tay gần/xa liên tục ~2 phút, depth phải đổi đúng chiều và không
     nhảy loạn khi tay đứng yên. Script này LÀ deliverable của Giai đoạn B,
     không phải phần phụ.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pytest

from core.gesture_event import HandLabel
from vision.depth_estimator import DepthEstimator, MIDDLE_FINGER_MCP, WRIST
from vision.hand_tracker import HandTracker


def _make_landmarks(
    wrist_xy: tuple[float, float] = (0.5, 0.8),
    middle_mcp_xy: tuple[float, float] = (0.5, 0.6),
) -> tuple:
    """
    21 landmark giả lập theo thứ tự chuẩn MediaPipe Hands. Chỉ WRIST và
    MIDDLE_FINGER_MCP mang giá trị có ý nghĩa (2 điểm DepthEstimator dùng),
    các điểm còn lại là placeholder hợp lệ vì không được dùng tới ở Layer 1.
    """
    points = [(0.5, 0.5, 0.0)] * 21
    points[WRIST] = (wrist_xy[0], wrist_xy[1], 0.0)
    points[MIDDLE_FINGER_MCP] = (middle_mcp_xy[0], middle_mcp_xy[1], 0.0)
    return tuple(points)


# ---------------------------------------------------------------------------
# DepthEstimator
# ---------------------------------------------------------------------------

class TestDepthEstimator:
    def test_larger_span_means_smaller_depth(self):
        """Tay lớn trong khung hình (gần camera) -> depth phải nhỏ hơn tay xa."""
        estimator = DepthEstimator()
        close_hand = _make_landmarks(wrist_xy=(0.5, 0.90), middle_mcp_xy=(0.5, 0.60))
        far_hand = _make_landmarks(wrist_xy=(0.5, 0.65), middle_mcp_xy=(0.5, 0.60))

        assert estimator.estimate(close_hand) < estimator.estimate(far_hand)

    def test_output_always_in_unit_range(self):
        estimator = DepthEstimator()
        extreme_close = _make_landmarks(wrist_xy=(0.5, 1.0), middle_mcp_xy=(0.5, 0.0))
        extreme_far = _make_landmarks(wrist_xy=(0.5, 0.501), middle_mcp_xy=(0.5, 0.5))

        for landmarks in (extreme_close, extreme_far):
            depth = estimator.estimate(landmarks)
            assert 0.0 <= depth <= 1.0

    def test_stable_when_hand_still(self):
        """Tay đứng yên hoàn toàn nhiều frame liên tiếp -> depth không được đổi (DoD Giai đoạn B)."""
        estimator = DepthEstimator()
        landmarks = _make_landmarks()

        depths = [estimator.estimate(landmarks) for _ in range(30)]

        assert all(d == depths[0] for d in depths)

    def test_direction_correct_as_hand_moves_away(self):
        """Tay lùi dần đều -> depth không bao giờ được giảm (đúng chiều, DoD Giai đoạn B)."""
        estimator = DepthEstimator()
        depths = []
        for step in range(20):
            span = 0.30 - step * 0.01  # span giảm dần đều = tay lùi dần đều
            landmarks = _make_landmarks(wrist_xy=(0.5, 0.5 + span), middle_mcp_xy=(0.5, 0.5))
            depths.append(estimator.estimate(landmarks))

        assert all(b >= a for a, b in zip(depths, depths[1:]))

    def test_rejects_invalid_range_config(self):
        with pytest.raises(ValueError):
            DepthEstimator(default_near_span=0.05, default_far_span=0.3)


# ---------------------------------------------------------------------------
# HandTracker
# ---------------------------------------------------------------------------

class TestHandTracker:
    def test_blank_frame_returns_no_hands_safely(self):
        """Frame không có tay (hoặc tay vừa mất khỏi khung hình) -> list rỗng, không raise."""
        tracker = HandTracker()
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        events = tracker.read_frame(blank_frame)

        assert events == []
        tracker.close()

    def test_smoothing_reduces_single_frame_jitter(self):
        """EMA: giá trị mượt phải nằm giữa giá trị cũ và mới, không nhảy thẳng (alpha < 1)."""
        tracker = HandTracker(smoothing_alpha=0.3)

        first = tracker._smooth(HandLabel.RIGHT, 0.5, 0.5, 0.5)
        jumped = tracker._smooth(HandLabel.RIGHT, 0.9, 0.9, 0.9)

        assert first == (0.5, 0.5, 0.5)
        assert 0.5 < jumped[0] < 0.9
        tracker.close()

    def test_missing_hand_state_resets_after_streak(self):
        """Tay biến mất đủ lâu -> xóa hẳn smoothing state, không kéo lê vị trí cũ."""
        tracker = HandTracker(reset_after_missing_frames=3)
        tracker._smooth(HandLabel.LEFT, 0.2, 0.2, 0.2)
        assert HandLabel.LEFT in tracker._smoothed

        for _ in range(3):
            tracker._decay_missing_hands(seen=set())

        assert HandLabel.LEFT not in tracker._smoothed
        tracker.close()

    def test_rejects_invalid_smoothing_alpha(self):
        with pytest.raises(ValueError):
            HandTracker(smoothing_alpha=0.0)
        with pytest.raises(ValueError):
            HandTracker(smoothing_alpha=1.5)

    def test_mirrored_view_swaps_label_exactly_once(self):
        """
        Regression test cho bug đã gặp thực tế: lật frame (cv2.flip) VÀ
        swap nhãn cùng lúc = 2 lần bù, triệt tiêu nhau thành sai ngược.

        Test này giả lập object handedness của MediaPipe (không cần chạy
        detection thật) để khóa chặt đúng 1 quy tắc:
          - mirrored_view=True (frame gốc, chưa lật)  -> label bị đảo.
          - mirrored_view=False (frame đã tự lật rồi) -> label giữ nguyên.
        """

        class _FakeClassification:
            def __init__(self, label: str) -> None:
                self.label = label
                self.score = 0.99

        class _FakeHandedness:
            def __init__(self, label: str) -> None:
                self.classification = [_FakeClassification(label)]

        raw_says_left = _FakeHandedness("Left")

        tracker_raw_frame = HandTracker(mirrored_view=True)
        assert tracker_raw_frame._resolve_hand_label(raw_says_left) == HandLabel.RIGHT
        tracker_raw_frame.close()

        tracker_preflipped_frame = HandTracker(mirrored_view=False)
        assert tracker_preflipped_frame._resolve_hand_label(raw_says_left) == HandLabel.LEFT
        tracker_preflipped_frame.close()


# ---------------------------------------------------------------------------
# Live demo script - dùng để tự tay verify DoD Giai đoạn B (cần camera thật)
# ---------------------------------------------------------------------------

def _run_live_demo(camera_index: int = 0) -> None:
    import cv2

    # KHÔNG cv2.flip() frame ở đây: demo này chỉ in số ra console, không
    # hiển thị hình nên không cần lật để "nhìn giống gương". HandTracker
    # nhận frame gốc (mirrored_view=True, mặc định) và tự bù nhãn tay nội
    # bộ. Nếu sau này rendering/ hiển thị hình dạng gương thật, việc lật
    # ảnh nên chỉ xảy ra ở bước VẼ (rendering), còn frame đưa vào
    # HandTracker.read_frame() luôn giữ nguyên gốc - tách biệt input xử lý
    # khỏi cách hiển thị, đúng nguyên tắc kiến trúc 4 lớp độc lập.
    tracker = HandTracker(mirrored_view=True)
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Không mở được camera index={camera_index}")

    print("Đưa tay gần/xa camera liên tục trong ~2 phút. Ctrl+C để dừng.")
    print(f"{'hand':<6}{'x':>8}{'y':>8}{'depth':>8}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue

            for event in tracker.read_frame(frame):
                print(f"{event.hand.name:<6}{event.x:>8.3f}{event.y:>8.3f}{event.depth:>8.3f}")
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
            "File này chứa pytest tests. Chạy `pytest tests/test_vision.py` để test tự động,\n"
            "hoặc `python tests/test_vision.py --live` để demo camera thật (verify DoD Giai đoạn B)."
        )