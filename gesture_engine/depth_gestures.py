"""
gesture_engine/depth_gestures.py

Layer 2 - Gesture Engine (Giai đoạn C): các gesture "rời rạc" - khác với
state_machine.py (trạng thái nắm/xòe THƯỜNG TRỰC mỗi frame), các gesture ở
đây chỉ xuất hiện đúng frame xảy ra sự kiện:

  - Pinch: ngón cái + ngón trỏ chạm nhau -> PINCH_START, tách ra -> PINCH_END.
  - Rotation: delta góc vector cổ tay -> gốc ngón giữa giữa 2 frame liên tiếp.
  - Push / Pull: gesture đặc thù AR - tay tiến/lùi theo depth qua 1 cửa sổ
    vài frame gần nhất (không phải delta 2 frame liên tiếp, vì depth ước
    lượng từ ảnh 2D nhiễu hơn x/y nhiều - xem ghi chú trong
    vision/depth_estimator.py).

Mỗi detector: 1 instance = trạng thái riêng cho 1 tay, giống quy ước
HandPoseStateMachine ở state_machine.py.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

# Landmark index theo chuẩn MediaPipe Hands.
WRIST = 0
MIDDLE_FINGER_MCP = 9
THUMB_TIP = 4
INDEX_TIP = 8

Point3 = tuple[float, float, float]


def _dist2(a: Point3, b: Point3) -> float:
    dx, dy = a[0] - b[0], a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


class PinchDetector:
    """
    Pinch = khoảng cách đầu ngón cái <-> đầu ngón trỏ, chuẩn hóa theo hand
    span (wrist -> middle_mcp, cùng cách chuẩn hóa với
    vision/depth_estimator.py) để không phụ thuộc tay gần hay xa camera.

    Có debounce theo SỐ FRAME liên tiếp (không phải theo thời gian tuyệt
    đối, vì tốc độ camera có thể đổi) để không nhận nhầm khi khoảng cách
    ngón tay chỉ thoáng lướt qua ngưỡng do nhiễu detection. Ngưỡng chạm và
    ngưỡng tách dùng hysteresis (tách phải xa hơn chạm) để tránh flicker
    start/end liên tục khi 2 đầu ngón đứng yên đúng ngay biên ngưỡng.
    """

    PINCH_RATIO_THRESHOLD = 0.35    # khoảng cách ngón / hand span <= ngưỡng này -> coi là chạm
    RELEASE_RATIO_THRESHOLD = 0.5   # ngưỡng tách ra CAO hơn ngưỡng chạm (hysteresis)
    DEBOUNCE_FRAMES = 3

    def __init__(self) -> None:
        self._is_pinching = False
        self._candidate_state: Optional[bool] = None
        self._candidate_frames = 0

    def update(self, landmarks: tuple[Point3, ...]) -> Optional[str]:
        """Trả về "start", "end", hoặc None (không đổi trạng thái frame này)."""
        span = _dist2(landmarks[WRIST], landmarks[MIDDLE_FINGER_MCP])
        if span < 1e-9:
            return None

        pinch_dist = _dist2(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
        ratio = pinch_dist / span

        threshold = self.RELEASE_RATIO_THRESHOLD if self._is_pinching else self.PINCH_RATIO_THRESHOLD
        raw_pinching = ratio <= threshold

        if raw_pinching == self._is_pinching:
            # Không có thay đổi ứng viên - reset bộ đếm debounce (tránh 1
            # frame nhiễu đơn lẻ rồi vài frame sau lại nhiễu tiếp bị cộng dồn).
            self._candidate_state = None
            self._candidate_frames = 0
            return None

        if self._candidate_state != raw_pinching:
            self._candidate_state = raw_pinching
            self._candidate_frames = 1
        else:
            self._candidate_frames += 1

        if self._candidate_frames < self.DEBOUNCE_FRAMES:
            return None

        self._is_pinching = raw_pinching
        self._candidate_state = None
        self._candidate_frames = 0
        return "start" if raw_pinching else "end"

    def reset(self) -> None:
        self._is_pinching = False
        self._candidate_state = None
        self._candidate_frames = 0


class RotationDetector:
    """
    Delta góc (radian, dấu +/- theo chiều atan2 chuẩn) của vector cổ tay ->
    gốc ngón giữa, so với frame TRƯỚC ĐÓ của CHÍNH tay này. Chỉ so 2 frame
    liên tiếp (không tích lũy cửa sổ dài như push/pull) vì rotation là tốc
    độ xoay tức thời - x/y landmark ổn định hơn depth nhiều nên không cần
    lọc noise bằng cửa sổ dài (xem vision/hand_tracker.py: EMA đã làm mượt
    x/y từ Layer 1 rồi).

    Bỏ qua delta quá nhỏ (rung tay tự nhiên / nhiễu detection dưới ngưỡng)
    để không spam ROTATE event khi tay gần như đứng yên.
    """

    MIN_DELTA_RADIANS = 0.03  # ~1.7 độ - dưới ngưỡng này coi là noise, không phải rotate thật

    def __init__(self) -> None:
        self._prev_angle: Optional[float] = None

    def update(self, landmarks: tuple[Point3, ...]) -> Optional[float]:
        """Trả về delta góc (radian) nếu vượt ngưỡng noise, ngược lại None."""
        wrist = landmarks[WRIST]
        middle_mcp = landmarks[MIDDLE_FINGER_MCP]
        angle = math.atan2(middle_mcp[1] - wrist[1], middle_mcp[0] - wrist[0])

        if self._prev_angle is None:
            self._prev_angle = angle
            return None

        delta = _angle_diff(angle, self._prev_angle)
        self._prev_angle = angle

        if abs(delta) < self.MIN_DELTA_RADIANS:
            return None
        return delta

    def reset(self) -> None:
        self._prev_angle = None


def _angle_diff(a: float, b: float) -> float:
    """Hiệu góc a - b, chuẩn hóa về (-pi, pi] để tránh nhảy giá trị giả khi
    góc đi qua biên -pi/pi (wraparound của atan2)."""
    diff = a - b
    return (diff + math.pi) % (2 * math.pi) - math.pi


class PushPullDetector:
    """
    Push / pull: gesture đặc thù AR dựa trên delta depth qua 1 CỬA SỔ vài
    frame gần nhất (không dùng delta 2 frame liên tiếp như rotation) - depth
    ước lượng từ ảnh 2D nhiễu hơn x/y nhiều, cửa sổ dài hơn giúp lọc noise
    (đúng ghi chú Giai đoạn B: "depth ước lượng từ ảnh 2D sẽ nhiễu hơn
    nhiều").

    Quy ước depth: 0 = gần camera nhất (core/gesture_event.py). Tay tiến về
    camera (push, tương tác/xác nhận) -> depth GIẢM dần qua cửa sổ. Tay lùi
    ra xa (pull, hủy/thu lại) -> depth TĂNG dần qua cửa sổ.

    Sau khi trigger, có cooldown (theo số frame) trước khi trigger lại -
    tránh 1 lần đẩy tay dài (nhiều frame) gây ra hàng loạt PUSH liên tiếp
    trong lúc tay vẫn còn đang di chuyển.
    """

    WINDOW_SIZE = 6
    DELTA_THRESHOLD = 0.18     # biên độ đổi depth tối thiểu trong cả cửa sổ để coi là push/pull thật
    COOLDOWN_FRAMES = 8

    def __init__(
        self,
        window_size: Optional[int] = None,
        delta_threshold: Optional[float] = None,
        cooldown_frames: Optional[int] = None,
    ) -> None:
        """
        Giai đoạn F - "tinh chỉnh ngưỡng push/pull cho cảm giác tự nhiên" là
        việc CẦN thử nhiều lần với tay thật (không suy ra được bằng công
        thức), nên 3 tham số quan trọng nhất mở constructor để calibrate mà
        không phải sửa trực tiếp source: `delta_threshold` thấp hơn -> push/
        pull nhạy hơn (dễ trigger, cũng dễ nhận nhầm khi tay run/di chuyển
        tự nhiên); `cooldown_frames` cao hơn -> chống double-trigger tốt
        hơn nhưng push/pull liên tiếp nhanh sẽ bị bỏ sót; `window_size` lớn
        hơn -> lọc noise depth tốt hơn nhưng độ trễ phản hồi cũng tăng.
        Không truyền gì -> giữ nguyên hằng số mặc định như trước Giai đoạn F.
        """
        self._window_size = window_size if window_size is not None else self.WINDOW_SIZE
        self._delta_threshold = delta_threshold if delta_threshold is not None else self.DELTA_THRESHOLD
        self._cooldown_frames = cooldown_frames if cooldown_frames is not None else self.COOLDOWN_FRAMES

        self._window: deque[float] = deque(maxlen=self._window_size)
        self._cooldown = 0

    def update(self, depth: float) -> Optional[str]:
        """Trả về "push", "pull", hoặc None."""
        self._window.append(depth)

        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        if len(self._window) < self._window_size:
            return None

        delta = self._window[-1] - self._window[0]  # + = depth tăng (lùi), - = depth giảm (tiến)

        if delta <= -self._delta_threshold:
            self._trigger_cooldown()
            return "push"
        if delta >= self._delta_threshold:
            self._trigger_cooldown()
            return "pull"
        return None

    def _trigger_cooldown(self) -> None:
        self._cooldown = self._cooldown_frames
        self._window.clear()

    def reset(self) -> None:
        self._window.clear()
        self._cooldown = 0