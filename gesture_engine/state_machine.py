"""
gesture_engine/state_machine.py

Layer 2 - Gesture Engine (Giai đoạn C): state machine nắm -> xòe.

Phân loại tư thế bàn tay mỗi frame thành 1 trong 3 trạng thái:
  CLOSED  - nắm chặt (các ngón cong lại)
  OPENING - đang chuyển tiếp giữa nắm và xòe (chưa ổn định ở 1 trong 2 đầu)
  OPEN    - xòe hết (các ngón duỗi thẳng)

Đây LÀ 1 state machine thật (nhớ trạng thái trước đó), không phải phân loại
tức thời độc lập từng frame - lý do: nếu chỉ dựa vào 1 ngưỡng duy nhất,
openness score dao động quanh biên ngưỡng sẽ làm trạng thái nhảy qua lại
liên tục (flicker) dù tay gần như đứng yên. Dùng hysteresis (2 ngưỡng khác
nhau cho chiều đóng->mở và mở->đóng, có vùng đệm OPENING ở giữa) để chống
hiện tượng này - đúng nguyên tắc "ổn định qua ít nhất 20 lần lặp" trong DoD.

World logic (Giai đoạn D) sẽ quan sát chuỗi CLOSED -> OPENING -> OPEN để
biết khi nào "kích hoạt" (vd: mở menu). State machine ở đây chỉ có trách
nhiệm phân loại trạng thái đúng và ổn định; KHÔNG tự quyết định hành động
gì xảy ra khi kích hoạt - đúng ranh giới 4 lớp, đó là việc của world/.
"""

from __future__ import annotations

from enum import Enum, auto

from core.gesture_event import GestureType

# Landmark index theo chuẩn MediaPipe Hands (giống vision/depth_estimator.py).
WRIST = 0

# Mỗi ngón (trừ cái): (MCP, TIP). Dùng MCP làm điểm gốc so sánh thay vì PIP
# vì ngón cái không có PIP theo đúng nghĩa như 4 ngón còn lại, và ngón cái
# CỐ Ý không tính vào openness score dưới đây - chuyển động của nó độc lập
# với 4 ngón kia, gộp chung sẽ chỉ gây nhiễu điểm số. Ngón cái được dùng
# riêng cho pinch ở depth_gestures.py.
_FINGER_JOINTS = {
    "index": (5, 8),
    "middle": (9, 12),
    "ring": (13, 16),
    "pinky": (17, 20),
}

# Tỉ lệ (khoảng cách đầu ngón -> cổ tay) / (khoảng cách MCP -> cổ tay) tối
# thiểu để coi 1 ngón là "duỗi thẳng". >1 vì ngón duỗi luôn xa cổ tay hơn
# gốc ngón của chính nó; 1.15 chừa biên an toàn cho nhiễu detection nhẹ.
_EXTENSION_RATIO = 1.15

Point3 = tuple[float, float, float]


class HandPose(Enum):
    CLOSED = auto()
    OPENING = auto()
    OPEN = auto()


_POSE_TO_GESTURE_TYPE = {
    HandPose.CLOSED: GestureType.HAND_CLOSED,
    HandPose.OPENING: GestureType.HAND_OPENING,
    HandPose.OPEN: GestureType.HAND_OPEN,
}


def openness_score(landmarks: tuple[Point3, ...]) -> float:
    """
    Tỉ lệ trong [0, 1]: 4/4 ngón (trừ cái) duỗi thẳng -> 1.0, cả 4 ngón cong
    lại (nắm đấm) -> 0.0.

    So sánh khoảng cách đầu ngón/MCP tới CỔ TAY (tương đối), không dùng trục
    y tuyệt đối trong ảnh - để không phụ thuộc hướng bàn tay trong khung
    hình (tay úp/ngửa/nghiêng bất kỳ góc nào vẫn tính đúng).
    """
    wrist = landmarks[WRIST]
    extended = 0
    for mcp_idx, tip_idx in _FINGER_JOINTS.values():
        mcp_dist = _dist2(landmarks[mcp_idx], wrist)
        tip_dist = _dist2(landmarks[tip_idx], wrist)
        if mcp_dist < 1e-9:
            continue
        if tip_dist / mcp_dist >= _EXTENSION_RATIO:
            extended += 1
    return extended / len(_FINGER_JOINTS)


def _dist2(a: Point3, b: Point3) -> float:
    dx, dy = a[0] - b[0], a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


class HandPoseStateMachine:
    """1 instance = state machine riêng cho 1 tay (giữ trạng thái trước đó).

    Không dùng chung 1 instance cho 2 tay - giống quy ước HandTracker ở
    Layer 1 (mỗi tay có state làm mượt riêng)."""

    # Hysteresis: ngưỡng "đủ mở" cao hơn ngưỡng "đủ đóng", vùng ở giữa luôn
    # là OPENING - tránh flicker khi openness_score dao động quanh 1 mốc.
    OPEN_THRESHOLD = 0.75
    CLOSED_THRESHOLD = 0.25

    def __init__(self) -> None:
        self._state = HandPose.CLOSED

    @property
    def state(self) -> HandPose:
        return self._state

    def update(self, landmarks: tuple[Point3, ...]) -> HandPose:
        score = openness_score(landmarks)

        if self._state is HandPose.CLOSED:
            if score >= self.OPEN_THRESHOLD:
                self._state = HandPose.OPEN
            elif score > self.CLOSED_THRESHOLD:
                self._state = HandPose.OPENING
        elif self._state is HandPose.OPEN:
            if score <= self.CLOSED_THRESHOLD:
                self._state = HandPose.CLOSED
            elif score < self.OPEN_THRESHOLD:
                self._state = HandPose.OPENING
        else:  # OPENING
            if score >= self.OPEN_THRESHOLD:
                self._state = HandPose.OPEN
            elif score <= self.CLOSED_THRESHOLD:
                self._state = HandPose.CLOSED
            # else: vẫn OPENING, giữ nguyên - đây chính là hysteresis.

        return self._state

    def gesture_type(self) -> GestureType:
        """Ánh xạ trạng thái hiện tại sang GestureType để gesture_engine/
        đóng gói thành GestureEvent - xem gesture_engine/__init__.py."""
        return _POSE_TO_GESTURE_TYPE[self._state]

    def reset(self) -> None:
        """Gọi khi tay biến mất khỏi khung hình lâu (HandTracker báo mất
        tay) - tránh việc tay quay lại đột ngột bị coi là chuyển tiếp mượt
        từ trạng thái cũ không còn ý nghĩa gì nữa."""
        self._state = HandPose.CLOSED