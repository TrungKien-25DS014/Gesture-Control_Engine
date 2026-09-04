"""
vision/depth_estimator.py

Layer 1 - Vision: ước lượng depth tương đối của bàn tay so với camera,
CHỈ dựa trên ảnh 2D (không dùng landmark.z do MediaPipe tự tính - giá trị
đó thực chất cũng suy ra từ kích thước tay nên không đáng tin hơn cách tự
làm ở đây, và tự làm thì kiểm soát/giải thích được rõ trong README).

Nguyên lý (theo kế hoạch Giai đoạn B): tay càng lớn trong khung hình thì
càng gần camera. Ta dùng khoảng cách giữa 2 landmark cố định - cổ tay
(WRIST, index 0) và gốc ngón giữa (MIDDLE_FINGER_MCP, index 9) - làm proxy
"kích thước tay nhìn thấy được". Khoảng cách này lớn -> gần camera -> depth
nhỏ (đúng quy ước GestureEvent.depth: 0 = gần camera nhất).

Không cần chính xác tuyệt đối theo mét, chỉ cần:
  1. Đơn điệu đúng chiều (tay tiến -> depth giảm, tay lùi -> depth tăng).
  2. Ổn định tương đối - không nhảy loạn khi tay đứng yên (phần làm mượt
     coordinate-level nằm ở hand_tracker.py, không lặp lại ở đây).

Auto-calibration nhẹ: mỗi người có kích thước tay và khoảng cách ngồi trước
camera khác nhau, nên thay vì hard-code 1 khoảng near/far cố định, class này
tự nới rộng dần range quan sát được (min/max span trong cửa sổ trượt gần
đây) rồi nội suy trong range đó. Trước khi có đủ mẫu, dùng range mặc định.
"""

from __future__ import annotations

from collections import deque

# Index landmark theo chuẩn MediaPipe Hands (21 điểm/tay).
WRIST = 0
MIDDLE_FINGER_MCP = 9

Point3 = tuple[float, float, float]


class DepthEstimator:
    def __init__(
        self,
        default_near_span: float = 0.30,
        default_far_span: float = 0.05,
        calibration_window: int = 90,
        min_samples_before_calibration: int = 15,
    ) -> None:
        if default_near_span <= default_far_span:
            raise ValueError("default_near_span phải lớn hơn default_far_span")

        self._default_near = default_near_span
        self._default_far = default_far_span
        self._min_samples = min_samples_before_calibration
        self._observed_spans: deque[float] = deque(maxlen=calibration_window)

    def estimate(self, landmarks: tuple[Point3, ...]) -> float:
        """
        landmarks: 21 điểm (x, y, z) normalized 0-1 theo MediaPipe, thứ tự
        chuẩn (index 0 = wrist ... index 9 = middle finger mcp ...).

        Trả về depth ước lượng trong [0, 1], 0 = gần camera nhất.
        """
        span = self._hand_span(landmarks)
        self._observed_spans.append(span)

        near, far = self._effective_range()
        span_clamped = min(max(span, far), near)

        # span lớn (gần camera) -> depth nhỏ: nội suy nghịch đảo tuyến tính.
        depth = (near - span_clamped) / (near - far)
        return min(max(depth, 0.0), 1.0)

    def reset_calibration(self) -> None:
        """Gọi khi đổi người dùng / đổi khoảng cách camera cố định để calibrate lại từ đầu."""
        self._observed_spans.clear()

    @staticmethod
    def _hand_span(landmarks: tuple[Point3, ...]) -> float:
        wrist = landmarks[WRIST]
        middle_mcp = landmarks[MIDDLE_FINGER_MCP]
        dx = wrist[0] - middle_mcp[0]
        dy = wrist[1] - middle_mcp[1]
        return (dx * dx + dy * dy) ** 0.5

    def _effective_range(self) -> tuple[float, float]:
        if len(self._observed_spans) < self._min_samples:
            return self._default_near, self._default_far

        lo = min(self._observed_spans)
        hi = max(self._observed_spans)
        if hi - lo < 1e-6:
            # Toàn bộ mẫu gần giống nhau (tay chưa di chuyển đủ để calibrate) -> giữ default.
            return self._default_near, self._default_far

        pad = (hi - lo) * 0.1
        near = hi + pad
        far = max(lo - pad, 1e-4)
        return near, far