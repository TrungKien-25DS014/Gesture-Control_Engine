"""
rendering/animation.py

Giai đoạn F - Polish tương tác: "Animation mượt khi grab/release/push
(spring/easing, không nhảy cứng)".

world/world_state.py (Giai đoạn D) vẫn là nguồn sự thật DUY NHẤT về vị trí
WorldAnchor - khi grab-move, anchor "nhảy" ngay theo tay từng frame (đúng,
vì đó là vị trí thật object đang ở). Cái "giật/nhảy cứng" mà Giai đoạn F
muốn sửa là ở tầng HIỂN THỊ: object/pulse/glow nên đổi mượt qua vài frame
thay vì vẽ đúng y giá trị target ngay lập tức.

Vì vậy SceneAnimator là 1 lớp trạng thái THUẦN TÚY nằm gọn trong rendering/,
không đụng đến world/ - đúng nguyên tắc "khi port sang AR hardware thật chỉ
rendering/ đổi, core/gesture_engine/world giữ nguyên" (xem README Giai đoạn
A). Input là list[WorldAnchor] + touch_states thật (target), output là
WorldAnchor "đã làm mượt" để build_render_items() dùng thay cho anchor gốc.

Kỹ thuật: exponential smoothing (current += (target - current) * (1 -
e^(-rate*dt))) - tương đương lò xo tới hạn (critically damped spring) bậc 1,
đơn giản hơn spring bậc 2 thật (không cần tune mass/stiffness/damping riêng)
nhưng vẫn cho cảm giác "đuổi theo mượt" đúng yêu cầu, và ổn định tuyệt đối
với mọi dt (không bao giờ overshoot/dao động) - phù hợp dt không đều của
QTimer thật hơn spring bậc 2 dễ dao động khi dt lớn bất thường (frame bị
giật/skip).

Thuần Python, không import Qt - test được bằng cách tự truyền dt, không cần
QApplication (đúng nguyên tắc "test/log độc lập từng lớp bằng dữ liệu giả
lập" xuyên suốt dự án).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from core.touch_state import TouchState
from core.world_anchor import WorldAnchor

# Tốc độ đuổi theo target, đơn vị 1/giây (rate lớn -> bám target nhanh hơn,
# ít độ trễ nhưng cũng ít "mượt" hơn). Giá trị chọn qua cảm nhận demo trên
# webcam ở FPS ~30 - đây là hằng số CẦN chỉnh lại bằng tay thật thay vì suy
# ra từ công thức, đúng ghi chú kế hoạch "tinh chỉnh cho cảm giác tự nhiên -
# thường tốn nhiều thời gian nhất, cần thử nhiều lần với tay thật" (Giai
# đoạn F). Có thể override qua constructor SceneAnimator để calibrate mà
# không phải sửa code.
POSITION_SMOOTH_RATE = 16.0
GLOW_SMOOTH_RATE = 10.0
PULSE_DECAY_RATE = 7.0

# Biên độ pulse mặc định khi push trúng object - object "nảy to" thêm 22%
# bán kính rồi tắt dần, làm feedback push rõ hơn glow tĩnh (theo ghi chú kế
# hoạch "feedback trực quan khi tay chạm đúng độ sâu của object").
DEFAULT_PUSH_PULSE_AMOUNT = 0.22

# Glow target (0-1) theo từng TouchState - dùng chung 1 chỗ để scene.py và
# overlay_window.py không phải tự định nghĩa lại mapping này.
GLOW_TARGET_BY_STATE: dict[TouchState, float] = {
    TouchState.NONE: 0.0,
    TouchState.TOUCHED: 0.45,
    TouchState.GRABBED: 1.0,
}


def _exp_smooth_step(current: float, target: float, dt: float, rate: float) -> float:
    """1 bước exponential smoothing - xem giải thích công thức ở docstring
    đầu file. dt<=0 (frame lỗi/trùng timestamp) -> giữ nguyên current, không
    chia cho 0 hay nhảy giá trị vô nghĩa."""
    if dt <= 0.0:
        return current
    alpha = 1.0 - math.exp(-rate * dt)
    return current + (target - current) * alpha


@dataclass
class _AnchorAnim:
    """Trạng thái animation riêng cho 1 anchor - key theo anchor.id."""

    x: float
    y: float
    z: float
    glow: float = 0.0
    pulse: float = 0.0


@dataclass
class SceneAnimator:
    """Giữ trạng thái "hiển thị mượt" cho từng WorldAnchor qua các frame.

    1 instance = 1 phiên render (giống 1 instance GestureEngine/WorldState =
    1 phiên xử lý) - main.py tạo 1 SceneAnimator dùng suốt vòng đời
    GesturePipeline, gọi update() mỗi frame trước khi build_render_items().
    """

    position_rate: float = POSITION_SMOOTH_RATE
    glow_rate: float = GLOW_SMOOTH_RATE
    pulse_decay_rate: float = PULSE_DECAY_RATE

    _anims: dict[str, _AnchorAnim] = field(default_factory=dict, repr=False)

    def update(
        self,
        anchors: list[WorldAnchor],
        touch_states: dict[str, TouchState],
        dt: float,
    ) -> None:
        """Cập nhật trạng thái smoothed theo target mới nhất từ world/. Gọi
        đúng 1 lần/frame, TRƯỚC render_items()."""
        seen_ids: set[str] = set()

        for anchor in anchors:
            seen_ids.add(anchor.id)
            anim = self._anims.get(anchor.id)

            if anim is None:
                # Anchor mới xuất hiện (add_anchor lần đầu hoặc vừa được
                # thêm giữa chừng) - khởi tạo current = target luôn, KHÔNG
                # animate từ (0,0,0) hay từ đâu đó ngẫu nhiên, tránh object
                # mới "bay" từ góc màn hình vào vị trí thật.
                x, y, z = anchor.position_3d
                anim = _AnchorAnim(x=x, y=y, z=z)
                self._anims[anchor.id] = anim

            target_x, target_y, target_z = anchor.position_3d
            anim.x = _exp_smooth_step(anim.x, target_x, dt, self.position_rate)
            anim.y = _exp_smooth_step(anim.y, target_y, dt, self.position_rate)
            anim.z = _exp_smooth_step(anim.z, target_z, dt, self.position_rate)

            glow_target = GLOW_TARGET_BY_STATE[touch_states.get(anchor.id, TouchState.NONE)]
            anim.glow = _exp_smooth_step(anim.glow, glow_target, dt, self.glow_rate)

            # pulse luôn đuổi về 0 (target cố định) - trigger_push() đẩy
            # current lên cao đột ngột, rồi mỗi frame tự tắt dần về đây.
            anim.pulse = _exp_smooth_step(anim.pulse, 0.0, dt, self.pulse_decay_rate)

        # Dọn anchor không còn tồn tại (world_state.remove_anchor đã xóa) -
        # tránh leak dict tăng dần vô hạn qua thời gian chạy dài.
        for stale_id in set(self._anims) - seen_ids:
            del self._anims[stale_id]

    def trigger_pulse(self, anchor_id: str, amount: float = DEFAULT_PUSH_PULSE_AMOUNT) -> None:
        """Gọi khi world_state.pop_pending_push() trả về 1 anchor_id - đẩy
        pulse lên `amount` ngay lập tức (không qua smoothing, vì đây LÀ giá
        trị khởi đầu cần nảy đột ngột), sau đó update() sẽ tự làm nó tắt dần.
        Không làm gì nếu anchor đã biến mất trước khi frame này chạy tới
        (an toàn, không raise)."""
        anim = self._anims.get(anchor_id)
        if anim is None:
            return
        anim.pulse = max(anim.pulse, amount)

    def glow_of(self, anchor_id: str) -> float:
        """Giá trị glow đã làm mượt (0-1) của 1 anchor - dùng khi cần đọc rời
        rạc thay vì qua smoothed_anchor()."""
        anim = self._anims.get(anchor_id)
        return anim.glow if anim is not None else 0.0

    def smoothed_anchor(self, anchor: WorldAnchor) -> WorldAnchor:
        """Trả về bản sao của `anchor` với position_3d đã làm mượt và scale
        đã cộng thêm hiệu ứng pulse - dùng thay cho anchor gốc khi đưa vào
        rendering.scene.build_render_items(). Nếu update() chưa từng thấy
        anchor này (gọi sai thứ tự) thì trả về nguyên bản, không raise."""
        anim = self._anims.get(anchor.id)
        if anim is None:
            return anchor
        return replace(
            anchor,
            position_3d=(anim.x, anim.y, anim.z),
            scale=anchor.scale * (1.0 + anim.pulse),
        )

    def smoothed_anchors(self, anchors: list[WorldAnchor]) -> list[WorldAnchor]:
        """Tiện ích: áp smoothed_anchor() cho cả danh sách, giữ nguyên thứ
        tự - overlay_window.py gọi hàm này ngay trước build_render_items()."""
        return [self.smoothed_anchor(anchor) for anchor in anchors]