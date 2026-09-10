"""
world/world_state.py

Layer 3 - World Logic (Giai đoạn D): implement Protocol WorldLogic khai báo
ở core/interfaces.py. Đây là lớp DUY NHẤT sở hữu và chỉnh sửa WorldAnchor
(đúng docstring core/world_anchor.py) - rendering/ (Giai đoạn E) chỉ đọc
get_anchors() để vẽ, không bao giờ ghi ngược lại.

Nhận GestureEvent đã phân loại từ gesture_engine/ (Giai đoạn C) qua
handle_event(), KHÔNG bao giờ thấy raw_landmarks - đúng ranh giới 4 lớp.

position_3d của tay (event.position_3d = x, y, depth) và của anchor
(WorldAnchor.position_3d) dùng CHUNG 1 hệ quy chiếu chuẩn hóa: x/y trong
[0, 1] theo khung hình, depth ước lượng trong [0, 1] (0 = gần camera nhất) -
giống hệt cách rendering/projection.py dùng position_3d trực tiếp không
cần quy đổi. Nhờ vậy hit-test 3D chỉ đơn giản là so khoảng cách Euclid thật
giữa 2 điểm (WorldAnchor.distance_to), không phải góc trên vòng tròn 2D
như bản launcher desktop cũ.

3 cơ chế chính, mỗi tay (HandLabel) có state riêng - không chia sẻ, đúng
quy ước "tay nào trigger menu, tay nào pinch/push" xuyên suốt dự án:

  - Hit-testing: mỗi frame xác định tay đang chạm anchor nào (nếu có),
    bán kính chạm nhân theo anchor.scale - object to có vùng chạm to hơn.
  - Grab-and-move: PINCH_START trong lúc đang chạm 1 anchor -> anchor bám
    theo delta di chuyển của tay trên cả 3 trục, tới khi PINCH_END.
  - Push để kích hoạt: PUSH trong lúc đang chạm 1 anchor có gắn action
    (anchor.metadata["action"], xem world/interactions.py) -> thực thi
    action đó - thay cho "thả ra ngoài bán kính" ở bản 2D cũ.

Mở rộng hợp lý thêm: PULL trong lúc đang grab -> hủy thao tác, trả anchor
về đúng vị trí lúc bắt đầu grab. Đây là world/ tự quyết định Ý NGHĨA hành
động cho ngữ nghĩa "push = xác nhận, pull = hủy/thu lại" mà Giai đoạn C đã
định nghĩa ở mức gesture thô - gesture_engine/ chỉ báo PULL xảy ra, còn
PULL "hủy cái gì" là quyết định của world logic, đúng ranh giới 4 lớp.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from core.gesture_event import GestureEvent, GestureType, HandLabel
from core.world_anchor import Vector3, WorldAnchor
from world.interactions import ActionResult, WorldAction

# Bán kính hit-test cơ bản, cùng đơn vị chuẩn hóa với position_3d (x/y trong
# [0,1], depth trong [0,1] - xem rendering/projection.py). Nhân với
# anchor.scale khi hit-test từng anchor cụ thể.
BASE_HIT_RADIUS = 0.12


@dataclass
class _GrabState:
    anchor_id: str
    last_position: Vector3
    start_position: Vector3   # vị trí gốc của anchor lúc bắt đầu grab - dùng để hủy khi PULL


class WorldState:
    """Implement Protocol WorldLogic (core/interfaces.py)."""

    def __init__(self, hit_radius: float = BASE_HIT_RADIUS) -> None:
        self._anchors: dict[str, WorldAnchor] = {}
        self._hit_radius = hit_radius
        self._grabs: dict[HandLabel, _GrabState] = {}
        self._touched: dict[HandLabel, Optional[str]] = {
            HandLabel.LEFT: None,
            HandLabel.RIGHT: None,
        }
        self._last_action_result: Optional[ActionResult] = None

    # ---- quản lý anchor ----

    def add_anchor(self, anchor: WorldAnchor) -> WorldAnchor:
        self._anchors[anchor.id] = anchor
        return anchor

    def remove_anchor(self, anchor_id: str) -> None:
        self._anchors.pop(anchor_id, None)
        # Tay nào đang grab đúng anchor vừa bị xóa thì dọn state luôn -
        # tránh grab "treo" vào 1 anchor không còn tồn tại.
        for hand, grab in list(self._grabs.items()):
            if grab.anchor_id == anchor_id:
                del self._grabs[hand]

    def get_anchors(self) -> list[WorldAnchor]:
        """Protocol WorldLogic - rendering/ (Giai đoạn E) đọc qua đây."""
        return list(self._anchors.values())

    def get_anchor(self, anchor_id: str) -> Optional[WorldAnchor]:
        return self._anchors.get(anchor_id)

    @property
    def last_action_result(self) -> Optional[ActionResult]:
        """Kết quả action gần nhất do push kích hoạt - chủ yếu để test/log
        (xem world/interactions.py.ActionResult)."""
        return self._last_action_result

    def touched_anchor_id(self, hand: HandLabel) -> Optional[str]:
        """Anchor đang bị tay `hand` chạm ở frame gần nhất, hoặc None - tách
        riêng để test hit-testing độc lập với grab/push."""
        return self._touched[hand]

    def is_grabbing(self, hand: HandLabel) -> bool:
        return hand in self._grabs

    # ---- xử lý event (Protocol WorldLogic) ----

    def handle_event(self, event: GestureEvent) -> None:
        """Mọi GestureEvent (kể cả trạng thái thường trực HAND_CLOSED/
        OPENING/OPEN) đều mang position_3d, nên hit-test và grab-move luôn
        chạy trước, sau đó mới xử lý riêng theo gesture_type rời rạc."""
        hand = event.hand
        position = event.position_3d

        touched_id = self._hit_test(position)
        self._touched[hand] = touched_id

        if hand in self._grabs:
            self._apply_grab_move(hand, position)

        if event.gesture_type is GestureType.PINCH_START:
            self._start_grab(hand, position, touched_id)
        elif event.gesture_type is GestureType.PINCH_END:
            self._end_grab(hand)
        elif event.gesture_type is GestureType.PUSH:
            self._trigger_push(touched_id)
        elif event.gesture_type is GestureType.PULL:
            self._cancel_grab(hand)

    # ---- hit-testing 3D ----

    def _hit_test(self, position: Vector3) -> Optional[str]:
        """Trả về id anchor GẦN NHẤT mà `position` đang nằm trong bán kính
        chạm, hoặc None nếu không chạm anchor nào. So khoảng cách 3D thật
        (WorldAnchor.distance_to) - không phải góc trên vòng tròn 2D."""
        best_id: Optional[str] = None
        best_distance = float("inf")

        for anchor in self._anchors.values():
            distance = anchor.distance_to(position)
            radius = self._hit_radius * anchor.scale
            if distance <= radius and distance < best_distance:
                best_distance = distance
                best_id = anchor.id

        return best_id

    # ---- grab-and-move ----

    def _start_grab(self, hand: HandLabel, position: Vector3, touched_id: Optional[str]) -> None:
        if touched_id is None or hand in self._grabs:
            return  # không chạm object nào, hoặc tay này đang grab sẵn rồi

        anchor = self._anchors.get(touched_id)
        if anchor is None:
            return

        self._grabs[hand] = _GrabState(
            anchor_id=touched_id,
            last_position=position,
            start_position=anchor.position_3d,
        )

    def _apply_grab_move(self, hand: HandLabel, position: Vector3) -> None:
        grab = self._grabs[hand]
        anchor = self._anchors.get(grab.anchor_id)
        if anchor is None:
            # Anchor bị xóa (remove_anchor) trong lúc đang grab - dọn state,
            # không crash. (remove_anchor cũng tự dọn, đây là an toàn kép.)
            del self._grabs[hand]
            return

        delta = tuple(p - lp for p, lp in zip(position, grab.last_position))
        self._anchors[anchor.id] = anchor.translated(delta)
        grab.last_position = position

    def _end_grab(self, hand: HandLabel) -> None:
        self._grabs.pop(hand, None)

    def _cancel_grab(self, hand: HandLabel) -> None:
        """PULL trong lúc đang grab -> hủy, trả anchor về đúng vị trí lúc
        bắt đầu grab (xem giải thích ngữ nghĩa ở docstring đầu file)."""
        grab = self._grabs.pop(hand, None)
        if grab is None:
            return  # PULL không trong lúc grab - không có gì để hủy

        anchor = self._anchors.get(grab.anchor_id)
        if anchor is not None:
            self._anchors[anchor.id] = replace(anchor, position_3d=grab.start_position)

    # ---- push để kích hoạt ----

    def _trigger_push(self, touched_id: Optional[str]) -> None:
        if touched_id is None:
            return  # push không trúng object nào - không làm gì, không lỗi

        anchor = self._anchors.get(touched_id)
        if anchor is None:
            return

        action: Optional[WorldAction] = anchor.metadata.get("action")
        if action is None:
            return  # anchor không gắn action nào - push không làm gì, không lỗi

        self._last_action_result = action.execute()