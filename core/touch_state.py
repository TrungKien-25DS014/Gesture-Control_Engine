"""
core/touch_state.py

Giai đoạn F - Polish tương tác: TouchState là kiểu dữ liệu RANH GIỚI giữa
world/ (Giai đoạn D, nguồn sự thật duy nhất về việc tay có đang chạm/grab
object nào) và rendering/ (Giai đoạn E/F, chỉ đọc để vẽ feedback trực quan
"đổi màu/glow khi depth tay khớp depth object" theo kế hoạch).

Đặt ở core/ (không phải world/ hay rendering/) vì đây đúng là 1 kiểu dữ liệu
chia sẻ qua ranh giới lớp - giống WorldAnchor/GestureEvent, không package nào
được phép import thẳng implementation của package khác (xem core/interfaces.py).
"""

from __future__ import annotations

from enum import Enum, auto


class TouchState(Enum):
    """Trạng thái "chạm" của 1 WorldAnchor, tổng hợp từ tất cả các tay.

    NONE    - không tay nào đang chạm đúng độ sâu/vị trí của object.
    TOUCHED - có tay đang chạm (hit-test 3D trúng) nhưng CHƯA grab - dùng để
              hiện glow nhẹ báo "tay đã khớp depth với object, có thể pinch
              để nhấc lên hoặc push để kích hoạt".
    GRABBED - object đang bị 1 tay pinch-giữ (grab-and-move) - glow đậm hơn
              TOUCHED, ưu tiên hiển thị GRABBED nếu 1 object vừa được tay
              này grab vừa được tay kia chạm cùng lúc.
    """

    NONE = auto()
    TOUCHED = auto()
    GRABBED = auto()