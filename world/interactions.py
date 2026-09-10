"""
world/interactions.py

Layer 3 - World Logic (Giai đoạn D): các loại "world action" có thể gắn vào
1 WorldAnchor (qua anchor.metadata["action"]), kích hoạt khi tay PUSH trúng
object đó (xem world_state.py._trigger_push).

Thiết kế mở theo đúng ghi chú kế hoạch Giai đoạn D: "mở app thật là một loại
world action trong nhiều loại hành động có thể gắn vào anchor (mở link,
chạy script, hiện thông tin...)". launcher.py (LaunchAppAction) chỉ là 1
implementation cụ thể của WorldAction định nghĩa ở đây - thêm loại action
mới = viết class mới implement execute() -> ActionResult, không cần sửa
world_state.py.
"""

from __future__ import annotations

import subprocess
import webbrowser
from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class ActionResult:
    """Kết quả thực thi 1 action.

    Chỉ dùng để log/test - world_state.py không đọc `success` để quyết định
    logic world tiếp theo (world/ không quan tâm action thành công về mặt
    nghiệp vụ thế nào, chỉ ghi lại kết quả gần nhất để debug/verify)."""

    success: bool
    message: str = ""


class WorldAction(Protocol):
    """Ranh giới action: mọi loại action (mở app, mở link, chạy script, hiện
    info...) chỉ cần implement execute() -> ActionResult. WorldState chỉ biết
    đến Protocol này khi push trigger, không biết chi tiết implementation -
    giữ đúng nguyên tắc kiến trúc "không lớp nào biết chi tiết lớp khác"."""

    def execute(self) -> ActionResult: ...


@dataclass(frozen=True)
class OpenLinkAction:
    """Mở 1 URL bằng browser mặc định của hệ thống."""

    url: str

    def execute(self) -> ActionResult:
        try:
            opened = webbrowser.open(self.url)
            return ActionResult(success=bool(opened), message=f"đã mở link: {self.url}")
        except Exception as exc:
            # An toàn: lỗi mở browser (không có display, không có browser...)
            # không được làm crash world logic - chỉ log lại là thất bại.
            return ActionResult(success=False, message=f"lỗi mở link {self.url}: {exc}")


@dataclass(frozen=True)
class RunScriptAction:
    """Chạy 1 script/command hệ thống. Không chờ kết quả (Popen, không
    .wait()) - world logic phải tiếp tục xử lý frame tiếp theo ngay, không
    được block chờ script chạy xong."""

    command: tuple[str, ...]

    def execute(self) -> ActionResult:
        try:
            subprocess.Popen(list(self.command))
            return ActionResult(success=True, message=f"đã chạy: {' '.join(self.command)}")
        except Exception as exc:
            return ActionResult(success=False, message=f"lỗi chạy script {self.command}: {exc}")


@dataclass(frozen=True)
class ShowInfoAction:
    """Hiện thông tin gắn vào anchor.

    Giai đoạn D chưa có UI thật (rendering/ là Giai đoạn E), nên gọi qua 1
    callback (mặc định `print`, dùng cho log/CLI demo) thay vì vẽ trực tiếp -
    rendering/ ở Giai đoạn E có thể truyền callback vẽ overlay info thật vào
    đây mà world_state.py không cần đổi gì."""

    text: str
    on_show: Callable[[str], None] = print

    def execute(self) -> ActionResult:
        self.on_show(self.text)
        return ActionResult(success=True, message=f"đã hiện info: {self.text}")