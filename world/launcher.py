"""
world/launcher.py

Layer 3 - World Logic (Giai đoạn D): LaunchAppAction - mở 1 app thật trên
máy. Đây CHỈ là 1 loại WorldAction cụ thể (xem world/interactions.py) trong
nhiều loại hành động có thể gắn vào anchor - không có gì đặc biệt hơn
OpenLinkAction/RunScriptAction ngoài việc tự chọn đúng lệnh mở app theo
từng hệ điều hành, thay vì bắt người dùng tự biết command.

Đúng nguyên tắc kiến trúc: world_state.py chỉ gọi action.execute() qua
Protocol WorldAction, không biết và không cần biết đây là app launcher.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field

from world.interactions import ActionResult


@dataclass(frozen=True)
class LaunchAppAction:
    """Mở app thật bằng tên/đường dẫn.

    - macOS   : `open -a <app_name> <args>`
    - Windows : `start "" <app_name> <args>` (qua cmd /c)
    - Linux   : gọi trực tiếp binary/lệnh <app_name> <args>

    Không dùng .wait() - mở app xong world logic phải xử lý frame tiếp theo
    ngay, không block chờ app khởi động/đóng."""

    app_name: str
    args: tuple[str, ...] = field(default_factory=tuple)

    def execute(self) -> ActionResult:
        try:
            subprocess.Popen(self._build_command())
            return ActionResult(success=True, message=f"đã mở app: {self.app_name}")
        except Exception as exc:
            # An toàn: app không tồn tại / lỗi hệ thống không được crash
            # world logic - chỉ log lại là thất bại, giống các action khác.
            return ActionResult(success=False, message=f"lỗi mở app {self.app_name}: {exc}")

    def _build_command(self) -> list[str]:
        system = platform.system()
        if system == "Darwin":
            return ["open", "-a", self.app_name, *self.args]
        if system == "Windows":
            # "" đầu tiên sau start là tham số title, bắt buộc khi app_name
            # có khoảng trắng/đứng trong dấu ngoặc.
            return ["cmd", "/c", "start", "", self.app_name, *self.args]
        return [self.app_name, *self.args]