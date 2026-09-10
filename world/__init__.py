"""
world/__init__.py

Layer 3 - World Logic (Giai đoạn D): facade công khai của package.

Import từ đây (`from world import WorldState, ...`) thay vì import trực
tiếp từng submodule - giống quy ước gesture_engine/__init__.py.
"""

from __future__ import annotations

from world.interactions import ActionResult, OpenLinkAction, RunScriptAction, ShowInfoAction, WorldAction
from world.launcher import LaunchAppAction
from world.world_state import WorldState

__all__ = [
    "WorldState",
    "WorldAction",
    "ActionResult",
    "OpenLinkAction",
    "RunScriptAction",
    "ShowInfoAction",
    "LaunchAppAction",
]