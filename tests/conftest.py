"""
tests/conftest.py

Fixture dùng chung cho toàn bộ test suite.

qt_app: chỉ cần cho tests/test_rendering.py::TestOverlayWindowSmoke - phần
DUY NHẤT trong dự án bắt buộc phải có QApplication thật (paintEvent()
không chạy nếu không có event loop Qt). Ép QT_QPA_PLATFORM=offscreen TRƯỚC
khi import PySide6 để test chạy được trên máy CI/container không có màn
hình thật, không ảnh hưởng gì đến main.py chạy thật (không set biến này).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="session")
def qt_app():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app