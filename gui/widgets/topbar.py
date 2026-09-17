from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from gui.theme import COLORS, TOPBAR_H
from gui.widgets.common import Badge, Chip, ProgressBar, Segmented

# ============================================================
# AOI Console top bar
# ============================================================

SCREEN_TITLES = {
    "monitor": "監控模式",
    "ccd": "CCD 控制",
    "run": "執行檢測",
    "designer": "Recipe 設計",
    "results": "檢測結果",
    "batch_dashboard": "批量數據圖表",
}

MODE_OPTIONS = [("op", "OP"), ("eng", "工程"), ("admin", "管理")]


class TopBar(QWidget):
    image_chip_clicked = Signal()
    recipe_chip_clicked = Signal()
    mode_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("topbar")
        self.setFixedHeight(TOPBAR_H)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(12)

        self.title_label = QLabel(SCREEN_TITLES["run"])
        self.title_label.setObjectName("topbarTitle")
        layout.addWidget(self.title_label)

        divider = QFrame()
        divider.setObjectName("topbarDivider")
        divider.setFixedHeight(20)
        layout.addWidget(divider)

        self.image_chip = Chip("image", "圖片")
        self.image_chip.clicked.connect(self.image_chip_clicked.emit)
        layout.addWidget(self.image_chip)

        self.recipe_chip = Chip("recipe", "Recipe")
        self.recipe_chip.clicked.connect(self.recipe_chip_clicked.emit)
        layout.addWidget(self.recipe_chip)

        self._progress_widget = QWidget()
        progress_layout = QHBoxLayout(self._progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(8)
        self._progress_widget.setFixedWidth(180)
        self.progress_bar = ProgressBar()
        progress_layout.addWidget(self.progress_bar, 1)
        self.progress_label = QLabel("0%")
        self.progress_label.setProperty("mono", "true")
        self.progress_label.setStyleSheet(f"color: {COLORS['text_2']};")
        progress_layout.addWidget(self.progress_label)
        self._progress_widget.setVisible(False)
        layout.addWidget(self._progress_widget)

        layout.addStretch(1)

        self.backend_badge = Badge("CPU", kind="neutral")
        self.backend_badge.setToolTip("目前使用 CPU")
        self.backend_badge.setAccessibleName("實際運算後端：CPU")
        layout.addWidget(self.backend_badge)

        self.mode_switch = Segmented(MODE_OPTIONS, value="op")
        self.mode_switch.currentChanged.connect(self.mode_changed.emit)
        layout.addWidget(self.mode_switch)

    def set_screen(self, screen_id: str) -> None:
        self.title_label.setText(SCREEN_TITLES.get(screen_id, ""))

    def set_running(self, running: bool, pct: int = 0) -> None:
        self._progress_widget.setVisible(running)
        self.progress_bar.setValue(pct)
        self.progress_label.setText(f"{pct}%")
        self.image_chip.setEnabled(not running)
        self.recipe_chip.setEnabled(not running)

    def set_mode(self, mode: str) -> None:
        self.mode_switch.setCurrent(mode)

    def set_backend_status(self, status: dict | None) -> None:
        status = status or {}
        if status.get("active"):
            device = str(status.get("device_name") or "CUDA").strip()
            text = f"CUDA · {device}" if device.upper() != "CUDA" else "CUDA"
            kind = "accent"
            tooltip = f"目前使用 CUDA\n裝置：{device}"
        elif status.get("requested"):
            text = "CPU FALLBACK"
            kind = "ng"
            reason = str(status.get("fallback_reason") or status.get("reason") or "CUDA 不可用")
            tooltip = f"目前使用 CPU fallback\n原因：{reason}"
        else:
            text = "CPU"
            kind = "neutral"
            tooltip = "目前使用 CPU"
        if status.get("vram_total_bytes"):
            from gui.performance_summary import VRAM_LOW_NOTICE, format_bytes

            tooltip += (
                f"\n整圖上傳前可用顯示卡記憶體：{format_bytes(status.get('vram_free_bytes'))}"
                f"／{format_bytes(status.get('vram_total_bytes'))}"
            )
            if status.get("vram_low"):
                tooltip += f"\n注意：{VRAM_LOW_NOTICE}"
        self.backend_badge.setText(text)
        self.backend_badge.set_kind(kind)
        self.backend_badge.setToolTip(tooltip)
        self.backend_badge.setAccessibleName(f"實際運算後端：{text}")
