from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from gui import icons
from gui.theme import COLORS, RAIL_W

# ============================================================
# AOI Console left navigation rail
# ============================================================

NAV_ITEMS = [
    ("run", "play", "執行檢測"),
    ("monitor", "eye", "監控模式"),
    ("ccd", "camera", "CCD 控制"),
    ("designer", "designer", "Recipe 設計"),
    ("results", "table", "檢測結果"),
    ("batch_dashboard", "bar_chart", "批量數據圖表"),
]


class NavRail(QWidget):
    screen_changed = Signal(str)
    settings_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("rail")
        self.setFixedWidth(RAIL_W)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 14, 0, 14)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        logo = QLabel("AOI")
        logo.setObjectName("railLogo")
        logo.setFixedSize(34, 34)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(14)

        self._buttons: dict[str, QToolButton] = {}
        for screen_id, icon_name, tooltip in NAV_ITEMS:
            button = QToolButton()
            button.setProperty("role", "rail-btn")
            button.setProperty("active", "false")
            button.setToolTip(tooltip)
            button.setIcon(icons.icon(icon_name, size=20, color=COLORS["text_3"]))
            button.setIconSize(button.iconSize().__class__(20, 20))
            button.setFixedSize(40, 40)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, sid=screen_id: self.screen_changed.emit(sid))
            layout.addWidget(button, 0, Qt.AlignmentFlag.AlignHCenter)
            self._buttons[screen_id] = button

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        layout.addWidget(spacer)

        self.settings_button = QToolButton()
        self.settings_button.setProperty("role", "rail-btn")
        self.settings_button.setToolTip("設定")
        self.settings_button.setIcon(icons.icon("gear", size=20, color=COLORS["text_3"]))
        self.settings_button.setIconSize(self.settings_button.iconSize().__class__(20, 20))
        self.settings_button.setFixedSize(40, 40)
        self.settings_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.settings_button.clicked.connect(self.settings_clicked.emit)
        layout.addWidget(self.settings_button, 0, Qt.AlignmentFlag.AlignHCenter)

        self._current = "run"
        self.set_active("run")

    def set_active(self, screen_id: str) -> None:
        self._current = screen_id
        for sid, button in self._buttons.items():
            button.setProperty("active", "true" if sid == screen_id else "false")
            button.style().unpolish(button)
            button.style().polish(button)

    def set_designer_visible(self, visible: bool) -> None:
        button = self._buttons.get("designer")
        if button is not None:
            button.setVisible(visible)
            if not visible and self._current == "designer":
                self.set_active("run")
                self.screen_changed.emit("run")

    def set_visible_screens(self, screen_ids: set[str]) -> None:
        for sid, button in self._buttons.items():
            button.setVisible(sid in screen_ids)
        if self._current not in screen_ids and screen_ids:
            next_screen = "monitor" if "monitor" in screen_ids else next(iter(screen_ids))
            self.set_active(next_screen)
            self.screen_changed.emit(next_screen)

    def set_settings_visible(self, visible: bool) -> None:
        self.settings_button.setVisible(visible)
