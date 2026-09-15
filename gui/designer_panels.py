from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QButtonGroup,
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QRadioButton,
    QVBoxLayout,
)

from core.gpu_runtime import GpuRuntime
from gui.theme import COLORS
from gui.widgets.common import Toggle
from gui.widgets.panel import Panel


def _form_grid() -> QFormLayout:
    form = QFormLayout()
    form.setHorizontalSpacing(12)
    form.setVerticalSpacing(8)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return form


def _label(text: str) -> QLabel:
    widget = QLabel(text)
    widget.setProperty("role", "form-label")
    return widget


class RecipeInfoPanel(Panel):
    def __init__(self, parent=None):
        super().__init__(title="Recipe 資訊", parent=parent)
        form = _form_grid()
        self.recipe_name_edit = QLineEdit("PRODUCT_A_CIRCLE_401_1_AOI_01")
        self.product_id_edit = QLineEdit("PRODUCT_A")
        self.machine_id_edit = QLineEdit("AOI_01")
        self.version_edit = QLineEdit("0.1.0")
        self.pixel_size_um_edit = QLineEdit()
        for widget in (
            self.recipe_name_edit,
            self.product_id_edit,
            self.machine_id_edit,
            self.version_edit,
            self.pixel_size_um_edit,
        ):
            widget.setProperty("mono", "true")
        self.pixel_size_um_edit.setPlaceholderText("未填則 CSV 保持 px²")
        validator = QDoubleValidator(0.000000001, 1_000_000_000.0, 9, self.pixel_size_um_edit)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.pixel_size_um_edit.setValidator(validator)
        self.pixel_size_um_edit.setToolTip("1 px 對應的微米數；CSV 面積會乘上此數值的平方。")
        form.addRow(_label("Recipe 名稱"), self.recipe_name_edit)
        form.addRow(_label("產品 Product"), self.product_id_edit)
        form.addRow(_label("機台 Machine"), self.machine_id_edit)
        form.addRow(_label("版本 Version"), self.version_edit)
        form.addRow(_label("精度 (µm/px)"), self.pixel_size_um_edit)
        self.add_layout(form)


class BackendChoiceCard(QFrame):
    """One selectable execution-policy option: a radio title plus what it actually does."""

    def __init__(self, value: str, title: str, description: str, group: QButtonGroup, parent=None):
        super().__init__(parent)
        self.value = value
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        self.radio = QRadioButton(title)
        self.radio.setStyleSheet(
            "QRadioButton { font-size: 12px; font-weight: 600; spacing: 8px; }"
            "QRadioButton::indicator { width: 10px; height: 10px; border-radius: 7px; "
            f"border: 2px solid {COLORS['border_strong']}; background: {COLORS['surface']}; }}"
            "QRadioButton::indicator:checked { "
            f"border: 4px solid {COLORS['accent']}; width: 6px; height: 6px; }}"
        )
        group.addButton(self.radio)
        layout.addWidget(self.radio)
        self.description_label = QLabel(description)
        self.description_label.setWordWrap(True)
        self.description_label.setContentsMargins(22, 0, 0, 0)
        self.description_label.setStyleSheet(f"color: {COLORS['text_2']}; font-size: 11px;")
        layout.addWidget(self.description_label)
        self.radio.toggled.connect(lambda _checked: self._restyle())
        self._restyle()

    def mousePressEvent(self, event) -> None:
        self.radio.click()
        super().mousePressEvent(event)

    def _restyle(self) -> None:
        # Selection is shown by the radio mark and a heavier border, not by colour alone.
        selected = self.radio.isChecked()
        border = COLORS["accent"] if selected else COLORS["border"]
        background = COLORS["accent_softer"] if selected else COLORS["surface"]
        width = 2 if selected else 1
        self.setStyleSheet(
            f"BackendChoiceCard {{ border: {width}px solid {border}; border-radius: 6px; "
            f"background: {background}; }}"
        )


class GpuSettingsPanel(Panel):
    """Recipe GPU settings with one explicit execution policy instead of mode + fallback switches.

    ``gpu.mode`` and ``gpu.fallback_to_cpu`` combine at runtime (``RecipeManager``): ``cpu`` never
    loads CUDA, ``auto`` with fallback may restart a Detector on CPU, and both ``cuda`` and ``auto``
    without fallback are strict. The old combo box plus toggle exposed four combinations for three
    behaviours, so the panel now offers the three behaviours directly and keeps a loaded Recipe's
    exact values until the operator picks a different policy.
    """

    POLICY_CPU = "cpu"
    POLICY_GPU_FALLBACK = "gpu_fallback"
    POLICY_GPU_STRICT = "gpu_strict"
    _CANONICAL = {
        POLICY_CPU: ("cpu", True),
        POLICY_GPU_FALLBACK: ("auto", True),
        POLICY_GPU_STRICT: ("cuda", False),
    }
    POLICY_TITLES = {
        POLICY_CPU: "僅 CPU",
        POLICY_GPU_FALLBACK: "GPU 優先，失敗改用 CPU",
        POLICY_GPU_STRICT: "僅 GPU（嚴格）",
    }

    policy_changed = Signal(str)

    def __init__(self, refresh_status, refresh_detector_status, parent=None):
        super().__init__(title="運算後端 GPU / CPU", parent=parent)
        self._policy_group = QButtonGroup(self)
        self._policy_group.setExclusive(True)
        options = (
            (self.POLICY_CPU, "不載入 CUDA DLL，所有步驟都在 CPU 執行；結果即為正確性基準。"),
            (
                self.POLICY_GPU_FALLBACK,
                "已開啟 GPU 的 Detector 優先使用 CUDA；CUDA 不可用或執行失敗時，"
                "整個 Detector 自動改用 CPU 重跑，檢測不中斷。",
            ),
            (
                self.POLICY_GPU_STRICT,
                "已開啟 GPU 的 Detector 必須成功使用 CUDA；CUDA 不可用或失敗時檢測直接報錯，"
                "不會改用 CPU。適合驗證 GPU 環境。",
            ),
        )
        self.policy_cards: dict[str, BackendChoiceCard] = {}
        for value, description in options:
            card = BackendChoiceCard(value, self.POLICY_TITLES[value], description, self._policy_group)
            card.radio.setProperty("policy", value)
            card.radio.toggled.connect(lambda checked, v=value: self._on_policy_toggled(v, checked))
            self.policy_cards[value] = card
            self.add_widget(card)

        self.detector_hint_label = QLabel("實際是否使用 GPU，還需在 Detector 清單開啟各 Detector 的 GPU 開關。")
        self.detector_hint_label.setWordWrap(True)
        self.detector_hint_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 11px;")
        self.add_widget(self.detector_hint_label)

        self.advanced_caption = QLabel("GPU 進階設定")
        self.advanced_caption.setProperty("role", "form-label")
        self.add_widget(self.advanced_caption)
        form = _form_grid()
        self.tiling_toggle = Toggle(checked=False)
        self.tiling_toggle.setToolTip("整圖上傳 GPU 後，切小圖（ROI）也在 GPU 上處理。")
        self.display_toggle = Toggle(checked=False)
        self.display_toggle.setToolTip("GUI 載入影像預覽時，色彩轉換使用 GPU。")
        self.dll_path_edit = QLineEdit(GpuRuntime.DEFAULT_DLL)
        self.dll_path_edit.setProperty("mono", "true")
        self._advanced_labels = (
            _label("切小圖使用 GPU"), _label("GUI 預覽使用 GPU"), _label("CUDA DLL 路徑"),
        )
        form.addRow(self._advanced_labels[0], self.tiling_toggle)
        form.addRow(self._advanced_labels[1], self.display_toggle)
        form.addRow(self._advanced_labels[2], self.dll_path_edit)
        self.add_layout(form)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 11px;")
        self.add_widget(self.status_label)

        self._refresh_status = refresh_status
        self._refresh_detector_status = refresh_detector_status
        self._loaded_values: tuple[str, bool] | None = None
        self._setting_policy = False
        self.set_gpu_values("auto", True)
        self.dll_path_edit.editingFinished.connect(refresh_status)
        self.tiling_toggle.toggled.connect(lambda _checked: refresh_status())
        self.display_toggle.toggled.connect(lambda _checked: refresh_status())

    @classmethod
    def policy_for(cls, mode: str, fallback_to_cpu: bool) -> str:
        mode = str(mode or "auto").lower()
        if mode == "cpu":
            return cls.POLICY_CPU
        if mode == "cuda" or not fallback_to_cpu:
            return cls.POLICY_GPU_STRICT
        return cls.POLICY_GPU_FALLBACK

    def policy(self) -> str:
        checked = self._policy_group.checkedButton()
        return str(checked.property("policy")) if checked is not None else self.POLICY_GPU_FALLBACK

    def set_gpu_values(self, mode: str, fallback_to_cpu: bool) -> None:
        """Show a Recipe's policy without emitting a user change or rewriting its values."""
        self._loaded_values = (str(mode or "auto").lower(), bool(fallback_to_cpu))
        self._setting_policy = True
        try:
            self.policy_cards[self.policy_for(mode, fallback_to_cpu)].radio.setChecked(True)
        finally:
            self._setting_policy = False
        self._apply_policy_enablement()

    def gpu_values(self) -> tuple[str, bool]:
        """Return ``(mode, fallback_to_cpu)``: the loaded pair while its policy is unchanged."""
        policy = self.policy()
        if self._loaded_values is not None and self.policy_for(*self._loaded_values) == policy:
            return self._loaded_values
        return self._CANONICAL[policy]

    def legacy_note(self) -> str:
        if self._loaded_values == ("auto", False) and self.policy() == self.POLICY_GPU_STRICT:
            return "此 Recipe 原設定為 mode=auto 且關閉失敗回退，行為等同「僅 GPU（嚴格）」；未變更時會原樣保存。"
        return ""

    def _on_policy_toggled(self, value: str, checked: bool) -> None:
        if not checked:
            return
        self._apply_policy_enablement()
        if self._setting_policy:
            return
        self.policy_changed.emit(value)
        self._refresh_status()
        self._refresh_detector_status()

    def _apply_policy_enablement(self) -> None:
        gpu_allowed = self.policy() != self.POLICY_CPU
        for widget in (self.tiling_toggle, self.display_toggle, self.dll_path_edit, *self._advanced_labels):
            widget.setEnabled(gpu_allowed)
        self.detector_hint_label.setVisible(gpu_allowed)


class PreviewPanel(Panel):
    def __init__(self, preview_label, parent=None):
        super().__init__(title="切圖預覽", parent=parent)
        self.preview_label = preview_label
        self.add_widget(self.preview_label)
        self.status_label = QLabel("尚未預覽")
        self.status_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 9pt;")
        self.status_label.setWordWrap(True)
        self.add_widget(self.status_label)
