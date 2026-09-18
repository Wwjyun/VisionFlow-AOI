from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from devices.ccd_models import CameraConnectionSettings
from gui import icons
from gui.theme import COLORS

# ============================================================
# Sapera 位置選擇對話框（Todo.md P11「以 Qt 對話框取代 AcqConfigDlg」）。
#
# `xx_ccd` 沒有固定的 server／resource／CCF 值：`CameraSettings` 預設為空，由機台上的
# AcqConfigDlg 選取後存進該機台。這個對話框做同一件事——透過機台自己的 Sapera 列舉
# server／resource／CCF，選取結果交由 CCD 頁既有的「Sapera 位置」欄位與「套用」流程
# 寫入機台設定檔，不新增第二條設定路徑。
#
# `probe` 是注入的 callable（正式路徑由 `CcdController` 提供），測試永遠不需要真的 Sapera。
# 對話框本身不 raise：Sapera 不可用時顯示原因短碼，只停用「確定」。
# ============================================================

ACQ_KIND = "Acq"
ACQ_DEVICE_KIND = "AcqDevice"


@dataclass(frozen=True)
class SaperaLocationCatalog:
    """What the machine's own Sapera reported for location selection."""

    servers: tuple[str, ...] = ()
    acq_resources: dict[str, tuple[str, ...]] = field(default_factory=dict)
    acq_devices: dict[str, tuple[str, ...]] = field(default_factory=dict)
    ccf_files: tuple[str, ...] = ()
    ccf_dir: str = ""
    ccf_problem: str = ""
    unavailable_reason: str = ""

    @property
    def available(self) -> bool:
        return not self.unavailable_reason

    def acq_count(self, server: str) -> int:
        return len(self.acq_resources.get(server, ()))

    def device_options(self) -> tuple[tuple[str, int, str], ...]:
        """Every AcqDevice on the machine as ``(server, index, name)``, in enumeration order."""

        options: list[tuple[str, int, str]] = []
        for server in self.servers:
            for index, name in enumerate(self.acq_devices.get(server, ())):
                options.append((server, index, name))
        return tuple(options)

    def all_ccf_candidates(self, extra: str = "") -> tuple[str, ...]:
        candidates = list(self.ccf_files)
        if extra and extra not in candidates:
            candidates.append(extra)
        return tuple(candidates)


@dataclass(frozen=True)
class SaperaLocationResult:
    """The operator's choice; written into the CCD screen's existing Sapera location fields."""

    server_name: str
    resource_index: int
    config_file_path: str
    device_feature_server_name: str = ""
    device_feature_resource_index: int = -1

    def connection(self, fallback: CameraConnectionSettings | None = None) -> CameraConnectionSettings:
        base = fallback or CameraConnectionSettings()
        return CameraConnectionSettings(
            server_name=self.server_name or base.server_name,
            resource_index=self.resource_index if self.server_name else base.resource_index,
            config_file_path=self.config_file_path or base.config_file_path,
            device_feature_server_name=self.device_feature_server_name,
            device_feature_resource_index=self.device_feature_resource_index,
        ).normalized()


def _ccf_candidates(catalog: SaperaLocationCatalog, configured: str) -> tuple[str, ...]:
    return catalog.all_ccf_candidates(configured)


class SaperaLocationDialog(QDialog):
    """Admin-only picker for the machine's Sapera server, Acq resource, AcqDevice and CCF file."""

    def __init__(
        self,
        probe: Callable[[], SaperaLocationCatalog],
        *,
        current: CameraConnectionSettings | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("選擇 Sapera 位置")
        self.setModal(True)
        self.setMinimumWidth(520)
        self._current = (current or CameraConnectionSettings()).normalized()
        self.catalog = self._probe(probe)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self.reason_label = QLabel()
        self.reason_label.setWordWrap(True)
        self.reason_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.reason_label.setVisible(False)
        layout.addWidget(self.reason_label)

        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self.server_combo = QComboBox()
        for server in self.catalog.servers:
            self.server_combo.addItem(server, server)
        form.addRow("擷取伺服器", self.server_combo)

        self.acq_combo = QComboBox()
        form.addRow("Acq Resource", self.acq_combo)

        ccf_row = QWidget()
        ccf_layout = QHBoxLayout(ccf_row)
        ccf_layout.setContentsMargins(0, 0, 0, 0)
        ccf_layout.setSpacing(6)
        self.ccf_edit = QLineEdit(self._current.config_file_path)
        self.ccf_edit.setProperty("mono", "true")
        browse = QPushButton("瀏覽")
        browse.setProperty("variant", "secondary")
        browse.setProperty("size", "sm")
        browse.setCursor(Qt.CursorShape.PointingHandCursor)
        browse.setIcon(icons.icon("folder", size=14, color=COLORS["text_2"]))
        browse.clicked.connect(self._browse_ccf)
        ccf_layout.addWidget(self.ccf_edit, 1)
        ccf_layout.addWidget(browse)
        form.addRow("CCF 檔案", ccf_row)
        layout.addLayout(form)

        self.ccf_list = QListWidget()
        self.ccf_list.setMaximumHeight(96)
        for candidate in _ccf_candidates(self.catalog, self.ccf_edit.text()):
            self.ccf_list.addItem(QListWidgetItem(candidate))
        self.ccf_list.setVisible(self.ccf_list.count() > 0)
        self.ccf_list.itemClicked.connect(lambda item: self.ccf_edit.setText(item.text()))
        layout.addWidget(_hint("Sapera CamFiles\\User 下找到的 CCF（點選即填入）："))
        layout.addWidget(self.ccf_list)

        self.device_label = _hint("相機功能（AcqDevice）")
        layout.addWidget(self.device_label)
        self.device_group: list[QRadioButton] = []
        for server, index, name in self.catalog.device_options():
            button = QRadioButton(f"{server}／{name}（resource {index}）")
            button.setChecked(self._is_current_device(server, index))
            self.device_group.append(button)
            layout.addWidget(button)

        self.device_hint = _hint()
        self.device_hint.setWordWrap(True)
        layout.addWidget(self.device_hint)
        self._auto_select_device()

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText("套用到 CCD 頁")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.save_button = self.buttons.button(QDialogButtonBox.StandardButton.Save)
        layout.addWidget(self.buttons)

        self.server_combo.currentIndexChanged.connect(self._on_server_changed)
        self.ccf_edit.textChanged.connect(self._refresh_ready)
        self._on_server_changed()
        self._apply_catalog_state()

    # ------------------------------------------------------------------
    @staticmethod
    def _probe(probe: Callable[[], SaperaLocationCatalog]) -> SaperaLocationCatalog:
        """Never raise: a broken prober becomes an unavailable catalog with a copyable reason."""

        try:
            catalog = probe()
        except Exception as exc:  # noqa: BLE001 - the dialog must stay usable
            return SaperaLocationCatalog(unavailable_reason=f"E-0401 列舉 Sapera server 失敗：{exc}")
        if catalog is None:
            return SaperaLocationCatalog(unavailable_reason="E-0401 列舉 Sapera server 失敗：沒有回傳結果")
        return catalog

    def _is_current_device(self, server: str, index: int) -> bool:
        return (
            self._current.device_feature_server_name == server
            and self._current.device_feature_resource_index == index
        )

    def _apply_catalog_state(self) -> None:
        if not self.catalog.available:
            self.reason_label.setText(f"無法列舉 Sapera 位置：{self.catalog.unavailable_reason}")
            self.reason_label.setStyleSheet(f"color: {COLORS['warn']}; font-size: 11px;")
            self.reason_label.setVisible(True)
        elif not self.catalog.servers:
            self.reason_label.setText(
                "Sapera 回報 0 個 server（E-0402）；請確認擷取卡驅動與 Sapera LT 安裝，"
                "或先按「執行相機診斷」取得短碼。"
            )
            self.reason_label.setStyleSheet(f"color: {COLORS['warn']}; font-size: 11px;")
            self.reason_label.setVisible(True)
        elif self.catalog.ccf_problem:
            self.reason_label.setText(self.catalog.ccf_problem)
            self.reason_label.setStyleSheet(f"color: {COLORS['text_2']}; font-size: 11px;")
            self.reason_label.setVisible(True)
        else:
            self.reason_label.setVisible(False)
        if self.catalog.ccf_dir:
            self.device_hint.setToolTip(self.catalog.ccf_dir)
        self._refresh_ready()

    def _auto_select_device(self) -> None:
        """Exactly one AcqDevice anywhere: select it and say so. Several: the operator must choose."""

        options = self.catalog.device_options()
        if len(options) == 1:
            self.device_group[0].setChecked(True)
            server, index, name = options[0]
            self.device_hint.setText(
                f"只找到一個 AcqDevice（{server}／{name}，resource {index}），已自動選取；"
                "多個 AcqDevice 時需自行選擇。"
            )
            self.device_hint.setStyleSheet(f"color: {COLORS['text_2']}; font-size: 11px;")
            return
        if len(options) > 1:
            self.device_hint.setText(
                f"找到 {len(options)} 個 AcqDevice，請明確選擇相機功能位置（不自動猜測）。"
            )
            self.device_hint.setStyleSheet(f"color: {COLORS['warn']}; font-size: 11px;")
            return
        self.device_hint.setText("找不到 AcqDevice；Exposure、Gain、Line Rate 將無法寫入相機。")
        self.device_hint.setStyleSheet(f"color: {COLORS['warn']}; font-size: 11px;")

    def _on_server_changed(self, *_args) -> None:
        server = self.selected_server()
        previous = int(self._current.resource_index) if server == self._current.server_name else 0
        names = self.catalog.acq_resources.get(server, ())
        self.acq_combo.clear()
        for index, name in enumerate(names):
            self.acq_combo.addItem(f"{index}：{name}" if name else f"resource {index}", index)
        if self.acq_combo.count() == 0:
            # Sapera reported no Acq resource for this server; keep the field usable and say why.
            self.acq_combo.addItem("（Sapera 未回報 Acq resource）", previous)
        position = self.acq_combo.findData(previous)
        self.acq_combo.setCurrentIndex(position if position >= 0 else 0)
        self._refresh_ready()

    def _refresh_ready(self, *_args) -> None:
        can_save = bool(self.catalog.servers) and bool(self.server_combo.count())
        self.save_button.setEnabled(can_save)

    def _browse_ccf(self) -> None:
        start = self.ccf_edit.text() or self.catalog.ccf_dir
        path, _ = QFileDialog.getOpenFileName(self, "選擇 CCF 檔案", start, "CCF 檔案 (*.ccf)")
        if path:
            self.ccf_edit.setText(path)

    # ------------------------------------------------------------------
    def selected_server(self) -> str:
        return str(self.server_combo.currentData() or self.server_combo.currentText() or "")

    def selected_resource_index(self) -> int:
        data = self.acq_combo.currentData()
        try:
            return int(data)
        except (TypeError, ValueError):
            return 0

    def selected_device(self) -> tuple[str, int]:
        for index, button in enumerate(self.device_group):
            if button.isChecked():
                server, resource_index, _name = self.catalog.device_options()[index]
                return server, resource_index
        return "", -1

    def result_location(self) -> SaperaLocationResult:
        server, resource_index = self.selected_device()
        return SaperaLocationResult(
            server_name=self.selected_server(),
            resource_index=self.selected_resource_index(),
            config_file_path=self.ccf_edit.text().strip(),
            device_feature_server_name=server,
            device_feature_resource_index=resource_index,
        )


def _hint(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 11px;")
    return label
