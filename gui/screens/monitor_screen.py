from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from core.batch_dashboard import BatchDashboardBuilder
from gui import icons
from gui.theme import COLORS
from gui.widgets.common import EmptyState, ProgressBar, Segmented
from gui.widgets.panel import Panel
from gui.widgets.scatter_chart import ImageScatterChart
from gui.table_models import RowTableModel, StatusFilterProxyModel, TableColumn


MONITOR_HISTORY_LIMIT = 200
MONITOR_SEQUENCE_SCATTER_LIMIT = 50
MONITOR_SOURCE_FOLDER = "folder"
MONITOR_SOURCE_CAMERA = "camera"
MONITOR_SOURCES = (MONITOR_SOURCE_FOLDER, MONITOR_SOURCE_CAMERA)


def _format_duration(value: object) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "-"
    if seconds <= 0:
        return "-"
    return f"{seconds:.2f}s" if seconds < 10 else f"{seconds:.1f}s"


class MonitorControlPanel(Panel):
    choose_folder_requested = Signal()
    choose_move_folder_requested = Signal()
    source_changed = Signal(str)
    start_requested = Signal()
    stop_requested = Signal()

    def __init__(self, parent=None):
        self.source_segmented = Segmented(
            [(MONITOR_SOURCE_FOLDER, "監控資料夾"), (MONITOR_SOURCE_CAMERA, "相機直連")],
            value=MONITOR_SOURCE_FOLDER,
        )
        self.source_segmented.setToolTip("選擇監控影像來源")
        super().__init__(title="監控來源", actions=self.source_segmented, parent=parent)
        self._folder: str | None = None
        self._source = MONITOR_SOURCE_FOLDER
        self.source_segmented.currentChanged.connect(self.source_changed.emit)

        self.folder_label = QLabel("尚未選擇資料夾")
        self.folder_label.setProperty("mono", "true")
        self.folder_label.setWordWrap(True)
        self.folder_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 12px;")
        self.add_widget(self.folder_label)

        move_label = QLabel("處理後影像保留在監控資料夾")
        move_label.setProperty("mono", "true")
        move_label.setWordWrap(True)
        move_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 12px;")
        self.move_folder_label = move_label
        self.add_widget(move_label)

        self.camera_status_label = QLabel("相機：離線")
        self.camera_status_label.setProperty("mono", "true")
        self.camera_status_label.setWordWrap(True)
        self.camera_status_label.setStyleSheet(f"color: {COLORS['text_2']}; font-size: 12px;")
        self.camera_status_label.setVisible(False)
        self.add_widget(self.camera_status_label)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)

        self.choose_button = QPushButton("選擇資料夾")
        self.choose_button.setProperty("variant", "secondary")
        self.choose_button.setProperty("size", "sm")
        self.choose_button.setIcon(icons.icon("folder", size=14, color=COLORS["text_2"]))
        self.choose_button.clicked.connect(self.choose_folder_requested.emit)
        button_layout.addWidget(self.choose_button)

        self.move_folder_button = QPushButton("搬移至")
        self.move_folder_button.setProperty("variant", "secondary")
        self.move_folder_button.setProperty("size", "sm")
        self.move_folder_button.setIcon(icons.icon("folder", size=14, color=COLORS["text_2"]))
        self.move_folder_button.clicked.connect(self.choose_move_folder_requested.emit)
        button_layout.addWidget(self.move_folder_button)

        self.start_button = QPushButton("啟動")
        self.start_button.setProperty("variant", "primary")
        self.start_button.setProperty("size", "sm")
        self.start_button.setIcon(icons.icon("play", size=14, color="#ffffff"))
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_requested.emit)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("停止")
        self.stop_button.setProperty("variant", "danger-ghost")
        self.stop_button.setProperty("size", "sm")
        self.stop_button.setIcon(icons.icon("x", size=14, color=COLORS["ng"]))
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch(1)
        self.add_widget(button_row)

        progress_row = QWidget()
        progress_layout = QHBoxLayout(progress_row)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(10)
        self.progress_bar = ProgressBar()
        progress_layout.addWidget(self.progress_bar, 1)
        self.progress_pct_label = QLabel("0%")
        self.progress_pct_label.setProperty("mono", "true")
        self.progress_pct_label.setFixedWidth(38)
        self.progress_pct_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        progress_layout.addWidget(self.progress_pct_label)
        self.add_widget(progress_row)

        self.message_label = QLabel("等待選擇資料夾與 Recipe")
        self.message_label.setProperty("mono", "true")
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet(f"color: {COLORS['text_2']}; font-size: 11px;")
        self.add_widget(self.message_label)

    def set_folder(self, folder: str | None) -> None:
        self._folder = folder
        self.folder_label.setText(folder or "尚未選擇資料夾")
        self.folder_label.setStyleSheet(
            f"color: {COLORS['text_2'] if folder else COLORS['text_3']}; font-size: 12px;"
        )

    def set_move_folder(self, folder: str | None) -> None:
        text = f"處理後影像搬移至：{folder}" if folder else "處理後影像保留在監控資料夾"
        self.move_folder_label.setText(text)
        self.move_folder_label.setStyleSheet(
            f"color: {COLORS['text_2'] if folder else COLORS['text_3']}; font-size: 12px;"
        )

    def set_source(self, source: str) -> None:
        self._source = source if source in MONITOR_SOURCES else MONITOR_SOURCE_FOLDER
        self.source_segmented.setCurrent(self._source)
        is_folder = self._source == MONITOR_SOURCE_FOLDER
        for widget in (self.folder_label, self.move_folder_label, self.choose_button, self.move_folder_button):
            widget.setVisible(is_folder)
        self.camera_status_label.setVisible(not is_folder)

    def source(self) -> str:
        return self._source

    def set_source_selectable(self, selectable: bool) -> None:
        self.source_segmented.setEnabled(selectable)

    def set_camera_status_text(self, text: str) -> None:
        self.camera_status_label.setText(text)

    def set_ready(self, ready: bool, running: bool) -> None:
        self.choose_button.setEnabled(not running)
        self.move_folder_button.setEnabled(not running)
        self.start_button.setEnabled(ready and not running)
        self.stop_button.setEnabled(running)
        self.start_button.setText("監控中" if running else "啟動")

    def set_progress(self, pct: int, message: str) -> None:
        pct = max(0, min(100, int(pct)))
        self.progress_bar.setValue(pct)
        self.progress_pct_label.setText(f"{pct}%")
        self.message_label.setText(message)


class MonitorStatsPanel(Panel):
    def __init__(self, parent=None):
        super().__init__(title="監控狀態", parent=parent)
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self._labels: dict[str, QLabel] = {}
        for key, label in (("processed", "已處理"), ("pass", "PASS"), ("ng", "NG"), ("error", "ERROR")):
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(2)
            value_label = QLabel("0")
            value_label.setProperty("mono", "true")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label.setStyleSheet("font-size: 20px; font-weight: 700;")
            name_label = QLabel(label)
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 10px;")
            cell_layout.addWidget(value_label)
            cell_layout.addWidget(name_label)
            layout.addWidget(cell, 1)
            self._labels[key] = value_label
        self.add_widget(row)

    def set_counts(self, items: list[dict]) -> None:
        counts = {
            "processed": len(items),
            "pass": sum(1 for item in items if item.get("final_result") == "PASS"),
            "ng": sum(1 for item in items if item.get("final_result") == "NG"),
            "error": sum(1 for item in items if item.get("final_result") == "ERROR"),
        }
        for key, value in counts.items():
            self._labels[key].setText(str(value))


class MonitorTablePanel(Panel):
    open_original_requested = Signal(dict)

    def __init__(self, parent=None):
        self.filter_segmented = Segmented(
            [("all", "全部"), ("pass", "PASS"), ("ng", "NG"), ("error", "ERROR")],
            value="all",
        )
        super().__init__(title="已處理影像", actions=self.filter_segmented, flush=True, parent=parent)
        self._items: list[dict] = []
        self.model = RowTableModel(
            [
                TableColumn("時間", "processed_at"),
                TableColumn("影像", "image_name", tooltip_key="image_path"),
                TableColumn("結果", "final_result"),
                TableColumn("缺陷", "defect_count", align_right=True),
                TableColumn("NG", "ng_count", align_right=True),
                TableColumn("端到端耗時", "duration_sec", formatter=_format_duration, align_right=True),
            ]
        )
        self.proxy = StatusFilterProxyModel(parent=self)
        self.proxy.setSourceModel(self.model)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        self.filter_segmented.currentChanged.connect(self._on_filter_changed)
        self.add_widget(self.table, 1)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

    def set_items(self, items: list[dict]) -> None:
        self._items = list(items)
        self.model.set_rows(items)

    def prepend_item(self, item: dict, limit: int) -> None:
        self._items.insert(0, item)
        self._items = self._items[:limit]
        self.model.prepend(item, limit)

    def selected_row_index(self) -> int:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return -1
        return selected[0].row()

    def selected_item(self) -> dict | None:
        return self.proxy.row_dict(self.selected_row_index())

    def _on_filter_changed(self, status: str) -> None:
        self.proxy.set_status(status)
        if self.proxy.rowCount() > 0:
            self.table.selectRow(0)
        else:
            self.table.clearSelection()

    def _show_context_menu(self, pos) -> None:
        row = self.table.rowAt(pos.y())
        item = self.proxy.row_dict(row)
        if item is None:
            return
        self.table.selectRow(row)
        menu = QMenu(self.table)
        open_action = menu.addAction("開啟原始影像")
        action = menu.exec(self.table.viewport().mapToGlobal(pos))
        if action == open_action:
            self.open_original_requested.emit(item)


class MonitorScreen(QWidget):
    choose_folder_requested = Signal()
    choose_move_folder_requested = Signal()
    source_changed = Signal(str)
    open_original_requested = Signal(dict)
    start_requested = Signal()
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[dict] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(12)

        self.control_panel = MonitorControlPanel()
        self.stats_panel = MonitorStatsPanel()
        top_layout.addWidget(self.control_panel, 2)
        top_layout.addWidget(self.stats_panel, 1)
        layout.addWidget(top_row)

        self.data_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.data_splitter.setChildrenCollapsible(False)
        self.table_panel = MonitorTablePanel()
        self.data_splitter.addWidget(self.table_panel)

        sequence_panel = Panel(title="監控序列散佈圖")
        self.sequence_chart = ImageScatterChart(
            x_label="tile x",
            y_label="tile y",
            empty_text="尚無累積切圖點位",
            defect_radius_scale=0,
        )
        sequence_panel.add_widget(self.sequence_chart, 1)
        self.data_splitter.addWidget(sequence_panel)

        scatter_panel = Panel(title="所選影像切圖散佈圖")
        self.scatter_chart = ImageScatterChart()
        scatter_panel.add_widget(self.scatter_chart, 1)
        self.data_splitter.addWidget(scatter_panel)
        self.data_splitter.setSizes([520, 320, 320])
        layout.addWidget(self.data_splitter, 1)

        self.empty_state = QFrame()
        empty_layout = QVBoxLayout(self.empty_state)
        empty_layout.setContentsMargins(0, 0, 0, 0)
        empty_layout.addWidget(EmptyState("eye", "尚無已處理影像", "啟動監控後，放入監控資料夾的影像會顯示在這裡。"))
        layout.addWidget(self.empty_state)

        self.control_panel.choose_folder_requested.connect(self.choose_folder_requested.emit)
        self.control_panel.choose_move_folder_requested.connect(self.choose_move_folder_requested.emit)
        self.control_panel.source_changed.connect(self.source_changed.emit)
        self.control_panel.start_requested.connect(self.start_requested.emit)
        self.control_panel.stop_requested.connect(self.stop_requested.emit)
        self.table_panel.table.selectionModel().selectionChanged.connect(self._on_table_selection_changed)
        self.table_panel.open_original_requested.connect(self.open_original_requested.emit)
        self._refresh()

    def set_folder(self, folder: str | None) -> None:
        self.control_panel.set_folder(folder)

    def set_move_folder(self, folder: str | None) -> None:
        self.control_panel.set_move_folder(folder)

    def set_source(self, source: str) -> None:
        self.control_panel.set_source(source)

    def source(self) -> str:
        return self.control_panel.source()

    def set_source_selectable(self, selectable: bool) -> None:
        self.control_panel.set_source_selectable(selectable)

    def set_camera_status_text(self, text: str) -> None:
        self.control_panel.set_camera_status_text(text)

    def set_ready(self, ready: bool, running: bool) -> None:
        self.control_panel.set_ready(ready, running)

    def set_progress(self, pct: int, message: str) -> None:
        self.control_panel.set_progress(pct, message)

    def clear_items(self) -> None:
        self._items = []
        self._refresh()

    def add_item(self, item: dict) -> None:
        self._items.insert(0, item)
        self._items = self._items[:MONITOR_HISTORY_LIMIT]
        self.stats_panel.set_counts(self._items)
        self.table_panel.prepend_item(item, MONITOR_HISTORY_LIMIT)
        self.sequence_chart.set_model(BatchDashboardBuilder.build_monitor_sequence_scatter(self._sequence_items()))
        self.data_splitter.setVisible(True)
        self.empty_state.setVisible(False)
        self.table_panel.table.selectRow(0)
        self._render_selected_scatter()

    def items(self) -> list[dict]:
        return list(self._items)

    def _refresh(self) -> None:
        has_items = bool(self._items)
        self.stats_panel.set_counts(self._items)
        self.table_panel.set_items(self._items)
        self.sequence_chart.set_model(BatchDashboardBuilder.build_monitor_sequence_scatter(self._sequence_items()))
        self.data_splitter.setVisible(has_items)
        self.empty_state.setVisible(not has_items)
        if has_items:
            self.table_panel.table.selectRow(0)
            self._render_selected_scatter()
        else:
            self.scatter_chart.set_model(BatchDashboardBuilder.build_image_scatter(None))

    def _on_table_selection_changed(self, *_args) -> None:
        self._render_selected_scatter()

    def _render_selected_scatter(self) -> None:
        item = self.table_panel.selected_item()
        self.scatter_chart.set_model(BatchDashboardBuilder.build_image_scatter(item))

    def _sequence_items(self) -> list[dict]:
        return self._items[:MONITOR_SEQUENCE_SCATTER_LIMIT]
