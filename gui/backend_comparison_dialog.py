from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout

from gui.theme import COLORS

STAGE_LABELS = {
    "end_to_end": "端到端",
    "image_load": "讀圖",
    "initialization": "初始化／整圖上傳",
    "tiling": "切圖",
    "detectors_total": "Detector 合計",
    "aggregation": "彙總",
    "reporting_total": "報表輸出",
}
SCOPE_LABELS = {"total": "總計", "pipeline": "流程", "detector": "Detector"}


def comparison_headline(summary: dict) -> tuple[str, str]:
    """Short operator text and notice level for a finished CPU/GPU comparison."""
    total = next((row for row in summary.get("stages", []) if row.get("stage") == "end_to_end"), {})
    speedup = total.get("speedup")
    timing = (
        f"端到端 CPU {total.get('cpu_ms', 0):,.0f} ms／GPU 設定 {total.get('gpu_ms', 0):,.0f} ms"
        + (f"（{speedup:.2f}×）" if speedup else "")
    )
    if not summary.get("gpu_active"):
        reason = summary.get("gpu_fallback_reason") or "Recipe 未讓任何 Detector 實際使用 CUDA"
        return f"CPU／GPU 對照：GPU 設定本次未實際使用 CUDA（{reason}），兩次都是 CPU 結果；{timing}", "warning"
    if summary.get("decision_equal"):
        drift = summary.get("drift_counts") or {}
        drift_text = f"；非判定診斷值有尾數差異（最大 {summary.get('worst_drift', 0):.3g}）" if drift else ""
        return f"CPU／GPU 對照完成：判定欄位一致{drift_text}；{timing}", "success"
    return (
        f"CPU／GPU 對照完成：判定欄位不一致（CPU {summary.get('cpu_final')}／{summary.get('cpu_defects')} 個缺陷，"
        f"GPU {summary.get('gpu_final')}／{summary.get('gpu_defects')} 個缺陷）；{timing}",
        "warning",
    )


class BackendComparisonDialog(QDialog):
    """Non-modal, read-only report of one CPU/GPU comparison; it never edits the Recipe."""

    COLUMNS = ("範圍", "階段", "CPU (ms)", "GPU 設定 (ms)", "倍數")

    def __init__(self, summary: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CPU／GPU 對照結果")
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(640, 460)
        layout = QVBoxLayout(self)

        headline, level = comparison_headline(summary)
        self.headline_label = QLabel(headline)
        self.headline_label.setWordWrap(True)
        color = COLORS["pass"] if level == "success" else COLORS["ng"]
        self.headline_label.setStyleSheet(f"color: {color}; font-weight: 600;")
        layout.addWidget(self.headline_label)

        self.detail_label = QLabel(
            f"影像：{summary.get('image_name', '')}　Recipe：{summary.get('recipe_name', '')}\n"
            f"CPU：{summary.get('cpu_final')}／{summary.get('cpu_defects')} 個缺陷　"
            f"GPU 設定：{summary.get('gpu_final')}／{summary.get('gpu_defects')} 個缺陷"
        )
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)

        difference = summary.get("first_difference")
        self.difference_label = QLabel(
            f"第一個不同的判定欄位（第 {difference['index'] + 1} 項）：CPU {difference['cpu']}／GPU {difference['gpu']}"
            if difference else "判定欄位（PASS/NG、缺陷數、bbox、area、confidence、metadata、Tile 位置）逐項相同。"
        )
        self.difference_label.setWordWrap(True)
        self.difference_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.difference_label)

        rows = summary.get("stages", [])
        self.table = QTableWidget(len(rows), len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row_index, row in enumerate(rows):
            speedup = row.get("speedup")
            values = (
                SCOPE_LABELS.get(row.get("scope"), row.get("scope", "")),
                STAGE_LABELS.get(row.get("stage"), row.get("stage", "")),
                f"{row.get('cpu_ms', 0):,.1f}",
                f"{row.get('gpu_ms', 0):,.1f}",
                f"{speedup:.2f}×" if speedup else "-",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column >= 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row_index, column, item)
        self.table.resizeColumnsToContents()
        layout.addWidget(self.table, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("關閉")
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
