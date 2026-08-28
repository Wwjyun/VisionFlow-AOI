from __future__ import annotations

import argparse
import csv
import filecmp
import hashlib
import math
import re
import shutil
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


DEFAULT_RANGES = "200-400\n401-500"
TOOL_VERSION = "1.0.0"
UNMATCHED_FOLDER = "_未落入區間"
RANGE_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(?:-|~|～|—)\s*(\d+(?:\.\d+)?)\s*$"
)
AGGREGATIONS = {
    "最大面積": "max",
    "面積總和": "sum",
    "最小面積": "min",
}


@dataclass(frozen=True)
class AreaRange:
    lower: float
    upper: float
    folder_name: str


@dataclass(frozen=True)
class TileArea:
    image_path: Path
    sidecar_path: Path | None
    csv_path: Path
    tile_id: str
    area: float
    area_unit: str


@dataclass(frozen=True)
class ScanResult:
    records: tuple[TileArea, ...]
    csv_count: int
    invalid_row_count: int
    missing_image_count: int


@dataclass(frozen=True)
class ClassificationSummary:
    csv_count: int
    tile_count: int
    copied_count: int
    unmatched_count: int
    skipped_unmatched_count: int
    invalid_row_count: int
    missing_image_count: int
    units: tuple[str, ...]
    output_dir: Path


def parse_ranges(text: str) -> list[AreaRange]:
    entries = [part.strip() for part in re.split(r"[,;\r\n]+", text) if part.strip()]
    if not entries:
        raise ValueError("請至少輸入一個面積區間，例如 200-400。")

    ranges: list[AreaRange] = []
    for entry in entries:
        match = RANGE_RE.fullmatch(entry)
        if not match:
            raise ValueError(f"無法辨識面積區間「{entry}」，請使用 200-400 格式。")
        lower = float(match.group(1))
        upper = float(match.group(2))
        if lower > upper:
            raise ValueError(f"面積區間「{entry}」的起始值不可大於結束值。")
        folder_name = f"{_format_number(lower)}-{_format_number(upper)}"
        ranges.append(AreaRange(lower, upper, folder_name))

    ranges.sort(key=lambda item: (item.lower, item.upper))
    for previous, current in zip(ranges, ranges[1:]):
        if current.lower <= previous.upper:
            raise ValueError(
                f"面積區間「{previous.folder_name}」與「{current.folder_name}」重疊；"
                "區間包含上下限，請調整後再執行。"
            )
    return ranges


def scan_ng_tiles(root_dir: Path, aggregation: str = "max") -> ScanResult:
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"根資料夾不存在：{root_dir}")
    if aggregation not in {"max", "sum", "min"}:
        raise ValueError(f"不支援的同圖面積計算方式：{aggregation}")

    csv_paths = _find_defect_csvs(root_dir)
    if not csv_paths:
        raise ValueError(
            "所選資料夾下找不到包含 tile_id 與 area 欄位的缺陷 CSV。"
        )

    image_index = _build_ng_tile_index(root_dir)
    records: list[TileArea] = []
    invalid_rows = 0
    missing_images = 0

    for csv_path in csv_paths:
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                tile_id = str(row.get("tile_id", "") or "").strip()
                area_text = str(row.get("area", "") or "").strip()
                area_unit = _normalize_area_unit(row.get("area_unit"))
                try:
                    area = float(area_text)
                except ValueError:
                    invalid_rows += 1
                    continue
                if not tile_id or not math.isfinite(area) or area < 0:
                    invalid_rows += 1
                    continue
                grouped[(tile_id, area_unit)].append(area)

        for (tile_id, area_unit), values in grouped.items():
            image_path = _find_ng_tile(csv_path, tile_id, image_index)
            if image_path is None:
                missing_images += 1
                continue
            records.append(
                TileArea(
                    image_path=image_path,
                    sidecar_path=(
                        image_path.with_suffix(".json")
                        if image_path.with_suffix(".json").is_file()
                        else None
                    ),
                    csv_path=csv_path,
                    tile_id=tile_id,
                    area=_aggregate(values, aggregation),
                    area_unit=area_unit,
                )
            )

    return ScanResult(
        records=tuple(records),
        csv_count=len(csv_paths),
        invalid_row_count=invalid_rows,
        missing_image_count=missing_images,
    )


def classify_ng_tiles(
    root_dir: Path,
    output_dir: Path,
    ranges: list[AreaRange],
    *,
    aggregation: str = "max",
    include_unmatched: bool = True,
    copy_sidecars: bool = True,
) -> ClassificationSummary:
    root_dir = Path(root_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not ranges:
        raise ValueError("請至少提供一個面積區間。")

    scan = scan_ng_tiles(root_dir, aggregation=aggregation)
    if not scan.records:
        details = []
        if scan.invalid_row_count:
            details.append(f"{scan.invalid_row_count} 筆面積資料無效")
        if scan.missing_image_count:
            details.append(f"{scan.missing_image_count} 張 NG Tile 找不到")
        suffix = f"（{'、'.join(details)}）" if details else ""
        raise ValueError(f"CSV 中沒有可分類且能對應到圖片的資料{suffix}。")

    units = tuple(sorted({record.area_unit for record in scan.records}))
    split_by_unit = len(units) > 1
    copied = 0
    unmatched = 0
    skipped_unmatched = 0

    for record in scan.records:
        area_range = next(
            (item for item in ranges if item.lower <= record.area <= item.upper),
            None,
        )
        if area_range is None:
            unmatched += 1
            if not include_unmatched:
                skipped_unmatched += 1
                continue
            category = UNMATCHED_FOLDER
        else:
            category = area_range.folder_name

        destination_dir = output_dir
        if split_by_unit:
            destination_dir /= _unit_folder(record.area_unit)
        destination_dir /= category
        destination_dir.mkdir(parents=True, exist_ok=True)

        destination = _destination_for(
            record.image_path,
            destination_dir,
            root_dir,
        )
        shutil.copy2(record.image_path, destination)
        if copy_sidecars and record.sidecar_path is not None:
            shutil.copy2(record.sidecar_path, destination.with_suffix(".json"))
        copied += 1

    return ClassificationSummary(
        csv_count=scan.csv_count,
        tile_count=len(scan.records),
        copied_count=copied,
        unmatched_count=unmatched,
        skipped_unmatched_count=skipped_unmatched,
        invalid_row_count=scan.invalid_row_count,
        missing_image_count=scan.missing_image_count,
        units=units,
        output_dir=output_dir,
    )


def _find_defect_csvs(root_dir: Path) -> list[Path]:
    paths = []
    for path in root_dir.rglob("*.csv"):
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                fields = set(next(csv.reader(handle), []))
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
        if {"tile_id", "area"}.issubset(fields):
            paths.append(path)
    return sorted(paths, key=lambda path: str(path).lower())


def _build_ng_tile_index(root_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for directory in root_dir.rglob("*"):
        if not directory.is_dir() or directory.name.lower() != "ng_tiles":
            continue
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                index[path.name.lower()].append(path)
    return index


def _find_ng_tile(
    csv_path: Path,
    tile_id: str,
    image_index: dict[str, list[Path]],
) -> Path | None:
    base_name = f"{csv_path.stem}_{tile_id}"
    expected_dir = csv_path.parent.parent / "ng_tiles"
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = expected_dir / f"{base_name}{suffix}"
        if candidate.is_file():
            return candidate

    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        matches = image_index.get(f"{base_name}{suffix}".lower(), [])
        if matches:
            return sorted(matches, key=lambda path: str(path).lower())[0]
    return None


def _aggregate(values: list[float], aggregation: str) -> float:
    if aggregation == "max":
        return max(values)
    if aggregation == "min":
        return min(values)
    return sum(values)


def _normalize_area_unit(value: object) -> str:
    text = str(value or "").strip().lower().replace("²", "^2")
    if not text:
        return "px^2"
    if text in {"um2", "um^2", "µm2", "µm^2", "μm2", "μm^2"}:
        return "um^2"
    if text in {"px2", "px^2"}:
        return "px^2"
    return text


def _unit_folder(unit: str) -> str:
    if unit == "px^2":
        return "px2"
    if unit == "um^2":
        return "um2"
    safe = re.sub(r'[<>:"/\\|?*]+', "_", unit).strip(" .")
    return safe or "unknown_unit"


def _destination_for(source: Path, destination_dir: Path, root_dir: Path) -> Path:
    destination = destination_dir / source.name
    if not destination.exists() or filecmp.cmp(source, destination, shallow=False):
        return destination

    try:
        source_key = str(source.relative_to(root_dir))
    except ValueError:
        source_key = str(source)
    digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:8]
    return destination.with_name(f"{destination.stem}__{digest}{destination.suffix}")


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def format_summary(summary: ClassificationSummary) -> str:
    unit_note = "、".join(summary.units)
    lines = [
        f"完成：掃描 {summary.csv_count} 份 CSV，找到 {summary.tile_count} 張 NG Tile。",
        f"已複製 {summary.copied_count} 張；未落入區間 {summary.unmatched_count} 張。",
        f"CSV 面積單位：{unit_note}",
        f"輸出資料夾：{summary.output_dir}",
    ]
    if len(summary.units) > 1:
        lines.insert(3, "偵測到多種面積單位，已先分成 px2／um2 等單位資料夾。")
    if summary.skipped_unmatched_count:
        lines.insert(2, f"其中 {summary.skipped_unmatched_count} 張依設定未複製。")
    if summary.invalid_row_count or summary.missing_image_count:
        lines.append(
            f"略過：無效 CSV 列 {summary.invalid_row_count} 筆，"
            f"找不到圖片 {summary.missing_image_count} 張。"
        )
    return "\n".join(lines)


class ClassificationWorker(QObject):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        root_dir: Path,
        output_dir: Path,
        ranges: list[AreaRange],
        aggregation: str,
        include_unmatched: bool,
        copy_sidecars: bool,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.ranges = ranges
        self.aggregation = aggregation
        self.include_unmatched = include_unmatched
        self.copy_sidecars = copy_sidecars

    @Slot()
    def run(self) -> None:
        try:
            summary = classify_ng_tiles(
                self.root_dir,
                self.output_dir,
                self.ranges,
                aggregation=self.aggregation,
                include_unmatched=self.include_unmatched,
                copy_sidecars=self.copy_sidecars,
            )
        except Exception as exc:  # pragma: no cover - GUI error path
            traceback.print_exc()
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(summary)


class NgTileAreaClassifierWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"NG Tile 面積分類工具 v{TOOL_VERSION}")
        self.resize(760, 560)
        self.setMinimumSize(680, 520)
        self._thread: QThread | None = None
        self._worker: ClassificationWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setHorizontalSpacing(12)
        self.root_folder = QLineEdit()
        root_row = QHBoxLayout()
        root_row.addWidget(self.root_folder, 1)
        root_button = QPushButton("選擇…")
        root_button.clicked.connect(self._select_root)
        root_row.addWidget(root_button)
        form.addRow("根資料夾", root_row)

        self.output_folder = QLineEdit()
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_folder, 1)
        output_button = QPushButton("選擇…")
        output_button.clicked.connect(self._select_output)
        output_row.addWidget(output_button)
        form.addRow("分類輸出", output_row)
        layout.addLayout(form)

        layout.addWidget(QLabel("面積區間（每行一個）"))
        self.range_text = QTextEdit()
        self.range_text.setPlainText(DEFAULT_RANGES)
        self.range_text.setPlaceholderText("例如：\n200-400\n401-500")
        layout.addWidget(self.range_text, 1)

        options = QHBoxLayout()
        options.addWidget(QLabel("同一張 Tile 有多筆缺陷時："))
        self.aggregation = QComboBox()
        self.aggregation.addItems(tuple(AGGREGATIONS))
        options.addWidget(self.aggregation)
        options.addSpacing(12)
        self.include_unmatched = QCheckBox("未落入區間也複製")
        self.include_unmatched.setChecked(True)
        options.addWidget(self.include_unmatched)
        options.addSpacing(12)
        self.copy_sidecars = QCheckBox("連同 JSON sidecar 複製")
        self.copy_sidecars.setChecked(True)
        options.addWidget(self.copy_sidecars)
        options.addStretch(1)
        layout.addLayout(options)

        note = QLabel(
            "程式會往下搜尋缺陷 CSV，依 tile_id 對應 ng_tiles 圖片。"
            "原始檔不會被移動；CSV 沒有 area_unit 時視為 px²。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #555555;")
        layout.addWidget(note)

        actions = QHBoxLayout()
        self.start_button = QPushButton("開始分類")
        self.start_button.clicked.connect(self._start)
        actions.addWidget(self.start_button)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(220)
        self.progress.hide()
        actions.addWidget(self.progress)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.status = QLabel("請選擇包含 csv 與 ng_tiles 的根資料夾。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    @Slot()
    def _select_root(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "選擇 AOI 輸出根資料夾")
        if not folder:
            return
        self.root_folder.setText(folder)
        self.output_folder.setText(str(Path(folder) / "area_classified"))
        self.status.setText("已選擇根資料夾，可以開始分類。")

    @Slot()
    def _select_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "選擇分類輸出資料夾")
        if folder:
            self.output_folder.setText(folder)

    @Slot()
    def _start(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        try:
            root_text = self.root_folder.text().strip()
            if not root_text:
                raise ValueError("請先選擇根資料夾。")
            root_dir = Path(root_text)
            output_text = self.output_folder.text().strip()
            output_dir = Path(output_text) if output_text else root_dir / "area_classified"
            ranges = parse_ranges(self.range_text.toPlainText())
            aggregation = AGGREGATIONS[self.aggregation.currentText()]
        except Exception as exc:
            QMessageBox.critical(self, "設定錯誤", str(exc))
            return

        self.output_folder.setText(str(output_dir))
        self.start_button.setEnabled(False)
        self.progress.show()
        self.status.setText("正在掃描 CSV 並複製分類，請稍候…")

        self._thread = QThread(self)
        self._worker = ClassificationWorker(
            root_dir,
            output_dir,
            ranges,
            aggregation,
            self.include_unmatched.isChecked(),
            self.copy_sidecars.isChecked(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.succeeded.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.succeeded.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._worker.succeeded.connect(self._finish_success)
        self._worker.failed.connect(self._finish_error)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._clear_worker)
        self._thread.start()

    @Slot(object)
    def _finish_success(self, summary: ClassificationSummary) -> None:
        self._set_idle()
        text = format_summary(summary)
        self.status.setText(text.replace("\n", "  "))
        QMessageBox.information(self, "分類完成", text)

    @Slot(str)
    def _finish_error(self, message: str) -> None:
        self._set_idle()
        self.status.setText(f"分類失敗：{message}")
        QMessageBox.critical(self, "分類失敗", message)

    @Slot()
    def _clear_worker(self) -> None:
        self._thread = None
        self._worker = None

    def _set_idle(self) -> None:
        self.progress.hide()
        self.start_button.setEnabled(True)

    def closeEvent(self, event) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.warning(self, "分類進行中", "請等待目前的圖片分類完成後再關閉視窗。")
            event.ignore()
            return
        event.accept()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="依缺陷 CSV 面積分類 AOI NG Tile 圖片。")
    parser.add_argument("--input", "-i", type=Path, help="包含 csv 與 ng_tiles 的根資料夾。")
    parser.add_argument("--output", "-o", type=Path, help="分類輸出資料夾。")
    parser.add_argument("--smoke-test", action="store_true", help="建立 GUI 後立即結束，供打包驗證。")
    parser.add_argument("--version", action="version", version=f"%(prog)s {TOOL_VERSION}")
    parser.add_argument(
        "--ranges",
        default=DEFAULT_RANGES.replace("\n", ","),
        help="面積區間，以逗號分隔，例如 200-400,401-500。",
    )
    parser.add_argument("--aggregation", choices=("max", "sum", "min"), default="max")
    parser.add_argument("--skip-unmatched", action="store_true")
    parser.add_argument("--no-sidecars", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.smoke_test:
        app = QApplication.instance() or QApplication(sys.argv)
        window = NgTileAreaClassifierWindow()
        window.show()
        QTimer.singleShot(0, app.quit)
        return app.exec()
    if args.input:
        output_dir = args.output or args.input / "area_classified"
        summary = classify_ng_tiles(
            args.input,
            output_dir,
            parse_ranges(args.ranges),
            aggregation=args.aggregation,
            include_unmatched=not args.skip_unmatched,
            copy_sidecars=not args.no_sidecars,
        )
        print(format_summary(summary))
        return 0

    app = QApplication.instance() or QApplication(sys.argv)
    window = NgTileAreaClassifierWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
