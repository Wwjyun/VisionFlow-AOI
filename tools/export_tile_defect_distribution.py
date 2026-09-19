"""Export an AOI ``csv/summary.csv`` as a self-contained tile distribution report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from tkinter import StringVar, Tk, filedialog, messagebox, ttk


DEFAULT_OUTPUT_NAME = "tile_defect_distribution.html"
TOOL_VERSION = "1.1.0"
GRID_TILE_RE = re.compile(r"^r(\d+)_c(\d+)$", re.IGNORECASE)
MAX_HEATMAP_CELLS = 3600
UNKNOWN_TILE_ID = "（未提供 tile_id）"
UNKNOWN_IMAGE_NAME = "（未提供 image_name）"
UNKNOWN_DETECTOR_ID = "（未提供 detector_id）"
UNKNOWN_DEFECT_TYPE = "（未提供 defect_type）"
UNKNOWN_RECIPE_NAME = "（未提供 recipe_name）"
UNKNOWN_MACHINE_ID = "（未提供 machine_id）"
UNKNOWN_PRODUCT_ID = "（未提供 product_id）"
UNKNOWN_FINAL_RESULT = "（未提供 final_result）"
UNKNOWN_AREA_UNIT = "（無面積單位）"
LEGACY_AREA_UNIT = "px^2"


# Chart map for the portable dashboard.  The browser recomputes every visual
# from the same filtered defect-row payload so cards, charts, and the table
# always reconcile.
CHART_MAP = {
    "tile-heatmap": "Tile coordinate matrix / defect row count",
    "tile-ng-heatmap": "Tile coordinate matrix / NG inspections divided by inspections",
    "tile-ng-ranking": "Tile ranking / NG rate with inspection denominator",
    "tile-pareto": "Tile ranking / defect row count and cumulative share",
    "grid-profile": "Grid position profile / row and column defect counts",
    "detector-type": "Detector composition / defect type stacked count",
    "tile-type-heatmap": "Tile and defect type matrix / defect row count",
    "defect-treemap": "Detector to defect type hierarchy / defect row count",
    "defect-share": "Defect type composition / row share",
    "image-ranking": "Affected image ranking / defect row count",
    "image-defect-histogram": "Image distribution / defect rows per affected image",
    "area-histogram": "Area distribution / one area unit at a time",
    "area-boxplot": "Area distribution by defect type / one unit at a time",
    "score-histogram": "Detector score distribution / valid numeric rows",
    "area-score-scatter": "Defect relationship / area versus detector score",
    "tile-scatter": "Tile relationship / affected images versus defect rows",
}


@dataclass(frozen=True)
class DefectRecord:
    """One normalized defect row embedded into the interactive dashboard."""

    image_name: str
    tile_id: str
    detector_id: str
    defect_type: str
    recipe_name: str
    machine_id: str
    product_id: str
    final_result: str
    area: float | None
    area_unit: str
    score: float | None


@dataclass(frozen=True)
class TileInspectionRecord:
    """One PASS/NG Tile inspection recovered from a sibling AOI JSON report."""

    image_name: str
    tile_id: str
    tile_result: str
    recipe_name: str
    machine_id: str
    product_id: str
    final_result: str


@dataclass
class TileDistribution:
    """Aggregated defects for one AOI tile identifier."""

    tile_id: str
    defect_count: int = 0
    image_names: set[str] = field(default_factory=set)
    detector_counts: Counter[str] = field(default_factory=Counter)
    defect_type_counts: Counter[str] = field(default_factory=Counter)
    area_values_by_unit: dict[str, list[float]] = field(default_factory=dict)
    score_values: list[float] = field(default_factory=list)

    @property
    def affected_image_count(self) -> int:
        return len(self.image_names)


@dataclass
class SummaryDistribution:
    """The statistics extracted from one AOI defect summary CSV."""

    source_path: Path
    fieldnames: tuple[str, ...] = ()
    total_defects: int = 0
    records: list[DefectRecord] = field(default_factory=list)
    inspection_records: list[TileInspectionRecord] = field(default_factory=list)
    image_names: set[str] = field(default_factory=set)
    tiles: dict[str, TileDistribution] = field(default_factory=dict)
    detector_counts: Counter[str] = field(default_factory=Counter)
    defect_type_counts: Counter[str] = field(default_factory=Counter)
    recipe_counts: Counter[str] = field(default_factory=Counter)
    machine_counts: Counter[str] = field(default_factory=Counter)
    product_counts: Counter[str] = field(default_factory=Counter)
    final_result_counts: Counter[str] = field(default_factory=Counter)
    area_unit_counts: Counter[str] = field(default_factory=Counter)
    rows_without_tile_id: int = 0
    rows_without_image_name: int = 0
    rows_with_area: int = 0
    rows_with_score: int = 0
    invalid_area_rows: int = 0
    invalid_score_rows: int = 0
    json_directory: Path | None = None
    json_reports_scanned: int = 0
    json_parse_errors: int = 0
    json_unknown_tile_results: int = 0

    @property
    def tile_count(self) -> int:
        return len(self.tiles)

    @property
    def image_count(self) -> int:
        return len(self.image_names)

    @property
    def top_tile(self) -> TileDistribution | None:
        return next(iter(sorted(self.tiles.values(), key=_tile_sort_key)), None)

    @property
    def sorted_tiles(self) -> list[TileDistribution]:
        return sorted(self.tiles.values(), key=_tile_sort_key)


def load_summary_distribution(summary_path: Path) -> SummaryDistribution:
    """Read one AOI ``summary.csv`` and aggregate its defect rows by ``tile_id``.

    AOI CSV output contains one row per defect, so a tile's count is the number of
    defect rows for that tile.  It does not infer PASS tile counts that are absent
    from the CSV.
    """

    summary_path = Path(summary_path)
    if not summary_path.is_file():
        raise FileNotFoundError(f"找不到 summary.csv：{summary_path}")

    distribution = SummaryDistribution(source_path=summary_path)
    try:
        with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            normalized_fieldnames = tuple(
                name.strip() for name in (reader.fieldnames or []) if name and name.strip()
            )
            fieldnames = set(normalized_fieldnames)
            if "tile_id" not in fieldnames:
                raise ValueError("檔案不是 AOI summary.csv：缺少 tile_id 欄位。")
            distribution.fieldnames = normalized_fieldnames

            for row in reader:
                if not any(str(value or "").strip() for value in row.values()):
                    continue
                _add_defect_row(distribution, row)
    except UnicodeDecodeError as exc:
        raise ValueError("summary.csv 必須是 UTF-8 或 UTF-8 BOM 編碼。") from exc
    except csv.Error as exc:
        raise ValueError(f"無法讀取 CSV：{exc}") from exc

    _load_sibling_tile_inspections(distribution)
    return distribution


def _add_defect_row(distribution: SummaryDistribution, row: dict[str | None, str | None]) -> None:
    raw_tile_id = str(row.get("tile_id") or "").strip()
    tile_id = raw_tile_id or UNKNOWN_TILE_ID
    raw_image_name = str(row.get("image_name") or "").strip()
    image_name = raw_image_name or UNKNOWN_IMAGE_NAME
    detector_id = str(row.get("detector_id") or "").strip() or UNKNOWN_DETECTOR_ID
    defect_type = str(row.get("defect_type") or "").strip() or UNKNOWN_DEFECT_TYPE
    recipe_name = str(row.get("recipe_name") or "").strip() or UNKNOWN_RECIPE_NAME
    machine_id = str(row.get("machine_id") or "").strip() or UNKNOWN_MACHINE_ID
    product_id = str(row.get("product_id") or "").strip() or UNKNOWN_PRODUCT_ID
    final_result = str(row.get("final_result") or "").strip() or UNKNOWN_FINAL_RESULT

    raw_area = str(row.get("area") or "").strip()
    area = _parse_optional_number(raw_area, minimum=0.0)
    raw_area_unit = str(row.get("area_unit") or "").strip()
    area_unit = raw_area_unit or (LEGACY_AREA_UNIT if area is not None else UNKNOWN_AREA_UNIT)
    raw_score = str(row.get("score") or "").strip()
    score = _parse_optional_number(raw_score)

    record = DefectRecord(
        image_name=image_name,
        tile_id=tile_id,
        detector_id=detector_id,
        defect_type=defect_type,
        recipe_name=recipe_name,
        machine_id=machine_id,
        product_id=product_id,
        final_result=final_result,
        area=area,
        area_unit=area_unit,
        score=score,
    )
    distribution.records.append(record)

    tile = distribution.tiles.setdefault(tile_id, TileDistribution(tile_id=tile_id))
    tile.defect_count += 1
    tile.image_names.add(image_name)
    tile.detector_counts[detector_id] += 1
    tile.defect_type_counts[defect_type] += 1
    if area is not None:
        tile.area_values_by_unit.setdefault(area_unit, []).append(area)
    if score is not None:
        tile.score_values.append(score)

    distribution.total_defects += 1
    distribution.image_names.add(image_name)
    distribution.detector_counts[detector_id] += 1
    distribution.defect_type_counts[defect_type] += 1
    distribution.recipe_counts[recipe_name] += 1
    distribution.machine_counts[machine_id] += 1
    distribution.product_counts[product_id] += 1
    distribution.final_result_counts[final_result] += 1
    distribution.area_unit_counts[area_unit] += 1
    if not raw_tile_id:
        distribution.rows_without_tile_id += 1
    if not raw_image_name:
        distribution.rows_without_image_name += 1
    if area is not None:
        distribution.rows_with_area += 1
    elif raw_area:
        distribution.invalid_area_rows += 1
    if score is not None:
        distribution.rows_with_score += 1
    elif raw_score:
        distribution.invalid_score_rows += 1


def _parse_optional_number(value: object, *, minimum: float | None = None) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        return None
    return number


def _load_sibling_tile_inspections(distribution: SummaryDistribution) -> None:
    """Load complete Tile PASS/NG denominators from the run's sibling ``json`` directory."""

    summary_path = distribution.source_path
    run_root = summary_path.parent.parent if summary_path.parent.name.casefold() == "csv" else summary_path.parent
    json_directory = run_root / "json"
    distribution.json_directory = json_directory
    if not json_directory.is_dir():
        return

    for json_path in sorted(json_directory.rglob("*.json"), key=lambda path: _natural_sort_key(str(path))):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            distribution.json_parse_errors += 1
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("tiles"), list):
            continue
        distribution.json_reports_scanned += 1
        image_name = str(payload.get("image_name") or "").strip() or UNKNOWN_IMAGE_NAME
        recipe_name = str(payload.get("recipe_name") or "").strip() or UNKNOWN_RECIPE_NAME
        machine_id = str(payload.get("machine_id") or "").strip() or UNKNOWN_MACHINE_ID
        product_id = str(payload.get("product_id") or "").strip() or UNKNOWN_PRODUCT_ID
        final_result = str(payload.get("final_result") or "").strip() or UNKNOWN_FINAL_RESULT
        for tile_result in payload["tiles"]:
            if not isinstance(tile_result, dict):
                distribution.json_unknown_tile_results += 1
                continue
            tile = tile_result.get("tile") or {}
            if not isinstance(tile, dict):
                tile = {}
            tile_id = str(tile.get("tile_id") or "").strip() or UNKNOWN_TILE_ID
            result = _resolve_tile_result(tile_result)
            if result not in {"PASS", "NG"}:
                distribution.json_unknown_tile_results += 1
                continue
            distribution.inspection_records.append(
                TileInspectionRecord(
                    image_name=image_name,
                    tile_id=tile_id,
                    tile_result=result,
                    recipe_name=recipe_name,
                    machine_id=machine_id,
                    product_id=product_id,
                    final_result=final_result,
                )
            )


def _resolve_tile_result(tile_result: dict[str, object]) -> str:
    explicit = str(tile_result.get("result") or "").strip().upper()
    if explicit in {"PASS", "NG"}:
        return explicit
    detectors = tile_result.get("detectors") or []
    if not isinstance(detectors, list) or not detectors:
        return ""
    states = [detector.get("pass") for detector in detectors if isinstance(detector, dict)]
    if any(state is False for state in states):
        return "NG"
    if states and all(state is True for state in states):
        return "PASS"
    return ""


def default_output_path(summary_path: Path) -> Path:
    """Place the report beside an AOI run folder instead of inside ``csv``."""

    summary_path = Path(summary_path)
    if summary_path.parent.name.casefold() == "csv":
        return summary_path.parent.parent / DEFAULT_OUTPUT_NAME
    return summary_path.parent / DEFAULT_OUTPUT_NAME


def export_html_report(
    summary_path: Path,
    output_path: Path | None = None,
) -> tuple[Path, SummaryDistribution]:
    """Generate an atomic, standalone HTML report and return its statistics."""

    summary_path = Path(summary_path)
    output_path = Path(output_path) if output_path else default_output_path(summary_path)
    if output_path.resolve() == summary_path.resolve():
        raise ValueError("報表輸出位置不可覆寫 summary.csv。")
    if output_path.suffix.casefold() not in {".html", ".htm"}:
        raise ValueError("報表輸出檔必須使用 .html 或 .htm 副檔名。")

    distribution = load_summary_distribution(summary_path)
    rendered = render_html_report(distribution)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    temporary_path.write_text(rendered, encoding="utf-8", newline="\n")
    temporary_path.replace(output_path)
    return output_path, distribution


def render_html_report(distribution: SummaryDistribution) -> str:
    """Return a self-contained Plotly dashboard backed by normalized CSV rows."""

    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    top_tile = distribution.top_tile
    top_tile_text = f"{top_tile.tile_id} · {top_tile.defect_count}" if top_tile else "—"
    records_payload = _safe_json([_record_payload(record) for record in distribution.records])
    inspections_payload = _safe_json(
        [_inspection_payload(record) for record in distribution.inspection_records]
    )
    source_meta_payload = _safe_json(
        {
            "jsonReports": distribution.json_reports_scanned,
            "jsonParseErrors": distribution.json_parse_errors,
            "unknownTileResults": distribution.json_unknown_tile_results,
        }
    )
    plotly_javascript = _plotly_javascript().replace("</script", "<\\/script")
    source_fields = "、".join(distribution.fieldnames) or "tile_id"
    fallback_rows = _render_tile_rows(distribution)
    missing_notice = _data_quality_notice(distribution)

    template = r"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AOI Tile 缺陷互動分析報表</title>
  <style>
    :root { color-scheme:light; --ink:#172033; --muted:#667085; --line:#d9e1ec; --panel:#fff; --bg:#f4f6fa; --brand:#165dce; --brand-dark:#123c80; --accent:#e67e22; --soft:#eef4ff; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.55 "Microsoft JhengHei","Noto Sans TC",Arial,sans-serif; }
    header { color:#fff; background:linear-gradient(120deg,var(--brand-dark),var(--brand)); padding:34px max(24px,calc((100% - 1380px)/2)); }
    header h1 { margin:0 0 8px; font-size:30px; } header p { margin:0; color:#e5efff; overflow-wrap:anywhere; }
    main { max-width:1380px; margin:0 auto; padding:24px 20px 48px; }
    .panel,.card { background:var(--panel); border:1px solid var(--line); border-radius:12px; box-shadow:0 1px 3px #17203312; }
    .panel { padding:18px; margin-top:18px; } h2 { margin:0 0 6px; font-size:20px; } h3 { margin:0 0 4px; font-size:16px; }
    .subtitle,.help { color:var(--muted); margin:0 0 12px; } .mono { font-variant-numeric:tabular-nums; }
    .filter-panel { position:sticky; top:0; z-index:20; box-shadow:0 8px 20px #17203314; }
    .filter-head,.toolbar { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; }
    .filters { display:grid; grid-template-columns:repeat(5,minmax(150px,1fr)); gap:10px; margin-top:12px; }
    .filter label { display:block; color:var(--muted); font-size:12px; margin-bottom:4px; }
    select,input[type=search],button { font:inherit; } select,input[type=search] { width:100%; border:1px solid #b9c5d5; border-radius:7px; padding:8px 9px; background:#fff; color:var(--ink); }
    button { border:1px solid #9db1ce; border-radius:7px; padding:8px 12px; background:#fff; color:#23446f; cursor:pointer; } button:hover { background:var(--soft); }
    .active-summary { margin-top:10px; color:#34435a; font-size:13px; }
    .cards { display:grid; grid-template-columns:repeat(7,minmax(135px,1fr)); gap:12px; margin-top:18px; }
    .card { padding:15px; } .card .label { color:var(--muted); font-size:12px; } .card .value { margin-top:5px; font-size:23px; font-weight:750; overflow-wrap:anywhere; } .card .note { color:var(--muted); font-size:11px; margin-top:3px; }
    .notice { border-left:4px solid #d79a17; background:#fff8df; padding:10px 12px; margin-top:14px; border-radius:5px; }
    .insights { margin:8px 0 0; padding-left:21px; } .insights li { margin:5px 0; }
    .chart-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }
    .chart-panel.full { grid-column:1/-1; } .plot { width:100%; height:390px; } .plot.tall { height:480px; }
    .quality-grid { display:grid; grid-template-columns:repeat(4,minmax(150px,1fr)); gap:12px; margin-top:12px; }
    .quality-item { background:#f8fafc; border:1px solid #e6ebf2; border-radius:8px; padding:11px; } .quality-item dt { color:var(--muted); font-size:12px; } .quality-item dd { margin:3px 0 0; font-weight:700; }
    .table-wrap { overflow:auto; border:1px solid var(--line); border-radius:8px; margin-top:12px; max-height:620px; }
    table { width:100%; border-collapse:collapse; min-width:1120px; } th,td { text-align:left; padding:9px 10px; border-bottom:1px solid #edf0f5; vertical-align:top; }
    th { background:#f7f9fc; color:#44516a; font-size:12px; position:sticky; top:0; z-index:2; } th button { border:0; padding:0; background:none; color:inherit; font-weight:700; }
    td.number { text-align:right; font-variant-numeric:tabular-nums; } .tile-button { padding:2px 5px; border-color:transparent; font-weight:700; }
    .table-status { color:var(--muted); margin-top:9px; font-size:12px; }
    details { margin-top:12px; } summary { cursor:pointer; font-weight:700; } code { overflow-wrap:anywhere; }
    footer { color:var(--muted); font-size:12px; margin-top:20px; }
    noscript .fallback { display:block; } .fallback { display:none; }
    @media (max-width:1050px) { .filters { grid-template-columns:repeat(3,1fr); } .cards { grid-template-columns:repeat(3,1fr); } }
    @media (max-width:760px) { header { padding:25px 18px; } header h1 { font-size:24px; } main { padding:16px 12px 36px; } .filter-panel { position:static; } .filters,.cards,.quality-grid,.chart-grid { grid-template-columns:1fr; } .chart-panel.full { grid-column:auto; } .filter-head,.toolbar { flex-direction:column; } .plot,.plot.tall { height:360px; } }
    @media print { body { background:#fff; } header { color:var(--ink); background:#fff; padding:0 0 16px; border-bottom:2px solid var(--ink); } header p { color:#44516a; } main { max-width:none; padding:14px 0; } .filter-panel,.modebar,.toolbar button { display:none!important; } .panel,.card { box-shadow:none; break-inside:avoid; } .chart-grid { display:block; } .chart-panel { margin-top:16px; } .table-wrap { max-height:none; } }
  </style>
  <script>__PLOTLY_JS__</script>
</head>
<body>
  <header>
    <h1>AOI Tile 缺陷互動分析報表</h1>
    <p>來源：__SOURCE_PATH__<br>產生時間：__GENERATED_AT__　｜　資料粒度：summary.csv 每列為一筆缺陷。</p>
  </header>
  <main>
    <section class="panel filter-panel" aria-labelledby="filter-title">
      <div class="filter-head"><div><h2 id="filter-title">全域篩選</h2><p class="subtitle">所有 KPI、圖表、洞察與明細會同步更新；點擊熱圖、Pareto、圖片排行或散點也能快速篩選。</p></div><button id="reset-filters" type="button">重設全部</button></div>
      <div class="filters">
        <div class="filter" data-filter-wrap="tile"><label for="filter-tile">Tile</label><select id="filter-tile"></select></div>
        <div class="filter" data-filter-wrap="image"><label for="filter-image">圖片</label><select id="filter-image"></select></div>
        <div class="filter" data-filter-wrap="detector"><label for="filter-detector">Detector</label><select id="filter-detector"></select></div>
        <div class="filter" data-filter-wrap="type"><label for="filter-type">缺陷類型</label><select id="filter-type"></select></div>
        <div class="filter" data-filter-wrap="recipe"><label for="filter-recipe">Recipe</label><select id="filter-recipe"></select></div>
        <div class="filter" data-filter-wrap="machine"><label for="filter-machine">機台</label><select id="filter-machine"></select></div>
        <div class="filter" data-filter-wrap="product"><label for="filter-product">產品</label><select id="filter-product"></select></div>
        <div class="filter" data-filter-wrap="result"><label for="filter-result">最終結果</label><select id="filter-result"></select></div>
        <div class="filter" data-filter-wrap="areaUnit"><label for="filter-area-unit">面積單位</label><select id="filter-area-unit"></select></div>
      </div>
      <div id="active-filter-summary" class="active-summary" aria-live="polite"></div>
    </section>

    <section class="cards" aria-label="篩選後統計總覽">
      __METRIC_CARDS__
    </section>
    __DATA_NOTICE__

    <section class="panel">
      <h2>自動摘要</h2>
      <p class="subtitle">從目前篩選結果計算，協助快速辨識集中位置、主因與資料涵蓋率。</p>
      <ul id="insights" class="insights"></ul>
    </section>

    <section class="panel">
      <h2>缺陷數最多的圖片</h2>
      <p class="subtitle">依目前篩選後的缺陷筆數排序，顯示前 15 名；點圖片名稱可進一步篩選整份報表。</p>
      <div class="table-wrap"><table><thead><tr><th>排名</th><th>圖片名稱</th><th>缺陷筆數</th><th>受影響 Tile</th><th>主要 Detector</th><th>主要缺陷類型</th><th>面積合計</th></tr></thead><tbody id="top-image-rows"></tbody></table></div>
    </section>

    <section class="chart-grid" aria-label="互動圖表">
      <section class="panel chart-panel full"><h3>Tile 座標熱圖</h3><p class="subtitle">色彩為缺陷列數；空白座標表示 summary.csv 沒有該 Tile 的缺陷列，不等同已知 PASS。</p><div id="tile-heatmap" class="plot tall"></div></section>
      <section class="panel chart-panel full"><h3>Tile NG 率熱圖</h3><p class="subtitle">由同一輸出目錄 json/ 的完整 Tile PASS／NG 結果計算；NG 率＝NG 次數 ÷ 檢測次數。</p><div id="tile-ng-heatmap" class="plot tall"></div></section>
      <section class="panel chart-panel full"><h3>Tile NG 率排行</h3><p class="subtitle">同時顯示分母，避免少量檢測造成的高百分比被誤讀；依 NG 率、檢測次數排序，顯示前 25 名。</p><div id="tile-ng-ranking" class="plot tall"></div></section>
      <section class="panel chart-panel full"><h3>Tile Pareto 與累積占比</h3><p class="subtitle">依缺陷列數排序，最多顯示前 30 個 Tile，其餘合併為「其他」；右軸為累積占比。</p><div id="tile-pareto" class="plot tall"></div></section>
      <section class="panel chart-panel full"><h3>Row／Column 位置剖面</h3><p class="subtitle">加總網格列與欄的缺陷筆數，用來辨識橫向或縱向帶狀集中。</p><div id="grid-profile" class="plot"></div></section>
      <section class="panel chart-panel"><h3>Detector × 缺陷類型</h3><p class="subtitle">堆疊缺陷列數；缺陷類型最多保留前 5 類，其餘合併。</p><div id="detector-type" class="plot"></div></section>
      <section class="panel chart-panel"><h3>Tile × 缺陷類型矩陣</h3><p class="subtitle">顯示前 30 個 Tile 與前 12 類缺陷的密度；點擊格子可同時篩選。</p><div id="tile-type-heatmap" class="plot"></div></section>
      <section class="panel chart-panel"><h3>Detector → 缺陷類型階層</h3><p class="subtitle">Treemap 方塊面積代表缺陷筆數，可點擊向下探索組成。</p><div id="defect-treemap" class="plot"></div></section>
      <section class="panel chart-panel"><h3>缺陷類型組成</h3><p class="subtitle">顯示缺陷列數占比；點擊區塊可篩選缺陷類型。</p><div id="defect-share" class="plot"></div></section>
      <section class="panel chart-panel"><h3>受影響圖片排行</h3><p class="subtitle">每張圖片出現的缺陷列數，顯示前 20 張。</p><div id="image-ranking" class="plot"></div></section>
      <section class="panel chart-panel"><h3>每張圖片缺陷數分布</h3><p class="subtitle">以圖片為觀察單位，檢視缺陷數量的分散與長尾。</p><div id="image-defect-histogram" class="plot"></div></section>
      <section class="panel chart-panel"><h3>Tile 重複影響關係</h3><p class="subtitle">每個 Tile 的受影響圖片數對缺陷列數；少於 8 個 Tile 時不畫散點，避免過度解讀。</p><div id="tile-scatter" class="plot"></div></section>
      <section class="panel chart-panel"><h3>缺陷面積分布</h3><p class="subtitle">混合面積單位時請先選擇一個單位，避免把 px² 與 µm² 放在同一尺度比較。</p><div id="area-histogram" class="plot"></div></section>
      <section class="panel chart-panel"><h3>各缺陷類型面積箱型圖</h3><p class="subtitle">單位一致且有效列足夠時，比較中位數、四分位距與離群值。</p><div id="area-boxplot" class="plot"></div></section>
      <section class="panel chart-panel"><h3>Detector score 分布</h3><p class="subtitle">只納入可解析的數值 score；此值是 Detector 輸出，不一定等同機率。</p><div id="score-histogram" class="plot"></div></section>
      <section class="panel chart-panel"><h3>缺陷面積 × Detector score</h3><p class="subtitle">至少 8 筆、面積單位一致時才繪製；超過 10,000 點時採等距取樣，用來檢視關係或離群點。</p><div id="area-score-scatter" class="plot"></div></section>
    </section>

    <section class="panel">
      <div class="toolbar"><div><h2>各 Tile 明細</h2><p class="subtitle">預設依缺陷列數降冪；點欄名排序，點 Tile 可套用全域篩選。</p></div><input id="table-search" type="search" placeholder="搜尋 Tile、Detector 或缺陷類型" aria-label="搜尋 Tile 明細"></div>
      <div class="table-wrap"><table><thead><tr>
        <th><button type="button" data-sort="tile">Tile ID</button></th>
        <th><button type="button" data-sort="count">缺陷筆數</button></th>
        <th><button type="button" data-sort="share">缺陷貢獻占比</button></th>
        <th><button type="button" data-sort="images">受影響圖片</button></th>
        <th><button type="button" data-sort="perImage">缺陷／圖片</button></th>
        <th>主要 Detector</th><th>主要缺陷類型</th><th>平均 score</th><th>面積合計</th>
      </tr></thead><tbody id="tile-rows"></tbody></table></div>
      <div id="table-status" class="table-status"></div>
    </section>

    <section class="panel">
      <h2>資料品質與統計口徑</h2>
      <dl class="quality-grid">
        <div class="quality-item"><dt>面積有效列</dt><dd id="quality-area">—</dd></div>
        <div class="quality-item"><dt>score 有效列</dt><dd id="quality-score">—</dd></div>
        <div class="quality-item"><dt>網格格式 Tile</dt><dd id="quality-grid">—</dd></div>
        <div class="quality-item"><dt>缺少 Tile ID</dt><dd id="quality-missing-tile">—</dd></div>
        <div class="quality-item"><dt>JSON 報告</dt><dd id="quality-json">—</dd></div>
        <div class="quality-item"><dt>Tile 檢測分母</dt><dd id="quality-inspections">—</dd></div>
      </dl>
      <ul class="help">
        <li>缺陷筆數＝summary.csv 資料列數；有缺陷 Tile／圖片＝至少出現一筆缺陷的不同 ID 數。</li>
        <li>summary.csv 不含零缺陷 Tile，因此本報表不能計算 PASS Tile 數、缺陷率、良率或完整網格覆蓋率。</li>
        <li>舊 CSV 有 area 但缺少 area_unit 時，依 VisionFlow 相容規則視為 px²；不同單位不互相加總。</li>
      </ul>
      <details><summary>來源欄位與工具資訊</summary><p>CSV 欄位：__SOURCE_FIELDS__</p><p>Tile NG 率來源：__JSON_SOURCE__</p><p>工具版本：__TOOL_VERSION__；Plotly 已內嵌，報表不需網路連線。</p></details>
    </section>

    <noscript><section class="panel fallback"><h2>無 JavaScript 摘要</h2><p class="notice">瀏覽器停用了 JavaScript，因此互動圖表無法顯示；以下保留產生時的 Tile 摘要。</p><div class="table-wrap"><table><thead><tr><th>Tile ID</th><th>缺陷筆數</th><th>受影響圖片數</th><th>主要缺陷類型</th><th>Detector 分布</th></tr></thead><tbody>__FALLBACK_ROWS__</tbody></table></div></section></noscript>
    <footer>本報表為唯讀離線 HTML，不上傳資料、不執行 AOI Detector；Plotly 圖表可縮放、框選、查看 hover 資訊並下載 PNG，也可直接列印／另存 PDF。</footer>
  </main>
  <script id="dashboard-data" type="application/json">__RECORDS_JSON__</script>
  <script id="inspection-data" type="application/json">__INSPECTIONS_JSON__</script>
  <script id="source-meta" type="application/json">__SOURCE_META_JSON__</script>
  <script>
  (() => {
    'use strict';
    const sourceRows = JSON.parse(document.getElementById('dashboard-data').textContent);
    const inspectionRows = JSON.parse(document.getElementById('inspection-data').textContent);
    const sourceMeta = JSON.parse(document.getElementById('source-meta').textContent);
    const GRID_RE = /^r(\d+)_c(\d+)$/i;
    const MAX_HEATMAP_CELLS = 3600;
    const MAX_PARETO_TILES = 31;
    const MAX_SCATTER_POINTS = 10000;
    const TABLE_LIMIT = 200;
    const COLORS = { blue:'#165dce', blueDark:'#123c80', orange:'#e67e22', gold:'#d79a17', olive:'#738a2f', pink:'#c45b83', grey:'#8b97a8', light:'#e7edf5' };
    const PALETTE = [COLORS.blue,COLORS.orange,COLORS.gold,COLORS.olive,COLORS.pink,COLORS.grey];
    const CONFIG = { responsive:true, displaylogo:false, scrollZoom:true, toImageButtonOptions:{ format:'png', filename:'aoi-tile-analysis', scale:2 } };
    const filterSpecs = [
      ['tile','filter-tile','Tile'],['image','filter-image','圖片'],['detector','filter-detector','Detector'],['type','filter-type','缺陷類型'],
      ['recipe','filter-recipe','Recipe'],['machine','filter-machine','機台'],['product','filter-product','產品'],['result','filter-result','結果'],['areaUnit','filter-area-unit','面積單位']
    ];
    const filters = Object.fromEntries(filterSpecs.map(([key,id]) => [key,document.getElementById(id)]));
    let tableSort = { key:'count', direction:-1 };

    const fmt = value => new Intl.NumberFormat('zh-TW',{maximumFractionDigits:2}).format(value);
    const pct = value => `${(value * 100).toFixed(value >= 0.1 ? 1 : 2)}%`;
    const natural = (a,b) => String(a).localeCompare(String(b),'zh-Hant',{numeric:true,sensitivity:'base'});
    const unique = values => [...new Set(values)].sort(natural);
    const countBy = (rows,key) => { const map=new Map(); rows.forEach(row => map.set(row[key],(map.get(row[key])||0)+1)); return map; };
    const rankedCounts = (rows,key) => [...countBy(rows,key).entries()].sort((a,b) => b[1]-a[1] || natural(a[0],b[0]));
    const sum = values => values.reduce((total,value) => total+value,0);
    const average = values => values.length ? sum(values)/values.length : null;
    const dominant = map => [...map.entries()].sort((a,b) => b[1]-a[1] || natural(a[0],b[0]))[0] || ['—',0];
    const create = (tag,text,className) => { const element=document.createElement(tag); if(text!==undefined) element.textContent=text; if(className) element.className=className; return element; };

    function baseLayout(extra={}) {
      return Object.assign({
        autosize:true, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#fff', margin:{l:58,r:24,t:18,b:62},
        font:{family:'Microsoft JhengHei, Noto Sans TC, Arial, sans-serif',color:'#34435a',size:12},
        hoverlabel:{bgcolor:'#172033',font:{color:'#fff'}}, legend:{orientation:'h',y:1.08,x:0},
        xaxis:{automargin:true,gridcolor:'#edf0f5',zerolinecolor:'#aeb8c6'}, yaxis:{automargin:true,gridcolor:'#edf0f5',zerolinecolor:'#aeb8c6'}
      },extra);
    }

    function populateFilters() {
      filterSpecs.forEach(([key,id,label]) => {
        const select=document.getElementById(id);
        select.replaceChildren();
        const all=create('option',`全部${label}`); all.value='*'; select.append(all);
        const denominatorValues=['tile','image','recipe','machine','product','result'].includes(key)?inspectionRows.map(row=>row[key]):[];
        const values=unique([...sourceRows.map(row => row[key]),...denominatorValues]);
        values.forEach(value => { const option=create('option',value); option.value=value; select.append(option); });
        const wrapper=document.querySelector(`[data-filter-wrap="${key}"]`);
        if(wrapper) wrapper.hidden=values.length<=1 && key!=='tile' && key!=='image' && key!=='detector' && key!=='type';
      });
    }

    function filteredRows() {
      return sourceRows.filter(row => filterSpecs.every(([key]) => filters[key].value==='*' || row[key]===filters[key].value));
    }

    function filteredInspections() {
      const applicable=['tile','image','recipe','machine','product','result'];
      return inspectionRows.filter(row => applicable.every(key => filters[key].value==='*' || row[key]===filters[key].value));
    }

    function setFilter(key,value) {
      setFilters({[key]:value});
    }

    function setFilters(values) {
      let changed=false;
      Object.entries(values).forEach(([key,value])=>{
        if(value===null || value===undefined || !filters[key]) return;
        const exists=[...filters[key].options].some(option=>option.value===String(value));
        if(exists && filters[key].value!==String(value)){ filters[key].value=String(value); changed=true; }
      });
      if(changed){ updateDashboard(); document.querySelector('.filter-panel').scrollIntoView({behavior:'smooth',block:'start'}); }
    }

    function aggregateTiles(rows) {
      const map=new Map();
      rows.forEach(row => {
        if(!map.has(row.tile)) map.set(row.tile,{tile:row.tile,count:0,images:new Set(),detectors:new Map(),types:new Map(),scores:[],areas:new Map()});
        const tile=map.get(row.tile); tile.count+=1; tile.images.add(row.image);
        tile.detectors.set(row.detector,(tile.detectors.get(row.detector)||0)+1);
        tile.types.set(row.type,(tile.types.get(row.type)||0)+1);
        if(Number.isFinite(row.score)) tile.scores.push(row.score);
        if(Number.isFinite(row.area)) { if(!tile.areas.has(row.areaUnit)) tile.areas.set(row.areaUnit,[]); tile.areas.get(row.areaUnit).push(row.area); }
      });
      return [...map.values()].sort((a,b) => b.count-a.count || natural(a.tile,b.tile));
    }

    function aggregateInspectionTiles(rows) {
      const map=new Map();
      rows.forEach(row => {
        if(!map.has(row.tile)) map.set(row.tile,{tile:row.tile,total:0,ng:0,pass:0,images:new Set()});
        const tile=map.get(row.tile); tile.total+=1; tile.images.add(row.image);
        if(row.tileResult==='NG') tile.ng+=1; else if(row.tileResult==='PASS') tile.pass+=1;
      });
      return [...map.values()].map(tile=>({...tile,rate:tile.total?tile.ng/tile.total:0})).sort((a,b)=>b.rate-a.rate || b.total-a.total || natural(a.tile,b.tile));
    }

    function aggregateImages(rows) {
      const map=new Map();
      rows.forEach(row=>{
        if(!map.has(row.image)) map.set(row.image,{image:row.image,count:0,tiles:new Set(),detectors:new Map(),types:new Map(),areas:new Map()});
        const image=map.get(row.image); image.count+=1; image.tiles.add(row.tile);
        image.detectors.set(row.detector,(image.detectors.get(row.detector)||0)+1); image.types.set(row.type,(image.types.get(row.type)||0)+1);
        if(Number.isFinite(row.area)){ if(!image.areas.has(row.areaUnit)) image.areas.set(row.areaUnit,[]); image.areas.get(row.areaUnit).push(row.area); }
      });
      return [...map.values()].sort((a,b)=>b.count-a.count || natural(a.image,b.image));
    }

    function updateCards(rows,tiles,inspectionTiles) {
      const images=new Set(rows.map(row => row.image));
      const top=tiles[0];
      const areaCount=rows.filter(row => Number.isFinite(row.area)).length;
      const inspected=sum(inspectionTiles.map(tile=>tile.total)),ng=sum(inspectionTiles.map(tile=>tile.ng));
      const values={
        'metric-defects':fmt(rows.length),'metric-tiles':fmt(tiles.length),'metric-images':fmt(images.size),
        'metric-average':tiles.length ? fmt(rows.length/tiles.length) : '—','metric-top':top ? `${top.tile} · ${fmt(top.count)}` : '—',
        'metric-area':rows.length ? pct(areaCount/rows.length) : '—','metric-ng-rate':inspected?pct(ng/inspected):'—'
      };
      Object.entries(values).forEach(([id,value]) => document.getElementById(id).textContent=value);
    }

    function updateFilterSummary(rows,inspections) {
      const active=filterSpecs.filter(([key]) => filters[key].value!=='*').map(([key,,label]) => `${label}：${filters[key].value}`);
      document.getElementById('active-filter-summary').textContent=`目前顯示 ${fmt(rows.length)} / ${fmt(sourceRows.length)} 筆缺陷；Tile 檢測分母 ${fmt(inspections.length)} / ${fmt(inspectionRows.length)}`+(active.length ? `｜${active.join('｜')}` : '｜未套用篩選');
    }

    function updateInsights(rows,tiles,inspectionTiles) {
      const list=document.getElementById('insights'); list.replaceChildren();
      if(!rows.length) { list.append(create('li','目前篩選條件下沒有缺陷資料。')); return; }
      const topTile=tiles[0]; const topFive=sum(tiles.slice(0,5).map(tile => tile.count));
      const topType=rankedCounts(rows,'type')[0]; const topDetector=rankedCounts(rows,'detector')[0];
      const areaCount=rows.filter(row => Number.isFinite(row.area)).length; const scoreCount=rows.filter(row => Number.isFinite(row.score)).length;
      const statements=[
        `最高缺陷 Tile 為 ${topTile.tile}，共 ${fmt(topTile.count)} 筆，占目前缺陷的 ${pct(topTile.count/rows.length)}。`,
        `前 5 個 Tile 合計占 ${pct(topFive/rows.length)}，可用 Pareto 圖判斷缺陷是否集中。`,
        `主要缺陷類型為 ${topType[0]}（${fmt(topType[1])} 筆，${pct(topType[1]/rows.length)}）；主要 Detector 為 ${topDetector[0]}（${fmt(topDetector[1])} 筆）。`,
        `面積欄位涵蓋 ${pct(areaCount/rows.length)}，score 欄位涵蓋 ${pct(scoreCount/rows.length)}；缺值不納入各自分布圖。`
      ];
      if(inspectionTiles.length){ const inspected=sum(inspectionTiles.map(tile=>tile.total)),ng=sum(inspectionTiles.map(tile=>tile.ng)),topNg=inspectionTiles[0]; statements.splice(2,0,`完整 JSON 分母下，整體 Tile NG 率為 ${pct(ng/inspected)}（${fmt(ng)} / ${fmt(inspected)}）；最高為 ${topNg.tile} 的 ${pct(topNg.rate)}（${topNg.ng}/${topNg.total}）。`); }
      else statements.splice(2,0,'未找到完整 JSON Tile 檢測資料，因此只呈現缺陷貢獻占比，不推算 Tile NG 率。');
      statements.forEach(text => list.append(create('li',text)));
    }

    function renderTopImages(rows) {
      const images=aggregateImages(rows).slice(0,15),body=document.getElementById('top-image-rows'); body.replaceChildren();
      images.forEach((image,index)=>{
        const detector=dominant(image.detectors),type=dominant(image.types),row=document.createElement('tr');
        row.append(create('td',String(index+1),'number'));
        const nameCell=document.createElement('td'),button=create('button',image.image,'tile-button'); button.type='button'; button.dataset.image=image.image; nameCell.append(button); row.append(nameCell);
        [[fmt(image.count),'number'],[fmt(image.tiles.size),'number'],[`${detector[0]} × ${detector[1]}`,''],[`${type[0]} × ${type[1]}`,''],[formatAreas(image.areas),'number']].forEach(([text,className])=>row.append(create('td',text,className)));
        body.append(row);
      });
      if(!images.length){ const row=document.createElement('tr'),cell=create('td','目前條件下沒有缺陷圖片。'); cell.colSpan=7; row.append(cell); body.append(row); }
    }

    function chartMessage(id,message) {
      return Plotly.react(id,[],baseLayout({xaxis:{visible:false},yaxis:{visible:false},annotations:[{text:message,x:.5,y:.5,xref:'paper',yref:'paper',showarrow:false,font:{color:COLORS.grey,size:14}}]}),CONFIG);
    }

    function drawHeatmap(rows) {
      const gridCounts=new Map();
      rows.forEach(row => { const match=GRID_RE.exec(row.tile); if(match) gridCounts.set(row.tile,{row:Number(match[1]),col:Number(match[2]),count:(gridCounts.get(row.tile)?.count||0)+1}); });
      const cells=[...gridCounts.values()];
      if(!cells.length) return chartMessage('tile-heatmap','沒有 r####_c#### 格式的 Tile 可繪製熱圖');
      const minRow=Math.min(...cells.map(cell=>cell.row)),maxRow=Math.max(...cells.map(cell=>cell.row));
      const minCol=Math.min(...cells.map(cell=>cell.col)),maxCol=Math.max(...cells.map(cell=>cell.col));
      if((maxRow-minRow+1)*(maxCol-minCol+1)>MAX_HEATMAP_CELLS) return chartMessage('tile-heatmap','Tile 座標範圍超過 3,600 格，請使用 Pareto 與明細表');
      const xs=Array.from({length:maxCol-minCol+1},(_,index)=>index+minCol);
      const ys=Array.from({length:maxRow-minRow+1},(_,index)=>index+minRow);
      const z=ys.map(()=>xs.map(()=>null)),custom=ys.map(()=>xs.map(()=>null));
      gridCounts.forEach((cell,tile) => { z[cell.row-minRow][cell.col-minCol]=cell.count; custom[cell.row-minRow][cell.col-minCol]=tile; });
      return Plotly.react('tile-heatmap',[{type:'heatmap',x:xs,y:ys,z,customdata:custom,zmin:0,colorscale:[[0,'#fff4e8'],[.35,'#f5c08b'],[1,'#c2410c']],colorbar:{title:'缺陷筆數'},hoverongaps:false,hovertemplate:'%{customdata}<br>缺陷 %{z} 筆<extra></extra>'}],baseLayout({margin:{l:58,r:80,t:18,b:55},xaxis:{title:'Column',tickprefix:'C',dtick:1,automargin:true},yaxis:{title:'Row',tickprefix:'R',dtick:1,autorange:'reversed',automargin:true}}),CONFIG);
    }

    function drawNgHeatmap(inspectionTiles) {
      const cells=inspectionTiles.map(tile => { const match=GRID_RE.exec(tile.tile); return match?{...tile,row:Number(match[1]),col:Number(match[2])}:null; }).filter(Boolean);
      if(!inspectionRows.length) return chartMessage('tile-ng-heatmap','找不到同層 json/ 完整結果，無法計算真實 Tile NG 率');
      if(!cells.length) return chartMessage('tile-ng-heatmap','JSON 中沒有 r####_c#### 格式的 Tile');
      const minRow=Math.min(...cells.map(cell=>cell.row)),maxRow=Math.max(...cells.map(cell=>cell.row));
      const minCol=Math.min(...cells.map(cell=>cell.col)),maxCol=Math.max(...cells.map(cell=>cell.col));
      if((maxRow-minRow+1)*(maxCol-minCol+1)>MAX_HEATMAP_CELLS) return chartMessage('tile-ng-heatmap','Tile 座標範圍超過 3,600 格，請使用 NG 率排行');
      const xs=Array.from({length:maxCol-minCol+1},(_,index)=>index+minCol),ys=Array.from({length:maxRow-minRow+1},(_,index)=>index+minRow);
      const z=ys.map(()=>xs.map(()=>null)),custom=ys.map(()=>xs.map(()=>null));
      cells.forEach(cell=>{ z[cell.row-minRow][cell.col-minCol]=cell.rate; custom[cell.row-minRow][cell.col-minCol]=[cell.tile,cell.ng,cell.total]; });
      return Plotly.react('tile-ng-heatmap',[{type:'heatmap',x:xs,y:ys,z,customdata:custom,zmin:0,zmax:1,colorscale:[[0,'#eef4ff'],[.45,'#8eb5ef'],[1,'#123c80']],colorbar:{title:'NG 率',tickformat:'.0%'},hoverongaps:false,hovertemplate:'%{customdata[0]}<br>NG %{customdata[1]} / %{customdata[2]}<br>NG 率 %{z:.1%}<extra></extra>'}],baseLayout({margin:{l:58,r:80,t:18,b:55},xaxis:{title:'Column',tickprefix:'C',dtick:1,automargin:true},yaxis:{title:'Row',tickprefix:'R',dtick:1,autorange:'reversed',automargin:true}}),CONFIG);
    }

    function drawNgRanking(inspectionTiles) {
      if(!inspectionRows.length) return chartMessage('tile-ng-ranking','找不到同層 json/ 完整結果，無法計算真實 Tile NG 率');
      const ranked=inspectionTiles.slice(0,25).reverse();
      if(!ranked.length) return chartMessage('tile-ng-ranking','目前篩選條件下沒有 Tile 檢測分母');
      return Plotly.react('tile-ng-ranking',[{type:'bar',orientation:'h',x:ranked.map(tile=>tile.rate),y:ranked.map(tile=>tile.tile),customdata:ranked.map(tile=>[tile.tile,tile.ng,tile.total]),text:ranked.map(tile=>`${pct(tile.rate)} (${tile.ng}/${tile.total})`),textposition:'auto',marker:{color:ranked.map(tile=>tile.rate),cmin:0,cmax:1,colorscale:[[0,'#cbdcf6'],[1,'#123c80']],line:{color:COLORS.blueDark,width:1}},hovertemplate:'%{customdata[0]}<br>NG %{customdata[1]} / %{customdata[2]}<br>NG 率 %{x:.1%}<extra></extra>'}],baseLayout({margin:{l:145,r:35,t:18,b:55},xaxis:{title:'NG 率',range:[0,1],tickformat:'.0%',gridcolor:'#edf0f5'},yaxis:{automargin:true},showlegend:false}),CONFIG);
    }

    function drawGridProfile(rows) {
      const rowCounts=new Map(),colCounts=new Map();
      rows.forEach(item=>{ const match=GRID_RE.exec(item.tile); if(!match) return; const row=Number(match[1]),col=Number(match[2]); rowCounts.set(row,(rowCounts.get(row)||0)+1); colCounts.set(col,(colCounts.get(col)||0)+1); });
      if(!rowCounts.size) return chartMessage('grid-profile','沒有 r####_c#### 格式的 Tile 可計算位置剖面');
      const rowEntries=[...rowCounts.entries()].sort((a,b)=>a[0]-b[0]),colEntries=[...colCounts.entries()].sort((a,b)=>a[0]-b[0]);
      return Plotly.react('grid-profile',[
        {type:'bar',name:'Row',x:rowEntries.map(item=>`R${item[0]}`),y:rowEntries.map(item=>item[1]),xaxis:'x',yaxis:'y',marker:{color:COLORS.blue},hovertemplate:'%{x}<br>%{y} 筆<extra></extra>'},
        {type:'bar',name:'Column',x:colEntries.map(item=>`C${item[0]}`),y:colEntries.map(item=>item[1]),xaxis:'x2',yaxis:'y2',marker:{color:COLORS.orange},hovertemplate:'%{x}<br>%{y} 筆<extra></extra>'}
      ],baseLayout({grid:{rows:1,columns:2,pattern:'independent',xgap:.12},margin:{l:50,r:30,t:30,b:68},xaxis:{title:'Grid Row',automargin:true},yaxis:{title:'缺陷筆數',rangemode:'tozero',gridcolor:'#edf0f5'},xaxis2:{title:'Grid Column',automargin:true},yaxis2:{title:'缺陷筆數',rangemode:'tozero',gridcolor:'#edf0f5'},legend:{orientation:'h',y:1.15,x:0}}),CONFIG);
    }

    function drawPareto(rows) {
      let ranked=rankedCounts(rows,'tile');
      if(!ranked.length) return chartMessage('tile-pareto','沒有資料');
      if(ranked.length>MAX_PARETO_TILES) ranked=[...ranked.slice(0,MAX_PARETO_TILES-1),['其他',sum(ranked.slice(MAX_PARETO_TILES-1).map(item=>item[1]))]];
      let running=0; const total=sum(ranked.map(item=>item[1]));
      const labels=ranked.map(item=>item[0]),counts=ranked.map(item=>item[1]),cumulative=counts.map(value=>(running+=value)/total);
      const custom=labels.map(label=>label==='其他'?null:label);
      return Plotly.react('tile-pareto',[
        {type:'bar',name:'缺陷筆數',x:labels,y:counts,customdata:custom,marker:{color:COLORS.blue,line:{color:COLORS.blueDark,width:1}},hovertemplate:'%{x}<br>%{y} 筆<extra></extra>'},
        {type:'scatter',mode:'lines+markers',name:'累積占比',x:labels,y:cumulative,customdata:custom,yaxis:'y2',line:{color:COLORS.orange,width:2},marker:{color:'#fff',line:{color:COLORS.orange,width:2}},hovertemplate:'%{x}<br>累積 %{y:.1%}<extra></extra>'}
      ],baseLayout({margin:{l:58,r:66,t:28,b:95},xaxis:{tickangle:-45,automargin:true},yaxis:{title:'缺陷筆數',rangemode:'tozero',gridcolor:'#edf0f5'},yaxis2:{title:'累積占比',overlaying:'y',side:'right',range:[0,1.05],tickformat:'.0%',showgrid:false},legend:{orientation:'h',y:1.12,x:0}}),CONFIG);
    }

    function drawDetectorType(rows) {
      const detectors=rankedCounts(rows,'detector').slice(0,15).map(item=>item[0]);
      const types=rankedCounts(rows,'type');
      if(!detectors.length) return chartMessage('detector-type','沒有資料');
      const selectedTypes=types.slice(0,5).map(item=>item[0]); const hasOther=types.length>5;
      const groups=[...selectedTypes,...(hasOther?['其他']:[])];
      const matrix=new Map(); rows.forEach(row=>{ const group=selectedTypes.includes(row.type)?row.type:'其他'; const key=`${row.detector}\u0000${group}`; matrix.set(key,(matrix.get(key)||0)+1); });
      const traces=groups.map((group,index) => ({
        type:'bar',name:group,x:detectors,y:detectors.map(detector => matrix.get(`${detector}\u0000${group}`)||0),
        customdata:detectors.map(detector=>({detector,type:group==='其他'?null:group})),marker:{color:PALETTE[index%PALETTE.length]},hovertemplate:`Detector %{x}<br>${group} %{y} 筆<extra></extra>`
      }));
      return Plotly.react('detector-type',traces,baseLayout({barmode:'stack',margin:{l:52,r:20,t:32,b:90},xaxis:{tickangle:-35,automargin:true},yaxis:{title:'缺陷筆數',rangemode:'tozero',gridcolor:'#edf0f5'},showlegend:groups.length>1,legend:{orientation:'h',y:1.15,x:0}}),CONFIG);
    }

    function drawTileTypeHeatmap(rows) {
      const tiles=rankedCounts(rows,'tile').slice(0,30).map(item=>item[0]),types=rankedCounts(rows,'type').slice(0,12).map(item=>item[0]);
      if(!tiles.length || !types.length) return chartMessage('tile-type-heatmap','沒有資料');
      const matrix=new Map(); rows.forEach(row=>{ const key=`${row.tile}\u0000${row.type}`; matrix.set(key,(matrix.get(key)||0)+1); });
      const z=types.map(type=>tiles.map(tile=>matrix.get(`${tile}\u0000${type}`)||0));
      const custom=types.map(type=>tiles.map(tile=>[tile,type]));
      return Plotly.react('tile-type-heatmap',[{type:'heatmap',x:tiles,y:types,z,customdata:custom,zmin:0,colorscale:[[0,'#f5f8fc'],[.35,'#8eb5ef'],[1,'#123c80']],colorbar:{title:'缺陷筆數'},hovertemplate:'%{customdata[0]}<br>%{customdata[1]}<br>%{z} 筆<extra></extra>'}],baseLayout({margin:{l:135,r:65,t:18,b:95},xaxis:{tickangle:-40,automargin:true},yaxis:{automargin:true}}),CONFIG);
    }

    function drawTreemap(rows) {
      if(!rows.length) return chartMessage('defect-treemap','沒有資料');
      const detectorCounts=rankedCounts(rows,'detector'),labels=['全部缺陷'],ids=['root'],parents=[''],values=[rows.length],custom=[[null,null]];
      detectorCounts.forEach(([detector,detectorCount])=>{
        const detectorId=`detector:${detector}`; labels.push(detector); ids.push(detectorId); parents.push('root'); values.push(detectorCount); custom.push([detector,null]);
        rankedCounts(rows.filter(row=>row.detector===detector),'type').forEach(([type,count])=>{ labels.push(type); ids.push(`${detectorId}:type:${type}`); parents.push(detectorId); values.push(count); custom.push([detector,type]); });
      });
      return Plotly.react('defect-treemap',[{type:'treemap',labels,ids,parents,values,customdata:custom,branchvalues:'total',marker:{colorscale:[[0,'#dce9fb'],[1,'#165dce']]},textinfo:'label+value+percent parent',hovertemplate:'%{label}<br>%{value} 筆<br>上層占比 %{percentParent:.1%}<extra></extra>'}],baseLayout({margin:{l:10,r:10,t:10,b:10}}),CONFIG);
    }

    function drawDefectShare(rows) {
      let ranked=rankedCounts(rows,'type');
      if(!ranked.length) return chartMessage('defect-share','沒有資料');
      if(ranked.length>8) ranked=[...ranked.slice(0,7),['其他',sum(ranked.slice(7).map(item=>item[1]))]];
      const labels=ranked.map(item=>item[0]);
      return Plotly.react('defect-share',[{type:'pie',hole:.56,labels,values:ranked.map(item=>item[1]),customdata:labels.map(label=>label==='其他'?null:label),sort:false,marker:{colors:labels.map((_,index)=>PALETTE[index%PALETTE.length]),line:{color:'#fff',width:2}},textinfo:'percent',hovertemplate:'%{label}<br>%{value} 筆 · %{percent}<extra></extra>'}],baseLayout({margin:{l:20,r:20,t:20,b:45},legend:{orientation:'h',y:-.08,x:0}}),CONFIG);
    }

    function drawImageRanking(rows) {
      const ranked=rankedCounts(rows,'image').slice(0,20).reverse();
      if(!ranked.length) return chartMessage('image-ranking','沒有資料');
      return Plotly.react('image-ranking',[{type:'bar',orientation:'h',x:ranked.map(item=>item[1]),y:ranked.map(item=>item[0]),customdata:ranked.map(item=>item[0]),marker:{color:COLORS.blue,line:{color:COLORS.blueDark,width:1}},hovertemplate:'%{y}<br>%{x} 筆<extra></extra>'}],baseLayout({margin:{l:150,r:28,t:18,b:48},xaxis:{title:'缺陷筆數',rangemode:'tozero',gridcolor:'#edf0f5'},yaxis:{automargin:true}}),CONFIG);
    }

    function drawImageDefectHistogram(rows) {
      const counts=rankedCounts(rows,'image').map(item=>item[1]);
      if(!counts.length) return chartMessage('image-defect-histogram','沒有資料');
      return Plotly.react('image-defect-histogram',[{type:'histogram',x:counts,nbinsx:Math.min(30,Math.max(5,Math.ceil(Math.sqrt(counts.length)))),marker:{color:COLORS.gold,line:{color:'#8a6310',width:1}},hovertemplate:'每張圖片缺陷數 %{x}<br>%{y} 張圖片<extra></extra>'}],baseLayout({margin:{l:58,r:24,t:18,b:58},xaxis:{title:'每張圖片的缺陷筆數',rangemode:'tozero',gridcolor:'#edf0f5'},yaxis:{title:'圖片數',rangemode:'tozero',gridcolor:'#edf0f5'},showlegend:false}),CONFIG);
    }

    function drawArea(rows) {
      const valid=rows.filter(row=>Number.isFinite(row.area)); const units=unique(valid.map(row=>row.areaUnit));
      if(!valid.length) return chartMessage('area-histogram','沒有可解析的 area 數值');
      if(filters.areaUnit.value==='*' && units.length>1) return chartMessage('area-histogram',`偵測到 ${units.length} 種面積單位，請先從全域篩選選擇一種`);
      const unit=filters.areaUnit.value==='*'?units[0]:filters.areaUnit.value; const values=valid.filter(row=>row.areaUnit===unit).map(row=>row.area);
      return Plotly.react('area-histogram',[{type:'histogram',x:values,nbinsx:Math.min(30,Math.max(5,Math.ceil(Math.sqrt(values.length)))),marker:{color:COLORS.orange,line:{color:'#a84d0e',width:1}},hovertemplate:`面積區間 %{x}<br>%{y} 筆<extra></extra>`}],baseLayout({margin:{l:55,r:20,t:18,b:58},xaxis:{title:`面積 (${unit})`,rangemode:'tozero',gridcolor:'#edf0f5'},yaxis:{title:'缺陷筆數',rangemode:'tozero',gridcolor:'#edf0f5'},showlegend:false}),CONFIG);
    }

    function areaRowsForOneUnit(rows,id) {
      const valid=rows.filter(row=>Number.isFinite(row.area)),units=unique(valid.map(row=>row.areaUnit));
      if(!valid.length) { chartMessage(id,'沒有可解析的 area 數值'); return null; }
      if(filters.areaUnit.value==='*' && units.length>1) { chartMessage(id,`偵測到 ${units.length} 種面積單位，請先選擇一種`); return null; }
      const unit=filters.areaUnit.value==='*'?units[0]:filters.areaUnit.value;
      return {unit,rows:valid.filter(row=>row.areaUnit===unit)};
    }

    function drawAreaBoxplot(rows) {
      const selected=areaRowsForOneUnit(rows,'area-boxplot'); if(!selected) return Promise.resolve();
      if(selected.rows.length<5) return chartMessage('area-boxplot',`目前只有 ${selected.rows.length} 筆有效面積；至少 5 筆才繪製箱型圖`);
      const types=rankedCounts(selected.rows,'type').slice(0,10).map(item=>item[0]);
      const traces=types.map((type,index)=>({type:'box',name:type,y:selected.rows.filter(row=>row.type===type).map(row=>row.area),boxpoints:'outliers',marker:{color:PALETTE[index%PALETTE.length]},line:{color:PALETTE[index%PALETTE.length]},hovertemplate:`${type}<br>${selected.unit} %{y}<extra></extra>`}));
      return Plotly.react('area-boxplot',traces,baseLayout({margin:{l:62,r:20,t:28,b:95},xaxis:{tickangle:-35,automargin:true},yaxis:{title:`面積 (${selected.unit})`,rangemode:'tozero',gridcolor:'#edf0f5'},showlegend:false}),CONFIG);
    }

    function drawScore(rows) {
      const values=rows.filter(row=>Number.isFinite(row.score)).map(row=>row.score);
      if(!values.length) return chartMessage('score-histogram','沒有可解析的 score 數值');
      return Plotly.react('score-histogram',[{type:'histogram',x:values,nbinsx:Math.min(30,Math.max(5,Math.ceil(Math.sqrt(values.length)))),marker:{color:COLORS.olive,line:{color:'#4d611a',width:1}},hovertemplate:'score 區間 %{x}<br>%{y} 筆<extra></extra>'}],baseLayout({margin:{l:55,r:20,t:18,b:58},xaxis:{title:'Detector score',gridcolor:'#edf0f5'},yaxis:{title:'缺陷筆數',rangemode:'tozero',gridcolor:'#edf0f5'},showlegend:false}),CONFIG);
    }

    function drawAreaScoreScatter(rows) {
      const selected=areaRowsForOneUnit(rows,'area-score-scatter'); if(!selected) return Promise.resolve();
      const valid=selected.rows.filter(row=>Number.isFinite(row.score));
      if(valid.length<8) return chartMessage('area-score-scatter',`目前只有 ${valid.length} 筆同時具有 area 與 score；至少 8 筆才繪製散點`);
      const stride=Math.max(1,Math.ceil(valid.length/MAX_SCATTER_POINTS)),sampled=valid.filter((_,index)=>index%stride===0).slice(0,MAX_SCATTER_POINTS);
      const topTypes=rankedCounts(valid,'type').slice(0,5).map(item=>item[0]); const groups=[...topTypes,...(new Set(valid.map(row=>row.type)).size>5?['其他']:[])];
      const traces=groups.map((group,index)=>{ const points=sampled.filter(row=>group==='其他'?!topTypes.includes(row.type):row.type===group); return {type:'scatter',mode:'markers',name:group,x:points.map(row=>row.area),y:points.map(row=>row.score),text:points.map(row=>row.tile),customdata:points.map(row=>[row.tile,group==='其他'?null:row.type,row.detector]),marker:{size:8,color:PALETTE[index%PALETTE.length],line:{color:'#fff',width:1},opacity:.75},hovertemplate:`%{text}<br>面積 %{x} ${selected.unit}<br>score %{y}<br>${group}<extra></extra>`}; });
      return Plotly.react('area-score-scatter',traces,baseLayout({margin:{l:62,r:22,t:30,b:58},xaxis:{title:`面積 (${selected.unit})`,rangemode:'tozero',gridcolor:'#edf0f5'},yaxis:{title:'Detector score',gridcolor:'#edf0f5'},legend:{orientation:'h',y:1.15,x:0}}),CONFIG);
    }

    function drawTileScatter(tiles) {
      if(tiles.length<8) return chartMessage('tile-scatter',`目前只有 ${tiles.length} 個 Tile；少於 8 個時請改看 Pareto 與明細`);
      return Plotly.react('tile-scatter',[{type:'scatter',mode:'markers',x:tiles.map(tile=>tile.images.size),y:tiles.map(tile=>tile.count),text:tiles.map(tile=>tile.tile),customdata:tiles.map(tile=>tile.tile),marker:{size:10,color:COLORS.blue,line:{color:COLORS.blueDark,width:1},opacity:.78},hovertemplate:'%{text}<br>受影響圖片 %{x}<br>缺陷 %{y} 筆<extra></extra>'}],baseLayout({margin:{l:58,r:24,t:18,b:58},xaxis:{title:'受影響圖片數',rangemode:'tozero',dtick:1,gridcolor:'#edf0f5'},yaxis:{title:'缺陷筆數',rangemode:'tozero',gridcolor:'#edf0f5'},showlegend:false}),CONFIG);
    }

    function formatAreas(areaMap) {
      if(!areaMap.size) return '—';
      return [...areaMap.entries()].sort((a,b)=>natural(a[0],b[0])).map(([unit,values])=>`${fmt(sum(values))} ${unit}`).join('；');
    }

    function renderTable(rows,tiles) {
      const query=document.getElementById('table-search').value.trim().toLocaleLowerCase();
      const total=rows.length || 1;
      let items=tiles.map(tile => {
        const detector=dominant(tile.detectors),type=dominant(tile.types),score=average(tile.scores);
        return {tile:tile.tile,count:tile.count,share:tile.count/total,images:tile.images.size,perImage:tile.count/tile.images.size,detector:`${detector[0]} × ${detector[1]}`,type:`${type[0]} × ${type[1]}`,score,area:formatAreas(tile.areas)};
      });
      if(query) items=items.filter(item=>`${item.tile} ${item.detector} ${item.type}`.toLocaleLowerCase().includes(query));
      const key=tableSort.key,direction=tableSort.direction;
      items.sort((a,b) => { const left=a[key],right=b[key]; const comparison=typeof left==='number'&&typeof right==='number'?left-right:natural(left,right); return comparison*direction; });
      const body=document.getElementById('tile-rows'); body.replaceChildren();
      items.slice(0,TABLE_LIMIT).forEach(item => {
        const row=document.createElement('tr');
        const tileCell=document.createElement('td'); const tileButton=create('button',item.tile,'tile-button'); tileButton.type='button'; tileButton.dataset.tile=item.tile; tileCell.append(tileButton); row.append(tileCell);
        [[fmt(item.count),'number'],[pct(item.share),'number'],[fmt(item.images),'number'],[fmt(item.perImage),'number'],[item.detector,''],[item.type,''],[item.score===null?'—':fmt(item.score),'number'],[item.area,'number']].forEach(([text,className])=>row.append(create('td',text,className)));
        body.append(row);
      });
      if(!items.length) { const row=document.createElement('tr'),cell=create('td','目前條件下沒有可顯示的 Tile。'); cell.colSpan=9; row.append(cell); body.append(row); }
      document.getElementById('table-status').textContent=items.length>TABLE_LIMIT?`符合 ${fmt(items.length)} 個 Tile；為維持瀏覽效能，表格顯示前 ${TABLE_LIMIT} 個。`:`顯示 ${fmt(items.length)} 個 Tile。`;
    }

    function updateQuality(rows,tiles,inspections) {
      const validArea=rows.filter(row=>Number.isFinite(row.area)).length,validScore=rows.filter(row=>Number.isFinite(row.score)).length;
      const gridTiles=tiles.filter(tile=>GRID_RE.test(tile.tile)).length,missingTile=rows.filter(row=>row.tile==='（未提供 tile_id）').length;
      document.getElementById('quality-area').textContent=`${fmt(validArea)} / ${fmt(rows.length)}`;
      document.getElementById('quality-score').textContent=`${fmt(validScore)} / ${fmt(rows.length)}`;
      document.getElementById('quality-grid').textContent=`${fmt(gridTiles)} / ${fmt(tiles.length)}`;
      document.getElementById('quality-missing-tile').textContent=fmt(missingTile);
      document.getElementById('quality-json').textContent=`${fmt(sourceMeta.jsonReports)} 份`+(sourceMeta.jsonParseErrors?`（${fmt(sourceMeta.jsonParseErrors)} 份錯誤）`:'');
      document.getElementById('quality-inspections').textContent=`${fmt(inspections.length)} 筆`+(sourceMeta.unknownTileResults?`（另 ${fmt(sourceMeta.unknownTileResults)} 筆未知）`:'');
    }

    function bindChartClick(id,handler) {
      const chart=document.getElementById(id); if(chart.dataset.clickBound) return; chart.dataset.clickBound='true'; chart.on('plotly_click',event=>{ const point=event.points&&event.points[0]; if(point) handler(point); });
    }

    function bindChartClicks() {
      bindChartClick('tile-heatmap',point=>setFilter('tile',point.customdata));
      bindChartClick('tile-ng-heatmap',point=>setFilter('tile',point.customdata&&point.customdata[0]));
      bindChartClick('tile-ng-ranking',point=>setFilter('tile',point.customdata&&point.customdata[0]));
      bindChartClick('tile-pareto',point=>setFilter('tile',point.customdata));
      bindChartClick('detector-type',point=>{ const value=point.customdata; if(value) setFilters({detector:value.detector,type:value.type}); });
      bindChartClick('tile-type-heatmap',point=>{ const value=point.customdata; if(value) setFilters({tile:value[0],type:value[1]}); });
      bindChartClick('defect-treemap',point=>{ const value=point.customdata; if(value) setFilters({detector:value[0],type:value[1]}); });
      bindChartClick('defect-share',point=>setFilter('type',point.customdata));
      bindChartClick('image-ranking',point=>setFilter('image',point.customdata));
      bindChartClick('tile-scatter',point=>setFilter('tile',point.customdata));
      bindChartClick('area-score-scatter',point=>{ const value=point.customdata; if(value) setFilters({tile:value[0],type:value[1]}); });
    }

    function updateDashboard() {
      const rows=filteredRows(),tiles=aggregateTiles(rows),inspections=filteredInspections(),inspectionTiles=aggregateInspectionTiles(inspections);
      updateFilterSummary(rows,inspections); updateCards(rows,tiles,inspectionTiles); updateInsights(rows,tiles,inspectionTiles); updateQuality(rows,tiles,inspections); renderTopImages(rows); renderTable(rows,tiles);
      Promise.all([drawHeatmap(rows),drawNgHeatmap(inspectionTiles),drawNgRanking(inspectionTiles),drawPareto(rows),drawGridProfile(rows),drawDetectorType(rows),drawTileTypeHeatmap(rows),drawTreemap(rows),drawDefectShare(rows),drawImageRanking(rows),drawImageDefectHistogram(rows),drawTileScatter(tiles),drawArea(rows),drawAreaBoxplot(rows),drawScore(rows),drawAreaScoreScatter(rows)]).then(bindChartClicks);
    }

    populateFilters();
    filterSpecs.forEach(([,id])=>document.getElementById(id).addEventListener('change',updateDashboard));
    document.getElementById('reset-filters').addEventListener('click',()=>{ filterSpecs.forEach(([key])=>filters[key].value='*'); document.getElementById('table-search').value=''; updateDashboard(); });
    document.getElementById('table-search').addEventListener('input',()=>renderTable(filteredRows(),aggregateTiles(filteredRows())));
    document.querySelector('thead').addEventListener('click',event=>{ const button=event.target.closest('[data-sort]'); if(!button) return; const key=button.dataset.sort; tableSort=tableSort.key===key?{key,direction:-tableSort.direction}:{key,direction:key==='tile'?1:-1}; renderTable(filteredRows(),aggregateTiles(filteredRows())); });
    document.getElementById('tile-rows').addEventListener('click',event=>{ const button=event.target.closest('[data-tile]'); if(button) setFilter('tile',button.dataset.tile); });
    document.getElementById('top-image-rows').addEventListener('click',event=>{ const button=event.target.closest('[data-image]'); if(button) setFilter('image',button.dataset.image); });
    updateDashboard();
  })();
  </script>
</body>
</html>
"""

    cards = "".join(
        [
            _metric_card("缺陷筆數", str(distribution.total_defects), "篩選後缺陷列"),
            _metric_card("有缺陷的 Tile", str(distribution.tile_count), "不同 tile_id"),
            _metric_card("受影響圖片", str(distribution.image_count), "不同 image_name"),
            _metric_card(
                "平均缺陷／Tile",
                _format_number(distribution.total_defects / distribution.tile_count)
                if distribution.tile_count
                else "—",
                "只計有缺陷 Tile",
            ),
            _metric_card("最高缺陷 Tile", top_tile_text, "Tile ID · 缺陷筆數"),
            _metric_card(
                "面積欄位涵蓋率",
                _format_percent(distribution.rows_with_area, distribution.total_defects),
                "有效 area／缺陷列",
            ),
            _metric_card(
                "Tile NG 率",
                _format_percent(
                    sum(record.tile_result == "NG" for record in distribution.inspection_records),
                    len(distribution.inspection_records),
                ),
                "JSON：NG／全部檢測",
            ),
        ]
    )

    replacements = {
        "__SOURCE_PATH__": _escape(str(distribution.source_path.resolve())),
        "__GENERATED_AT__": _escape(generated_at),
        "__METRIC_CARDS__": cards,
        "__DATA_NOTICE__": missing_notice,
        "__SOURCE_FIELDS__": _escape(source_fields),
        "__JSON_SOURCE__": _escape(
            f"{distribution.json_directory}（已讀取 {distribution.json_reports_scanned} 份；"
            f"解析失敗 {distribution.json_parse_errors} 份）"
        ),
        "__TOOL_VERSION__": _escape(TOOL_VERSION),
        "__FALLBACK_ROWS__": fallback_rows,
        "__RECORDS_JSON__": records_payload,
        "__INSPECTIONS_JSON__": inspections_payload,
        "__SOURCE_META_JSON__": source_meta_payload,
        "__PLOTLY_JS__": plotly_javascript,
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def _metric_card(label: str, value: str, note: str) -> str:
    metric_id = {
        "缺陷筆數": "metric-defects",
        "有缺陷的 Tile": "metric-tiles",
        "受影響圖片": "metric-images",
        "平均缺陷／Tile": "metric-average",
        "最高缺陷 Tile": "metric-top",
        "面積欄位涵蓋率": "metric-area",
        "Tile NG 率": "metric-ng-rate",
    }[label]
    return (
        '<div class="card">'
        f'<div class="label">{_escape(label)}</div>'
        f'<div class="value mono" id="{metric_id}">{_escape(value)}</div>'
        f'<div class="note">{_escape(note)}</div></div>'
    )


def _record_payload(record: DefectRecord) -> dict[str, object]:
    return {
        "image": record.image_name,
        "tile": record.tile_id,
        "detector": record.detector_id,
        "type": record.defect_type,
        "recipe": record.recipe_name,
        "machine": record.machine_id,
        "product": record.product_id,
        "result": record.final_result,
        "area": record.area,
        "areaUnit": record.area_unit,
        "score": record.score,
    }


def _inspection_payload(record: TileInspectionRecord) -> dict[str, object]:
    return {
        "image": record.image_name,
        "tile": record.tile_id,
        "tileResult": record.tile_result,
        "recipe": record.recipe_name,
        "machine": record.machine_id,
        "product": record.product_id,
        "result": record.final_result,
    }


def _safe_json(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


@lru_cache(maxsize=1)
def _plotly_javascript() -> str:
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:  # pragma: no cover - dependency/packaging failure
        raise RuntimeError("缺少 Plotly，請安裝 requirements.lock.txt 後重新產生報表。") from exc
    return get_plotlyjs()


def _format_number(value: float) -> str:
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _format_percent(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.1%}" if denominator else "—"


def _data_quality_notice(distribution: SummaryDistribution) -> str:
    notices: list[str] = []
    if distribution.rows_without_tile_id:
        notices.append(f"{distribution.rows_without_tile_id} 筆缺少 tile_id")
    if distribution.rows_without_image_name:
        notices.append(f"{distribution.rows_without_image_name} 筆缺少 image_name")
    if distribution.invalid_area_rows:
        notices.append(f"{distribution.invalid_area_rows} 筆 area 無法解析")
    if distribution.invalid_score_rows:
        notices.append(f"{distribution.invalid_score_rows} 筆 score 無法解析")
    if not distribution.inspection_records:
        notices.append("找不到同層 json/ 的完整 Tile PASS／NG 分母，因此 Tile NG 率不顯示")
    elif distribution.json_parse_errors:
        notices.append(f"{distribution.json_parse_errors} 份 JSON 無法解析，Tile NG 率分母可能不完整")
    if distribution.json_unknown_tile_results:
        notices.append(
            f"{distribution.json_unknown_tile_results} 筆 JSON Tile 缺少可判定的 PASS／NG 結果"
        )
    if not notices:
        return ""
    return '<p class="notice">資料品質提醒：' + _escape("；".join(notices)) + "。這些列仍保留於可用的其他統計中。</p>"


def _render_heatmap(distribution: SummaryDistribution) -> str:
    positioned: list[tuple[TileDistribution, int, int]] = []
    non_grid_tiles = 0
    for tile in distribution.sorted_tiles:
        coordinate = _tile_coordinate(tile.tile_id)
        if coordinate is None:
            non_grid_tiles += 1
            continue
        row, col = coordinate
        positioned.append((tile, row, col))

    if not positioned:
        return '<div class="grid-unavailable">沒有可用 r####_c#### 格式的 tile_id，因此以下方明細表呈現統計。</div>'

    min_row = min(row for _, row, _ in positioned)
    max_row = max(row for _, row, _ in positioned)
    min_col = min(col for _, _, col in positioned)
    max_col = max(col for _, _, col in positioned)
    row_count = max_row - min_row + 1
    col_count = max_col - min_col + 1
    if row_count * col_count > MAX_HEATMAP_CELLS:
        return (
            '<div class="grid-unavailable">Tile 座標範圍過大，為避免產生難以閱讀的網格，'
            '已改以以下方明細表呈現統計。</div>'
        )

    max_count = max(tile.defect_count for tile, _, _ in positioned)
    cells: list[str] = []
    for tile, row, col in positioned:
        intensity = tile.defect_count / max_count if max_count else 0.0
        lightness = round(96 - intensity * 41)
        title = (
            f"{tile.tile_id}：{tile.defect_count} 筆缺陷，"
            f"{tile.affected_image_count} 張受影響圖片"
        )
        cells.append(
            '<div class="tile-cell" '
            f'style="grid-row:{row - min_row + 1};grid-column:{col - min_col + 1};background:hsl(7 74% {lightness}%);" '
            f'title="{_escape(title)}">'
            f'<span class="count">{tile.defect_count}</span>'
            f'<span class="label">R{row} C{col}</span></div>'
        )
    extra = (
        f'<p class="help">另有 {non_grid_tiles} 個非網格格式的 tile_id，請見明細表。</p>'
        if non_grid_tiles
        else ""
    )
    return (
        f'<div class="heatmap" style="--columns:{col_count};--rows:{row_count}">{"".join(cells)}</div>'
        '<div class="legend"><i></i><span>較少缺陷 → 較多缺陷（數字為缺陷筆數）</span></div>'
        f"{extra}"
    )


def _render_tile_rows(distribution: SummaryDistribution) -> str:
    if not distribution.tiles:
        return '<tr><td colspan="5">此 summary.csv 沒有缺陷資料列。</td></tr>'
    rows = []
    for tile in distribution.sorted_tiles:
        rows.append(
            "<tr>"
            f"<td>{_escape(tile.tile_id)}</td>"
            f'<td class="number">{tile.defect_count}</td>'
            f'<td class="number">{tile.affected_image_count}</td>'
            f'<td class="breakdown">{_escape(_format_counts(tile.defect_type_counts))}</td>'
            f'<td class="breakdown">{_escape(_format_counts(tile.detector_counts))}</td>'
            "</tr>"
        )
    return "".join(rows)


def _render_rank_rows(counts: Counter[str], label: str) -> str:
    if not counts:
        return '<p class="help">沒有可統計的資料。</p>'
    rows = "".join(
        f'<li><span>{_escape(name)}</span><span class="count">{count} 筆</span></li>'
        for name, count in _sorted_counts(counts)
    )
    return f'<ul class="rank" aria-label="{_escape(label)} 統計">{rows}</ul>'


def _tile_coordinate(tile_id: str) -> tuple[int, int] | None:
    match = GRID_TILE_RE.fullmatch(tile_id.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _tile_sort_key(tile: TileDistribution) -> tuple[int, tuple[tuple[int, object], ...]]:
    return (-tile.defect_count, _natural_sort_key(tile.tile_id))


def _sorted_counts(counts: Counter[str]) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda item: (-item[1], _natural_sort_key(item[0])))


def _format_counts(counts: Counter[str]) -> str:
    return "；".join(f"{name} × {count}" for name, count in _sorted_counts(counts))


def _natural_sort_key(value: str) -> tuple[tuple[int, object], ...]:
    parts: list[tuple[int, object]] = []
    for part in re.split(r"(\d+)", str(value).casefold()):
        if part:
            parts.append((0, int(part)) if part.isdigit() else (1, part))
    return tuple(parts) or ((1, ""),)


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


class TileDefectDistributionApp:
    """Small Tk GUI for selecting one summary and writing one HTML report."""

    def __init__(self) -> None:
        self.root = Tk()
        self.root.title(f"AOI Tile 缺陷分布報表 v{TOOL_VERSION}")
        self.root.geometry("720x285")
        self.root.minsize(640, 260)
        self.summary_path = StringVar(value="")
        self.output_path = StringVar(value="")
        self.status = StringVar(value="選擇 AOI 輸出資料夾中的 csv/summary.csv。")
        self._build_ui()

    def run(self) -> None:
        self.root.mainloop()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=18)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Summary CSV").grid(row=0, column=0, sticky="w", pady=(0, 10))
        ttk.Entry(frame, textvariable=self.summary_path).grid(
            row=0, column=1, sticky="ew", padx=8, pady=(0, 10)
        )
        ttk.Button(frame, text="選擇檔案…", command=self._select_summary).grid(
            row=0, column=2, pady=(0, 10)
        )

        ttk.Label(frame, text="HTML 報表").grid(row=1, column=0, sticky="w", pady=(0, 14))
        ttk.Entry(frame, textvariable=self.output_path).grid(
            row=1, column=1, sticky="ew", padx=8, pady=(0, 14)
        )
        ttk.Button(frame, text="儲存位置…", command=self._select_output).grid(
            row=1, column=2, pady=(0, 14)
        )

        ttk.Button(frame, text="產生 HTML 報表", command=self._export).grid(
            row=2, column=1, sticky="w", pady=(0, 14)
        )
        ttk.Label(frame, textvariable=self.status, wraplength=650).grid(
            row=3, column=0, columnspan=3, sticky="ew"
        )

    def _select_summary(self) -> None:
        selected = filedialog.askopenfilename(
            title="選擇 AOI csv/summary.csv",
            filetypes=[("CSV 檔案", "*.csv"), ("所有檔案", "*.*")],
        )
        if not selected:
            return
        path = Path(selected)
        self.summary_path.set(str(path))
        if not self.output_path.get().strip():
            self.output_path.set(str(default_output_path(path)))
        self.status.set("已選擇 summary.csv，按下「產生 HTML 報表」。")

    def _select_output(self) -> None:
        initial = self.output_path.get().strip() or DEFAULT_OUTPUT_NAME
        selected = filedialog.asksaveasfilename(
            title="儲存 HTML 報表",
            initialfile=Path(initial).name,
            defaultextension=".html",
            filetypes=[("HTML 報表", "*.html"), ("所有檔案", "*.*")],
        )
        if selected:
            self.output_path.set(selected)

    def _export(self) -> None:
        summary_text = self.summary_path.get().strip()
        if not summary_text:
            messagebox.showerror("尚未選擇 Summary", "請先選擇 AOI 輸出的 csv/summary.csv。")
            return
        try:
            output_path, distribution = export_html_report(
                Path(summary_text),
                Path(self.output_path.get().strip()) if self.output_path.get().strip() else None,
            )
        except Exception as exc:  # pragma: no cover - GUI error path
            traceback.print_exc()
            messagebox.showerror("產生失敗", str(exc))
            self.status.set(f"產生失敗：{exc}")
            return
        self.output_path.set(str(output_path))
        self.status.set(
            f"已產生 {distribution.total_defects} 筆缺陷、{distribution.tile_count} 個 tile 的 HTML 報表。"
        )
        messagebox.showinfo("報表完成", f"已儲存：\n{output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read one AOI csv/summary.csv and export an offline HTML tile defect distribution report."
    )
    parser.add_argument("--input", "-i", type=Path, help="AOI csv/summary.csv path.")
    parser.add_argument("--output", "-o", type=Path, help="HTML output path.")
    parser.add_argument("--smoke-test", action="store_true", help="Create and close the GUI for package validation.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {TOOL_VERSION}")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        TileDefectDistributionApp().run()
        return 0

    parser = build_parser()
    args = parser.parse_args(raw_argv)
    if args.smoke_test:
        app = TileDefectDistributionApp()
        app.root.update_idletasks()
        app.root.destroy()
        return 0
    if args.input is None:
        parser.error("CLI 模式必須提供 --input AOI csv/summary.csv。")

    output_path, distribution = export_html_report(args.input, args.output)
    print(
        f"已產生 {distribution.total_defects} 筆缺陷、{distribution.tile_count} 個 tile 的報表：{output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
