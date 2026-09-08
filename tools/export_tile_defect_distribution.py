"""Export an AOI ``csv/summary.csv`` as a self-contained tile distribution report."""

from __future__ import annotations

import argparse
import csv
import html
import re
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tkinter import StringVar, Tk, filedialog, messagebox, ttk


DEFAULT_OUTPUT_NAME = "tile_defect_distribution.html"
TOOL_VERSION = "1.0.0"
GRID_TILE_RE = re.compile(r"^r(\d+)_c(\d+)$", re.IGNORECASE)
MAX_HEATMAP_CELLS = 3600
UNKNOWN_TILE_ID = "（未提供 tile_id）"
UNKNOWN_IMAGE_NAME = "（未提供 image_name）"
UNKNOWN_DETECTOR_ID = "（未提供 detector_id）"
UNKNOWN_DEFECT_TYPE = "（未提供 defect_type）"


@dataclass
class TileDistribution:
    """Aggregated defects for one AOI tile identifier."""

    tile_id: str
    defect_count: int = 0
    image_names: set[str] = field(default_factory=set)
    detector_counts: Counter[str] = field(default_factory=Counter)
    defect_type_counts: Counter[str] = field(default_factory=Counter)

    @property
    def affected_image_count(self) -> int:
        return len(self.image_names)


@dataclass
class SummaryDistribution:
    """The statistics extracted from one AOI defect summary CSV."""

    source_path: Path
    total_defects: int = 0
    image_names: set[str] = field(default_factory=set)
    tiles: dict[str, TileDistribution] = field(default_factory=dict)
    detector_counts: Counter[str] = field(default_factory=Counter)
    defect_type_counts: Counter[str] = field(default_factory=Counter)
    rows_without_tile_id: int = 0

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
            fieldnames = {name.strip() for name in (reader.fieldnames or []) if name}
            if "tile_id" not in fieldnames:
                raise ValueError("檔案不是 AOI summary.csv：缺少 tile_id 欄位。")

            for row in reader:
                if not any(str(value or "").strip() for value in row.values()):
                    continue
                _add_defect_row(distribution, row)
    except UnicodeDecodeError as exc:
        raise ValueError("summary.csv 必須是 UTF-8 或 UTF-8 BOM 編碼。") from exc
    except csv.Error as exc:
        raise ValueError(f"無法讀取 CSV：{exc}") from exc

    return distribution


def _add_defect_row(distribution: SummaryDistribution, row: dict[str | None, str | None]) -> None:
    raw_tile_id = str(row.get("tile_id") or "").strip()
    tile_id = raw_tile_id or UNKNOWN_TILE_ID
    image_name = str(row.get("image_name") or "").strip() or UNKNOWN_IMAGE_NAME
    detector_id = str(row.get("detector_id") or "").strip() or UNKNOWN_DETECTOR_ID
    defect_type = str(row.get("defect_type") or "").strip() or UNKNOWN_DEFECT_TYPE

    tile = distribution.tiles.setdefault(tile_id, TileDistribution(tile_id=tile_id))
    tile.defect_count += 1
    tile.image_names.add(image_name)
    tile.detector_counts[detector_id] += 1
    tile.defect_type_counts[defect_type] += 1

    distribution.total_defects += 1
    distribution.image_names.add(image_name)
    distribution.detector_counts[detector_id] += 1
    distribution.defect_type_counts[defect_type] += 1
    if not raw_tile_id:
        distribution.rows_without_tile_id += 1


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
    """Return a portable report with a grid heatmap and per-tile detail table."""

    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    top_tile = distribution.top_tile
    top_tile_text = (
        f"{top_tile.tile_id}（{top_tile.defect_count} 筆）" if top_tile else "—"
    )
    heatmap = _render_heatmap(distribution)
    tile_rows = _render_tile_rows(distribution)
    detector_rows = _render_rank_rows(distribution.detector_counts, "Detector")
    type_rows = _render_rank_rows(distribution.defect_type_counts, "缺陷類型")
    missing_tile_notice = ""
    if distribution.rows_without_tile_id:
        missing_tile_notice = (
            '<p class="notice">'
            f'有 <strong>{distribution.rows_without_tile_id}</strong> 筆資料未提供 tile_id，'
            '已集中列為「未提供 tile_id」。</p>'
        )

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AOI Tile 缺陷分布報表</title>
  <style>
    :root {{ color-scheme: light; --ink:#172033; --muted:#667085; --line:#d9e1ec; --panel:#fff; --bg:#f5f7fb; --brand:#165dce; --danger:#c43232; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 "Microsoft JhengHei", "Noto Sans TC", Arial, sans-serif; }}
    header {{ color:#fff; background:linear-gradient(120deg,#123c80,#165dce); padding:38px max(24px, calc((100% - 1180px)/2)); }}
    header h1 {{ margin:0 0 8px; font-size:30px; }}
    header p {{ margin:0; color:#e5efff; word-break:break-all; }}
    main {{ max-width:1180px; margin:0 auto; padding:26px 20px 48px; }}
    .cards {{ display:grid; grid-template-columns:repeat(5,minmax(135px,1fr)); gap:14px; margin-bottom:22px; }}
    .card,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; box-shadow:0 1px 2px #1720330d; }}
    .card {{ padding:16px; }} .card .label {{ color:var(--muted); font-size:12px; }} .card .value {{ margin-top:5px; font-size:25px; font-weight:700; overflow-wrap:anywhere; }}
    .panel {{ padding:20px; margin-top:20px; }} h2 {{ margin:0 0 7px; font-size:20px; }} h3 {{ margin:0 0 12px; font-size:16px; }}
    .help,.subtitle {{ color:var(--muted); margin:0 0 16px; }} .notice {{ border-left:4px solid #e6a700; background:#fff8dd; padding:10px 12px; margin:16px 0 0; }}
    .heatmap {{ display:grid; grid-template-columns:repeat(var(--columns),minmax(36px,1fr)); grid-template-rows:repeat(var(--rows),52px); gap:5px; min-width:100%; overflow:auto; }}
    .tile-cell {{ border:1px solid #ffffff; border-radius:7px; min-width:36px; padding:5px; color:#541414; font-size:11px; overflow:hidden; cursor:default; }}
    .tile-cell .count {{ display:block; font-weight:800; font-size:16px; line-height:18px; }} .tile-cell .label {{ display:block; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .legend {{ display:flex; gap:8px; align-items:center; color:var(--muted); margin:12px 0 0; }} .legend i {{ width:25px; height:12px; border-radius:3px; background:linear-gradient(90deg,hsl(7 74% 96%),hsl(7 74% 55%)); }}
    .grid-unavailable {{ background:#f7f9fc; border:1px dashed var(--line); padding:15px; color:var(--muted); border-radius:8px; }}
    .split {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
    .rank {{ list-style:none; padding:0; margin:0; }} .rank li {{ display:flex; justify-content:space-between; border-top:1px solid #edf0f5; padding:8px 0; gap:12px; }} .rank li:first-child {{ border-top:0; }} .rank .count {{ font-weight:700; white-space:nowrap; }}
    .toolbar {{ display:flex; justify-content:space-between; align-items:center; gap:12px; margin:0 0 12px; }} .toolbar input {{ border:1px solid #b9c5d5; border-radius:7px; padding:8px 10px; width:min(320px,100%); font:inherit; }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:8px; }} table {{ width:100%; border-collapse:collapse; min-width:720px; }} th,td {{ text-align:left; padding:10px 12px; border-bottom:1px solid #edf0f5; vertical-align:top; }} th {{ background:#f7f9fc; color:#44516a; font-size:12px; position:sticky; top:0; }} td.number {{ text-align:right; font-variant-numeric:tabular-nums; }} .breakdown {{ color:#44516a; max-width:300px; overflow-wrap:anywhere; }}
    footer {{ color:var(--muted); font-size:12px; margin-top:20px; }}
    @media (max-width:760px) {{ header {{ padding:26px 20px; }} header h1 {{ font-size:25px; }} .cards {{ grid-template-columns:repeat(2,1fr); }} .split {{ grid-template-columns:1fr; }} .toolbar {{ align-items:flex-start; flex-direction:column; }} }}
    @media print {{ body {{ background:#fff; }} header {{ color:#172033; background:#fff; padding:0 0 18px; border-bottom:2px solid #172033; }} header p {{ color:#44516a; }} main {{ max-width:none; padding:18px 0; }} .panel,.card {{ box-shadow:none; }} .toolbar {{ display:none; }} }}
  </style>
</head>
<body>
  <header>
    <h1>AOI Tile 缺陷分布報表</h1>
    <p>來源：{_escape(str(distribution.source_path.resolve()))}<br>產生時間：{_escape(generated_at)}　｜　每一列 summary.csv 代表一筆缺陷。</p>
  </header>
  <main>
    <section class="cards" aria-label="統計總覽">
      {_metric_card("缺陷總筆數", str(distribution.total_defects))}
      {_metric_card("有缺陷的 Tile", str(distribution.tile_count))}
      {_metric_card("受影響圖片", str(distribution.image_count))}
      {_metric_card("Detector 數", str(len(distribution.detector_counts)))}
      {_metric_card("最高缺陷 Tile", _escape(top_tile_text))}
    </section>
    {missing_tile_notice}
    <section class="panel">
      <h2>Tile 缺陷熱度分布</h2>
      <p class="help">色彩越深代表該 tile 在這份 summary.csv 內累計的缺陷越多；空白位置不代表 PASS，因 summary.csv 不包含無缺陷 tile。</p>
      {heatmap}
    </section>
    <section class="split">
      <section class="panel"><h3>依 Detector</h3>{detector_rows}</section>
      <section class="panel"><h3>依缺陷類型</h3>{type_rows}</section>
    </section>
    <section class="panel">
      <div class="toolbar"><div><h2>各 Tile 明細</h2><p class="subtitle">依缺陷筆數由高至低排序；圖片數為至少出現一筆缺陷的不同 image_name 數量。</p></div><input id="tile-filter" type="search" placeholder="搜尋 tile、detector 或缺陷類型" aria-label="搜尋各 Tile 明細"></div>
      <div class="table-wrap"><table><thead><tr><th>Tile ID</th><th>缺陷筆數</th><th>受影響圖片數</th><th>主要缺陷類型</th><th>Detector 分布</th></tr></thead><tbody id="tile-rows">{tile_rows}</tbody></table></div>
    </section>
    <footer>本報表為離線 HTML，沒有上傳資料、沒有執行 AOI Detector；可直接用瀏覽器開啟或列印／另存 PDF。</footer>
  </main>
  <script>
    const filter = document.getElementById('tile-filter');
    const rows = [...document.querySelectorAll('#tile-rows tr')];
    filter.addEventListener('input', () => {{ const query = filter.value.trim().toLocaleLowerCase(); rows.forEach(row => {{ row.hidden = query && !row.textContent.toLocaleLowerCase().includes(query); }}); }});
  </script>
</body>
</html>
"""


def _metric_card(label: str, value: str) -> str:
    return f'<div class="card"><div class="label">{_escape(label)}</div><div class="value">{value}</div></div>'


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
