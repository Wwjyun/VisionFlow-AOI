# 獨立工具原始碼

此資料夾集中五支可單獨執行、也可打包成 one-file EXE 的後處理／切圖工具：

- `export_ng_tiles_by_area.py`：依缺陷面積分類 NG Tile。
- `export_pattern_grid_tiles.py`：使用 Pattern Anchor Grid 批量切圖。
- `export_matrix_summary.py`：彙整矩陣 CSV。
- `export_scatter_plots.py`：由 JSON／CSV 匯出散點圖。
- `export_tile_defect_distribution.py`：由 AOI `csv/summary.csv` 匯出各 tile 缺陷分布 HTML 報表。

從 repository 根目錄使用模組方式執行：

```powershell
.\env\Scripts\python.exe -m tools.export_ng_tiles_by_area
.\env\Scripts\python.exe -m tools.export_pattern_grid_tiles
.\env\Scripts\python.exe -m tools.export_matrix_summary
.\env\Scripts\python.exe -m tools.export_scatter_plots
.\env\Scripts\python.exe -m tools.export_tile_defect_distribution
```

個別 PyInstaller 建置仍由根目錄的 `build_*` 腳本負責；五支工具的合集 ZIP 由 `build_utility_tools.ps1` 建立在 `release_artifacts/`。
