# VisionFlow Utility Tools v1.1.0

這是 VisionFlow AOI 獨立工具合集的第二個版本。v1.1.0 在既有四支工具之外，新增 Tile 缺陷分布 HTML 報表工具；原有工具的操作方式維持不變。

## 新增功能

- `Tile-Defect-Distribution-Exporter.exe`：讀取 AOI 執行輸出中的 `csv/summary.csv`，依 `tile_id` 統計缺陷筆數、受影響圖片數、Detector 與缺陷類型。
- 報表為單一離線 HTML，包含統計總覽、`r####_c####` Tile 熱度網格、各 Tile 明細與文字搜尋。
- 報表可直接使用瀏覽器開啟、列印或另存 PDF，不會上傳資料，也不會執行 AOI Detector。
- `summary.csv` 只記錄缺陷，因此熱度圖空白位置不等同 PASS；報表不會推測 CSV 中不存在的 Tile 檢測結果。

## 內含工具

- `NG-Tile-Area-Tool.exe`：依缺陷 CSV 面積區間分類並複製 NG Tile 圖片。
- `Pattern-Grid-Tile-Exporter.exe`：以 Pattern 模板錨點與固定網格批次輸出小圖及座標清單。
- `Matrix-Summary-Exporter.exe`：整合資料夾內的矩陣 CSV。
- `Scatter-Plot-Exporter.exe`：從 AOI JSON 或 CSV 報告輸出散點圖 PNG。
- `Tile-Defect-Distribution-Exporter.exe`：從 AOI `csv/summary.csv` 輸出 Tile 缺陷分布 HTML。

## 使用方式

解壓縮 ZIP 後，直接雙擊任一 EXE 開啟圖形介面。工具也保留命令列模式，可用 `--help` 查看參數、`--version` 查看各工具版本。

Tile 缺陷分布工具可在圖形介面選擇 `summary.csv`；命令列範例如下：

```powershell
.\Tile-Defect-Distribution-Exporter.exe --input C:\AOI\output\csv\summary.csv
```

未指定 `--output` 時，報表會建立在該次 AOI 執行資料夾，檔名為 `tile_defect_distribution.html`。

## 相容性與驗證

- Windows 10／11 x64。
- CPU-only，不包含 CUDA DLL，不需要 NVIDIA GPU。
- 不執行 AOI Detector，不影響 VisionFlow AOI 主程式版本。
- 五支工具皆為可獨立執行的 one-file EXE，不需要另外安裝 Python。
- 本次只重建新增的 `Tile-Defect-Distribution-Exporter.exe`；其 SHA-256 為 `c9994dac65983b094fca28c81424a4fefbefe164c9b62f9bb482bf86f330551d`。
- 其餘四支未變更 EXE 直接沿用已發布的 v1.0.0；來源 ZIP 已從 GitHub Release 重新下載，並以大小 `195,812,209` bytes 與 SHA-256 `5ca76da7de38171c829c40c7a615eade8d4e3fd85c3fe0c4df0de980945ddfcf` 驗證。
- 五支 packaged GUI smoke 均為 exit 0；Tile 報表工具另以 3 筆缺陷、2 個 tile 的 summary.csv 實際產生 HTML 並核對內容。
- 專案完整 312 tests、compileall、CUDA source preflight、CLI synthetic PASS 與 GUI offscreen smoke 均通過。
- 發行 ZIP 大小為 `207,093,401` bytes，SHA-256 為 `62e4355835d8f0b44b01e4943b27558af9a486248e20f652836385198c8f4b5f`。
- EXE 尚未進行程式碼簽章，Windows SmartScreen 可能顯示「未知的發行者」。

請保留原始資料備份，並先用少量資料確認輸出符合預期。
