VisionFlow AOI 獨立小工具
=========================

內容
----

1. NG-Tile-Area-Tool.exe
   依 AOI 缺陷 CSV 面積區間分類並複製 NG Tile 圖片。

2. Pattern-Grid-Tile-Exporter.exe
   以 Pattern 模板錨點與固定網格批次輸出 PNG 小圖及座標清單。

3. Matrix-Summary-Exporter.exe
   整合資料夾內的矩陣 CSV，輸出 matrix_summary.csv。

4. Scatter-Plot-Exporter.exe
   從 AOI JSON 或 CSV 報告批次輸出散點圖 PNG。

5. Tile-Defect-Distribution-Exporter.exe
   讀取 AOI 輸出 csv/summary.csv，輸出內嵌 Plotly、可離線篩選的互動 HTML；
   包含 Tile 熱圖／Pareto、Detector／缺陷類型、圖片排行、面積／score、明細等分析。
   若同一輸出資料夾有 json/ 完整報告，會以全部 Tile 檢測為分母計算 Tile NG 率；
   缺少 json/ 時只呈現 summary.csv 能支持的缺陷貢獻統計，不推算 NG 率。

使用方式
--------

直接雙擊任一 EXE 可開啟圖形介面。五個工具也保留命令列模式；可在 PowerShell
執行「工具檔名.exe --help」查看參數。

相容性與安全
------------

- Windows x64
- CPU-only，不需要 NVIDIA GPU 或 CUDA DLL
- 不需要另外安裝 Python
- 工具不會執行 AOI Detector
- EXE 未進行程式碼簽章；Windows SmartScreen 可能顯示未知發行者
- 請保留原始資料備份，並先以少量資料確認輸出是否符合預期
