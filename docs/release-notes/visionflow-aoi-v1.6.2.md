# VisionFlow AOI v1.6.2

這是 Windows x64 CUDA-enabled 效能更新，包含完整 `VisionFlow AOI` PyInstaller 資料夾與以 CUDA 13.3、MSVC x64、`sm_86` 編譯的 `gpu/visionflow_cuda.dll`。

## GPU 改進

- `202-CS-SN-1` 的 morphology、排除區、connected components、面積／邊界過濾與 ring CNR 全部留在 resident GPU ROI，只下載每個候選的統計紀錄。
- component 像素、ring 背景收集與 float32 pairwise 葉節點改為平行計算；256×256 起 GPU 路徑已穩定較快，12000×2000 ROI 的 ring 路徑為 10.43×。
- 24-bit BMP 使用最多 8 workers 平行分段讀取。GPU resident 模式保留 BMP bottom-up 檔案列序，整圖一次上傳後在 GPU 翻列，省略 CPU 對 639 MB 影像的全圖翻轉。
- 新增 optional `vf_context_upload_u8_file_order`；舊 DLL 缺少 export 時會安全回退既有連續影像上傳，ABI v1 不變。
- build provenance 在 process 內快取，冷路徑合併成一次 Git 查詢，連續檢測不再為每張圖重跑 Git subprocess。

## RTX 3090 正式尺寸基準

16384×13000 BMP、一列 6 個高 12000×寬 2000 ROI、`202-CS-SN-1`，warm-up 1＋量測 3 輪：

| 範圍 | CPU median／P95 | GPU median／P95 | 倍數 |
| --- | ---: | ---: | ---: |
| 端到端 | 5453.3／5684.8 ms | **397.7／418.9 ms** | **13.71×** |
| Detector | 5147.0／5399.7 ms | **86.3／86.9 ms** | 59.66× |
| Tiling | 67.0／67.5 ms | **2.45／2.51 ms** | 27.30× |
| 影像讀取 | 175.8／190.8 ms | **102.6／116.9 ms** | 1.71× |

每張圖只有一次 638,976,000-byte H2D；六次候選 export 與一次 anchor match 合計 D2H 22,340 bytes，共 8 次 native calls。3/3 輪 PASS/NG、tile／defect 數量、type、bbox、area、confidence 與 decision metadata 完全相同。

## 相容性與限制

- CUDA DLL 以 RTX 3090／compute capability 8.6 驗證；其他 GPU 必須支援包內 `sm_86` binary。沒有相容 NVIDIA GPU 時可選「僅 CPU」或「GPU 優先，失敗改用 CPU」。
- BMP 快速路徑支援未壓縮 24-bit 與 8-bit palette、top-down／bottom-up；其他格式與不支援的 BMP 變體回退 OpenCV。
- 效能數據來自固定 seed 合成影像；真實產線影像與判定門檻仍需在目標產線驗收。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。
- TensorRT、production YOLOX 權重與人工標註 acceptance set 尚未完成。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
