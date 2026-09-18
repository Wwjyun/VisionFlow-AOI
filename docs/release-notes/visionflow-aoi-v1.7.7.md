# VisionFlow AOI v1.7.7

這是 Windows x64 CUDA-enabled 修正版，包含完整 `VisionFlow AOI` PyInstaller 資料夾與以 CUDA 13.3、MSVC x64、`sm_86` 編譯的 `gpu/visionflow_cuda.dll`（沿用 v1.6.3 起同一個 DLL，未重新編譯）。

檢測流程、Detector、Recipe 語意與 PASS/NG 判定皆未變更，CUDA source／header／ABI 未修改。本版修正相機機台第六次回報的 `060601 070702`，當時 CamExpert 顯示 `AcquisitionLineRate` 為 n/a。

## 修正：連續模式先讓相機回到 free-run，再寫線速率

相機停在外部觸發（attached camera 的 Trigger mode = On）時，`AcquisitionLineRate` 不可用，CamExpert 顯示 n/a；相機也會一直等 CC1 的觸發脈衝。前一版在連續模式下先寫線速率、後把 TriggerMode 切回 Off（與參考程式同順序），所以寫入當下線速率仍是 n/a。

- 連續模式改為**先把相機 TriggerMode 切回 Off 並送出**，再寫線速率；線速率仍在曝光、增益之前寫。
- 切不回 Off 時回報新碼 **`E-0609`**，報告附上 TriggerMode 讀回值。以前這個失敗只記成備註，不會顯示。

## 修正：S7 等待上限依影像長度計算

30 Hz 掃 720 線要 24 秒，舊的固定 5 秒上限必定逾時。S7 現在以「長度 ÷ 線速率」算出一張影像的時間，等 1.5 倍再加 2 秒，最少 5 秒、最多 60 秒，並在報告寫出實際上限。

## 現場操作

1. **先把 CCD 頁的線速率改成實際使用的值**（例如 5000 Hz）。30 Hz 對 Linea 16K 太低，掃一張 720 線的影像要 24 秒。
2. 相機按「斷線」（米輪可保持連線）。
3. 「Sapera 診斷」→「執行相機診斷」，把數字短碼**整列**抄回。
4. 若出現 `0609`：在 CamExpert 的 attached camera → I/O controls 把 Trigger mode 設為 Off、板卡 Line Sync Source 設為 None，再跑一次。

## 相容性與限制

- 真實 Linea Mono 16K 在 TriggerMode Off 後的線速率寫入與 S7 取像仍待相機機台確認。
- CUDA DLL 與 v1.7.6 相同，其他 GPU／無 GPU 電腦驗收仍待目標環境。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
