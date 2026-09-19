# VisionFlow AOI v1.6.1

這是 Windows x64 CUDA-enabled 修正版，包含完整 `VisionFlow AOI` PyInstaller 資料夾與 `gpu/visionflow_cuda.dll`。v1.6.0 之後 CUDA 原始碼沒有變動，DLL 與 v1.6.0 同為 CUDA 13.3、MSVC x64、`sm_86` 編譯並在 RTX 3090 驗證的同一份二進位檔。

## 修正

- **GPU 切圖變慢**：GPU mode 與「切小圖使用 GPU」開啟、但沒有 Detector 開啟 GPU（或切圖模式非 grid）時，v1.6.0 會對每張 tile 重傳整張原圖，GPU 反而比 CPU 慢。RTX 3090 正式尺寸實測切圖由 871.7 ms 恢復為 84.9 ms。「GPU 優先」模式下改用 CPU 切圖，並在結果的 `execution.gpu.tiling.reason` 說明；「僅 GPU（嚴格）」維持 CUDA 裁切。
- **GPU Anchor 定位實際未執行**：v1.6.0 在整圖已上傳 GPU 時，Template Anchor Grid 定位仍走 CPU。現已在 GPU 上執行，strict CUDA 下失敗會直接報錯；CPU 參考只轉換搜尋區灰階。正式尺寸 anchor 定位 CPU 58.8 → 6.95 ms、GPU 3.29 ms，tile 座標與 CPU 完全相同。
- **執行位置回報**：`device_host_split.anchor_localization` 改依實際執行的 backend 回報，不再只因整圖上傳就標示為 GPU。

## 新功能

- **GPU 預熱**：「檢測控制」面板新增「GPU 預熱」按鈕，建立 CUDA context 並以目前影像試跑一次（不輸出任何檔案），讓第一張檢測不承擔初始化成本。RTX 3090 正式尺寸第一張由 2075.5 ms 降為 1859.2 ms。
- **運算後端選擇重新設計**：Recipe Designer 改為「僅 CPU」「GPU 優先，失敗改用 CPU」「僅 GPU（嚴格）」三個含行為說明的選項，取代原本的下拉選單加回退開關。狀態列會顯示 CUDA 是否可用、有幾個 Detector 開啟 GPU，以及「切小圖使用 GPU」無效的組合。舊 Recipe 設定未變更時原值保存。

## RTX 3090 正式尺寸基準

16384×13000 合成圖、一列 6 個高 12000×寬 2000 ROI、`202-CS-SN-1`，warm-up 1＋量測 3 輪（median／P95）：

| 範圍 | CPU | GPU | 倍數 |
| --- | ---: | ---: | ---: |
| 端到端 | 6315.7／6524.9 ms | 1957.5／2112.7 ms | 3.23× |
| Anchor 定位 | 6.95 ms | 3.29 ms | 2.11× |
| Tiling 合計 | 90.2 ms | 3.4 ms | 26.2× |

CPU 端也因搜尋區灰階而變快，所以倍數比 v1.6.0 的 3.38× 略低；GPU 端到端本身比 v1.6.0 快約 54 ms。3/3 輪 PASS/NG、defect、tile 座標與 `match_bbox` 完全相同。

## 使用方式

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。CUDA DLL 以 RTX 3090／compute capability 8.6 為驗證目標；其他 GPU 必須支援包內 `sm_86` binary，否則請選「僅 CPU」或「GPU 優先，失敗改用 CPU」。內附 tiny YOLOX model 只供軟體流程測試。

若 GPU mode 下仍比 CPU 慢，請確認需要加速的 Detector 已在 Detector 清單開啟 GPU，並提供 `outputs\logs\aoi.log` 中該次檢測的 `Inspection performance` 與 `CUDA host metrics` 兩行。

## 已知限制

- connected components 與 ring CNR 仍在 CPU；每輪會下載 gray 與候選遮罩各 144 MB（6 ROI）。
- `device_host_split` 其餘步驟與 `execution.gpu.metrics` 在 GUI／批次共用 session 時為累計值，可能把先前影像的 GPU 呼叫算入本次。
- 在 Designer 儲存 Recipe 後 GPU session 會重建，需要重新預熱；OP 模式沒有預熱按鈕。
- 效能數據來自合成影像；真實產線影像與門檻仍需在目標產線驗收。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。
- TensorRT、production YOLOX 權重與人工標註 acceptance set 尚未完成。
