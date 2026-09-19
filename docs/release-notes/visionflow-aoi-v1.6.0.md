# VisionFlow AOI v1.6.0

這是 Windows x64 CUDA-enabled 發行版，包含完整 `VisionFlow AOI` PyInstaller 資料夾，以及針對同一 release commit 以 CUDA 13.3、MSVC x64 與 `sm_86` 重編的 `gpu/visionflow_cuda.dll`。

## 本版重點：新 GPU mode

- 影像仍在 CPU 解碼；GPU 模式下整張圖只做一次 resident upload，之後 ROI 產生與前處理直接在 device 上執行，不再逐 tile 複製 CPU 影像。Template Anchor Grid 定位在本版完整 pipeline 中仍由 CPU 執行（已知接線問題，見下方已知限制）。
- 新增 `vf_cnr_mask_u8_roi`：`202-CS-SN-1` 的 automatic CNR（BGR→gray、float32 Gaussian、residual、exact median/MAD、threshold、candidate mask）直接讀 resident ROI，只下載 mask 與三個診斷純量。
- 新增 bit-exact GPU float32 median（cub radix sort）、支援 sigma 的 float32 Gaussian、融合 residual mask export、GPU Anchor Grid 定位 export（本版 pipeline 尚未實際啟用），以及與 OpenCV 逐點等價的 `RETR_LIST` contour 追蹤（目前未接入 Detector，因稀疏影像仍慢於 CPU）。
- 依實測 crossover 將低成本前處理留在 CPU；每個 pipeline 步驟的實際 device/host 分工改由執行期呼叫紀錄回報，而非由 Recipe 請求推論。
- CUDA sticky context error、kernel error 與 OOM 後可乾淨復原或停用 CUDA 至重啟；`Resize(area)` 與 OpenCV `INTER_AREA` 完全一致。
- 其他：可選的平行 CPU crop、大圖直接以 OpenCV 解碼、monitor 端到端計時修正、202 CNR 平手改以 bounding box 決定排序。

## RTX 3090 效能與等價

正式尺寸：16384×13000 原圖、一列 6 個高 12000×寬 2000 ROI、`202-CS-SN-1`，warm-up 1 次、量測 3 次（median）。

| 範圍 | CPU (ms) | GPU (ms) | 倍數 |
| --- | ---: | ---: | ---: |
| 完整 pipeline | 6792.8 | 2011.1 | 3.38× |
| detectors_total | 5649.8 | 859.2 | 6.58× |
| automatic CNR mask | 5260.7 | 477.0 | 11.03× |
| connected components + CNR | 292.5 | 278.8 | 1.05× |
| tiling | 135.6 | 58.9 | 2.30× |

- 3/3 輪 PASS/NG、defect 數、bbox、area、confidence 與 metadata 判定欄位完全一致；resident CNR 15/15 與既有 GPU chain 逐位元相同、15/15 candidate mask 與 CPU 相同。
- connected components 與 ring CNR 仍在 CPU；GPU CCL 原型在完整 pipeline 僅改善約 0.4% 且傳輸更多，已撤回。完整量測與後續條件見 `gpu/README.md`。

## 使用方式

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`。請勿只複製 EXE；相鄰的 `_internal`、`recipes`、`models` 與 `gpu` 目錄都是執行所需內容。

本版 CUDA DLL 以 RTX 3090／compute capability 8.6 為驗證目標。其他 GPU 必須支援包內 `sm_86` binary；若不相容，請將 Recipe 設為 `gpu.mode: cpu` 或允許 CPU fallback。`gpu.mode=auto` 在 CUDA 不可用時完整回退 CPU；strict `gpu.mode=cuda` 要求 CUDA 成功，不會靜默 fallback。內附 tiny YOLOX model 只供軟體流程測試，不是 production 缺陷模型。

## 已知限制

- 整圖已上傳 GPU 時，pipeline 傳給切圖器的 runtime 為空，GPU Anchor 定位不會執行而改用 CPU（結果正確、未加速）；執行結果的 `device_host_split.anchor_localization` 仍會誤報為 `device`。將於後續版本修正。
- 效能數據來自合成的正式尺寸影像；production 真實 PASS／NG 影像與門檻仍需在目標產線驗收。
- GPU 預設啟用仍需產線影像的等價、穩定性與端到端效能證據。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。
- TensorRT、production YOLOX 權重與人工標註 acceptance set 尚未完成。
