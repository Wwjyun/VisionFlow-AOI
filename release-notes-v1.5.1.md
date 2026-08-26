# VisionFlow AOI v1.5.1

這是 Windows x64 CUDA-enabled 發行版，包含完整 `VisionFlow AOI` PyInstaller 資料夾，以及針對同一 release commit 以 CUDA 13.3、MSVC x64 與 `sm_86` 重編的 `gpu/visionflow_cuda.dll`。

## 本版重點

- 延續 v1.5.0 的 `503-CS-SN-1` 與 `506-CS-SN-1`：固定一般二值化門檻 `200`、中心與四邊 MASK、多邊形偵測、面積尺寸限制，以及工程外參／管理內參分層。
- 新增 `401-CS-SN-1`：Gray → Adaptive Mean 一般二值化（設定 block `156`、有效 block `157`、C `-56`）→ 四邊 MASK → 輪廓；抓到即 NG、未抓到即 PASS。
- 內建十個傳統 CV Detector 與 YOLOX Detector。
- CUDA DLL 維持 ABI v1 與 detector-neutral primitive／linear plan／DAG／resident ROI 架構，不新增 detector 專屬 CUDA workflow。
- `gpu.mode=auto` 在 CUDA 不可用時可依設定完整回退 CPU；strict `gpu.mode=cuda` 要求 CUDA 成功，不會靜默 fallback。

## 使用方式

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`。請勿只複製 EXE；相鄰的 `_internal`、`recipes`、`models` 與 `gpu` 目錄都是執行所需內容。

本版 CUDA DLL 以 RTX 3090／compute capability 8.6 為驗證目標。其他 GPU 必須支援包內 `sm_86` binary；若不相容，請使用 v1.5.0 CPU-compatible 版，或將 Recipe 設為允許 CPU fallback。內附 tiny YOLOX model 只供軟體流程測試，不是 production 缺陷模型。

## 已知限制

- production 真實 PASS／NG 影像與正式門檻仍需依產品、光源、治具及 Recipe 在目標產線驗收。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。
- TensorRT、production YOLOX 權重與人工標註 acceptance set 尚未完成。
