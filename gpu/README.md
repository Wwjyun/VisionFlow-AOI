# VisionFlow CUDA DLL

`visionflow_cuda.dll` 是 VisionFlow AOI 的可選 CUDA backend。CPU executor 是 OpenCV
正確性基準；CUDA 必須保留 recipe、PASS/NG、座標、defect metadata 與輸出語意。
`gpu.mode: auto` 可在 CUDA 失敗時整個 detector 回到 CPU，`gpu.mode: cuda` 則禁止
隱藏 fallback。

## 架構

Detector 以 `core/preprocess_plan.py` 的 backend-neutral operators 描述前處理。
CUDA backend 支援：

- ABI v1 stateless primitives。
- Persistent context、grow-only buffers 與 non-blocking stream。
- Detector-neutral linear `VfPlanDescV1`。
- Shared-gray/multi-output `VfDagPlanDescV1`。
- Resident image/ROI 與 coordinate ROI batch。
- `VfCudaTimingsV1` CUDA event 分項。
- 舊版 `vf_preprocess_401_2_u8` compatibility adapter。

Gaussian kernel 3/5/7/9 使用與 OpenCV 相同的固定係數，其他 kernel 使用 OpenCV
自動 sigma 規則；全部 kernel 都以相同的 8-bit fixed-point 誤差擴散與兩階段
rounding 執行。BGR→Gray 使用 OpenCV 8-bit 路徑相同的 15-bit BT.601 fixed-point
係數，避免 ±1 灰階差異在 threshold 後放大成 binary mask 差異。

`Resize(area)` 只支援兩軸不放大的單通道縮小，並逐分支重現 OpenCV `INTER_AREA`：相同尺寸
copy、2×2 `(sum+2)>>2`、其他整數倍 float 平均，以及非整數倍的 `computeResizeAreaTab`
權重表與 half-even 捨入；權重表在 plan create 時上傳一次。`cuda_project.json` 的
`nvcc.fmad: false` 讓 build script 傳入 `--fmad=false`，避免 GPU 將乘加融合成 FMA 而破壞
float 累加的逐像素一致性，不可移除。

`gpu.mode: auto`（允許 CPU fallback）時，`PlanCrossoverPolicy` 會對每個前處理 plan 與輸入
尺寸實測數次 CUDA 與 CPU 後凍結較快的後端；小 tile 或便宜 operator 可能改走逐像素相同的 CPU
plan，tile metadata 以 `cpu_crossover` 路線與 `preprocess_routes` 標示。`gpu.mode: cuda`
不啟用此路由。

## 已完成並接入產線、以及尚未接入的步驟

以 RTX 3090 完成等價量測後才可接入產線；未接入者在 `execution.gpu.device_host_split` 中
一律回報為 cpu。**接入與否以本節與 `Todo.md` 為準，不以 export 存在為準。**

- `vf_match_template_gray_u8`（Template Anchor Grid 定位）：**已接入** `core/tiler.py`。
  與 `cv2.matchTemplate` 的定位座標在 9 個場景 9/9 相同、分數差 ≤ 4.2e-7、逐次執行決定性。
  形狀界線內（template 每邊 ≤ 128 px 且搜尋面積 ≥ 256×256）比 CPU 快 1.6～3.9 倍，
  界線外或失敗時回 CPU 參考；界線見 `core/tiler.py` 的 `gpu_anchor_shapes_supported`。
- `vf_find_contours_u8` / `vf_find_contours_download`（輪廓抽取）：**正確但全面較慢，
  已判定不接入產線**。與 `cv2.findContours(RETR_LIST/RETR_EXTERNAL,
  CHAIN_APPROX_SIMPLE)` 在 `tools/check_contour_equivalence.py` 的 **314 個案例全部逐點
  相同且決定性**（含輪廓數、每條 shape、點順序、子區域座標契約），但**在每一個量測形狀都
  慢於 cv2**（GPU／CPU 毫秒）：512×512 稀疏 0.24→1.06（0.22×）、512×512 密集
  0.23→3.91（0.06×）、2048×2048 密集 3.04→23.17（0.13×）、2000×12000 稀疏
  13.77→34.16（0.40×）、中型 18.63→136.54（0.14×）、大型 26.97→261.85（0.10×）、
  `RETR_EXTERNAL` 13.83→1234.75（0.01×）。**沒有任何形狀勝出，故不訂啟用界線、維持停用**；
  此結論已更正先前的錯誤記載，日後若要重啟需先提出新的演算法（例如 tile 化後只下載輪廓點，
  而非整張 label 影像）。
- `vf_median_f32`（float32 精確中位數，供 202-CS-SN-1 的 median／MAD 使用）：**已接入
  `detectors/detector_202_1.py` 的 `_exact_median`**（`gpu.mode: cuda`／支援時才走 GPU，
  否則整個呼叫回 `np.median`）。除以 float32 轉 monotone uint32 key 後用
  `cub::DeviceRadixSort` 排序、只讀回中間 1～2 個 key，再於 host 以 float32 平均，
  因此與 `np.median(float32)` **位元相同**：34/34 案例一致（其中 32 個連排序後的中間值
  圖樣也相同）、NaN 6/6、不改動輸入、決定性。4M float32 為 GPU 2.92～6.53 ms、
  CPU 34.5～36.3 ms（**5.6～11.8×**）；202 端到端 PASS/NG、缺陷數、bbox、area、
  confidence、metadata **完全相同**，305.5 → 188.2 ms（**1.62×**）。median 未接線前
  約佔該 Detector 75% 時間，是該 Detector GPU 化的首要目標，現已完成。
- `vf_gaussian_blur_f32` / `vf_gaussian_blur_f32_roi`（float32 Gaussian，供 202-CS-SN-1 的
  CNR 背景）：**已接入** `detectors/detector_202_1.py` 的 `_background_blur`，需同時具備
  `supports_gaussian_blur_f32` 與 `supports_gaussian_f32_sigma`；缺 export、舊版 DLL 或任何
  device 錯誤都整段回 `cv2.GaussianBlur`。sigma 語意與 OpenCV 相同（`sigma>0` 直接使用、
  `sigma<=0` 走自動規則、NaN／±inf 拒絕），係數對 `cv2.getGaussianKernel` 在 441 組
  (ksize, sigma) **位元相同**；1008 個 sigma 掃描案例 worst `max|diff|` 7.629e-05
  （claimed 4.0e-04）、`mean|diff|` 3.052e-05（claimed 5.0e-05）。
  **語意差異（必須揭露）**：device 的加法順序與 OpenCV 不同，因此 `residual` 尾位漂移，
  `mad`／`residual_median`／`residual_threshold`／`robust_noise_sigma` 四個診斷值會有
  約 1e-5 的差異（202 矩陣最大 3.4e-05），其餘 40 個 metadata 欄位完全相同。
  **判定不受影響**：47 個場景的 PASS/NG、缺陷數、結構欄位皆相同，且**候選遮罩 47/47 位元相同**。
  報表會以 `metadata.background_backend`（`opencv_cpu`／`cuda_f32`）與
  `metadata.background_precision_note` 明確標示該次執行用的是哪一條路徑。
  **效能**：單獨呼叫沒有收益（4000×2000 ksize=51 為 13.5 vs 13.8 ms，H2D＋D2H 佔 96%），
  但 **kernel 本體只有 0.606 ms**；接線後 202 的 `automatic_cnr_mask` 836.6 → 438.8 ms，
  產線形狀端到端 1337 → 1019 ms（**1.31×**）。真正的大幅收益需把 residual 留在 device。
- **connected components 與 ring CNR 統計**：**尚未 GPU 化**。`tools/connected_components_reference.py`
  已可與 `cv2.connectedComponentsWithStats` 在結構化遮罩上完全一致（含 label 0 描述背景像素、
  無像素標籤的 sentinel／NaN centroid 語意），但**隨機遮罩的標籤編號順序仍不同**；
  在該缺口修正前，此參考不得作為 GPU 化的黃金標準（細節與已排除的假設見 `Todo.md`）。
  ring CNR 的背景 mean/std 目前以 NumPy 在 host 計算，會與 GPU 化後的加法順序有
  ULP 級差異，判定邊界案例必須另外量測。

`vf_match_template_debug_*` 與 `vf_find_contours_*` 的下載介面只供等價驗證與診斷使用，
不屬於產線路徑。

## 檔案

```text
gpu/
├── include/
│   ├── visionflow_cuda.h
│   ├── visionflow_cuda_errors.h
│   └── visionflow_cuda_internal.cuh
├── cuda_project.json          # 明確分離 DLL 與 test source manifest
├── visionflow_cuda.cu
├── test_cuda_api.cu
├── preflight_cuda_build.py
├── validate_cuda_dll.py
├── validate_cuda_fault_injection.py
└── build_cuda_dll.ps1
```

GitHub hosted runner 只能編譯、檢查 exports/dependencies，沒有 NVIDIA GPU 時不能宣稱
通過 runtime validation。下載 artifact 時只部署核准的 DLL/LIB/EXE 與 evidence
manifest，不得用 standalone Action 專案版本覆蓋 repository 內的 build、preflight、
validator 或 profiler。

## RTX 3090 本機編譯

在 Visual Studio x64 Native Tools PowerShell 執行：

```powershell
.\gpu\build_cuda_dll.ps1 -Architecture sm_86
```

建置流程會：

1. 執行 header/source/runtime/smoke preflight。
2. 依 `cuda_project.json` 分開編譯 DLL 與測試 EXE，不使用 `*.cu` glob。
3. 在 `outputs_validation/cuda_build_stage/` 產生 staging artifacts。
4. 通過 `dumpbin /exports` 與 `/dependents` 後才發布至 `gpu/`。
5. 保存 source manifest、exports、dependencies 及
   `cuda_build_evidence.json`（工具版本、commit、binary SHA-256）。

正式二進位產物不納入 Git。

## RTX runtime 驗證

```powershell
.\gpu\test_cuda_api.exe

.\env\Scripts\python.exe gpu\validate_cuda_dll.py `
  --dll gpu\visionflow_cuda.dll `
  --warmup 5 `
  --benchmark 20 `
  --crossover `
  --morphology-profile `
  --stress 10 100 1000 `
  --resize-area-pipeline `
  --json-output outputs_validation\rtx3090_benchmark.json
```

`--roi-batch-matrix` 在 16384×13000 resident 原圖上以 256²／512²／1024² ROI 測 batch
8／16／32／64 的全像素正確性、建立與下載時間及 VRAM 回收，並以 66 個 2000×12000 ROI 驗證依可用
記憶體自動分批。

`--resize-area-pipeline` 以正式 `PRODUCT_A_CIRCLE_401_1_AOI_01.yaml` 在多個
`process_scale` 下比對合成 PASS／NG 圖的完整 CPU/GPU Pipeline。

正式 validator 覆蓋 structured/non-contiguous primitives、linear/DAG plan、resident
ROI、coordinate batches、context reuse、4K benchmark 與 persistent-plan stress。
五份 production recipe 的 PASS/NG acceptance 仍需提供可追溯真實樣本 manifest。

實機故障注入不使用 fake DLL：

```powershell
.\env\Scripts\python.exe gpu\validate_cuda_fault_injection.py `
  --dll gpu\visionflow_cuda.dll `
  --vram-pressure `
  --json-output outputs_validation\fault_injection\report.json
```

- `init_failure`：子程序設定 `CUDA_VISIBLE_DEVICES=-1`，確認 Detector／`gpu.mode: auto`
  Pipeline 與 CPU 完全一致且零 CUDA 呼叫，`gpu.mode: cuda` 明確失敗。
- `kernel_launch_error`：1 像素寬、高度超過 `65535 × 16` 列的影像使 kernel grid
  無效，真實 launch 失敗後整顆 Detector CPU 重跑；同一 runtime／session 的下一張圖
  必須恢復 CUDA 且不得沿用上一張的 fallback 狀態。
- `device_oom`：配置超過專用＋共用 GPU 記憶體的 ROI batch 取得真實 OOM，之後同一
  context 的小批次與 resident plan 必須立即成功並與 CPU 相同。
- `sticky_context`：隔離子程序以 NVRTC 編譯故意越界寫入的 kernel，產生真實 CUDA 700
  illegal address。runtime 收到 sticky 錯誤碼（214、220、226、700、702、709、710、714～719）
  後標記 CUDA context 損毀：該次 Detector 整顆 CPU 重跑，之後同一程序不再呼叫 CUDA，
  `gpu.mode: auto` 回報需重新啟動的原因，`gpu.mode: cuda` 明確失敗。
- `--vram-pressure`：另一個程序佔住可用專用 VRAM。Windows 驅動預設的 CUDA sysmem
  fallback 會讓配置溢出到共用記憶體而非回傳 OOM，因此此項驗證結果等價與時間變化，
  不代表 OOM 失敗路徑。

高度介於 1,048,561～1,048,576 列的影像會觸發上述 kernel grid 限制並由 CPU fallback
處理；OpenCV 預設讀圖上限為 1,048,576 列。

完整驗收進度以 [`Todo.md`](../Todo.md) 為準。
