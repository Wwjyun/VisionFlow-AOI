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
- `--vram-pressure`：另一個程序佔住可用專用 VRAM。Windows 驅動預設的 CUDA sysmem
  fallback 會讓配置溢出到共用記憶體而非回傳 OOM，因此此項驗證結果等價與時間變化，
  不代表 OOM 失敗路徑。

高度介於 1,048,561～1,048,576 列的影像會觸發上述 kernel grid 限制並由 CPU fallback
處理；OpenCV 預設讀圖上限為 1,048,576 列。

完整驗收進度以 [`Todo.md`](../Todo.md) 為準。
