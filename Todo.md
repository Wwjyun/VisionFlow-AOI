# VisionFlow AOI 統一開發清單

本文件是專案唯一的工作清單，涵蓋 CPU、GPU、CUDA、Detector、GUI、打包、CI 與實機驗收。完成程式修改時必須同步更新對應 checkbox；不得再建立分散的 CPU/GPU Todo 文件。

- [x] 2026-08-18：建立 Traditional CV Tuning Tool v1.0.0 獨立 Windows 發佈契約：加入單一來源版本、PyInstaller-safe launcher、Windows version resource、one-file GUI spec、專用 build script、完整解析度 packaged smoke、README 與 release notes；使用 `cv-tuning-tool-vX.Y.Z` Tag 與 `Traditional-CV-Tuning-Tool-vX.Y.Z-windows-x64.zip`，不混用 AOI 主程式或 Utility Tools 版本；成品為 CPU/OpenCV 處理、Qt OpenGL 顯示／raster fallback，且不含 CUDA DLL。

## 開發原則

- CPU 路徑是正確性基準，也是無 NVIDIA GPU、DLL 載入失敗、CUDA error 或顯存不足時的 fallback。
- GPU 最佳化不得改變 recipe 語意、PASS/NG、座標與輸出格式；允許差異必須先定義容差並加入測試。
- 不追求所有工作 GPU 化。YAML、彙總、少量 contour 幾何、GUI 控制、CSV/JSON、PNG 編碼與磁碟 I/O 預設留在 CPU。
- Detector 不得各自建立一套 CUDA workflow。Detector 只宣告 backend-neutral `PreprocessPlan`，由 CPU/CUDA executor 執行共用 operators。
- GPU 路徑應盡量一次 upload、連續執行多個 operators、最後只 download 必要 mask 或統計值。
- 新功能必須保持 OOP 邊界、CPU-only 可啟動、舊 DLL 相容與完整 detector CPU fallback。
- 只有 RTX 3090 實測通過數值等價、穩定性與端到端效能門檻的功能，才能預設啟用 GPU。

## 目前狀態摘要

- [x] CUDA DLL 已在 RTX 3090 編譯，並在另一台電腦確認可載入及顯示 CUDA active。
- [x] 已確認 CUDA active 不代表整條 AOI pipeline 都在 GPU；首次跨機測試端到端沒有加速。
- [x] 已有 CPU-only、缺 DLL fallback、GPU 呼叫統計及 detector 整體 CPU 重跑機制。
- [x] Gaussian 已改 separable kernels；Adaptive Mean 已改 64-bit integral image。
- [x] 已有 persistent context、grow-only buffers 與 401-2 fused preprocessing 原型。
- [x] 已建立通用 `PreprocessPlan`、typed operators、CPU/CUDA executors，401-2 已完成第一階段遷移。
- [x] 目前開發機已具備 RTX 3090（Driver 610.62）、CUDA 13.3 `nvcc` 與 Visual Studio 18 x64 工具，可經 `vcvars64.bat` 執行 `gpu/build_cuda_dll.ps1`（建置不需 CMake）；新增 CUDA 原始碼仍須每次重編並跑 native smoke／validator 後才可宣稱通過。
- [ ] 尚未完成固定 production 測試集、五個 recipes 全流程等價、長時間壓測與可信的 CPU/GPU benchmark。

## P0：正確性、CPU 基準與觀測能力

### Pipeline 與 profiler

- [x] 記錄 recipe setup、image load、tiling、各 detector、aggregation、reporting 與 end-to-end host wall time。
- [x] Reporter 分別記錄 overlay、NG tiles、CSV、matrix CSV 與 JSON 耗時。
- [x] 記錄 DLL load、同步呼叫、lock wait、估算 H2D/D2H bytes、round trips 與各 primitive 呼叫統計。
- [x] 加入 GUI 顯示、QImage/QPixmap 轉換與使用者實際等待時間計時。
- [x] DLL 加入 CUDA event，拆分 context、allocation、H2D、device copy、kernel、synchronize、D2H 與 free；RTX 數值驗證仍列在實機驗收清單。
- [x] benchmark JSON 保存 CPU、GPU、RAM、Driver、recipe、影像資訊與 commit hash；Toolkit 另由 runner environment artifact 保存。
- [ ] 在 RTX 3090 固定 production 測試集執行並建立可重現 baseline。（workflow_dispatch 已支援可選 production manifest；待真實樣本與 runner）
- [x] benchmark 分開記錄 cold、warm-up 次數、純檢測與既有 pipeline/report 端到端數據。
- [x] benchmark 記錄平均、median、P95、process CPU%、GPU utilization、VRAM、溫度與功耗快照。

### CPU 與 fallback 正確性

- [x] 缺少 DLL 時，CPU fallback 與純 CPU 的 PASS/NG、tiles、defects、bbox 與 metadata 完整一致。
- [x] fused GPU 呼叫失敗時不採用部分結果，整個 detector 重新從 CPU preprocess 開始執行。
- [x] 建立固定 random seed 合成測例：BGR、gray、全黑、全白、棋盤格與邊界像素。
- [ ] 補入固定真實 AOI 影像測例；manifest schema、路徑/標籤/coverage 驗證已完成，待取得可追蹤的生產樣本後執行。
- [x] 覆蓋奇數尺寸、極小圖、4K、non-contiguous stride、1/3 channels 與不同 ROI 尺寸。
- [ ] 五個 production recipes 各準備至少一張 PASS 與一張 NG 樣本；`gpu/production_manifest.example.yaml` 已固定所需 10 個 case，影像待提供。
- [x] 實機注入 kernel error、CUDA 初始化失敗與 OOM，確認 fallback 後無 stale pointer 或錯誤中間結果。（2026-09-14 RTX 3090 以 `gpu/validate_cuda_fault_injection.py` 完成：`CUDA_VISIBLE_DEVICES=-1`、超過 kernel grid 上限的真實 launch error、超過專用＋共用 GPU 記憶體的 ROI batch OOM；Detector／Pipeline 與 CPU 完全一致，同一 runtime／session 下一張圖恢復 CUDA。sticky context error 另列 P2 待辦）
- [x] `fallback_to_cpu: false` 且 CUDA DLL 不可用時必須明確失敗，不可回報假的 GPU success。

## P1：共用 Preprocess Plan 架構

### Python/OOP execution layer

- [x] 建立 immutable `PreprocessPlan`。
- [x] 建立 typed operators：Gray、Resize、Gaussian、Threshold、AdaptiveMean、Morphology。
- [x] 建立 `CpuPreprocessExecutor`，OpenCV 結果作為 fallback 與等價基準。
- [x] 建立 `CudaPreprocessExecutor`，支援 stateless primitives 與既有 401-2 fused compatibility adapter。
- [x] `BaseDetector.execute_preprocess_plan()` 統一選擇 executor。
- [x] 401-2 改為宣告 Gray → Gaussian → AdaptiveMean，不再直接呼叫 CUDA export。
- [x] CUDA 尚不能保持 `INTER_AREA` 語意時拒絕執行 Resize(area)，不可靜默改用 nearest-neighbor。
- [x] 將 plan 建立移出每次 tile 熱路徑，依 detector params、dtype 與 shape cache immutable plan，並以 bounded LRU 避免無限成長。
- [x] 加入 versioned operator/plan signature、輸入輸出 uint8 型別、channel、shape、順序與 operator 參數 validation。
- [x] 加入 versioned capability report，清楚列出 plan 為何走 fused、primitive、CPU 或 fallback，並寫入 detector execution metadata。
- [x] capability preflight 判定整份 plan/DAG 不支援 CUDA 時，在任何 primitive 執行前直接走完整 CPU fallback；關閉 fallback 時明確失敗。

### Generic native plan ABI

- [x] 定義 versioned C structs：operator kind、input node、參數、output node；不得包含 detector ID/name。
- [x] 新增 optional `vf_plan_create/execute/destroy` exports；ABI v1 primitives 與 401-2 adapter 保持相容。
- [x] plan create 階段驗證 operators、channel、shape、參數與輸出；execute 階段只處理資料。
- [x] 將 Gray、Gaussian、AdaptiveMean、Threshold、Morphology 接入通用 native executor。
- [x] 通用 native plan 達成一次 H2D、連續 kernels、最後一次必要 D2H。
- [x] 加入 plan capability query；任一 operator 不支援時整份 plan CPU fallback，避免反覆 CPU/GPU 傳輸。
- [x] 完成與 OpenCV 等價的 `INTER_AREA` resize 驗收後，才將 CUDA Resize(area) 視為 production ready。（2026-09-14 CUDA 改為逐分支重現 OpenCV：相同尺寸 copy、2×2 `(sum+2)>>2`、整數倍 float 平均、非整數倍 `computeResizeAreaTab` 權重表與 half-even 捨入，DLL 以 `--fmad=false` 建置；RTX 3090 隨機／邊界矩陣逐像素 0 差異，正式 `PRODUCT_A_CIRCLE_401_1_AOI_01.yaml` 在 `process_scale` 1.0／0.5／0.37／0.25 的合成 PASS／NG 全 Pipeline CPU/GPU 完全一致。真實樣本 PASS／NG 仍列於 RTX 驗收區。）
- [x] Python/CPU plan 擴充 topologically ordered DAG/multi-output，支援一份 gray 產生多張 masks。
- [x] CUDA/native plan 擴充 DAG/multi-output，讓 device gray 直接產生多張 masks。

## P2：CUDA kernels 與資源生命週期

### 已完成的核心 kernels

- [x] Gaussian 使用 horizontal/vertical separable kernels 與 float 中間 buffer。
- [x] Gaussian weights 使用 constant memory。
- [x] Adaptive Mean 使用 replicate-border 64-bit integral image，視窗查詢為 O(1)。
- [x] Integral image 使用 row scan、transpose、第二次 row scan，並檢查 allocation overflow。
- [x] 驗證工具已加入 Gaussian、Adaptive Mean、401-2 fused 與 4K benchmark 案例。
- [x] Gaussian 加入 shared-memory tile/halo，實測 kernel 45 收益與限制。（2026-09-14 RTX 3090：block-local tile/halo 融合兩段 pass 輸出完全相同但全面約慢 2 倍，未採用；改採保留 reflect101 邊界的內部像素無分支快速路徑，k45 在 4K／2300×12000 ROI 交錯 A/B 10/10 勝出、kernel 時間降約 40～47%，小 kernel 在雜訊範圍）
- [ ] 正式 Recipe 真圖量測 Gaussian 快速路徑對 Detector／端到端的實際占比與收益；512² tile 的 k45 kernel 僅約 0.09→0.08 ms，收益主要在大 ROI。
- [x] CUDA event 分別量測 Adaptive Mean integral/kernel、Gaussian passes 與 threshold kernel；待 RTX runner 回收實測數值。

### Persistent context 與 buffers

- [x] 保留 ABI v1 host-pointer primitives，使用 optional export probe 相容舊 DLL。
- [x] 新增 `vf_context_create/destroy/stats`。
- [x] context 擁有 grow-only uint8、float Gaussian 與 64-bit integral buffers。
- [x] 相同或較小尺寸的 401-2 fused 呼叫不再重複 `cudaMalloc/cudaFree`。
- [x] `GpuRuntime` 提供 `close()`、context manager、destructor 與 `RLock` 序列化。
- [x] 將 CUDA stream、morphology ping-pong 與所有 plan scratch 納入同一 context。
- [x] monitor/batch 跨多張影像重用同一個長生命週期 `GpuRuntime`/context。
- [x] 測試尺寸增減、channel 切換、參數改變、CUDA error/OOM 後的重用與釋放。（validator 覆蓋 shape grow/shrink、1/3 channel、參數切換與 warm allocation plateau；2026-09-14 RTX 3090 真實 launch error 後 context allocation 不再增加、真實 OOM 後連續三次小批次／resident plan 與 CPU 相同且 allocation count 不變、失敗 batch 不留下 native handle）
- [x] 偵測 sticky CUDA context error（例如 illegal memory access）後，明確停用或重建共用 `GpuExecutionSession`，並以 GUI／監控狀態提示重新啟動；目前每次 run 仍會嘗試 CUDA 後整顆 Detector CPU fallback，結果正確但會重複失敗。需在 RTX 以隔離子程序注入驗證。（2026-09-14：同一程序內重建無法恢復 primary context，因此 runtime 標記 CUDA 損毀並停用至重新啟動；RTX 3090 以 NVRTC 越界 kernel 實測 CUDA 700 通過）
- [ ] 評估 Windows 驅動預設 CUDA sysmem fallback：佔滿專用 VRAM 時配置溢出到共用記憶體而不回傳 OOM（2026-09-14 4K plan 結果等價、新 context 首次 51 ms），需以正式大圖／批次量測溢出後的端到端延遲，決定是否以 `recommended_roi_batch_size`／監控告警限制專用 VRAM 使用量。
- [ ] 評估 `cudaMallocAsync`/memory pool；只有相容且實測有收益時採用。

### Morphology

- [x] 量測 detector 401 多 iterations 的 morphology 占比：既有 close iterations 1/2/4/8 benchmark 加上 RTX 3090 合成 16384×13000／70 ROI／5×5 open iterations=10 基準，舊 kernel 的 morphology warm median 165.0 ms、占 GPU preprocessing 約 65%；正式真圖占比仍依 P4 驗收。
- [x] 評估矩形 kernel 的 horizontal/vertical separable min/max filter：5×5 改用同一 kernel 內的 shared-memory 水平／垂直 min/max，其他尺寸保留原 kernel；RTX 3090 合成圖交錯 A/B 10 組皆勝出，逐像素及完整 Tile 結果一致。正式真圖、其他 Recipe 與長時間驗收仍依 P4 待辦。
- [x] 多 iterations 使用 device ping-pong buffers，中間不得回傳 CPU。
- [ ] 小 kernel/少 iterations 建立 CPU/GPU crossover 規則。（validator 已輸出各 iterations 含傳輸 CPU/GPU median/P95/speedup；production threshold 待 RTX 數據）

## P3：Detector 遷移與 CPU/GPU 邊界

- [x] 401-CS-AP-1（舊 ID：401-1）遷移到 cached 共用 plan：Gray → Resize(area) → Gaussian → AdaptiveMean → Morphology；CUDA 無法保持 area 語意時整個 detector CPU fallback。
- [x] 401-AS-SN-1（舊 ID：401）遷移到 cached 共用 plan，保留 BGR Gaussian → Morphology → Gray → AdaptiveMean、threshold 與 contour 語意。
- [x] 401-CS-AP-2（舊 ID：401-2）preprocessing 已遷移到共用 plan，並保留 fused/legacy/CPU 路徑。
- [x] Detector 202 已從 DetectorManager、GUI 與正式 Recipe 契約移除；其二值化與屏蔽實作僅保留為 202-CS-SN-1 的內部共用基底，舊 `202` Recipe 不提供別名且會明確回報未註冊。
- [x] Detector 202-CS-SN-1（舊 ID：202-1）自動 CNR：依 AcceptanceChecker 參考實作執行尺度相關 Gaussian 背景、residual、MAD robust sigma、`max(8, 3×sigma)` mask、3×3 Open、8-connectivity components 與局部背景 ring CNR；候選依 CNR 降冪輸出，抓到即 NG。
- [x] Detector 203-AS-SN-1（舊 ID：203-AS-AP-1）：固定執行 Gray → Gaussian 3 → Adaptive Mean 反相（block 21、C 1）→ 3×3 Open 一次 → 四邊屏蔽 → LIST contours；四邊內縮與面積上下限可由 Recipe 調整，抓到任一符合輪廓即 NG。
- [x] Detector 401-CS-SN-1：固定執行 Gray → Adaptive Mean 一般二值化（設定 block 156 依 OpenCV 契約轉成有效 block 157、C -56）→ 四邊屏蔽 → LIST contours；四邊內縮與面積上下限可由 Recipe 調整，抓到任一有效輪廓即 NG、未抓到即 PASS。
- [x] Detector 503-CS-SN-1：固定執行 Gray → 一般二值化（預設門檻 200、不反相）→ 中心與四邊屏蔽 → LIST contours → 多邊形近似；中心半寬／半高、四邊內縮與面積上下限為工程外參，其餘屏蔽、影像與演算法設定為管理內參，抓到任一符合候選即 NG。
- [x] Detector 506-CS-SN-1：固定執行 Gray → 一般二值化（預設門檻 200、不反相）→ 中心與四邊屏蔽 → LIST contours → 多邊形近似；中心半寬／半高、四邊內縮與面積上下限為工程外參，其餘屏蔽、影像與演算法設定為管理內參，抓到任一符合候選即 NG。
- [x] Detector 505-AS-SN-1：固定執行 Gray → 一般反相二值化（預設門檻 120）→ 四邊屏蔽 → LIST contours → 多邊形近似；接受面積 100～100000 且至少 3 頂點的候選，抓到即 NG、未抓到即 PASS。
- [x] 900-CS-AP-1（舊 ID：900）遷移成 cached CPU DAG plan，共用一次 gray 產生 outer global 與 inner adaptive masks。
- [x] 900-CS-AP-1 DAG 接上 CUDA/native executor，共用 device gray 並只下載必要 masks。
- [x] 401/401-1/401-2 的 `findContours` 與少量幾何分析暫留 CPU，只下載 binary mask。
- [x] 401-2 contour mask 改為局部 bbox mask，避免每個 contour 配置整張 ROI mask。
- [ ] 評估 401-2 white-pixel reduction 移至 GPU，只下載統計值與必要 mask。（已拆出 `white_ratio_analysis` profiler；CPU bbox-local counting 改用 OpenCV countNonZero/bitwise_and，512² synthetic median 0.0343→0.0151 ms；GPU 搬移待 RTX/production 佔比證明）
- [x] 評估 connected components；bbox 雖可一致，但 pixel area、孔洞 contour 數與既有排序語意不等價，且固定 seed 4K CPU benchmark 無收益，因此不取代 contours。
- [ ] 全部 detector 遷移並通過 RTX 3090 驗收後，才評估移除 detector-specific compatibility adapter。

## P4：Tiling、ROI、Batch 與跨圖片重用

### Detector 401 GPU 效能執行順序（2026-07-20）

- [x] 建立 Template Anchor Grid 專用 profiler 與離線分析器，分離 detector、pipeline、CUDA events、CPU contours、fallback 與座標/PASS-NG gate；目前 RTX 回報 CPU detector median 約 1205 ms、GPU warm median 約 1400 ms（GPU 慢約 1.18x），GUI 約 3 秒因量測範圍與 cold session 不同，尚不能據此指定 kernel 瓶頸。
- [x] GUI 單張檢測跨相同 recipe 重用 `GpuExecutionSession`；首次 cold、後續 warm，recipe 路徑/mtime/size（涵蓋已儲存的 GPU 設定）改變時安全失效，關閉視窗時釋放 context，不跨執行共用 resident image generation；cache reuse/invalidation/close 測試已通過。
- [x] GUI 顯示實際使用者等待時間，並保留現有 `duration_sec` schema；profiler 同時輸出 detector-only、pipeline-before-report、reporting、end-to-end 與外層 wall time，避免把 401 1.4 秒和 GUI 3 秒直接比較。
- [x] 監控列表的逐圖 `duration_sec` 改為新檔案到達／首次輪詢觀測、連續穩定檢查、Pipeline、overlay／CSV／JSON 報告寫檔及處理後影像搬移完成的端到端時間；另保留 `timing` 分項供追查。
- [x] 離線分析器新增「計時口徑（請勿混用）」與 `scopes_ms`，比較 CPU/GPU detector、cold/warm pipeline、reporting、end-to-end 與非 detector overhead；重疊 CUDA events 仍只作瓶頸占比，不相加為總時間。
- [ ] 在 RTX 3090 以相同 image/recipe 重新執行 cold 1 次、warm 10 次與 GUI 連續 10 次；確認第二次以後 context/allocation 接近 0、GPU active、無 fallback、ROI/PASS-NG 完全一致，再決定下列分支。
- [ ] 若 launch/synchronize/ROI gather/D2H 為主：實作 detector-neutral `execute_plan_roi_batch` optional ABI，resident image 唯讀共享、一次提交一批 ROI、一次同步及批次 masks 下載；測 batch size 1/4/8/16/32/全部。
- [ ] 若 Morphology 為主：針對簡報第 12 頁的 5×5 open、iterations=10，先在 RTX 3090 以相同真圖／Recipe／ROI 記錄現有 CUDA 形態學、Detector 與端到端 warm median／P95，以及 kernel 啟動、顯存讀寫與 VRAM 峰值；70 ROI 單顆 401 曾量得形態學約 300 ms，須以本次基準重測。
- [ ] 建立可分離橫向／縱向形態學 CUDA 原型，並比較 shared-memory／合併有效 kernel 方案；保留矩形 kernel、OpenCV open 的侵蝕後膨脹次序、iterations、border、channel 與輸出語意。500→200 次鄰居讀取只是理論存取次數，不能當成 2.5 倍實測加速，且須計入新增的中間寫入與 kernel 啟動。
- [ ] 以 CPU OpenCV 與現有 CUDA 作參考，比對 5×5 open、iterations=10 的 binary mask，涵蓋 ROI 邊界、不同尺寸／channel、連續與非連續輸入，並驗證正式 Recipe 的 PASS／NG、缺陷 bbox／area／metadata、排序及失敗後整顆 Detector CPU fallback；等價不通過不得替換現有路徑。
- [ ] 在 RTX 3090 對相同真圖交錯量測舊／新 CUDA 路徑的形態學、Detector 與端到端 warm median／P95、VRAM 與長時間穩定性；只有完整等價且端到端有可重現收益才採用，否則保留原 kernel。若修改 `.cu`，須重編 DLL、跑 native smoke／CUDA validator 並確認舊 DLL 相容路由。
- [x] 先以生成的 16384×13000 BMP／70 ROI／401-AS-SN-1 在 RTX 3090 驗證 5×5 shared-memory 形態學：舊／新 DLL 交錯 A/B 各 10 次，morphology warm median 165.0→69.8 ms、Detector 413.8→325.2 ms、整張圖 1429.1→1341.2 ms，三個指標逐對均 10/10 勝出；1960 次 kernel launch 不變，ROI／PASS-NG／完整 Tile 輸出一致。此項只代表合成圖，不替代上方真圖及 production 驗收。
- [ ] Batch 單 stream 仍無法達標時才建立 2/4 execution slots；每 slot 獨立 stream/scratch/events/pinned output，縮小 Python lock 至 metadata/context lifecycle，不接受仍被全域 lock 序列化的 worker 數字。
- [ ] 以相同 ROI/輸入/輸出條件比較 OpenCV CPU、自製 CUDA、OpenCV CUDA/hybrid；Gray-first 僅作 feature-flag 實驗，golden mask/PASS-NG 不等價時不得採用。
- [ ] 最終 RTX 驗收：GPU warm median < 3.3 秒、目標 < 2 秒；ROI 座標與 PASS/NG 100% 相同、binary agreement >= 99.99%、無 silent fallback，完成 batch/stream/VRAM/100 次一致性/error/leak 報告後才調整 production backend。

- [x] 偵測重複 GPU crop round trips，記錄傳輸量並輸出負優化警告。
- [x] production/benchmark 在 device tiling 改善前預設關閉 GPU crop。
- [x] 原圖一次 upload，以 device offset/view 表示 grid ROI，不再每 tile 上傳完整原圖。
- [x] detector 可直接消費 device ROI；只有 CPU contour、GUI、debug 或存檔時才下載。
- [x] 新增 batch ROI API，以座標陣列產生連續 device buffers。
- [x] 新增 `run_batch(images/rois)` 或等價 detector batch 介面；CPU 預設實作可逐張執行。
- [x] 依影像尺寸與可用 VRAM 自動選 batch size，配置失敗時自動縮小批次且不留下 stale handle。
- [ ] RTX 3090 實機測試 8、16、32、64 ROI batch 的正確性、效能與 VRAM 平台。
- [x] 單張 GUI 採低延遲策略；資料夾、monitor、batch 採高吞吐策略。
- [x] 使用 bounded 單一 GPU queue，避免多個 CPU workers 同時搶 GPU 或無限制累積 VRAM。
- [ ] 評估 pinned host memory 與 CUDA streams，量測 upload/kernel/download 重疊收益。

### Phase2 GPU 加速簡報第 14 頁工作包對照（2026-09-14）

- A（ROI batch 進 plan）：上方已有 `execute_plan_roi_batch` optional ABI 待辦；現有 ROI batch API 只負責 ROI 資料配置／下載，不等於整批 plan 一次提交、一次同步。
- [ ] B（CUDA Graphs）：先以 profiler 確認 kernel launch／host 提交確實是瓶頸，再針對固定 shape、plan、ROI 批次與 context 的重複執行建立 capture／replay 原型；測 graph 建立與重建成本、參數／尺寸變更、context 釋放、錯誤後整顆 Detector fallback，並以相同資料交錯比較 warm median／P95 與端到端收益。未量得收益不加入正式路徑。
- C（可分離形態學）：上方已有 5×5 open、iterations=10 的基準、等價與實機收益待辦。
- D（pinned memory＋stream 重疊）：上方已有評估待辦；須以多張圖片的批次／監控流程量測 H2D、kernel、必要 D2H 的實際重疊、host RAM／VRAM 峰值與端到端吞吐，單張圖片的序列時間不能直接當成可重疊收益。
- [ ] E（向量化／`__restrict__`／`__ldg`）：以 profiler 選出受記憶體存取限制的 kernel，再分別試向量化載入／儲存及適用的編譯器讀取提示；確認對齊、stride、1／3 channel、ROI 邊界與 OpenCV 輸出語意，逐項測 kernel、Detector 和端到端收益。`__restrict__`／`__ldg` 不預設有效，沒有可重現收益即不採用。
- F（Gaussian shared memory）：2026-09-14 已在 RTX 3090 實測 shared-memory tile／halo 較慢而不採用，改以內部快速路徑取得 k45 kernel 約 40～47% 收益（見 P2）；正式 Recipe 真圖占比仍待量測。
- G（`INTER_AREA` 完整等價）：2026-09-14 已完成 CUDA Resize(area) 與 OpenCV 逐像素等價及合成 Recipe 端到端驗收（見 P1）；真實樣本仍依 RTX 驗收區。
- I（RTX 實機驗收）：上方效能 gate、下方 RTX 3090 的 production PASS／NG、GUI、打包與壓測待辦仍未完成；H（metrics／等價驗證）貫穿 A～G，不另估一份收益。

簡報的 30～45 週、約 1.2 倍及各工作包百分比是評估假設，不是本 Todo 的交付承諾；讀圖與 resident ROI 路徑已有後續改動，須以同版程式、同圖、同 Recipe 重建 CPU／GPU cold、warm 與端到端基準後才可重估工期和效益。A～G 的收益有重疊，不得直接相加。

### 大圖 GPU ROI 直通 Detector（2026-09-14 開始實作）

- [x] PNG／JPEG 等圖片保持 CPU 讀檔／OpenCV BGR 解碼；啟用原生 GPU Detector 的 grid／Template Anchor Grid 路徑在切 tile 前整圖只做一次 H2D upload，tile 以 resident device ROI 執行 D2D staging 與 CUDA plan，不逐 tile 下載 BGR 或重傳原圖；tile 同時保留不複製像素的 CPU 原圖 view 供 shape、fallback 與報表。非 grid 定位（例如 contour／pattern match）仍依其 CPU 定位語意另行評估。
- [x] GPU resident tile 不再預製所有 CPU tile 副本；PreprocessPlan／native linear 與 DAG 能力查詢及 device ROI 執行可用非連續 NumPy view 的 shape／dtype／channels 驗證，避免 `np.ascontiguousarray` 偷做等量複製。ROI inset、generation／bounds 與 batch／monitor 共用 session 沿用既有生命週期。
- [x] 混用 CPU Detector 時才按需建立獨立 CPU tile 副本供該 tile 的 CPU Detectors 共用；GPU 執行失敗時依既有政策從原圖 view 重跑完整 Detector。保留 `gpu.mode=cpu/auto/cuda`、舊 DLL 與 strict CUDA 路由，不混用部分 GPU 中間結果與 CPU 後續步驟。
- [ ] Overlay、NG tile／sidecar、debug 與 GUI 預覽所需的 CPU 像素按需取得；401 等目前仍需 CPU `findContours`／幾何判定的 Detector 只下載必要的 binary mask，檢查 PASS/NG、tile 順序／座標、defect bbox／area／confidence／metadata、輸出內容與 CPU 基準等價。
- [ ] 用同一批真實 16384×13000 圖、約六個 2300×12000 ROI 及正式 Recipe／輸出設定，在 RTX 3090 比較修改前後 cold、warm median／P95、image load、整圖 H2D、tile 建立、Detector、D2D／必要 D2H、Reporter、端到端耗時、RAM／VRAM 峰值與 100 次穩定性；未證明整體收益與完整等價前保持 production 預設不變。2026-09-14 合成尺寸基準：BGR 原圖約 609.4 MiB，六張 CPU tile 副本約 473.8 MiB，CPU 裁切 warm median 128.6 ms，整圖 H2D 85.0 ms，ROI descriptor 建立約 0.03 ms；預期省的是 CPU 副本及其約 129 ms 複製，不包含既有 H2D，端到端百分比須以目前每張總耗時為分母實測。

## P5：CPU 與整體 Pipeline 最佳化

- [x] 分別量測 `findContours`、幾何分析、Python tile/detector 迴圈、progress callback、aggregation 與 reporter。
- [x] 降低 progress callback 頻率，避免每個小 primitive 更新 GUI。
- [x] 移除不必要的 detector `image.copy()` 與完整尺寸 temporary masks；必要的 non-contiguous CUDA/QImage 邊界 copy 保留。
- [x] 相同 tile 的 CPU detectors 共用一次 gray；GPU detectors 共用 resident source，避免各自重傳原圖。
- [ ] RTX profiler 證明有收益後，再加入跨 detector 的 device-gray／完整 preprocessing result cache。
- [ ] 對小圖、小 ROI、少 tiles 建立 CPU/GPU crossover benchmark；低於門檻自動選 CPU。（64²～1024² native 401-style matrix 與穩定 1.0x/1.5x threshold 報告已完成；production policy 待 RTX 數據）
- [x] Overlay、NG tiles、CSV/JSON 與純檢測計時分離；目前各 reporter 與 `detectors_total` 已獨立計時，是否背景化由實測決定。
- [x] Pattern matching 目前維持 CPU；只有 RTX profiler 證明為主要熱點後才另案 GPU 化，並要求模板常駐與 CPU 等價路徑。
- [x] PNG 編碼、YAML、彙總、logging 與 GUI 控制邏輯維持 CPU，除非量測證明需要改變。

## P6：GUI、設定與部署

- [x] Recipe 與 GUI 可設定 GPU，並顯示 DLL/device/fallback 狀態。
- [x] GPU mode 統一為清楚的 `auto`、`cpu`、`cuda` 語意，並相容未含 mode 的舊 recipe。
- [ ] production 預設 mode 仍需由 RTX 3090 實機驗收決定。
- [x] GUI worker 不在 UI thread 等待 CUDA；monitor 取消、錯誤與進度以 stop callback／Qt signals 保持可回應。
- [x] GUI 顯示實際 backend，不得因 recipe 勾選 GPU 就顯示 CUDA active。
- [x] PyInstaller 有 DLL 時條件式包含 `gpu/visionflow_cuda.dll`，無 DLL 時建立 CPU-compatible package 且 runtime 可 fallback。
- [x] **GUI 狀態層級**：TopBar 只呈現全域進度，操作 panel 顯示目前步驟，Status Bar 僅保留短事件；就緒／執行中／PASS／NG／ERROR 狀態一致且不重複。
- [x] **實際 backend chip**：TopBar 固定顯示 CPU、CUDA device 或 CPU FALLBACK；fallback 原因可由 tooltip 查看，未實際啟用 CUDA 不得顯示 CUDA active。
- [x] **非阻塞提示**：可恢復警告與成功訊息改用既有視覺語言的 inline notice，只有阻止操作、不可恢復或離開確認保留 modal dialog。
- [x] **NG 導航與快捷鍵**：Results 支援上一個／下一個 NG、J／K 切換、Enter 聚焦 bbox，列表與 viewer 選取同步並自動捲動。
- [x] **Recipe dirty／validation 狀態**：Designer 顯示已儲存、未儲存、驗證失敗；切換 recipe 或關閉時僅在 dirty 狀態詢問，錯誤在畫面內顯示。
- [x] **CSV 面積精度換算**：Recipe Designer 可設定 `1 px = n µm` 並保存至 recipe；所有執行模式的缺陷 CSV 集中換算為 µm²並標示單位，留空及舊 recipe 維持 px²，Detector 判定門檻不變。
- [x] **GUI 操作環境記憶**：使用 QSettings 保存並恢復上次 recipe、影像／batch／monitor 資料夾、輸出選項、最後畫面、視窗 geometry/state、viewer zoom 與主要 splitter 比例；無效路徑安全忽略。
- [x] **GUI 權限管理器**：每次啟動固定進入 OP；工程／管理模式分別以預設密碼 `1234`／`5678` 驗證，驗證邏輯與密碼提示視窗採獨立 OOP 元件且可注入替換。
- [x] **Detector 內外參權限**：所有正式 Detector 由共用 parameter schema 明確標記外參／內參；工程模式只顯示面積、尺寸、間距與 ROI 幾何外參，管理模式完整顯示影像、光學、演算法與模型內參，未分類的新參數安全預設為管理者內參，舊 Recipe 隱藏值仍完整保留。
- [x] **大量資料操作**：Batch／Monitor 表格使用 model/view 與增量更新，提供 PASS／NG／ERROR 篩選；scatter 超過上限採 deterministic sampling，避免每筆結果重建整表。
- [x] **繁中一致性與可及性**：操作訊息統一繁體中文，PASS／NG／CPU／CUDA 等工業縮寫保留；狀態不得只依賴顏色，並補 tooltip／文字標籤與鍵盤操作測試。
- [ ] 有 GPU、無 GPU、DLL 缺少、DLL 版本不符、fallback 開/關各完成一次打包實機測試。（runtime tests 已覆蓋 missing/ABI mismatch/no-device/context-failure 與 fallback policy；無 NVIDIA/CUDA DLL 電腦已完成 CPU-compatible package build 與 5 recipes bundle，packaged smoke 進一步驗證 MainWindow、CPU-only pipeline、缺 DLL fallback 開啟時與 CPU 結果一致且 GPU call count=0、fallback 關閉/strict CUDA 明確失敗，EXE exit 0；有 GPU 與 packaged ABI mismatch 待實機）

## P7：CI、GitHub Actions 與發布

- [x] 一般 Windows runner 執行 unit tests、compileall、recipe/CLI/GUI smoke 與 CUDA headers/API 靜態檢查。
- [x] DLL 與 test EXE 使用明確 source manifest 分開編譯，不以 glob 無差別加入所有 `.cu`，並以 preflight 靜態驗證。
- [x] workflow 明確加入 `gpu/include/`，上傳 DLL、LIB、test EXE 與 build log artifacts。
- [x] CUDA runtime、CPU/GPU 等價、VRAM leak 與 benchmark 只在 GPU self-hosted runner 執行。
- [x] self-hosted runner 使用 `self-hosted`、`Windows`、`X64`、`gpu`、`rtx3090` labels。
- [ ] 在目前 `wjcudalearning/VisionFlow` repository 註冊或授權具有上述 labels 的 RTX self-hosted runner；2026-07-31 GitHub API 回報可用 runner 數量為 0。
- [x] 不允許不受信任的 fork PR 直接在可接觸本機資料的 self-hosted runner 執行。
- [x] GPU job 支援手動與 nightly；PR 至少完成 compile/static checks。
- [ ] 保存 benchmark JSON、Nsight report、Driver/Toolkit/GPU 與 commit hash，支援 commit 間比較。（JSON、環境與 commit 已完成；workflow 已加入可用時執行 nsys smoke capture 並記錄 skip/status，report 待 RTX runner）

## P8：產線安全、追溯與持續驗證

- [x] Detector 宣告共用參數 schema；recipe 載入嚴格拒絕未知 detector、未知參數、錯誤型別、越界值與非法 enum，GUI designer 使用同一份 schema 建立欄位。
- [x] Inspection 輸出保存原始 recipe SHA-256、套用 runtime override 後的 effective recipe SHA-256，以及 build commit/dirty provenance。
- [x] 每張 NG tile 旁產生 dataset metadata sidecar，包含 recipe provenance、detector/參數、局部與全域座標、來源影像及人工複判欄位。
- [x] 五份 production recipe 皆有可重現的合成 PASS/NG golden regression，斷言 final result、defect count、bbox 容差、area/confidence/metadata 與順序；四種 detector 各至少五個合成案例。
- [x] 建立 Windows 精確 dependency lock；hosted CI、RTX runner 與 PyInstaller build 使用同一份 lock，避免時間與機器造成版本漂移。
- [x] hosted CI 監測 RTX workflow 最近成功時間（超過 48 小時失敗）、benchmark 與 baseline 比較並 gate P95 退化，另有 weekly PyInstaller build + packaged smoke，Python 版本與部署版本一致。
- [x] Hypothesis 隨機產生影像與合法 PreprocessPlan，驗證 CPU executor 與直接 OpenCV reference；固定生成順序可供 RTX CPU/GPU fuzzing，recipe/designer schema 與 GPU ABI/metrics 已拆成可 headless 測試模組。

## P9：CPU 吞吐、輸出與可維護性優化

本區的效能功能不得改變 recipe 語意、PASS/NG、座標、缺陷 metadata 或預設輸出。無實機收益證據的平行策略維持 opt-in。

- [x] 單張純 CPU 檢測支援 opt-in tile 級平行：`performance.tile_workers`／`AOI_TILE_WORKERS` 啟用，thread-local detector 避免共享 instance state；GPU detector 或 resident image 強制保持序列。
- [x] CPU 切圖支援 `performance.crop_workers`／`AOI_CROP_WORKERS` 覆寫，預設 `auto` 依前 16 張 ROI 工作量決定序列或最多 4 workers；grid、Template Anchor Grid、contour 與 pattern_match 先固定座標，再以有界工作池保序複製 ROI，GPU crop／resident image 不進入 CPU 平行路徑。
- [x] Recipe 依 path、mtime、size 建立 thread-safe process cache，cache hit 回傳 deepcopy，檔案異動後重新解析與驗證。
- [x] Batch worker 上限調整為 `min(8, cpu_count, image_count)`，批次期間分配 OpenCV thread budget 並於結束時還原；`AOI_BATCH_WORKERS`／`max_workers` 可覆寫。
- [x] `gc.collect(0)` 改為 `AOI_BATCH_GC_INTERVAL` 可設定週期，預設每 8 張，`0` 停用。
- [x] Reporter 支援 bounded NG tile 平行寫檔、`png_compression`、overlay PNG/JPG、JPEG quality 與 preview max dimension；machine-readable 座標維持全解析度。
- [x] `output.save_debug_images` 可輸出 detector preprocess 中間影像；runtime payload 在 JSON 與公開 tile result 前移除，預設關閉，CPU fallback 也保留擷取。
- [x] 新增 `core/result_types.py` TypedDict 結果契約與 runtime contract test。
- [x] 新增 Windows Unicode 路徑安全的 P9 regression tests，固定 serial/parallel 結果等價、cache invalidation、batch/GC policy、輸出參數、overlay decode/downscale 與 debug payload 隔離。
- [ ] 使用固定 production 資料集量測 worker 上限、GC interval、PNG compression、NG write workers 的 median/P95、peak RSS 與檔案大小，再決定量產建議值。
- [ ] 在 RTX 3090 驗證 `AOI_TILE_WORKERS>1` 不會使 GPU detector/resident image 進入平行路徑，且 GPU queue/VRAM 無競爭或累積。

## P10：OOP 責任邊界重構

本區只調整 Python 責任分工與可測試性；必須保留 ABI v1、recipe、PASS／NG、座標、metadata、輸出格式、排序、CPU fallback 與既有 GUI 操作契約。外部相容 façade 在遷移期間不得移除。

- [x] 先以 contract／golden regression 固定 Reporter 輸出、Designer recipe round-trip、MainWindow workflow、Detector900 metadata、GpuRuntime fallback/lifecycle 與 Pipeline 公開結果。
- [x] 將 `Reporter` 改為 coordinator，拆出 Overlay／JSON／CSV／Matrix CSV／NG Tile／Debug Image writer；Detector900 debug overlay 改由 detector-specific renderer registry 提供。
- [x] 將 `DesignerScreen` 的 editor state、recipe mapping、runtime validation 與主要 panels 拆成可獨立測試元件，Widget 僅保留畫面組合與 signal routing。
- [x] 將 `MainWindow` 保留為 application shell／composition root，Batch、Monitor、Preview、Inspection 工作流程移入獨立 controllers，背景 worker 與 UI 狀態契約不變。
- [x] 將 `Detector900` 拆為 typed config/value objects、mask preprocessing、candidate analysis、pair geometry 與 result assembly；只在公開輸出邊界轉回既有 dict schema。
- [x] 將 `GpuRuntime` 保留為相容 façade，拆出 DLL bindings/capability、native plan descriptor/cache、resident image/ROI batch resource 與 metrics/lifecycle 協作者；舊 DLL optional probing、thread safety 與完整 detector CPU fallback 不變。
- [x] 將 `AOIPipeline._run()` 拆為明確的 recipe/runtime preparation、tile inspection、execution metadata/result assembly 階段，`AOIPipeline.run()` 保持單一 orchestration 入口。
- [x] 完成全部 unit tests、compileall、CUDA preflight、CLI synthetic smoke、GUI offscreen smoke 與 `git diff --check`，並記錄無法在本機執行的 RTX／實體 GPU 驗收。
- [x] 移除 `Detector900` domain 遷移後殘留的 16 個舊演算法方法，以及 `Reporter` renderer registry 遷移後殘留的 Detector900 debug renderer／格式化死碼；保留仍使用的通用 NG tile line-width helper。
- [x] 以 `DetectorManager.ai_performance_stats()` 與 detector 共用的 public `device_name` contract 取代 pipeline 對 `_ai_manager()`／YOLOX `_ai_execution` 的跨模組私有存取。
- [x] 完成 Reporter writer 責任反轉：`ReportWriteContext` 僅攜帶明確的公開設定、路徑、profiler 與輸出服務，writers 不得持有整個 `Reporter` 或呼叫其私有成員。
- [x] 增加 OOP 架構邊界與輸出回歸測試，並重跑全部 unit tests、compileall、CUDA preflight、CLI synthetic smoke 與 `git diff --check`。

## RTX 3090 編譯與實機驗收

### 環境與編譯

- [x] `nvidia-smi` 可看到 GPU，並記錄 Driver、CUDA compatibility 與 VRAM。
- [x] 安裝 CUDA Toolkit、VS 2022 C++ Build Tools、Windows SDK；確認 `nvcc --version` 與 `where.exe cl`。
- [x] 使用 x64 Native Tools PowerShell 執行 `gpu/build_cuda_dll.ps1 -Architecture sm_86`。
- [x] 產生 `visionflow_cuda.dll`、`visionflow_cuda.lib` 與 `test_cuda_api.exe`，沒有 link/architecture 錯誤。
- [x] `test_cuda_api.exe` 驗證 ABI、device、compute capability、grayscale、context 與 fused smoke。
- [x] `dumpbin /exports` 檢查所有預期 `vf_` exports；`dumpbin /dependents` 無缺少依賴。

### 2026-07-30 RTX 3090 實測阻擋項目

- [x] 修正 CUDA Gaussian 與 OpenCV `GaussianBlur` 的邊界及 rounding 語意：random odd-size kernel 3（max diff 6、超容差像素 19.6828%）、kernel 5（max diff 4、0.9675%）及 non-contiguous kernel 3（max diff 4、4.3701%）必須回到既定 `max_diff <= 2`、超容差像素 `<= 0.1%`。
- [x] 修正 native 401-style plan 的 Gray → Gaussian → Adaptive Mean 組合誤差；目前首次及 context reuse 均有 3.1047% binary mask mismatch，高於允許的 2%，修正後兩次結果都必須通過。
- [x] 修正 native 900 shared-gray DAG 的 outer/inner mask 差異；目前分別有 0.0895%／0.0977% mismatch，而此路徑要求與 CPU 完全一致。
- [x] 修正 resident ROI linear plan 與 DAG 的 0.0977% mismatch；逐項核對 ROI origin、width/height、host/device stride、邊界處理及 threshold rounding，並維持 resident execution 不增加 H2D。
- [x] 修正 GitHub Actions artifact 與 repository 的整合方式：artifact 只部署核准的 DLL/LIB/EXE/manifest，不得覆蓋 repository 版 `build_cuda_dll.ps1`、`preflight_cuda_build.py`、validator 或 profiler；恢復 repo-local import root 與 optional export contract，使直接執行 validator 及完整 unit tests 不再發生 import error。
- [x] 完成上述修正後，在本機 x64 Native Tools PowerShell 以 `gpu/build_cuda_dll.ps1 -Architecture sm_86` 重編 DLL/LIB/EXE，記錄 source/binary SHA-256、exports、dependencies 與工具版本；不得沿用未對應目前 commit 的舊二進位檔。
- [x] 本機重編產物必須讓正式 `gpu/validate_cuda_dll.py` 以零 soft bypass、零失敗完成全部 primitive、linear/DAG plan、resident ROI、context reuse、4K benchmark 及 10/100/1000 stress checkpoints，再進入 production PASS/NG 驗收。

### Primitive、plan 與效能

- [x] BGR→RGB、crop、threshold、morphology 與 CPU 完全一致。
- [x] BGR→Gray、resize、Gaussian 與 Adaptive Mean 通過既定像素容差。
- [x] Gaussian 覆蓋 kernel 3/5/15/25/45 與 structured/non-contiguous inputs。
- [x] Adaptive Mean 覆蓋 block 3/11/35、正負與小數 C、invert 及邊界輸入。
- [x] 401-2 fused 與 CPU plan 結果在容差內；相同尺寸連續執行 allocation count 不增加。
- [x] 通用 native plan 完成後，逐 operator 與完整 plan 對 CPU executor 建立等價矩陣。
- [ ] 記錄 4K primitives、preprocessing plan、純檢測與端到端 CPU/GPU speedup。
- [x] 連續執行三次完整驗證，沒有 CUDA error、崩潰或 VRAM 持續成長。

### Production recipes、GUI、打包與壓測

- [ ] `PRODUCT_A_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] `PRODUCT_A_CIRCLE_401_1_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] `PRODUCT_A_NEGATIVE_401_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] `PRODUCT_A_WHITE_RATIO_401_2_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] `PRODUCT_A_FRAME_900_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] 比較 tiles、PASS/NG、defect count、bbox、area、confidence、metadata 與 fallback log。
- [ ] GUI 的 recipe 儲存/載入、viewer backend、status、overlay、輸出與 fallback 正確。
- [ ] 打包版在有 NVIDIA GPU 與無 NVIDIA GPU 電腦均完成驗證。（目前無 GPU 電腦已完成 CPU-compatible package build 與 bundled recipe/MainWindow smoke；有 GPU 電腦待驗收）
- [ ] warm-up 5 張後測 10、100、1000 張；VRAM 穩定、GUI 可回應、無 crash/error。（validator/workflow 已加入 checkpoints、allocation/VRAM/median/P95；待 RTX 執行）

## 未來 AI Detector

### YOLOX Detector 規格與參數

- [x] 定義 detector ID 為 `yolox`、顯示名稱為「YOLOX 物件偵測」，輸入語意為目前 tile／ROI 的 BGR `uint8` 影像；命中指定缺陷類別即產生 defect，零筆 defect 為 PASS。
- [x] 建立受控的 YOLOX model registry；Recipe 只保存穩定的 `model_id`，不可直接依賴使用者電腦上的任意絕對路徑。每個模型 manifest 至少記錄模型名稱、版本、格式、SHA-256、class names、輸入尺寸、色彩順序、正規化方式、letterbox padding、輸出節點與 decoder/stride 規格。
- [x] 工程模式只開放尺寸／面積外參 `min_box_area_px`，用來濾除過小 bbox，`0` 代表停用；此參數與 NMS IoU 分開定義。
- [x] 管理模式開放 YOLOX 內參：`model_id`（由已驗證 `.onnx` 與 model registry 解析）、`confidence_threshold`、`nms_iou_threshold`、`target_class_ids`、`max_detections`、`inference_backend`、`precision` 與 `class_agnostic_nms`；模型輸入寬高由 manifest 唯讀帶入，不讓 Recipe 任意改成模型不支援的尺寸。
- [x] GUI 將 `nms_iou_threshold` 標示為「NMS 重疊率 (IoU)」，tooltip 說明其值為兩框交集除以聯集，不是像素交集面積；抑制規則固定為同類別較低分框在 `IoU > threshold` 時移除，邊界等於門檻時保留。
- [x] `ParameterSpec`／Recipe validation 驗證數值範圍、model ID 存在、類別 ID、backend/precision 相容性及 `max_detections > 0`；舊 Recipe 不含 YOLOX 時行為完全不變。

### 前處理、推論與結果契約

- [x] 建立可追溯的共用 DL 前處理：BGR/RGB 轉換、等比例 resize、letterbox、padding、dtype/normalization 與 NCHW 轉換皆由 model manifest 決定；保存 scale、padding 與原始/模型輸入 shape，供 bbox 精確映回 tile／ROI。
- [x] 先以 ONNX Runtime CPU 建立 correctness reference；使用固定輸入及已知 raw output 驗證 YOLOX grid/stride decode、`objectness × class_probability` 信心分數、class filter、NMS、clip 與反 letterbox 座標。
- [x] 每筆結果沿用既有 defect schema：`type` 使用 class name、`bbox_local` 為 `[x, y, width, height]` 整數、`area` 為 bbox 的 `width × height` px²、`confidence` 為最終合成分數；metadata 保存 `class_id`、`class_name`、objectness、class probability、model ID/version/SHA-256、閾值、原始 float bbox 與 letterbox 資訊。
- [x] detector execution metadata 明確記錄 requested/actual backend、device、precision、input/output shape、batch size、model load/warm-up/inference/postprocess 耗時及 fallback reason；不得把 Recipe 要求的 CUDA 誤報為實際 CUDA。
- [x] 輸出固定採 deterministic ordering：`confidence` 由高到低，再依 `class_id`、`y`、`x` 排序；同分與 NMS tie case 必須有穩定結果。
- [x] 明確限制第一版支援的 export contract；若同時支援 raw YOLOX output 與 end-to-end NMS model，必須由 manifest 指定 decoder，不可依 tensor shape 猜測後靜默套用。

### Model/session 生命週期與 fallback

- [x] 在 `core/` 建立 detector-neutral 的 AI model/session manager；cache key 至少包含 model SHA-256、backend、device、precision 與 input shape，GUI preview、單張檢測、batch 與 monitor 共用 session，不得由每個 worker 各載入一份模型。
- [ ] session 支援明確 `close()`、warm-up、bounded batch queue、VRAM budget、模型切換安全釋放及 cache invalidation；同一模型連續執行不得重複載入。
  - [x] 已完成 `close()`、warm-up、bounded inference queue、LRU cache 上限、模型/backend 選擇性 invalidation、安全等待 active inference 結束及 session/queue metrics；同模型連續執行與 batch/monitor 實測只載入一次。
  - [ ] production 模型的實際 VRAM budget 與模型切換峰值仍需在 RTX 3090 量測後定案。
- [x] 沿用 `gpu.mode` 與 `fallback_to_cpu`：`cpu` 只使用 ONNX Runtime CPU；`auto` 的 CUDA/TensorRT 初始化或推論失敗時，只有在存在相容 ONNX reference model 時才整個 detector 於 CPU 重跑；`cuda` 必須明確失敗且禁止 silent fallback。
- [ ] TensorRT engine 必須與 GPU compute capability、TensorRT/CUDA 版本及模型 SHA-256 綁定；不相容時不可載入舊 engine。FP16 通過精度驗收後才可選，INT8 需保存校正資料集版本與精度報告。
- [x] AI inference 與既有傳統 CV 共用 `GpuExecutionSession` execution scope；YOLOX 另有 bounded inference queue 與 metrics，batch/monitor 停止會在目前影像完成後收斂，CUDA stability validator 以 `nvidia-smi` 記錄 process VRAM，避免 YOLOX session 和 CUDA DLL 工作同時無限制搶占 GPU。

### 整合、測試與驗收順序

- [x] M0：完成 model manifest/schema、model registry、ONNX Runtime CPU session cache、獨立前處理/後處理單元測試與一個可散佈的 tiny 測試模型；此階段不接 GUI、不預設 GPU。
- [x] M1：新增 `DetectorYolox`、DetectorManager registration、Recipe round trip、繁中 detector label 與合成 CLI smoke；驗證 PASS/NG、defect count、class、confidence、bbox、area、metadata 與排序。
- [x] M2：Recipe Designer 加入 `.onnx` 模型檔案選擇視窗、信心門檻、NMS 重疊率、NG 類別、最大筆數與最小 bbox 面積；模型不存在、未登錄、checksum 錯誤或 backend 不可用時使用 inline notice，Recipe 不可在錯誤狀態下儲存。
- [ ] M3：接入 ONNX Runtime CUDA，再以相同 ONNX 模型比較 CPU/CUDA 的前處理、raw output、NMS 後 class/數量/座標/分數；先定義 bbox 與 confidence 容差，任何不等價都不得預設啟用。
  - [x] M3 軟體接入：provider 探測、CPU/CUDA 分離 session cache、禁止 CUDA EP 靜默 CPU fallback、初始化／OOM／推論失敗完整 CPU 重跑、strict CUDA 明確失敗及實機 validator 已完成。
  - [x] RTX workflow 接線：runner 環境會以 `onnxruntime-gpu==1.27.0` 執行 M3 等價、CUDA 1000 次 stability，並可選擇執行 production acceptance；JSON 全部上傳為 workflow artifact。
  - [ ] M3 RTX 3090 驗收：使用 `gpu/validate_yolox_ort.py` 實測 raw tensor `atol=1e-5`／`rtol=1e-5`、bbox 最大差 1 px、confidence 最大差 `1e-4`，且 class／數量／排序完全相同。
- [ ] M4：需要效能時才加入 TensorRT FP32/FP16；以固定資料集比較 ONNX Runtime CPU、ONNX Runtime CUDA 與 TensorRT 的 cold/warm median、P95、吞吐、peak VRAM、模型載入時間與端到端時間。
- [x] 測試 invalid model/manifest、缺少 execution provider、CUDA OOM、推論中斷、輸出 shape 錯誤、空 detection、單框、多框、高重疊同類／跨類、邊界框、非方形影像、極小 ROI、灰階輸入及 Unicode 路徑。
- [ ] 驗證 GUI、CLI、batch、monitor 與 PyInstaller package；無 NVIDIA GPU 電腦可使用 reference CPU model，有 GPU 電腦連續執行 1000 張後 session 數量與 VRAM 位於穩定平台，且停止/切換模型無 crash 或 stale result。
  - [x] CPU reference 已覆蓋 GUI Recipe、CLI、batch、monitor、Unicode 路徑與 PyInstaller bundled YOLOX smoke；本機 warm-up 5 後執行 1000 次，session/load count 固定 1、輸出 deterministic、RSS 由 69,177,344 增至 69,484,544 bytes，validator 通過。
  - [ ] RTX 3090 尚需以 production 模型連續 1000 張驗證 session 數量、VRAM 平台、停止與模型切換。
- [ ] 建立人工標註 acceptance set，以 precision、recall、mAP50、誤殺率、漏檢率及每類 confusion matrix 驗收；`confidence_threshold` 與 `nms_iou_threshold` 的 production 預設值必須由該資料集決定，不以範例預設值直接上線。
  - [x] 已建立 `gpu/yolox_acceptance.example.yaml` 與 `gpu/validate_yolox_acceptance.py`；檢查 PASS/NG coverage、標註 bbox 邊界、production/test-only 模型隔離、backend、單一 session，並輸出完整指標與 JSON 證據。
  - [ ] 尚待 production ONNX 權重、實際 AOI 標註影像與門檻，由正式 acceptance set 決定 production 預設值。

- [ ] 導入模型時比較 PyTorch CUDA、ONNX Runtime CUDA 與 TensorRT 的部署及效能。
- [ ] 模型/session 只載入一次並常駐 GPU；支援 batch inference 與固定輸入尺寸。
- [ ] 優先驗證 FP16；INT8 必須完成校正與精度驗收後才能啟用。
- [ ] AI 與傳統 CV 共用 GPU scheduler、VRAM budget、warm-up、metrics 與 fallback policy。
- [ ] 避免 GUI、monitor、batch worker 各自載入一份大型模型。

### 分類式 AI Detector（TensorRT 純推論）規劃

#### 系統邊界與模型交付

- [x] 固定責任邊界：資料集建立、標註、PyTorch／Transformers 訓練、資料增強、模型評估、門檻校準、ONNX 匯出與 TensorRT engine 建置／驗證全部由 VisionFlow 之外的獨立模型工具負責；VisionFlow 不嵌入訓練、微調、資料集生成、訓練圖表或 checkpoint 管理功能，只接收已驗證模型包並執行推論。
- [ ] 定義版本化分類模型包與 schema；至少包含 CPU correctness reference `model.onnx`、`manifest.json`，以及可選且已驗證的 TensorRT engine。量產包不攜帶 PyTorch `safetensors`、訓練程式或 Transformers 訓練相依套件。
- [ ] `manifest.json` 至少保存穩定 `model_id`、模型名稱／版本、ONNX SHA-256、class ID／名稱、PASS 類別、輸入／輸出節點、固定輸入尺寸、BGR/RGB、resize／crop／padding／插值、pixel scale、mean/std、logits／probability 語意、允許 backend／precision、模型來源與 acceptance 報告版本；Recipe 只保存 `model_id` 與受控判定參數，不保存任意絕對路徑。
- [ ] TensorRT engine cache key 與相容性檢查必須綁定 ONNX SHA-256、前處理／類別 manifest、precision、input profile、GPU compute capability、TensorRT／CUDA 版本；不相容、反序列化失敗或 checksum 錯誤時不得載入舊 engine。engine 不是 CPU reference，也不得只因檔名含 `fp16` 就宣稱 FP16。
- [ ] 外部模型工具必須明確以 TensorRT `FP16` builder 設定產生 FP16 engine，保留 FP32 baseline，並輸出 ONNX↔TensorRT raw logits／probability、Top-1／PASS-NG、精度、效能與模型包 SHA-256 驗證報告；INT8 另需校正集版本與精度報告。

#### Detector、判定與共用 runtime

- [ ] 定義分類 Detector 正式 ID／繁中名稱、輸入語意為目前 tile／ROI 的 BGR `uint8` 影像、固定 preprocessing contract、PASS/NG 規則、defect type、confidence、deterministic ordering 與容差；分類結果只代表整個 ROI，若輸出 bbox 必須使用完整 ROI 並在 metadata 標示 `localization=roi_level`，不得誤報為模型已定位瑕疵。
- [ ] PASS/NG 不只使用 argmax：由 manifest 定義 PASS 類別，Recipe 管理內參提供正常品門檻、目標 NG 類別／各類門檻與低信心處置；不確定區間須保守判 NG 或使用既有可追溯狀態，正式語意在實作前固定並加入 golden tests。
- [ ] 將現有 `AiModelSessionManager` 擴充為 detector-neutral 分類 session，而不是在 Detector、GUI、batch 或 monitor 內另建 runtime；同模型／backend／precision／input shape 只載入一次，共用 warm-up、bounded queue、LRU cache、明確 `close()`、安全 invalidation、GPU scheduler、VRAM budget 與 capability／performance metrics。
- [ ] 建立 manifest-driven 共用分類前處理與後處理；CPU reference、ONNX Runtime CUDA 與 TensorRT 必須使用完全相同的色彩、幾何、插值、正規化、NCHW、Softmax 與類別對照語意。DL inference 不強制塞入傳統 CV `PreprocessPlan`，但不得在各 worker 複製不同實作。
- [ ] 沿用 `gpu.mode=cpu|auto|cuda` 與 `fallback_to_cpu`：`cpu` 只載入 ONNX Runtime CPU；`auto` 在缺少 TensorRT、engine 不相容、初始化／OOM／推論失敗時，使用同一 ONNX 與相同前後處理完整重跑整個分類 Detector；`cuda` 明確失敗且禁止 silent CPU fallback。execution metadata 與 GUI 必須顯示實際 backend、precision、device 與 fallback reason。
- [ ] 支援固定輸入尺寸的 ROI batch inference，依 engine optimization profile 安全分批；量測 batch 1／4／8／16／全部 ROI 的 cold/warm median、P95、吞吐、H2D/D2H、peak VRAM 與 pipeline end-to-end，不得只以單張 kernel latency 決定預設值。
- [ ] 結果 metadata 保存 model ID/version/SHA-256、requested/actual backend、precision、device、原始／模型 input shape、完整 preprocessing contract、class ID/name、各類 probability、使用門檻、低信心處置、batch size、load/warm-up/inference/postprocess 耗時與 fallback reason；Reporter／sidecar／JSON compaction 不得遺失必要追溯欄位。
- [ ] Recipe Designer 依 `ParameterSpec.parameter_group` 將模型、PASS／NG 類別、confidence、backend、precision 與低信心策略列為管理內參；只有實體 ROI／尺寸接受條件可列工程外參。模型遺失、manifest／checksum 錯誤、類別或 backend 不相容時使用繁中 inline notice 並阻止儲存。

#### 驗證與量產導入門檻

- [ ] 先建立可散佈的小型分類 ONNX fixture，覆蓋 registry／manifest、前處理 pixel equivalence、logits／Softmax、PASS／NG、低信心、完整 ROI bbox、metadata、Recipe round trip、Unicode 路徑、空／灰階／非方形／極小 ROI、錯誤 shape／class count 與 deterministic ordering。
- [ ] 定義 ONNX Runtime CPU 為分類 correctness reference；分別驗證 ONNX Runtime CUDA、TensorRT FP32、TensorRT FP16 的 raw logits／probability、Top-1、PASS/NG 與門檻邊界容差。任何 class、PASS/NG、數量或排序不等價都不得預設啟用 GPU／FP16。
- [ ] 使用真實 AOI 人工標註 acceptance set，依產品、lot、日期、機台／相機與光源切分，避免同一生成器或近重複樣本跨 train/test；至少輸出正常品過殺率、所有缺陷合併漏檢率、各缺陷 precision／recall／F1、confusion matrix、confidence calibration 與 Wilson CI。合成資料與隨機同分布 100% accuracy 不可作為量產驗收。
- [ ] 先確認分類適用範圍：輸入必須是已定位且語意單一的元件／ROI；若同一 ROI 可能同時有多個缺陷、需要精確座標或大圖搜尋，改採 object detection／segmentation，不以 ROI-level classifier 取代定位模型。
- [ ] 在 RTX 3090 以 production 模型完成 warm-up 5 後 10／100／1000 張與模型切換／停止測試；確認 session/load count、VRAM 平台、queue、GUI 回應、無 crash/OOM/stale result，並與 ONNX Runtime CPU／CUDA 比較端到端效能。未達精度、穩定性或至少 1.5 倍目標加速時維持 CPU／GPU 預設關閉。
- [ ] 驗證 CLI、GUI 單張、batch、monitor、Reporter、NG tile／sidecar、PyInstaller CPU-compatible 與 CUDA-enabled package；無 NVIDIA GPU 可用 ONNX CPU，engine 缺少／不相容可安全 fallback，strict CUDA 明確失敗，且 VisionFlow 發行包不包含任何訓練功能。

## 最終驗收門檻

- [x] CPU-only 是完整受支援模式，沒有 CUDA/NVIDIA GPU 仍可啟動 GUI、CLI、batch 與 monitor。
- [ ] 五個 production recipes 通過 CPU/GPU 等價規則，沒有未解釋的 fallback。
- [x] 每個 GPU plan 原則上每張輸入最多一次 upload 與一次必要 download；resident ROI plan 額外 H2D 為零。
- [x] native plan/context 預留並重用 operator buffers，相同 shape warm-up 後不再逐 operator `cudaMalloc/cudaFree`。
- [ ] 連續 1000 張後 VRAM 位於穩定平台，沒有資源洩漏或程序崩潰。
- [ ] GPU 純檢測 median 與 P95 在目標資料集均優於 CPU；目標加速門檻為至少 1.5 倍。
- [x] 未達 RTX 效能門檻的 production recipe/operator 保持 CPU、GPU 預設關閉。
- [ ] 加速不得犧牲 GUI 回應、打包啟動、結果追溯、錯誤訊息或 CPU fallback。

## 完成紀錄

- [x] 2026-09-14：處理 sticky CUDA context error。RTX 3090 實測（隔離子程序以 NVRTC 編譯故意寫入無效 device 位址的 kernel，`cuCtxSynchronize` 回傳 700）確認 sticky 錯誤後原 runtime 每次呼叫都回 1700，同一程序新建 runtime 也因 context 建立失敗而無法恢復，但舊版 `available` 仍為 True、每張圖都會重試並失敗。`GpuRuntime` 將所有 native 失敗集中到 `_native_error()`，遇到 cudaError 214、220、226、700、702、709、710、714～719（999 不視為 sticky）即設定 `device_lost_reason`，`available`／所有 optional capability 轉為 False，原因以繁中提示需重新啟動程式並保留原始錯誤；session 的每 run 可恢復錯誤清除不會解除此狀態。另修正 `BaseDetector.run()` 改以本次開始時是否嘗試 GPU 決定 CPU 重跑，避免 runtime 在同一次 run 中轉為不可用時把例外往外拋。`validate_cuda_fault_injection.py` 新增 `sticky_context`：共用 session 的健康 run 使用 7 次 CUDA 呼叫、注入後的 run 僅 1 次呼叫即整顆 Detector CPU 重跑、下一次 run 0 次 CUDA 呼叫，三次 Pipeline 結果皆與 CPU 模式相同，strict `gpu.mode: cuda` 明確回報損毀原因；子程序輸出改用 UTF-8。新增 fake DLL 測試覆蓋 sticky 標記、非 sticky 碼（1001／1002／1999）維持可用、context 建立 1700 與後續 Detector 零呼叫 CPU 路由。完整 345 tests、fault injection（init failure、kernel launch error、device OOM、sticky context）、compileall、CUDA preflight、CLI 合成 NG（預期 exit 2）與 `git diff --check` 通過；未修改 CUDA source／ABI／DLL。

- [x] 2026-09-14：完成 CUDA Gaussian shared-memory tile／halo 實測並改採內部快速路徑。先依 5×5 morphology 模式實作單次 launch 的 block-local tile（reflect101 載入 halo、水平與垂直 pass 皆在 shared memory，radius≤32），27 組 512²／4K／2300×12000 ROI、1／3 通道、kernel 3～45 的輸出 SHA 與舊版完全相同，但 CUDA event 時間全面約慢 2 倍（4K 單通道 k45 1.145→2.179 ms、大 ROI 3 通道 k45 10.3→21.5 ms），因此移除不採用。改在原兩段式 kernel 對不需邊界反射的內部像素使用連續視窗、無 `reflect101` 分支的快速路徑，邊界像素維持原公式；同一程序交錯 A/B 各 10 輪：4K 單通道 k45 1.23→0.68 ms（9/10）、4K 3 通道 k45 3.16→1.74 ms（10/10）、4K 3 通道 k15 1.18→0.84 ms（8/10）、2300×12000 ROI 單通道 k45 3.59→2.06 ms、3 通道 k45 10.28→5.42 ms（皆 10/10），kernel 3～11 差異在雜訊範圍，所有輸出與 CPU OpenCV 0 差異。linear／DAG plan、stateless primitive 與 401-2 fused adapter 共用同一 kernels；stateless 4K k45 含傳輸 median 6.05 ms（CPU 12.23 ms）。以 CUDA 13.3、VS 18、`sm_86` 重編 DLL，native smoke、完整 validator（含 `--resize-area-pipeline`、benchmark、crossover、morphology profile、10／100／1000 stress）、fault injection、341 tests、compileall、CUDA preflight 與 `git diff --check` 均通過。512² tile 的 k45 kernel 僅約 0.09→0.08 ms，正式 Recipe 真圖收益另列待辦；ABI 未變。

- [x] 2026-09-14：CUDA `Resize(area)` 改為與 OpenCV `INTER_AREA` 逐像素一致。先以 Python 重建 OpenCV 5.0 CV_8UC1 行為並在 393 組隨機、二值、常數及大縮放比案例與 `cv2.resize` 完全相同；CUDA 依同一規則實作：相同尺寸 copy、2×2 `(sum+2)>>2`、其他整數倍 `sum × (float)(1/area)`、非整數倍在 plan create 以 host double 建立與 `computeResizeAreaTab` 相同的來源索引／float 權重表並上傳一次（execute 仍無配置、維持一次 H2D／D2H），kernel 依 `ResizeArea_Invoker` 的 float 累加順序計算並 half-even 捨入。stateless `vf_resize_gray_u8` 的縮小路徑共用相同實作，放大路徑不變。`cuda_project.json` 新增 `fmad: false`，build script 對 DLL 與 smoke 傳入 `--fmad=false`：同矩陣以預設 `--fmad=true` 重編時 324 組輸出、7,856 個像素不一致，關閉後 483 組、7,197 萬像素 0 差異。`validate_cuda_dll.py` 的 area 比對改為 0 容差並擴充為 179 組尺寸（copy／2×2／整數倍／單軸不變／1 像素邊界／4K／13000×2300 等，含非連續來源），新增 `--resize-area-pipeline` 以正式 `PRODUCT_A_CIRCLE_401_1_AOI_01.yaml` 在 `process_scale` 1.0／0.5／0.37／0.25 對合成 PASS／NG 圖跑完整 CPU/GPU Pipeline，8 組全部相同且 GPU 無 fallback。RTX 3090 含傳輸 median：4K 0.5 倍 GPU 1.85／CPU 0.91 ms、4K 0.37 倍 1.92／3.00 ms、16384×13000 0.37 倍 34.65／51.56 ms。以 CUDA 13.3、VS 18、`sm_86` 重編 DLL，native smoke、fault injection、完整 validator（benchmark、crossover、morphology profile、10／100／1000 stress）、340 tests、compileall、CUDA preflight 與 `git diff --check` 均通過；ABI v1 與 exports 未變，真實樣本驗收仍待提供。

- [x] 2026-09-14：完成 RTX 3090 實機 CUDA 故障注入並修正兩個因此發現的恢復缺陷。新增 `gpu/validate_cuda_fault_injection.py`（不使用 fake DLL）：`CUDA_VISIBLE_DEVICES=-1` 子程序中 Detector 與 `gpu.mode=auto` Pipeline 與 CPU 完全一致且零 CUDA 呼叫、`gpu.mode=cuda` 明確失敗；1×1,048,570 影像使 kernel grid 超過 65535 列，真實 launch 回傳 1001 後整顆 Detector CPU 重跑、strict 模式直接回報，同一 runtime 下一張圖恢復 CUDA 且 context allocation 不再增加；65535 張 4096² ROI batch 取得真實 1002 OOM，失敗 batch 不留 native handle，之後連續三次小批次與 resident plan 逐像素等於 CPU、allocation count 不變。缺陷一（DLL）：失敗的 `cudaMalloc` 會殘留 CUDA thread-local last error，下一次 kernel launch 檢查再回報同一個 OOM，使 OOM 後第一個小批次也失敗、`iter_roi_batches` 降批時每層被吃掉一次；`runtime_error()` 改為回報錯誤時同時消耗 last error，新增 native smoke（舊 DLL 實測回傳 exit 9／1002，重編後通過）與 source contract。缺陷二（Python）：共用 `GpuExecutionSession` 中一張圖的 GPU crop／resident upload 失敗會讓 `last_error` 永久殘留，之後正常圖片仍顯示 tiling CPU fallback 且不再嘗試 GPU crop（RTX 實測重現）；`runtime_for()` 現在於每次 Pipeline run 開始清除可恢復錯誤，新增 fake runtime 回歸測試（未修正前 2 項失敗）。另以子程序佔住 23,208 MiB 專用 VRAM，Windows 驅動預設 sysmem fallback 使 4K plan 配置溢出而非 OOM，結果仍等價、median 7.8→7.6 ms、新 context 首次 51 ms，已列 P2 待評估；sticky context error 需重建 session 另列待辦。以 CUDA 13.3、MSVC（VS 18）、`sm_86` 重編 DLL，native smoke、fault injection（含 VRAM pressure）、`validate_cuda_dll.py`（126 PASS、4K benchmark、10／100／1000 stress allocation 維持 44）、完整 339 tests、compileall、CUDA preflight、CLI 合成 NG（預期 exit 2）與 `git diff --check` 均通過。已知限制：高度 1,048,561～1,048,576 列的影像會觸發 kernel grid 上限並安全回退 CPU。ABI v1 與 exports 未變；production 真圖驗收仍待樣本。

- [x] 2026-09-14：`llm-delegate` MCP 的 GLM 與 Qwen provider 改經 OpenRouter（`https://openrouter.ai/api/v1`），共用 `OPENROUTER_API_KEY`，預設模型分別為 `z-ai/glm-5.3-flash`、`qwen/qwen3.8-flash`（仍可用 `GLM_MODEL`／`QWEN_MODEL`／`*_BASE_URL` 或 `model` 參數覆寫）；串流解析同時接受 DeepSeek `reasoning_content` 與 OpenRouter `reasoning` 欄位，並更新 `CLAUDE.md` 設定說明。本機假 OpenRouter SSE smoke 驗證 provider 清單、Bearer 授權、`/chat/completions` 路徑、模型 ID、`: OPENROUTER PROCESSING` 註解行略過、reasoning 計數與提案可套用；真實 OpenRouter 以合成 `clamp` 範例（不含 repo 程式碼）實測：`z-ai/glm-5.3-flash` TTFT 0.78 s、總時間 7.07 s、約 167 output tokens/s（含 621 reasoning tokens），`qwen/qwen3.8-flash` TTFT 0.55 s、總時間 3.39 s、約 302 output tokens/s；兩者提案皆可套用，產生的 5 個 unittest 均通過。未修改 runtime、Detector、Recipe、GUI、CUDA source／header／ABI／DLL。
- [x] 2026-09-14：啟用 resident GPU ROI 時，grid／Template Anchor Grid tile 改為 CPU 原圖零複製 view，原生 linear／DAG plan 的 capability 與 ROI 執行不再對非連續 view 做 `ascontiguousarray`；混用 CPU Detector 時才按需複製獨立 CPU tile。RTX 3090 的 16384×13000 合成 BGR／六個 2300×12000 ROI 基準，六張 CPU tile 複製 warm median 148.5→resident view 0.1 ms，整圖 H2D 78.2 ms，單張大 ROI native Gray plan 額外 H2D 為零；4K 合成圖交錯 A/B 的 GPU Pipeline warm median 296→268 ms（約 9.5%），tile 階段 15.3→0.6 ms，PASS/NG、Tile 與 defect count 相同。1K 合成圖 CPU/GPU overlay 與九張 NG tile PNG 逐像素相同。完整 336 tests、compileall、CUDA preflight、RTX native C ABI smoke／validator（含 ROI batch 8/16/32/64 與 10/100 stress）、strict GPU CLI 預期 NG exit 2、diff check 均通過。此為合成資料與局部執行證據，完整產線 Recipe／真圖等價、端到端收益及長時間穩定性仍待驗收，production 預設未變。
- [x] 2026-09-14：新增 Claude Code 外部模型程式碼委派 MCP：`.claude/mcp/llm_delegate_server.py` 為僅用 Python 標準函式庫的 stdio MCP server，由 `.mcp.json` 註冊，提供 `list_providers`、`delegate_code`、`apply_proposal`；支援 DeepSeek（預設 `deepseek-flash`）、GLM、Qwen 的 OpenAI-compatible API，API key 只從環境變數讀取且不回傳。遠端模型無工具權限，只收到明確列出的 repo 檔案，回傳 SEARCH/REPLACE 提案、unified diff、比對問題與 TTFT／tokens/s 計時；套用前重新比對、任一區塊失配則完全不寫入，並拒絕 repo 外、`.git`、`env` 等路徑。本機假 API smoke 覆蓋 MCP 握手、串流解析、diff 預覽、原子套用、過期提案拒絕與路徑限制；實際 DeepSeek 以合成 3 行範例（不含 repo 程式碼）測得 `deepseek-flash` TTFT 0.80 s、總時間 3.76 s、約 302 output tokens/s。未修改 runtime、Detector、Recipe、GUI、CUDA source／header／ABI／DLL。
- [x] 2026-09-14：建立 Claude Code 開發環境設定；新增 `CLAUDE.md` 以 `@AGENT.md` 載入既有規範並補充 Claude Code 操作注意事項，將 `codex-skills/` 五個 skills 複製為 `.claude/skills/` 專案 skills（`aoi-release` 發布腳本路徑改指 `.claude/skills/`，不含 Codex 專用 `agents/openai.yaml`），新增 `.claude/agents/aoi-coder.md` 實作用 subagent，限制其不得 commit／push／修改 `Todo.md` 或 CUDA ABI。僅新增開發工具設定與文件，未修改 runtime、Detector、Recipe、GUI、CUDA source／header／ABI／DLL。
- [x] 2026-09-14：完成大圖 GPU ROI 直通 Detector 的現況盤點、RTX 3090 合成尺寸量測與實作／驗收規劃，新增 P4 未完成項目；確認現有原生 GPU grid 路徑已整圖一次 H2D、device ROI 額外 H2D 為零，但仍預製 CPU tile，binary mask 仍依 Detector 需求 D2H。16384×13000、六個 2300×12000 ROI 的 CPU tile 複製 warm median 128.6 ms，整圖上傳 85.0 ms，約 473.8 MiB CPU 副本可望省去；此為切圖階段上限估算，尚未修改 runtime／Detector／CUDA、未驗證產線端到端加速。
- [x] 2026-09-14：主 Pipeline／GUI／切圖模板讀圖改由 OpenCV `imdecode` 直接產生 BGR，保留 Windows 中文路徑及 EXIF 旋轉；RGB 預覽僅在需要時轉換。OpenCV 載入前提高像素／寬高上限，調參工具亦套用相同設定。新增中文路徑、透明圖、EXIF、壞檔及小上限啟動回歸測試；本機 16384×50000 合成 PNG 成功解碼為 2.46 GB BGR，單次 OpenCV 3.84 秒、舊 Pillow→NumPy→OpenCV 11.01 秒，GUI 預覽 worker 亦成功建立完整尺寸 QImage（約 4.24 秒）。完整單元測試、compileall、CUDA preflight、中文檔名 CLI PASS、GUI offscreen smoke 與 diff check 通過；真實產線圖速度及可處理尺寸仍受 RAM 與後續處理額外配置影響。
- [x] 2026-09-14：新增 CPU opt-in 平行切圖，`performance.crop_workers`／`AOI_CROP_WORKERS` 可設定 worker 數，獨立 tiler 支援 `tile.crop_workers`；一般 grid、先整圖 Pattern Match 再依 offset／rows／cols／ROI／gap 裁切的 Template Anchor Grid、contour 與 pattern_match 均使用有界且保序的 ROI 裁切工作池。GPU crop／resident image 仍維持序列，0～1 張 tile 不建立工作池。四模式像素、座標、metadata 與排序等價、實際多執行緒、GPU 路由及 Pipeline 串接測試通過；完整 327 tests、compileall、CUDA ABI preflight、CPU CLI 合成 PASS smoke 與 `git diff --check` 通過。未改 CUDA source／ABI／DLL，實際產線資料集加速幅度仍待量測。
- [x] 2026-09-14：依使用者要求將 CPU 切圖預設升為工作量自動選擇：前 16 張 ROI 至少 4 張且總裁切量達 8 MiB 時最多用 4 workers，小批／小 Tile 維持序列，明確指定 worker 數仍可覆寫。本機 16 邏輯核心、3072×4096 合成圖切圖基準顯示 64×64 grid 序列 median 44.9 ms、4 workers 137.7 ms；512×512 grid 序列 9.5 ms、4 workers 6.2 ms，Template Anchor Grid 256×256 序列 10.1 ms、4 workers 11.5 ms。63 tiles 的合成全 Pipeline median 由序列 387 ms 降至 auto 375 ms，PASS/NG、Tile 座標與順序一致。完整 329 tests、compileall、CUDA ABI preflight、CPU CLI 合成 PASS smoke 與 `git diff --check` 通過；門檻為本機合成尺寸上的保守預設，產線最佳值仍須用固定 production 資料集驗證。
- [x] 2026-08-28：完成分類式 AI Detector 的 TensorRT 純推論規劃；固定 VisionFlow 只接收已驗證模型包並執行推論，資料集、標註、PyTorch／Transformers 訓練、評估、門檻校準、ONNX 匯出與 TensorRT engine 建置／驗證全部留在外部模型工具。新增版本化 ONNX／manifest／可選 engine 交付契約、engine 相容鍵、ROI-level 分類與 PASS/NG／低信心語意、detector-neutral 共用 session、CPU／auto／strict CUDA fallback、batch／metadata／Recipe Designer、真實 AOI acceptance set、FP32／FP16 等價、RTX 3090 1000 張穩定性及 PyInstaller 驗收待辦；除責任邊界決策外，其餘實作與硬體項目皆保持未勾選。完整 308 tests、compileall、CUDA source／ABI preflight 與 `git diff --check` 通過；本次只修改 `Todo.md`，未變更 runtime、Detector、Recipe、GUI、CUDA source／header／ABI／DLL，未執行分類模型 RTX runtime 或量產精度驗收。
- [x] 2026-08-28：整理本機發行工件與獨立工具原始碼；將根目錄 9 份既有版本化 ZIP 原封不動移至 `release_artifacts/`，新增索引並讓 Utility Tools 合集後續直接輸出至該資料夾；四支 `export_*.py` 移入 `tools/` package，改以 `python -m tools...` 執行，同步更新 imports、PyInstaller specs、build、CI、README、AGENT、release skill、歷史紀錄與 packaging contract tests。四支原始碼及重新打包 EXE 的 `--smoke-test` 均 exit 0，驗證 ZIP 結構與 CPU 說明正確；完整 308 tests、compileall、CUDA preflight、skill validator 與 `git diff --check` 通過。驗證用 `v0.0.0` ZIP／合集目錄已刪除，9 份正式 ZIP 保持原檔名與內容，未修改 CUDA source／header／ABI／DLL，亦未移動或提交既有未追蹤簡報、圖表、課程及架構圖產物。
- [x] 2026-08-28：整理 repository 文件結構；新增 `docs/README.md` 索引，將 8 份 release notes 依產品與版本統一移至 `docs/release-notes/`、4 份技術／階段報告移至 `docs/reports/`、2 份會收錄進發行工件的純文字說明移至 `docs/packaging/`。同步更新 README、AGENT、歷史紀錄、`aoi-release` skill 範例、Utility／NG Tile 建置腳本與 packaging contract tests；四支 Utility Tools 重建及 packaged `--smoke-test` exit 0，驗證 ZIP 內 README 正確，完整 308 tests、compileall、CUDA preflight、skill validator 與 `git diff --check` 通過。根目錄保留執行／開發入口與固定的 `weekly_reports/` 契約，未移動或提交既有未追蹤簡報與架構圖產物。
- [x] 2026-08-27：正式發布 VisionFlow AOI `v1.5.1` CUDA-enabled Windows x64；annotated tag 精準指向乾淨 release commit `805bfcb`（含 `401-CS-SN-1` 與 CUDA Gaussian weight stream ordering 修正）且該 commit 位於 `origin/main`。以 CUDA 13.3、MSVC x64、`sm_86` 在 RTX 3090 重編 DLL，C++ ABI smoke、exports/dependencies、primitive／linear plan／DAG／resident ROI、5 份 production Recipe 共 10 個合成 PASS/NG CPU/GPU 等價、10 個傳統 Detector 共 20 個 PASS/NG CPU/GPU 等價、1000 次 stress、benchmark、crossover 與 morphology profile 均通過；完整 307 tests、compileall、CUDA preflight、GUI offscreen smoke、PyInstaller 及獨立解壓 packaged smoke 亦通過。GitHub Release 為非草稿、非 prerelease、latest，僅含 `VisionFlow-AOI-v1.5.1-windows-x64.zip` 一項資產；ZIP 為 126,315,190 bytes、SHA-256 `EBA27ABB1B46469C386231B34C132545D5FAFA09DB96EC45CAC143F68A6375E6`，內含 6 份 Recipe 與 1 份 CUDA DLL（SHA-256 `D388CE5445C9E5761FA91387E2AB824449AF81AB5239CF42C61D3BB0BA5B6373`）。GitHub 回傳的資產大小／digest 與重新下載後的獨立驗證一致，下載版 packaged `--smoke-test` exit 0。
- [x] 2026-08-26：完成 VisionFlow AOI v1.5.1 CUDA-enabled Windows x64 package-only 發行工件；內容含新增的 `401-CS-SN-1`、合計十個傳統 CV Detector 與 YOLOX。以 CUDA 13.3、MSVC x64、`sm_86` 針對乾淨最終 commit 重編 DLL，C++ ABI smoke、exports/dependencies、primitive／plan／DAG／resident ROI／ROI batch、`401-CS-SN-1` 及五份 production Recipe 共 10 個合成 PASS/NG CPU/GPU 等價、1000 次 stress、benchmark、crossover 與 morphology profile 均通過；PyInstaller 與獨立解壓 packaged smoke exit 0，ZIP 內為 6 份 Recipe 與 1 份 CUDA DLL。此次僅建立本機 ZIP，未建立 tag 或 GitHub Release；最終工件 bytes／SHA-256 以交付回報為準。
- [x] 2026-08-26：新增 `401-CS-SN-1` 自適應輪廓 Detector；固定 Gray → Adaptive Mean 一般二值化（配方 block 156 依調參工具規則使用有效 block 157、C -56）→ 四邊 MASK → LIST contours，抓到即 NG、未抓到即 PASS。已整合 cached shared plan、DetectorManager、Recipe Designer 內外參、繁中／結果標籤與 metadata，並覆蓋調參工具逐像素／輪廓等價、CPU、native plan、legacy primitive、缺 primitive、strict CUDA、GPU failure 全 CPU restart、PASS／NG、bbox／area／排序及 Recipe round trip；完整 307 tests、compileall、CUDA preflight、GUI offscreen smoke、CPU CLI 合成 NG（exit 2）與 `git diff --check` 均通過；RTX 3090 新 DLL runtime 等價由 v1.5.1 package 完成紀錄覆蓋。
- [x] 2026-08-26：準備 VisionFlow AOI v1.5.1 CUDA-enabled Windows x64 發行原始碼；GUI Pipeline 版本同步為 1.5.1，發行範圍加入 `401-CS-SN-1`，合計十個傳統 CV Detector 與 YOLOX，並預定只收錄由同一乾淨 release commit 以 CUDA 13.3、MSVC x64、`sm_86` 重編且在 RTX 3090 驗證的 `gpu/visionflow_cuda.dll`。新增 `docs/release-notes/visionflow-aoi-v1.5.1.md` 說明 CUDA 適用環境、CPU fallback、安裝方式與限制；DLL 重編、ABI／exports／dependencies、primitive／plan／DAG／resident ROI、Detector CPU/GPU 等價、benchmark／stress、packaged smoke、ZIP 雜湊仍須在此 release commit 通過後才可完成打包；GitHub Release 發布未在本次 package-only 範圍內。
- [x] 2026-08-21：正式發布 VisionFlow AOI `v1.5.0` CPU-compatible Windows x64；annotated tag 精準指向乾淨 release commit `efe5ce0` 且該 commit 位於 `origin/main`。GitHub Release 為非草稿、非 prerelease、latest，僅含 `VisionFlow-AOI-v1.5.0-windows-x64.zip` 一項資產；ZIP 為 126,244,941 bytes、SHA-256 `DEDD95329555C0A066DB9E78C7E167A0AB034872FC79A2D8DBBB3FC276CDC327`，內含完整 PyInstaller 資料夾、1 個主程式 EXE、6 份 YAML，CUDA DLL 數量為 0。原始 `dist` 與從 GitHub 重新下載／解壓的 packaged `--smoke-test` 皆 exit 0，GitHub 回傳的資產大小／digest、下載後獨立 SHA-256、Recipe／DLL 數量全部一致；本版未簽章，且 `503-CS-SN-1`／`506-CS-SN-1` 的 RTX 3090 runtime 與 production 實圖驗收仍未宣稱完成。
- [x] 2026-08-21：準備 VisionFlow AOI v1.5.0 CPU-compatible Windows x64 發行；GUI Pipeline 版本同步為 1.5.0，發行內容新增獨立 `506-CS-SN-1` 與 `503-CS-SN-1` 固定門檻 200 多邊形 Detector、中心／四邊 MASK、面積尺寸限制及工程外參／管理內參分層。工作區現有 `gpu/visionflow_cuda.dll` 並非針對本次 release commit 重編及完成 RTX 3090 驗證，因此正式 CPU-compatible ZIP 明確排除該 DLL；CPU-only 完整受支援，`gpu.mode=auto` 在缺 DLL 時安全回退 CPU。完整 296 tests、compileall、CUDA source／ABI preflight、GUI offscreen smoke、CPU CLI 合成 PASS（exit 0）及 `git diff --check` 均通過；新增 `docs/release-notes/visionflow-aoi-v1.5.0.md` 記錄安裝方式與限制，正式 ZIP 仍須以乾淨 release commit 建置並完成 packaged smoke、雜湊與 GitHub Release 資產回讀驗證。
- [x] 2026-08-21：新增獨立 Detector `503-CS-SN-1`，沿用 `506-CS-SN-1` 已驗證的固定二值化多邊形共用契約，固定執行 Gray → 一般二值化（預設門檻 `200`、最大值 `255`、預設不反相）→ 中心與四邊 MASK → `RETR_LIST` contours → `2%` perimeter 多邊形近似；預設接受面積 `100`～`100000` 且至少 3 頂點，抓到即 NG。中心半寬／半高、四邊內縮與面積為工程外參，屏蔽開關、中心定位、threshold、反相、輪廓與近似設定為管理內參；已建立獨立 registry ID、preprocess/debug identity、defect type `503_cs_sn_1_polygon_ng`、GUI／結果標籤、README 與 P3 契約。4 項 503 identity 專屬測試加上共用完整 CPU／native／primitive／缺 DLL／strict／failing backend fallback 矩陣、完整 296 tests、compileall、CUDA source／ABI preflight、GUI offscreen smoke、CPU CLI 中心 MASK 排除一個候選且保留 1 筆 NG（預期 exit 2）及 `git diff --check` 通過；未修改 CUDA source／header／ABI／DLL，未執行 503 的 RTX 3090 runtime 等價驗收。
- [x] 2026-08-21（Detector ID 更正）：依規格將剛新增的 `503-CS-AP-1` 完整改名為 `506-CS-SN-1`，同步更新 Python 檔名／類別、DetectorManager registry、GUI 繁中標籤、defect type `506_cs_sn_1_polygon_ng`、preprocess/debug identity、測試模組、README 與 P3／完成紀錄；正式 runtime、Recipe schema 與文件不再含錯誤的 503 ID，且因尚無正式 Recipe 使用該暫時名稱，不保留 legacy alias，回歸測試固定舊 ID 明確回報未註冊。演算法、threshold 200、中心／四邊 MASK、尺寸與內外參語意不變。12 項專屬測試、完整 292 tests、compileall、CUDA source／ABI preflight、GUI offscreen smoke、CPU CLI `506-CS-SN-1` 合成 NG（defect type `506_cs_sn_1_polygon_ng`，1 筆缺陷，預期 exit 2）及 `git diff --check` 通過；未修改 CUDA source／header／ABI／DLL，未執行 RTX 3090 runtime 等價驗收。
- [x] 2026-08-21（506 MASK 補正）：`506-CS-SN-1` 在既有四邊 MASK 外新增中心 MASK，支援使用影像中心或管理模式自訂中心 X／Y；中心半寬／半高為工程外參，預設皆為 `0`，中心與四邊 MASK 開關預設啟用但未調尺寸時不排除有效像素。固定二值化後依序套用中心與四邊排除，再抓取多邊形；結果 metadata 新增中心座標、半寬高、實際裁切 bbox 與完整 mask order。中心／四邊逐像素、自訂中心、超界裁切、OpenCV reference、Recipe round-trip、Engineer／Admin 顯示、native／primitive／缺 DLL／strict／failing backend fallback 測試均通過；完整 292 tests、compileall、CUDA source／ABI preflight、GUI offscreen smoke、CPU CLI 中心 MASK 排除一個候選且保留一筆 NG（預期 exit 2）及 `git diff --check` 通過。未修改 CUDA source／header／ABI／DLL，未執行新增 MASK 契約的 RTX 3090 runtime 等價驗收。
- [x] 2026-08-21：新增 Detector `506-CS-SN-1`，固定執行 Gray → 一般二值化（預設門檻 `200`、最大值 `255`、預設不反相）→ 四邊屏蔽 → `RETR_LIST` contours → `2%` perimeter 多邊形近似；預設接受含邊界面積 `100`～`100000` 且至少 3 頂點的候選，抓到任一候選即 NG、未抓到即 PASS。四邊共同／個別內縮與面積屬工程外參，threshold、反相、輪廓與多邊形設定屬管理內參；已整合 DetectorManager、Recipe Designer 繁中名稱、結果缺陷標籤、metadata、cached shared plan、CPU／native／primitive／缺 DLL／strict／failing backend fallback 與 resident ROI 測試，並將 505 共用實作的 detector identity/default fallback 改為可由子類別安全覆寫而不改變既有行為。12 項專屬測試、完整 292 tests、compileall、CUDA source／ABI preflight、GUI offscreen smoke、CPU CLI 合成 NG（1 筆缺陷，預期 exit 2）及 `git diff --check` 通過；未修改 CUDA source／header／ABI／DLL，未執行新增 Detector 的 RTX 3090 runtime 等價驗收。
- [x] 2026-08-20：在既有 GitHub Release `v1.4.0` 保留原 CPU-compatible `VisionFlow-AOI-v1.4.0-windows-x64.zip`，另新增獨立 CUDA-enabled 資產 `VisionFlow-AOI-v1.4.0-windows-x64-cuda-sm86.zip`；PyInstaller provenance 為乾淨的 commit `ce1fd9e`，完整 ZIP 126,305,049 bytes、SHA-256 `8E67C8941BC958AC08277899B9255F189D9406F68DE9A362CC382F7E6B48A1F3`，包含 1 顆已驗證的 `sm_86` CUDA DLL（SHA-256 `7338497ADF40BC80016561DFD775FD61B2F1538C412395C651AEEFB041A9C1B9`）、5 份 production Recipes 與 1 份 YOLOX example Recipe。原始 `dist`、本機重新解壓及從 GitHub 重新下載後的 packaged `--smoke-test` 均 exit 0；GitHub 回傳的資產 bytes／digest、本機下載 hash、DLL 數量／hash與 Recipe 數量全部一致，Release notes 已改為清楚區分 CPU 與 CUDA 兩個包。其他電腦、正式 production PASS／NG 影像、五 Recipe 全流程與完整長時間 pipeline 驗收依使用者安排保留至外部環境執行，未勾選相關最終驗收門檻。
- [x] 2026-08-19：在 RTX 3090（Driver 610.62、CUDA 13.3、compute capability 8.6）以已推送的 VisionFlow AOI v1.4.0 source tree `67593af`、MSVC 19.51 與 `sm_86` 重編 `visionflow_cuda.dll`，維持 ABI v1 與既有 35 個通用 exports，最終本機 DLL SHA-256 為 `7338497ADF40BC80016561DFD775FD61B2F1538C412395C651AEEFB041A9C1B9`。排除 YOLOX 後逐一比對 7 個傳統 CV Detector（`202-CS-SN-1`、`203-AS-SN-1`、三個 401、`505-AS-SN-1`、`900-CS-AP-1`）各 3 組合成影像，共 21 組 CPU／GPU 的 PASS／NG、score、defects、bbox、area、confidence、metadata 與排序皆完全一致；前六者走 `native_plan`、900 走 `native_dag_plan`，全程 GPU active、零 fallback。C++ ABI／plan／resident ROI／ROI batch smoke、CUDA/OpenCV 數值 validator、4K 5 次 benchmark 與 10／100／1000 次 persistent-plan stress 均通過；1000 次時 allocation count 維持 22、VRAM 穩定，median 0.419 ms、P95 0.598 ms，無 CUDA error 或 crash。稽核確認所有 Detector 已由現有 Gray／Gaussian／Threshold／Adaptive Mean／Morphology／Resize(area)／DAG 通用能力覆蓋，因此未修改 Detector GPU routing、CUDA kernel、header 或 ABI；production 真實影像五 Recipe 驗收、完整 1000 張 pipeline 長測及 release package 收錄仍依各自門檻另行執行。
- [x] 2026-08-19：準備 VisionFlow AOI v1.4.0 CPU-compatible Windows x64 發行；GUI Pipeline 版本同步為 1.4.0，發行內容涵蓋 Detector 202 最終語意修正、新增 `202-CS-SN-1`／`203-AS-SN-1`／`505-AS-SN-1`、正式 Detector ID 與舊 Recipe alias 相容、工程／管理內外參權限分層，以及大量 Overlay／Results 顯示穩定性改善。此發行準備當時的 `gpu/visionflow_cuda.dll` 並非針對本次 release commit 重編及 RTX 3090 驗證，因此發行包明確排除該 DLL；CPU-only 完整受支援，`gpu.mode=auto` 在缺 DLL 時安全回退 CPU。後續針對同 commit 的 DLL 重編與實機驗證證據另見上方完成紀錄，是否改為 CUDA-enabled release package 仍須執行獨立 release 打包流程。
- [x] 2026-08-19：新增 Detector `505-AS-SN-1`，固定執行 Gray → 一般反相二值化（預設門檻 `120`、最大值 `255`）→ 四邊屏蔽 → `RETR_LIST` contours → `2%` perimeter 多邊形近似；預設接受含邊界面積 `100`～`100000` 且至少 3 頂點的候選，抓到任一候選即 NG、未抓到即 PASS。四邊共同／個別內縮與面積屬工程外參，threshold、反相、輪廓與多邊形設定屬管理內參；已整合 DetectorManager、Recipe Designer 繁中名稱、結果缺陷標籤、metadata、cached shared plan、CPU／native／primitive／缺 DLL／strict／failing backend fallback 與 resident ROI 測試。15 項專屬測試、完整 280 tests、compileall、CUDA source／ABI preflight、GUI offscreen smoke、CPU CLI 合成 NG（1 筆缺陷，預期 exit 2）及 `git diff --check` 通過；未修改 CUDA source／header／ABI／DLL。
- [x] 2026-08-18：將使用者提供的傳統 CV contour 調參小工具移入獨立 `contour_preprocess_tool/` 套件並完成 OOP 邊界；Qt-independent `ContourProcessingEngine`、Unicode image store、版本化 `visionflow-traditional-cv-tuning/v1` Recipe store、preview/save workers 與 OpenGL full-resolution viewer 分責。移除預設最長邊 1400 的 OpenCV 運算縮圖，preview 與 save 現在共用同一張 `original_full` 及同一 engine，fit-to-window 只改 OpenGL view transform；Windows OpenGL 4.6 context 實測有效，offscreen／無 context 時安全回退 Qt raster。另新增不套形狀篩選的「輪廓」模式，修正舊「全部」模式不等同 Detector 203 raw contours 的落差；工具與 `203-AS-SN-1` 的 mask 逐像素、輪廓數、bbox、area 及排序直接等價。同步建立匯出／載入調參 JSON、工具說明、Detector 開發契約與 4 項回歸測試；完整 264 tests、compileall、CUDA source/ABI preflight、主 GUI offscreen smoke、工具 offscreen／Windows OpenGL smoke及 `git diff --check` 均通過，未修改既有 Detector、Recipe、CUDA source／header／ABI／DLL。
- [x] 2026-08-18：補正 202／203 Detector 管理內參遺漏：`202-CS-SN-1` 將 Auto CNR 的 Gaussian 固定／自動核心與 Sigma、MAD／robust sigma、residual 門檻、候選遮罩、形態學、connected-components、背景取樣及 CNR noise floor 納入 Recipe schema，候選面積／比例、邊界距離與背景 Ring 尺寸列為工程外參，合計 15 外參／21 內參；`203-AS-SN-1` 將 Gaussian blur、Adaptive Mean block／C／最大值／反相、形態學及 contour mode 納入管理內參，合計 7 外參／10 內參。預設值維持既有 AcceptanceChecker／OpenCV 結果，舊 Recipe 缺少新欄位時由 Detector defaults 補入；管理者 GUI 顯示、修改與 Recipe 儲存均有回歸測試，微小比例／noise floor 支援 8 位小數。`AGENT.md` 稽核契約改為必須檢查 behavior-affecting hard-coded constants/formulas，不得只掃 `self.params`。完整 260 tests、compileall、CUDA preflight、GUI smoke、202＋203 自訂內參 CLI PASS 及 `git diff --check` 通過；未修改 CUDA source／header／ABI／DLL。
- [x] 2026-08-18（後續更正）：此輪只完成「已存在於 `default_params`／`PARAM_SPEC` 的參數」權限分類與 GUI 顯示稽核，沒有逐項納入 Detector runtime 中影響行為的硬編碼常數，因此原先宣稱「全部 Detector 參數權限 audit」及 `202-CS-SN-1` 7／5、`203-AS-SN-1` 7／1 的數量並不完整；202 Auto CNR 與 203 Adaptive Mean 遺漏已由後續完成紀錄修正。此輪其餘 `parameter_group` 權威化、Engineer 隱藏值 round-trip、YOLOX 權限修正及 257 tests 驗證紀錄仍有效。
- [x] 2026-08-18：完成所有 7 個正式 Detector 的內外參權限分層；共用 `ParameterSpec` 新增 `outer`／`inner` schema，未分類參數安全預設為管理者內參，並保留舊 `engineer_visible` metadata 相容輸出。Recipe Designer 不再依參數名稱猜測權限，工程模式只顯示面積、尺寸、間距與 ROI 幾何外參，管理模式依序顯示外參及影像／光學／演算法／模型內參；YOLOX 工程模式僅保留最小框面積，模型、信心、NMS、類別與後端改為管理者限定。舊 Recipe 隱藏內參載入再儲存時完整保留。完整 257 tests、compileall、CUDA source／ABI preflight、GUI offscreen smoke、`401-AS-SN-1` CPU CLI 合成 PASS、管理模式畫面檢查及 `git diff --check` 通過；未修改 Recipe 欄位、Detector 判定語意、CUDA source／header／ABI／DLL。
- [x] 2026-08-18：同步更新 README 的 Detector 文件：新增 7 組正式 ID 與舊 Recipe ID 對照表，說明舊 ID 載入別名、衝突拒絕、已移除 `202` 不提供別名、內建 Recipe 舊檔名及 defect type key 的相容策略；並補齊 Detector 專案樹、`900-CS-AP-1` 架構名稱、完整 compileall 指令，以及 YOLOX ONNX Runtime CPU／CUDA FP32 fallback 現況與限制。完整 253 tests、compileall、CUDA source／ABI preflight 及 `git diff --check` 通過；本次僅修改文件，未變更 runtime、Recipe、GUI 或 CUDA 實作。
- [x] 2026-08-18：依命名規格將 Detector `202-1`／`203-AS-AP-1`／`401`／`401-1`／`401-2`／`900` 正式 ID 更新為 `202-CS-SN-1`／`203-AS-SN-1`／`401-AS-SN-1`／`401-CS-AP-1`／`401-CS-AP-2`／`900-CS-AP-1`，YOLOX 維持 `yolox`；Detector `202` 已從 registry、GUI 與正式 Recipe 移除，僅保留為 202-CS-SN-1 的內部屏蔽共用基底。內建 Recipe、Designer 預設、GUI 繁中標籤、900 debug renderer、401 profiler 與文件已同步；RecipeManager 載入舊 Recipe 時會將六組舊 ID、decision 清單及舊預設顯示名稱正規化成新值，同時存在新舊 ID 時拒絕覆蓋，舊 `202` 則明確回報未註冊。完整 253 tests、compileall、CUDA source/ABI preflight、GUI offscreen smoke、`401-AS-SN-1` CPU CLI 合成 NG（1 筆缺陷，預期 exit 2）及 `git diff --check` 通過；未修改 CUDA source／header／ABI／DLL。

- [x] 2026-08-18：新增 Detector 203-AS-AP-1，固定執行 Gray → Gaussian 3 → Adaptive Mean 反相（block 21、C 1）→ 3×3 Morphology Open 一次，再套用共同／左／右／上／下四邊屏蔽與 LIST contours；整合 DetectorManager、Recipe Designer 繁中標籤、結果標籤、metadata、cached shared plan、CPU／native／primitive／missing／strict／failing backend fallback 與 resident ROI 測試。12 項專屬測試、完整 251 tests、compileall、CUDA source/ABI preflight、GUI offscreen smoke、CPU CLI 合成 NG（1 筆缺陷，預期 exit 2）及 `git diff --check` 通過；未修改 CUDA source／header／ABI／DLL。

- [x] 2026-08-17：新增 `docs/reports/DETECTOR_202_1_AUTO_CNR_EVALUATION.md` 技術評估，逐步說明 Detector 202-1 的 Gaussian 背景、residual、MAD robust sigma、自動門檻、Morphology Open、屏蔽、connected components、局部 ring 與 CNR 公式，並和 Detector 202 固定門檻 172 比較。以實際兩支 detector 對 240×160 合成影像做逐 1% 光衰與 Monte Carlo shot/read-noise 壓測；理想模型上限為固定 21%／Auto CNR 77%，較接近相機的 shot＋read 模型保守支持固定約 10%／Auto CNR 40%。報告明確將 Auto CNR 量產暫定設計值降為 30%，並記錄實機驗證方法、限制與不可直接視為保證規格的原因。公式／Wilson CI spot check、完整 239 tests、compileall、CUDA source/ABI preflight 與 `git diff --check` 通過；本次只新增文件與連結，未修改 detector runtime、Recipe、GUI、CUDA source／header／ABI／DLL，也未打包。

- [x] 2026-08-13：新增 Detector 202-1，依 `Wwjyun/AcceptanceChecker` commit `117fce4` 的自動 CNR 語意實作尺度相關 Gaussian 背景、residual、MAD robust sigma、`max(8, 3×sigma)` 異常 mask、3×3 Open、8-connectivity components、自動面積上下限與局部背景 ring CNR；沿用 Detector 202 的中心／四邊屏蔽，被排除像素不產生候選且不納入背景 ring，候選依 CNR 降冪輸出並抓到即 NG。關閉屏蔽時逐像素 mask 與逐候選 bbox／面積／CNR 對照一致；10 項專屬 tests、完整 239 tests、compileall、CUDA source/ABI preflight、GUI offscreen smoke、CPU CLI 合成 NG（1 筆缺陷，預期 exit 2）及 `git diff --check` 通過。未修改 CUDA source／header／ABI／DLL，未打包。

- [x] 2026-08-13：依新規格簡化 Detector 202：保留中心與四邊排除屏蔽，前處理改為 Gray → 一般二值化（預設門檻 172、最大值 255、預設不反相）→ 屏蔽 → 固定 LIST contours；輪廓固定以 2% epsilon 近似，只接受面積 5～100 且剛好 4 個頂點，不限制凹凸，缺陷類型更新為 `202_quadrilateral_ng`。舊 Adaptive Mean、Morphology、contour mode、頂點與凸性 Recipe 欄位仍可載入，但從 Designer 隱藏且完全忽略。Detector 202 專屬 15 tests、完整 229 tests、compileall、CUDA source/ABI preflight、GUI offscreen smoke、CPU CLI 合成 NG（1 筆缺陷，預期 exit 2）及 `git diff --check` 通過；未修改 CUDA source／header／ABI／DLL，依要求未打包。

- [x] 2026-08-12：彙整 2026-08-06 至 2026-08-12 的 8 筆功能／文件提交、22 個異動檔案、Detector 202 與調參工具逐像素等價、CSV summary 自動彙總、GUI 大量 Overlay／Results 延後載入、VisionFlow AOI v1.3.0／v1.3.1 發布及 228 項測試證據，新增 `WEEKLY_UPDATE_2026-08-06_to_2026-08-12.md` 流水帳報告；另明確記錄 v1.3.0／v1.3.1 tags 早於 Detector 202 最終語意修正，正式使用前應準備後續發行版。週報建立前 `main` 已與 `origin/main` 同步且工作目錄乾淨。

- [x] 2026-08-10：同步更新 `README.md` 與 `AGENT.md` 至目前實作：修正 Detector 202 為 Gray → Morphology Open → Adaptive Mean → 排除屏蔽 → LIST contours，記錄大量 Overlay 的批次重繪、Results 延後建立及每批 24 張 NG 縮圖行為，補齊 CSV summary／report artifacts／Detector 202 與四支獨立工具的文件結構；維護規範新增外部調參工具逐像素等價、GUI 大量繪製、獨立工具打包 smoke 與完整 compileall 範圍。完整 228 tests、擴充後 compileall、CUDA source/ABI preflight 與 `git diff --check` 通過；本次未變更 runtime、Recipe、CUDA source／header／ABI／DLL，也未勾選硬體或 production 驗收項目。

- [x] 2026-08-07：依排除屏蔽調參小工具的實際 recipe 順序修正 Detector 202，前處理改為 Gray → Morphology Open → Adaptive Mean → 中心／四邊屏蔽 → LIST contours；同一 400×1400 合成小圖直接呼叫小工具函式逐像素對照為 0 差異，中心 X/Y 半徑與共同內縮 0、左15／右26／上50／下20 亦完全一致。更新 OpenCV CPU reference、cached plan、native plan、resident ROI、舊 DLL primitives 與完整 detector CPU fallback 回歸；14 項 Detector 202 測試、完整 228 tests、CLI 合成 NG（1 筆缺陷，預期 exit 2）、compileall、CUDA source preflight 與 `git diff --check` 通過。未修改 CUDA source／header／ABI／DLL，依要求未打包。
- [x] 2026-08-06：修正單張檢測完成後 GUI 偶發向下延伸／殘影變形：`ImageViewer` 大量替換 overlay 時暫停逐項更新，改用 bounding-rect viewport update 並主動 invalidate／重繪；隱藏的 Results 頁延後到實際開啟時才建立內容，NG 縮圖改為每批 24 張。350／1000 個 overlay 的檢測完成 callback 由原先連同結果頁同步建構約 0.58／2.00 秒，降至約 0.033／0.076 秒；新增大量 overlay、延後結果頁及分批縮圖回歸測試，完整 228 tests、compileall、CUDA source preflight、GUI offscreen smoke 與 `git diff --check` 通過，未修改 CUDA source／header／ABI／DLL。

- [x] 2026-08-06：修正 Detector 202 與排除屏蔽調參小工具的語意差異；中心 X=100／Y=630 現為半徑，實際屏蔽 200×1260，處理順序改為 Adaptive Mean → Open → 中心／四邊屏蔽 → LIST contours，新增共同內縮、屏蔽開關、自訂中心及最多 12 頂點。Gray／Adaptive Mean／Open 合併為單一共用 plan，保留 CPU correctness、CUDA native／primitive routing 與 full-detector fallback；三種合成影像逐像素對照小工具皆為 0 差異，完整 226 tests、compileall、CUDA source preflight、GUI offscreen smoke、Detector 202 CLI 合成 NG（1 筆缺陷，預期 exit 2）及 `git diff --check` 均通過。未修改 CUDA source／header／ABI，因此不需重編 DLL。
- [x] 2026-08-06：準備 VisionFlow AOI v1.3.1 CUDA-enabled Windows x64 發行；GUI Pipeline 版本同步為 1.3.1，發行內容加入單張／批次分析完成及監控 Stop 後自動產生 `csv/summary.csv`。發行包只收錄針對同一 release commit 以 CUDA 13.3／`sm_86` 重編並在 RTX 3090 驗證的 `gpu/visionflow_cuda.dll`，不沿用 v1.3.0 舊二進位檔。
- [x] 2026-08-06：新增逐圖缺陷 CSV 自動彙總；單張與批次分析完成後，以及監控模式按下 Stop、工作執行緒停止後，會將同一執行目錄 `csv/` 內的 CSV 合併為 UTF-8 BOM `summary.csv`。合併會排除舊 summary、採欄位聯集並原子覆寫，避免重複累加或留下半寫入檔案。完整 223 tests、compileall、CUDA source preflight、GUI offscreen smoke、實際 CLI 合成 NG（5 筆缺陷／summary 5 列，預期 exit 2）與 `git diff --check` 均通過。
- [x] 2026-08-05：彙整 2026-07-30 至 2026-08-05 的 12 筆提交、69 個異動檔案、RTX 3090 CUDA 修復與實機驗收、VisionFlow AOI v1.2.0／Utility Tools v1.0.0 發布，以及 P10 OOP 責任邊界重構，新增 `WEEKLY_UPDATE_2026-07-30_to_2026-08-05.md` 流水帳報告；週報建立前 `main` 已與 `origin/main` 同步且工作目錄乾淨。
- [x] 2026-08-03：完成 P10 OOP 收尾。移除 Detector900 的 16 個舊演算法方法與 Reporter 的 Detector900 renderer／格式化死碼；Reporter 縮為 68 行 composition root，輸出行為拆為 image encoder、overlay renderer、NG tile、CSV、Matrix CSV、Debug Image、JSON 七個單一責任協作者，writers 僅依賴明確的 paths/config/profiler/public services。新增 `DetectorManager.ai_performance_stats()` 與 detector `device_name` contract，pipeline 不再跨模組存取 `_ai_manager()`／YOLOX `_ai_execution`；架構測試固定上述邊界與死碼不得回流。完整 207 tests、compileall、CUDA preflight、CPU CLI synthetic NG smoke（9 tiles、12 defects、7 sidecars）與 `git diff --check` 均通過；未變更 CUDA header/source/DLL，故無需重編 DLL 或新增 RTX runtime 驗收結論。
- [x] 2026-08-03：完成 P10 OOP 責任邊界重構。Reporter 改由 output strategy coordinator 組合並註冊 Detector900 專屬 debug renderer；Designer 抽出 editor state、recipe mapper／validator 與主要 panels；MainWindow 以五個 workflow controllers 管理 Qt worker/thread lifecycle；Detector900 以 typed config、candidate、pair geometry、mask preprocessor 與 result assembler 保持既有 metadata；GpuRuntime 保留 façade 並抽出 ABI bindings、capability、plan descriptor/cache、handle lifecycle；AOIPipeline 抽出 recipe/runtime preparation、tile inspection 與 result assembly。同步擴充 CUDA preflight 的 Python bridge source manifest與 4 項 OOP boundary contracts。完整 204 tests、compileall、CUDA preflight、GUI offscreen smoke、CPU CLI synthetic NG smoke（9 tiles、12 defects）與 `git diff --check` 均通過；未變更 CUDA header/source/DLL，故本次未重編 DLL 或新增 RTX runtime 驗收結論。
- [x] 2026-07-30：完成 RTX 3090 CUDA blocker 修復與本機實機驗收。GitHub artifact/repository 工具已改為 repo-local manifest 與受控發布；BGR→Gray 採 OpenCV 15-bit fixed-point 係數，Gaussian 3–127 採 OpenCV 8-bit bit-exact kernel、reflect101 與兩階段 fixed-point rounding。以 CUDA 13.3、MSVC 19.51、`sm_86` 重編 DLL/LIB/EXE，C++ smoke 確認 RTX 3090 compute capability 8.6、ABI/plan/DAG/resident ROI/batch 全通過；正式 validator 三次均以零失敗完成全部 correctness、4K benchmark、crossover、morphology profile 與 10/100/1000 stress，1000 checkpoint 的 allocation count 固定 25、reserved bytes 固定 222,435,236、free VRAM 不變。五份 production recipes 以合成影像執行 CPU/GPU pipeline 結果完全一致；完整 unit tests、compileall、preflight 與 diff check 通過。
- [x] 2026-07-24：YOLOX 已接入既有 RTX 3090 workflow：先將 CPU 版 ONNX Runtime 換成 `onnxruntime-gpu==1.27.0`，執行 M3 CPU/CUDA raw/NMS 等價與 CUDA warm-up 5＋1000 次 stability，並接受可選 `yolox_acceptance_manifest` 產生 CPU/CUDA production 指標；所有 JSON 納入 artifact。最新 RTX job `30037233901` 仍因 self-hosted runner 離線而 queued，heartbeat `30041558647` 失敗，故硬體項目保持未勾選。
- [x] 2026-07-24：完成 YOLOX session 治理與 CPU 交付驗證：加入 bounded inference queue、LRU session cache、模型/backend invalidation、安全 close、pipeline AI metrics；新增 production acceptance manifest/validator（precision、recall、mAP50、誤殺率、漏檢率、per-class/confusion matrix）及 stability validator。本機 fixture warm-up 5 後執行 1000 次維持單一 session/load、輸出 deterministic、RSS 僅增加 307,200 bytes；batch/monitor 共用 session 與 Unicode 路徑測試通過。CPU-compatible PyInstaller package 已重建為 288 個檔案、確認不含 CUDA DLL，bundled YOLOX registry／ONNX model／Recipe 存在且 EXE smoke exit 0。RTX 3090、production 權重與標註資料仍待外部資產驗收。
- [x] 2026-07-24：完成 YOLOX M3 軟體接入：新增 ONNX Runtime CUDA FP32 session、provider/active EP 驗證與 CPU/CUDA 分離 cache，CUDA session 明確停用 CPU EP fallback；`gpu.mode=auto` 在 provider 缺少、初始化失敗、OOM 或推論失敗時完整以 CPU 重跑，`gpu.mode=cuda` 維持嚴格失敗。GUI 單張、batch、monitor 透過既有 execution session 共用 AI manager，YOLOX 不再誤載 `visionflow_cuda.dll`；Recipe Designer 開放 YOLOX GPU toggle 並在 provider 不可用時阻止儲存。新增 raw/NMS 等價性容差與 `gpu/validate_yolox_ort.py`，本機僅有 CPUExecutionProvider，因此 RTX 3090 實機 M3 驗收仍未勾選。
- [x] 2026-07-24：完成 YOLOX M2 Recipe Designer 整合：動態讀取已驗證 model registry 建立模型下拉，工程模式顯示信心、NMS IoU、NG 類別、最大筆數與最小框面積，管理模式才顯示 backend／precision／class-agnostic NMS；模型輸入尺寸與 class mapping 唯讀顯示，NMS tooltip 固定嚴格 `IoU > threshold` 語意。遺失模型、SHA-256 錯誤及不支援 backend 皆以繁中 inline notice 顯示並阻止 Recipe 儲存；當時 YOLOX CUDA toggle 在 M3 前維持停用，後續已由 M3 軟體接入開放，動態參數也已納入 dirty tracking。
- [x] 2026-07-24：完成 YOLOX M0/M1 CPU reference 垂直切片：新增 checksum model registry、固定輸出 tiny ONNX fixture、ONNX Runtime CPU session cache、manifest-driven letterbox/NCHW 前處理、raw YOLOX decode、class-aware NMS、座標還原、`DetectorYolox`、Recipe/DetectorManager/繁中標籤整合、結果與 execution metadata、reference Recipe 及 9 項專屬測試；合成 CLI smoke 固定輸出 2 筆 NG。GUI 模型下拉、跨工作模式 session、CUDA、TensorRT 與 production 模型驗收仍未完成。
- [x] 2026-07-24：完成 YOLOX Detector 分階段規劃；固定 model registry、信心門檻、NMS IoU 重疊率、NG 類別、最大筆數與最小 bbox 面積等參數語意，並定義前後處理、結果 metadata、共享 session、CPU reference、GPU fallback、GUI、測試及 production acceptance 順序；尚未實作 detector 或導入模型。
- [x] 2026-07-13：建立 `cuda_practice/`、RTX 3090 `sm_86` 練習與編譯說明。
- [x] 2026-07-14：加入 recipe/detector/GUI GPU 開關、CUDA 狀態與安全 CPU fallback。
- [x] 2026-07-14：建立 CUDA DLL C ABI、ctypes bridge、build script、C++ smoke 與 Python 驗證工具。
- [x] 2026-07-14：完成 M0 第一批 profiler、Reporter 分項計時、傳輸統計、crop 警告及容差修正。
- [x] 2026-07-14：完成 CPU-only/缺 DLL fallback 等價回歸。
- [x] 2026-07-14：完成 M1 原始碼：separable Gaussian、constant weights、64-bit integral Adaptive Mean 與 structured tests。
- [x] 2026-07-14：完成 M2 第一個垂直切片：persistent context、grow-only buffers、context stats 與 401-2 fused preprocessing。
- [x] 2026-07-14：建立通用 `PreprocessPlan`、typed operators、CPU/CUDA executors，並遷移 401-2。
- [x] 2026-07-14：將 CPU、GPU、CUDA、GUI、CI、打包與 RTX 3090 驗收清單合併為唯一 `Todo.md`。
- [x] 2026-07-14：更新 `AGENT.md`，統一 Todo 紀律、模組責任、PreprocessPlan/CPU fallback 架構、驗證矩陣、安全 staging 與 commit/push 規範。
- [x] 2026-07-14：更新 `aoi-verify-push`，並新增 `aoi-detector-development`、`aoi-cuda-validate`、`aoi-release` skills；四者皆完成 metadata 與官方 validator 檢查。
- [x] 2026-07-15：整理 2026-07-09 至 2026-07-15 Git 紀錄、GPU/CUDA 進度與待驗收項目，完成本週流水帳報告。
- [x] 2026-07-15：更新並完整中文化根目錄 `README.md`，同步目前 CLI、GUI、配方、Detector、輸出、CUDA fallback、打包與驗證方式。
- [x] 2026-07-17：加入 GUI 預覽影像載入、色彩轉換、QImage/QPixmap、scene 顯示與使用者實際等待時間量測，並輸出至日誌及 viewer backend tooltip。
- [x] 2026-07-17：補齊固定 seed 合成影像、奇數/極小/4K、non-contiguous、1/3 channels、ROI 尺寸 CPU 測試，並驗證關閉 fallback 時缺少 CUDA DLL 會明確失敗；真實照片與 GPU 實機項目保留待辦。
- [x] 2026-07-17：新增 per-detector bounded LRU `PreprocessPlanCache`，依 shape、dtype 與參數 signature 重用 immutable plan；401-2 已移出逐 tile plan 建立熱路徑並加入 cache/失效測試。
- [x] 2026-07-17：加入 versioned operator/plan signature、tensor spec 推導及輸入輸出 dtype/channel/shape/order/參數驗證，CPU 與 CUDA executor 共用相同契約並以 fake runtime 覆蓋錯誤輸出。
- [x] 2026-07-17：加入 preprocess capability report，記錄 requested/selected backend、fused/primitive/CPU/fallback route、原因、plan signature 與不支援項目，並帶入 detector execution metadata。
- [x] 2026-07-17：將 401-1 遷移到 cached shared plan（Gray/Resize area/Gaussian/AdaptiveMean/Morphology），保留 process scale、ROI、contour 與 metadata 語意，area 不支援時維持 full-detector CPU fallback。
- [x] 2026-07-17：將 401 遷移到 cached shared plan，保留 BGR Gaussian/Morphology 後轉 Gray/AdaptiveMean 的既有逐像素順序，以及 ROI、contour、座標、排序與 metadata 語意。
- [x] 2026-07-17：新增 CPU DAG/multi-output plan/executor，900 改以 cached DAG 共用一次 Gray 產生 outer Threshold 與 inner AdaptiveMean masks；CUDA DAG/device gray 另列待辦。
- [x] 2026-07-17：401-2 contour white-ratio 改用局部 bbox mask，避免逐 contour 配置整張 ROI；CPU 測試確認逐像素統計、排序、ROI offset 與 metadata 均與舊 full-ROI 演算法一致。
- [x] 2026-07-17：完成 connected components CPU 評估；合成測試證明 pixel area 與孔洞/list contour 語意不等價，固定 seed 4K/350 blobs benchmark 的 findContours LIST median 3.562 ms、connectedComponentsWithStats 8.063 ms，因此 401/401-1/401-2 維持 CPU contours。
- [x] 2026-07-17：加入 CUDA build preflight 與 SHA-256 manifest，靜態核對 17 個 ABI v1 header/source/runtime/smoke exports；DLL、LIB、test EXE 改在 staging 成功編譯並通過 dumpbin exports/dependencies 後才發布，避免 stale artifacts。
- [x] 2026-07-17：修正 CUDA capability preflight routing；unsupported linear/DAG plan 不再先執行部分 GPU primitive，並讓 `fallback_to_cpu: false` 對 runtime/semantic failure 維持嚴格失敗。
- [x] 2026-07-17：完成 generic native linear plan 原始碼與 OOP routing：versioned detector-neutral structs、optional query/create/execute/destroy、compiled-plan cache、persistent buffers、Gray/Gaussian/Threshold/AdaptiveMean/Morphology 單次 H2D/D2H execution，並同步 Python bridge、fake-DLL lifecycle、C++ smoke、validator、preflight 與文件；RTX 3090 編譯/實測仍保留待辦。
- [x] 2026-07-17：persistent context 納入 non-blocking CUDA stream、plan scratch 與 morphology device ping-pong；新增 `GpuExecutionSession` 讓 batch/monitor 跨影像共用同一 runtime/context，並以 pipeline、batch、monitor 與 CUDA source contract 測試驗證生命週期；RTX 3090 runtime 驗證仍保留待辦。
- [x] 2026-07-17：新增 detector-neutral native DAG/multi-output ABI、compiled-plan cache 與 CUDA executor；900 以一次 root H2D 共用 device gray，僅下載 outer/inner masks 並同步一次，已覆蓋 descriptor、fake-DLL lifecycle、detector routing、C++ smoke、validator 與 source contract；RTX 3090 編譯/實測仍保留待辦。
- [x] 2026-07-17：新增 detector `run_batch(images/rois)` CPU 預設契約與 manager 介面；GpuRuntime 採 bounded queue 加單一序列化 execution，單張 pipeline 使用 latency depth=1，batch/monitor 使用可設定 throughput depth，production recipes 持續預設關閉負優化 GPU crop。
- [x] 2026-07-17：新增 Windows CPU/static CI 與受信任 RTX 3090 self-hosted manual/nightly workflow；PR 執行 tests、compileall、recipe/CLI/GUI smoke、CUDA contract，GPU job 使用專屬 labels 並上傳 DLL/LIB/EXE/build log、環境及含 commit 的 benchmark JSON；Nsight capture 保留實機待辦。
- [x] 2026-07-17：新增 context-owned resident image 與 linear/DAG device ROI ABI；grid pipeline 每張原圖只 upload 一次，以可組合子 ROI 對應 tile 與 detector inset，ROI plan 僅 D2D staging 並下載必要輸出；已覆蓋 generation/bounds、零額外 H2D、pipeline 單次 upload、C++ smoke、validator 與 source contract，RTX 3090 編譯/實測仍保留待辦。
- [x] 2026-07-17：新增 native ROI coordinate batch opaque API，以單一 3D gather kernel 產生連續 device buffers；Python OOP handle 支援 download/context cleanup，依 `cudaMemGetInfo`、ROI 工作集與 8/16/32/64 candidates 自動選批次，配置失敗逐級降批且無 stale handle；validator 已準備四種批次實測，RTX 3090 數據仍保留待辦。
- [x] 2026-07-17：細分 detector preprocess/findContours/geometry、Python tile loop overhead、progress callback、aggregation、純檢測與各 reporter 計時；相同 percent 的 progress callback 去重，移除四個 detector 無必要 input copy，並以測試固定 profiler schema 與 callback 行為。
- [x] 2026-07-17：新增 tile-scope CPU preprocess cache，401-1/401-2/900 共用一次 Gray；稽核並測試五種 GUI worker 均先 moveToThread 再執行、無 UI wait、monitor stop/error/progress 使用 callback/signals，以及 PyInstaller CUDA DLL 條件式收錄與 CPU-only build path。
- [x] 2026-07-17：擴充 RTX validator benchmark schema，分離 cold 與 warm-up、average/median/P95/process CPU%，並記錄 nvidia-smi utilization/VRAM/溫度/功耗/Driver、CPU/RAM/Python、recipe/影像與 commit；workflow 明確 warm-up 5 次，實際 baseline 數據待 RTX runner。
- [x] 2026-07-17：在無 nvcc/CUDA DLL/GPU 環境實跑 CLI（含 outputs）、GUI offscreen、單圖 batch 與 monitor 均成功；確認 production recipes 的 tiling/display/use_gpu 預設全關閉，並以 source/runtime tests 固定 native plan 單次傳輸與 warm-up buffer reuse。
- [x] 2026-07-17：首次手動 dispatch RTX 3090 workflow run `29574501971`；workflow active 且 request 成功，但持續 queued、updated_at 未變，確認目前 self-hosted RTX runner 尚未上線接單。
- [x] 2026-07-17：統一 GPU `auto/cpu/cuda` policy：auto 可安全 fallback、cpu 完全不要求/載入 CUDA、cuda 強制成功且禁止 fallback；recipe 驗證、pipeline、長生命週期 session、GUI preview/tiling worker 與設計器均共用同一語意，GUI/history 顯示實際 backend。
- [x] 2026-07-17：新增 `VfCudaTimingsV1` 與 `vf_context_last_timings`，persistent plan 以 CUDA events 拆分 H2D/D2D、kernel、D2H、Gaussian、Adaptive Mean、threshold 與 device total，host clock 補 context/allocation/synchronize/free；Python metrics、C++ smoke、preflight 與 source/runtime tests 已同步，數值正確性待 RTX runner 驗證。
- [x] 2026-07-17：RTX validator 新增 persistent native plan 累積壓測 checkpoints，workflow 固定 warm-up 5 後跑 10/100/1000 次並保存 allocation count、VRAM、telemetry、average/median/P95 與 CUDA metrics；fake DLL 測試確認 warm-up 後不再配置，且一次 execution error 後可安全重用同一 plan handle。
- [x] 2026-07-17：RTX validator 新增 64²、128²、256²、512²、1024² 的 401-style native plan CPU/GPU crossover matrix，包含 cold/warm-up/median/P95、含傳輸 speedup、穩定 1.0x/1.5x 門檻候選；只輸出證據、不在 RTX 驗收前改 production routing。
- [x] 2026-07-17：新增 production acceptance manifest 與 validator 入口，強制五份 production recipes 各具 PASS/NG、唯一 case id、有效檔案與標籤，逐案執行 CPU/GPU 完整 pipeline 等價並核對 expected final；example 已列出 10 個待提供的真實樣本位置。
- [x] 2026-07-17：`VfCudaTimingsV1` 新增 morphology CUDA event 分項，linear/DAG native plan 均量測完整 morphology passes；RTX validator 加入 detector-401-style close iterations 1/2/4/8 的 CPU/GPU cold/warm/median/P95、含傳輸 speedup 與 morphology/kernel 占比，separable kernel 與 routing threshold 仍待實機數據決策。
- [x] 2026-07-17：新增 persistent context reuse matrix，依序覆蓋 BGR shape grow、gray channel 切換、BGR shrink 與 plan parameter 改變，第二輪要求 allocation count 不再增加；source contract 證明 grow-only reserve 先成功配置 replacement 才釋放舊 pointer，因此單次 OOM 不會破壞既有 buffer，真實 CUDA error/OOM 注入仍待 RTX。
- [x] 2026-07-17：補齊 CUDA loader failure matrix，實跑 ABI mismatch、零 CUDA device 與 persistent context create failure；context failure 現在一致傳遞到 fused、native linear、native DAG capability reason，避免 fallback metadata 誤報成缺少 generic ABI。
- [x] 2026-07-17：RTX workflow 新增可選 `production_manifest` dispatch input，可直接執行五 recipes PASS/NG acceptance；新增 Nsight Systems smoke capture，runner 有 `nsys` 時產生 `.nsys-rep`，否則保存明確 skip status，兩者均納入 artifacts。
- [x] 2026-07-17：401-2 profiler 將 contour white-pixel mask/count 從 geometry 拆成 `white_ratio_analysis`；bbox-local 統計由 NumPy boolean temporaries 改成 OpenCV `bitwise_and`/`countNonZero`，保持 count/ratio/order/metadata 等價，512² synthetic microbenchmark median 由 0.0343 ms 降至 0.0151 ms；是否移至 GPU 留待 RTX production 占比。
- [x] 2026-07-17：完成 native linear `VF_PLAN_RESIZE_AREA` source routing：descriptor 固定 area target、query 拒絕放大/混合軸語意、compiled plan 追蹤 output shape，401-1 下採樣維持單次 H2D/D2H；同步 Python encoding、C++ smoke、RTX validator、舊 DLL fallback 與 fake handle/OOP lifecycle tests，真實 CUDA 等價待 RTX runner。
- [x] 2026-07-17：修正 `build_exe.ps1` 每次覆寫受版控 spec、導致 CUDA 條件式收錄規則遺失的缺陷；改由固定 `VisionFlow AOI.spec` 建置，新增 packaged `--smoke-test` 從 PyInstaller bundle 載入 recipe 並建立 MainWindow。CPU-compatible package 在目前無 GPU 電腦實跑 exit 0、5 recipes、無 CUDA DLL，validation ZIP 103,993,603 bytes、SHA-256 `5E4E833AEA184A7889F2911B56AB22DCFAD3F2E1A6E82D46D60C5C431A4C134F`。
- [x] 2026-07-17：擴充 packaged `--smoke-test` 為缺 DLL fallback policy 全 pipeline 矩陣；CPU-only 與 auto fallback 的 PASS/NG、tiles、defects、bbox、metadata 一致且 GPU call count=0，strict CUDA 明確回報 DLL 不存在；重建 CPU-compatible EXE（5,550,515 bytes、5 recipes、無 CUDA DLL）後實際 exit 0。validation ZIP 103,996,491 bytes、SHA-256 `7477496D9DC5FD47CA99752235D451A132C9C5BC0279F237760FD308471271AD`；Windows CI 通過，RTX workflow 因 repository 無 self-hosted runner 排隊中。
- [x] 2026-07-17：依目前 codebase 稽核並更新 `README.md` 與 `AGENT.md`，同步 Windows／RTX CI、shared preprocess plan、GPU session、CUDA preflight、打包 fallback smoke、專案模組地圖與實際驗證命令；未變更 runtime 行為或 RTX 實機驗收狀態。
- [x] 2026-07-17：修正 Windows CLI smoke 的 exit code 判斷，明確接受 PASS=0 與 NG=2，並讓未捕捉例外等其他 exit code 正確使 CI 失敗。
- [x] 2026-07-17：完成 P8 產線安全與持續驗證：strict detector schema/GUI 共用、recipe/build SHA-256/commit provenance、NG dataset sidecar、五配方與每 detector 至少五個合成 golden cases、Python 3.13 Windows lock、RTX 48h heartbeat/P95 15% gate/weekly package smoke、100-case Hypothesis preprocess fuzz，並拆分 GPU ABI 與 metrics；本機 CPU-compatible PyInstaller build 及 packaged smoke exit 0。
- [x] 2026-07-18：參考 `VisionFlow_GPU_CPU-main.zip` 完成 P9 修正版移植：recipe cache、opt-in CPU tile 平行、batch/OpenCV thread budget 與週期 GC、NG tile 平行寫檔、overlay 格式／品質／縮圖、debug preprocess images、TypedDict 結果契約及 Unicode-safe regression tests；修正參考快照在中文 Windows 路徑以 `cv2.imread` 造成的 3 個假失敗，並補 invalid output config 與 process-wide OpenCV lock。實跑 119 tests、compileall、CUDA preflight、`git diff --check`、4-worker CLI synthetic smoke（9 tiles，PASS）及 GUI offscreen smoke皆通過；RTX 3090 runtime 與 production tuning 仍保留未完成。
- [x] 2026-07-18：完成 P6 GUI 操作改善：統一全域／步驟／短事件狀態層級、加入實際 backend chip 與 inline notice、Results NG 鍵盤導覽及 bbox 聚焦、Recipe dirty／validation、QSettings 工作環境回復、Batch／Monitor model-view 篩選與 deterministic scatter sampling，並同步繁中、可及性、README、AGENT 與 6 項 GUI 工作流程測試。實跑 125 tests、compileall、CUDA preflight、CLI PASS smoke、GUI offscreen smoke及 `git diff --check` 均通過。
- [x] 2026-07-20：新增 Detector 401 Template Anchor Grid 專用 profiling harness；分離 template match/ROI generation，opt-in 累計整張圖所有 ROI 的 native CUDA events、kernel launches 與 peak persistent working set，輸出 CPU 10 次、cold GPU、warm GPU 10 次及 mean/median/P95/min/max，strict CUDA 禁止 silent fallback。已完成 CPU/fake-DLL/source 測試；目前機器無 `nvidia-smi`、`nvcc`、CUDA device，且未提供本次真實影像與 anchor-grid recipe，因此 RTX 3090 baseline、瓶頸結論與 kernel/架構優化仍保持未完成。
- [x] 2026-07-20：新增 `analyze_401_profile.py` 離線判讀器；無須上傳 production JSON 即可驗證座標/PASS-NG/fallback gate，比較 CPU、GPU cold/warm median/P95 與 2/3.3 秒門檻，依 launch/ROI、同步、Morphology、D2H、resident gather、Adaptive Mean、Gaussian、CPU contours、warm allocation 輸出證據與優化優先序，並明確避免把重疊 CUDA events 相加。規則已用有效、fallback、錯誤 schema 合成報告測試；實際瓶頸結論仍待 RTX 3090 profiler JSON。
- [x] 2026-07-20：依 Detector 401 profiling 執行順序完成 GUI 單張 `GpuExecutionSession` cache；相同 Recipe 後續執行重用 context，Recipe path/mtime/size 改變時失效，視窗關閉時釋放，resident image 仍逐次建立。GUI 顯示 user-wait，profiler/analyzer 分列 detector、pipeline-before-report、reporting、end-to-end、outer wall 與 non-detector overhead。專案虛擬環境完整 137 tests、compileall、CUDA source preflight、CLI synthetic smoke、GUI offscreen smoke 與 diff check 通過；本機仍無 `nvidia-smi`、`nvcc`、MSVC `cl`，下一步 RTX cold/warm/GUI 10 次實測保持未完成。
- [x] 2026-07-20：新增 OOP GUI 權限管理器與獨立密碼提示器；程式啟動不再恢復高權限模式，固定從 OP 開始，切換工程／管理模式分別驗證預設密碼 `1234`／`5678`，取消或密碼錯誤會維持原模式。專案虛擬環境完整 139 tests、compileall、CUDA source preflight、GUI offscreen smoke 與 diff check 通過。
- [x] 2026-07-20：彙整 2026-07-16 至 2026-07-20 的 48 筆提交與驗證證據，產出本週進度報告；區分已完成的 CPU／靜態／打包驗證與仍待 RTX 3090、真實 production 影像執行的項目，未將缺少硬體的 CUDA 工作誤列為完成。
- [x] 2026-07-21：依操作確認將 401-2 恢復為既有逐 contour 白像素比例定義；同步恢復 `contour_mode`、`min_area`、`max_area`、輪廓 bbox／area／metadata、recipe 0.1.0、CPU/CUDA 共用後處理說明與原始回歸測試。此回復會重新允許單一 tile 產生多筆 contour defects。
- [x] 2026-07-21：修正新版 generic native plan 在 401-1 預設 `morph_operation: none` 時誤報不支援；Python runtime 現在會在建立 native descriptor 前略過 `none`、零 iterations 與 1x1 kernel 等 morphology no-op，並重新編排線性節點，維持 CPU／primitive 語意且不需重編 CUDA DLL。完整 142 tests、compileall、CUDA source preflight、401-1 CLI PASS smoke、GUI offscreen smoke與 `git diff --check` 均通過；本機無 `nvidia-smi`、`nvcc`、MSVC `cl`，RTX 實機仍待使用者環境確認。
- [x] 2026-07-21：使用者於具 NVIDIA GPU 的實機確認 GPU 模式與 CPU 模式皆可正常執行；本次觀察兩者整體耗時相差無幾，尚未呈現明顯 GPU 加速。因未提供固定測試集、重複次數、median／P95 與完整硬體／環境數據，可信的 CPU／GPU benchmark 與至少 1.5 倍效能門檻仍維持未完成。
- [x] 2026-07-21：新增 `docs/reports/FEATURE_VALIDATION_VERSION_CONTROL_REPORT.md` 報告稿，依目前 codebase 與驗證證據整理專案架構、各功能、四種 Detector、GUI、輸出追溯、CPU／GPU／fallback、142 項測試、CI、打包、Git 版本控制、實機結果、限制及後續規劃；未將 GPU 可執行誤列為已達效能門檻。
- [x] 2026-07-22：新增 Recipe Designer「精度 (µm/px)」與 `output.pixel_size_um_per_px`；缺陷 CSV 依 `area_px × n²` 集中換算為 `um^2` 並新增 `area_unit`，未填及舊 recipe 維持 `px^2`，不改變 Detector 面積篩選、PASS／NG、JSON 或 GUI 結果。完整 146 tests、compileall、CUDA source preflight、GUI offscreen smoke、`git diff --check` 及精度 4 µm/px 的 CLI NG CSV smoke 均通過。
- [x] 2026-07-22：新增獨立 `tools/export_ng_tiles_by_area.py` PySide6 GUI／CLI 工具；可選 AOI 輸出根資料夾、輸入多個面積區間並依 CSV `tile_id` 對應複製 `ng_tiles` 圖片，支援同 Tile 最大／總和／最小面積、未分類資料夾、JSON sidecar、舊 CSV px² 相容及 px²／µm² 混合單位隔離，且不移動原始輸出。完整 151 tests、compileall、CUDA source preflight、主 GUI／分類 GUI offscreen smoke、現有 9 張 NG Tile 實際分類 smoke 與 `git diff --check` 均通過。
- [x] 2026-07-22：建立 NG Tile 面積分類小工具 v1.0.0 的獨立 PyInstaller one-file spec、建置腳本、繁中使用說明、版本資訊與 packaged `--smoke-test`；輸出採獨立 `NG-Tile-Area-Tool` 資料夾及 `ng-tile-area-tool-vX.Y.Z` Tag，不與主程式版本混用。CPU-only 單檔 EXE（不含 CUDA DLL）已完成 GUI 啟動及現有 9 張 NG Tile／9 份 JSON 實際分類驗證；程式未簽章，發布說明須明確標示。
- [x] 2026-07-22：準備 VisionFlow AOI v1.1.1 主程式發行，GUI Pipeline 版本同步為 1.1.1；發行內容包含 Recipe CSV 面積 µm/px 精度換算、`area_unit`、近期 GUI／CPU／fallback／CUDA routing 改善與獨立 NG Tile 面積分類工具原始碼。本機未提供已針對此 commit 驗證的 CUDA DLL，因此主程式 Windows x64 發行包明確採 CPU-compatible、缺 DLL 可安全 fallback 的配置。
- [x] 2026-07-23：依星期四至星期三週期，重新彙整 2026-07-16 至 2026-07-22 的 62 筆提交、驗證證據、限制與下週計畫；週報納入 7/21～7/22 的 401-2 契約恢復、morphology no-op routing、µm/px 面積換算、NG Tile 分類工具及 v1.1.1 發行，並保留 RTX 3090 與 production samples 未完成狀態。
- [x] 2026-07-27：新增獨立 `tools/export_pattern_grid_tiles.py` 批量切圖工具，重用 production Template Anchor Grid 邏輯，支援 recipe 或直接參數、遞迴資料夾、Unicode 路徑、逐圖錯誤隔離、PNG 小圖及座標／匹配分數 CSV manifest；187 tests、compileall、CUDA source preflight、實際 CLI 4-tile smoke 與 `git diff --check` 均通過。
- [x] 2026-07-27：替 Pattern 定位固定網格批量切圖工具新增 PySide6 GUI；不含影像預覽，提供輸入／輸出／recipe／模板選擇與全部模板網格參數，支援 recipe 回填後修改、背景執行、執行中關閉保護及工作參數保存，同時保留原 CLI；190 tests、compileall、CUDA source preflight、主 GUI／小工具 GUI offscreen smoke、CLI 4-tile smoke 與 `git diff --check` 均通過。
- [x] 2026-07-29：依星期四至星期三週期，彙整 2026-07-23 至 2026-07-29 的 9 筆提交與驗證證據，產出本週進度報告；內容涵蓋 YOLOX CPU reference、Recipe Designer、ONNX Runtime CUDA fallback、session／acceptance／stability／RTX workflow，以及 Pattern 定位固定網格批量切圖 GUI／CLI，並明確保留 production 權重、標註資料、CUDA provider 與 RTX 3090 實機驗收的未完成狀態。
- [x] 2026-07-30：YOLOX Recipe Designer 的模型選擇由下拉選單改為模型資料夾視窗；所選資料夾必須包含單一模型的 `registry.yaml` 與通過 SHA-256 驗證的權重，成功後安全切換 session registry、Recipe 維持只保存 `model_id`，GUI 偏好設定記住資料夾並在切換時清除既有單張 session cache。完整 192 tests、compileall、CUDA source preflight、GUI offscreen smoke、YOLOX CLI 固定 2 筆 NG smoke（預期 exit 2）、畫面截圖檢查與 `git diff --check` 均通過。
- [x] 2026-07-30：將 AOI 專案使用的 `aoi-verify-push`、`aoi-detector-development`、`aoi-cuda-validate`、`aoi-release` 四個 Codex skills 複製至 repository 的 `codex-skills/`，移除固定舊機使用者路徑，並加入新機安裝說明及預設不覆蓋既有內容的 PowerShell 安裝程式；四個 skills 官方 validator、隔離安裝／防覆蓋 smoke、192 tests、compileall、CUDA source preflight 與 `git diff --check` 均通過。
- [x] 2026-07-30：在 RTX 3090（Driver 610.62、CUDA 13.3、compute capability 8.6）驗證 GitHub Actions 產物；35 個 exports、dependencies、native ABI/plan/ROI batch smoke 及 1000 次 persistent-plan reuse 通過，但正式 CUDA validator 在 72 個數值 cases 中有 10 項 Gaussian、401-style、900 DAG 與 resident ROI 失敗，完整 unit tests 另因 artifact 覆蓋 repository preflight API 而有 1 項 import error。已新增明確阻擋與本機重編驗收項目，RTX production acceptance 保持未完成。
- [x] 2026-07-30：修正 GUI 全域 `QComboBox` 樣式，明確設定選單本體、停用狀態、下拉 popup 與選取項目的前景／背景色，避免 Windows 原生黑色 popup 與應用程式深色文字疊成黑底黑字；GPU Mode 與所有共用參數下拉選單同步生效，並新增 palette regression test。完整 196 tests、compileall、CUDA source preflight、GUI offscreen smoke、畫面渲染檢查與 `git diff --check` 均通過。
- [x] 2026-07-30：YOLOX Recipe Designer 的模型選擇改為直接挑選 `.onnx` 檔案，再以同資料夾 `registry.yaml` 驗證所選檔案唯一對應的 model ID 與 SHA-256；支援同 registry 多模型，不再要求整個資料夾只能有一個模型。`.pt`／`.pth` 因目前沒有 PyTorch 推論後端而明確拒絕，Recipe 仍只保存穩定 `model_id`。完整 197 tests、compileall、CUDA source preflight、GUI offscreen smoke、YOLOX CLI 固定 2 筆 NG smoke（預期 exit 2）與 `git diff --check` 均通過。
- [x] 2026-07-31：為 `tools/export_pattern_grid_tiles.py` 新增獨立 PyInstaller one-file spec 與建置腳本，輸出 `dist/Pattern-Grid-Tile-Exporter/export_pattern_grid_tiles.exe`；成品不需 Python 或 CUDA DLL，已完成無參數 GUI 啟動、CLI 合成影像 4-tile／manifest smoke、完整 197 tests、compileall、CUDA source preflight 與 `git diff --check`。
- [x] 2026-07-31：準備 VisionFlow AOI v1.2.0 CUDA-enabled Windows x64 發行；GUI Pipeline 版本同步為 1.2.0，發行包將收錄針對同一 release commit 以 CUDA 13.3／`sm_86` 重編並在 RTX 3090 驗證的 `gpu/visionflow_cuda.dll`。Git 僅追蹤 CUDA source、header 與建置腳本，DLL／LIB／EXP／native test EXE 改由 `.gitignore` 防止誤提交，正式二進位檔只放入版本化 ZIP 與 GitHub Release。
- [x] 2026-07-31：修正新 repository 的 GitHub Actions：Windows CI 強制 UTF-8，避免 Pattern Grid 中文輸出在英文 runner 被 `charmap` 誤判為影像失敗；短路徑測試統一比較 resolved path，YOLOX model directory 在寫入環境變數前正規化；Windows compileall 補齊 GUI launcher 與 standalone exporters。同步將 Windows／weekly／RTX workflows 限縮為 `contents: read`、使用 lock-file cache／乾淨 venv，RTX guard 由舊 owner 改為精確的 `wjcudalearning/VisionFlow`，heartbeat 加入 5 分鐘 timeout。四份 workflow 通過 actionlint（自訂 RTX labels 除外），完整 197 tests、compileall、CUDA source preflight、GUI offscreen smoke 與 `git diff --check` 均通過；目前 repository 尚無可用 self-hosted runner，RTX runtime 仍待註冊後執行。
- [x] 2026-07-31：手動執行 weekly packaging 後確認 windowed PyInstaller EXE 不會可靠更新 PowerShell `$LASTEXITCODE`，改以 `Start-Process -Wait -PassThru` 取得真實 smoke exit code；並依 GitHub Node 20 deprecation 警告，將 checkout／setup-python／upload-artifact／cache／github-script 更新至目前官方 Node 24 majors。
- [x] 2026-07-31：將 NG Tile 面積分類、Pattern 固定網格切圖、矩陣 CSV 彙總與散點圖匯出四個小工具統一為各自可雙擊 GUI／命令列使用的 PyInstaller one-file EXE；矩陣與散點圖補上獨立 spec／建置腳本，四支工具皆提供版本與非互動 `--smoke-test`。新增具語意版本檢查、拒絕覆寫、CPU-only／無 CUDA／未簽章說明的工具合集 ZIP 流程；四支 packaged GUI smoke exit 0，實際資料驗證完成 NG 複製 1 張、矩陣彙總 1 列、散點圖 1 張與 Pattern 切圖 4 張，完整 200 tests、compileall、CUDA source preflight、主 GUI offscreen smoke、主程式 PyInstaller build／packaged smoke exit 0 及 `git diff --check` 均通過。使用者已明確選定首版工具合集 `utility-tools-v1.0.0`，本提交作為該 Release 的版本來源。
- [x] 2026-08-06：新增 Detector 202 凸多邊形 NG 檢測；依小圖中心寬 100／高 630 屏蔽及左 15、右 26、上 50、下 20 內縮，使用 Adaptive Mean block 3／C 2、3×3 Morphology Open 6 次、LIST contours，僅接受面積 20～1000、2% epsilon、至少 3 頂點的凸多邊形。已整合 DetectorManager、Recipe Designer 繁中標籤、結果標籤、metadata、cached shared plans、resident ROI 安全路由與 CPU／primitive／native／fallback 測試；完整 218 tests、compileall、CUDA source preflight、GUI offscreen smoke、Detector 202 合成 CLI NG smoke（預期 exit 2）及 `git diff --check` 均通過。
- [x] 2026-08-06：準備 VisionFlow AOI v1.3.0 CUDA-enabled Windows x64 發行；GUI Pipeline 版本同步為 1.3.0，發行內容加入 Detector 202 與其 Recipe Designer／metadata／CPU-CUDA fallback 整合。發行包只收錄針對同一 release commit 以 CUDA 13.3／`sm_86` 重編並在 RTX 3090 驗證的 `gpu/visionflow_cuda.dll`，不沿用 v1.2.0 舊二進位檔。
- [x] 2026-09-08：新增獨立 Tile 缺陷分布 HTML 報表工具；讀取 AOI 輸出 `csv/summary.csv`，依 `tile_id` 統計缺陷筆數、受影響圖片數、Detector／缺陷類型，並輸出可離線瀏覽、搜尋、列印／另存 PDF 的熱度網格與明細 HTML。支援 GUI、CLI、`--smoke-test`、空 summary、缺少 tile_id、UTF-8 BOM 與非 AOI CSV 拒絕；新增 one-file PyInstaller spec／建置腳本並納入 Utility Tools 合集。完整 312 tests、compileall、CUDA preflight、`git diff --check`、新 EXE 建置及 packaged `--smoke-test` 均通過；未變更 AOI Pipeline、Detector 或 CUDA source／ABI。
- [x] 2026-09-08：準備 VisionFlow Utility Tools v1.1.0 增量發行；Tag 使用 `utility-tools-v1.1.0`，只重建新增的 Tile 缺陷分布 EXE，另外四支未變更工具取自 GitHub Release v1.0.0 並核對來源 ZIP 195,812,209 bytes／SHA-256 `5ca76da7de38171c829c40c7a615eade8d4e3fd85c3fe0c4df0de980945ddfcf`。五支 packaged smoke 與新工具 3 筆缺陷實際 HTML smoke 均通過；最終 CPU-only／零 CUDA DLL ZIP 為 207,093,401 bytes，SHA-256 `62e4355835d8f0b44b01e4943b27558af9a486248e20f652836385198c8f4b5f`。同步更新本機及 repository `aoi-release` skill，未來只有部分工具變更時預設重建異動工具，未變更 EXE 則從可驗證的前版 Release 沿用。
- [x] 2026-09-08：將 Tile 缺陷分布報表升級為內嵌 Plotly 7 的離線互動儀表板；九種全域篩選同步更新七張 KPI、前 15 名缺陷圖片表、Tile 明細、自動洞察與 16 個圖表／診斷面板，包含缺陷／NG 率熱圖、NG 率排行、Pareto、Row／Column 剖面、Tile × 類型矩陣、Treemap、面積／score 分布及關係圖。工具會自動讀取同一輸出目錄 `json/` 的完整 PASS／NG Tile 記錄，以 `NG 次數 ÷ 檢測次數` 計算真正的 Tile NG 率；缺少或損壞分母時明確停用該指標，不以 summary 缺陷列代替。468 筆缺陷／24 份 JSON／1,152 次 Tile 檢測的合成報表完成離線渲染與 1440×7000 視覺檢查，完整 314 tests、compileall、CUDA preflight、JS syntax／placeholder 檢查均通過；依增量規則只重建本工具 EXE，packaged smoke 與實際報表輸出 exit 0，EXE 39,247,086 bytes、SHA-256 `541ee8646b98ebd3cbb824df1d38bb68b6ba012cae5aceea7994f70a582dc011`，未重建其他四支工具。
- [x] 2026-09-11：修正監控模式逐圖耗時口徑；Pipeline `duration_sec` 延後至 Reporter 完成 overlay／NG tiles／CSV／matrix CSV／debug／JSON 寫檔後定值，JSON 自身在最後 writer 執行時保存已包含前序報告的近終值。監控器從新檔案的可用建立時間（位於前後兩次輪詢區間時）或首次觀測開始持續計時，涵蓋穩定檢查、等待前序影像、完整 Pipeline、結果壓縮與處理後影像搬移，ERROR 亦不再固定回報 0 秒；逐圖另輸出 `discovery_and_stability_wait_sec`、`queue_wait_sec`、`pipeline_and_reports_sec`、`processed_image_move_sec` 與 `end_to_end_sec`。完整 319 tests、compileall、CUDA source／ABI preflight、GUI offscreen smoke 與 `git diff --check` 通過；另以真實監控輪詢 smoke 驗證 overlay／CSV／JSON 均落盤、原圖完成搬移，該次發現與穩定等待 0.352 秒、佇列等待 0.001 秒、Pipeline＋報告 0.188 秒、搬移 0.002 秒、端到端 0.544 秒。
- [x] 2026-09-14：依 Phase2 GPU 加速簡報第 12 頁，將可分離形態學加入 Detector 401 GPU 效能待辦，明列基準量測、CUDA 原型、OpenCV／現有 CUDA 等價、真圖端到端收益及 DLL 重編驗收門檻；理論鄰居讀取量不視為實測加速。本次只更新規畫，未修改 CUDA source 或執行形態學優化。
- [x] 2026-09-14：對照 Phase2 GPU 加速簡報第 14 頁 A～G、I 工作包與既有 Todo，補列 CUDA Graphs 和向量化／讀取提示的條件式實驗及驗收，並標出 ROI batch、形態學、pinned memory／stream、Gaussian shared memory、`INTER_AREA` 與 RTX 驗收的現有待辦；工期和加速倍率保留為須重新量測的估算。本次僅更新 Todo，未修改執行程式或 CUDA DLL。
- [x] 2026-09-14：在 RTX 3090（Driver 610.62／CUDA 13.3／`sm_86`）將 5×5 矩形形態學改為單次 kernel launch 的 shared-memory 水平／垂直 min/max；1／3 channel、邊界與 open／close／erode／dilate 共 80 組舊 DLL／新 DLL／OpenCV primitive 和 linear plan 比對通過，正式 CUDA validator 另加 5×5 primitive／linear／DAG 案例（41 項、max diff 0，含非連續 BGR），舊／新 DLL validator、1000 次 persistent-plan stress、native ABI smoke、336 項 unit tests、compileall、preflight、strict CUDA 合成大圖 CLI（NG 為預期 exit 2）與 diff check 通過。生成的 16384×13000 BMP／70 ROI／401-AS-SN-1 交錯 A/B 各 10 次：形態學 median 165.0→69.8 ms、Detector 413.8→325.2 ms、端到端 1429.1→1341.2 ms，完整 Tile 輸出相同、逐對 10/10 勝出；新 DLL 連跑 100 張後 warm median 1350.1 ms、P95 1417.7 ms，context reserved 689,236,528 bytes、allocation count 9 均固定，無 CUDA error／fallback。另以 32 組生成尺寸先探測 `INTER_AREA`，最大像素差 1，新增 10 組 primitive／native 驗證但仍保留正式 Recipe gate。僅證明合成圖收益；真實 production 樣本、所有 Recipe 與正式發行 DLL 仍待驗收。
