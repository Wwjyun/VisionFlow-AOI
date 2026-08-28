# VisionFlow AOI 功能、驗證與版本控制報告

> 報告日期：2026-07-21
>
> 專案版本基準：Git `main` 分支，commit `05c89d6`
>
> 專案定位：以 Python、OpenCV、PySide6 與可選 CUDA DLL 建置的配方驅動 AOI（自動光學檢測）系統

## 一、專案摘要

VisionFlow AOI 將影像讀取、切圖、缺陷檢測、PASS／NG 判定、結果顯示與報表輸出整合成同一套檢測框架。不同產品不需要各自維護一份程式，而是透過 YAML Recipe 設定切圖方式、Detector、判定規則、輸出項目與 CPU／GPU 模式。

系統同時提供 CLI 與 Windows GUI，支援單張、批次與監控資料夾三種執行模式。CPU 是結果正確性的基準；具 NVIDIA GPU 的環境可使用 CUDA DLL。GPU 無法使用或執行失敗時，`auto` 模式可安全回退 CPU，`cuda` 嚴格模式則會明確失敗，不會把 CPU 結果誤報為 GPU 成功。

目前四種 Detector、五份 production Recipe、GUI、輸出、Windows 打包、自動化測試及 CI 架構均已建立。2026-07-21 的 GPU 實機操作確認 CPU 與 GPU 模式都能執行，但整體耗時相差無幾，尚未證明 GPU 有明顯加速，因此 production 預設仍以 CPU 與安全 fallback 為主。

## 二、系統目標與特色

- **配方驅動**：產品參數保存在 YAML，不需為每個產品改寫 Pipeline。
- **流程共用**：CLI、GUI、Batch 與 Monitor 共用同一套 `AOIPipeline`。
- **模組化 Detector**：新增檢測器時沿用統一輸入、結果與註冊介面。
- **完整追溯**：保存來源影像、Recipe hash、commit、Tile、Detector、bbox、面積、信心值與輸出路徑。
- **CPU／GPU 雙路徑**：CPU 為正確性基準，CUDA 為可選加速，不影響無 GPU 電腦的完整操作能力。
- **安全回退**：缺少 DLL、版本不符、無裝置、初始化失敗、kernel error 或 OOM 時，可依模式回退 CPU。
- **工程與產線分流**：GUI 提供 OP、Engineer、Admin 三種操作權限。
- **可驗證與可發布**：具備 unit tests、smoke tests、CUDA preflight、GitHub Actions、PyInstaller 與版本標籤流程。

## 三、系統架構與資料流程

```text
影像 + YAML Recipe
        │
        ▼
 RecipeManager ── 配方 schema／參數驗證／快取
        │
        ▼
  ImageLoader ── 支援中文 Windows 路徑
        │
        ▼
     Tiler ── Grid／模板定位網格／Contour／Pattern Match
        │
        ▼
 DetectorManager ── 401／401-1／401-2／900
        │
        ├── CPU Preprocess Executor（OpenCV 正確性基準）
        └── CUDA Preprocess Executor（可選，失敗時完整重跑 CPU）
        │
        ▼
 Result Mapper ── Tile 內座標轉回原圖座標
        │
        ▼
  Aggregator ── Tile 與整張影像 PASS／NG
        │
        ▼
   Reporter ── Overlay／NG Tiles／CSV／Matrix CSV／JSON／Logs
```

主要目錄責任如下：

| 目錄／檔案 | 責任 |
|---|---|
| `main.py` | CLI 與 GUI 主入口 |
| `core/` | Pipeline、Recipe、切圖、彙總、報表、效能與 GPU session |
| `detectors/` | 各 Detector 的特徵、輪廓、幾何與 NG 判定 |
| `gpu/` | CUDA C ABI、kernels、DLL 建置與 CPU／GPU 驗證工具 |
| `gui/` | PySide6 主視窗、畫面、元件及背景 workers |
| `recipes/` | 五份產品配方 |
| `tests/` | 單元、整合、回歸、fallback 與契約測試 |
| `.github/workflows/` | Windows CI、RTX 驗證、heartbeat 與每週打包 |
| `Todo.md` | 唯一開發、驗收與未完成事項清單 |

## 四、功能介紹

### 4.1 YAML Recipe 配方管理

Recipe 定義產品、機台、版本、切圖、Detector、PASS／NG 規則、輸出與 GPU 設定。同一套程式可藉由切換 Recipe 支援不同產品，降低換線時修改程式的風險。

配方載入時會嚴格檢查未知 Detector、未知參數、型別、範圍與 enum。GUI Recipe Designer 和 runtime 使用同一份參數 schema，避免畫面可填入但執行端不接受的落差。配方另有 path、mtime、size 快取；檔案變更後會自動失效，回傳值使用獨立副本，避免批次工作共享可變狀態。

### 4.2 影像載入與切圖

系統支援常用 JPG、PNG、BMP、TIF／TIFF，並處理 Windows 中文路徑。切圖後每個 Tile 都保存 ID、列欄、位置、尺寸及模式 metadata，以便將缺陷精確映射回原圖。

| 切圖方式 | 功能 | 適用情境 |
|---|---|---|
| Grid | 依寬高、重疊與步距掃描 | 全畫面或規則區域 |
| 模板定位網格 | 先找 anchor，再按列欄與間距建立 ROI | 規則排列但位置會小幅漂移的產品 |
| Contour | 依 threshold、morphology、輪廓條件找 ROI | ROI 可由形狀分離的產品 |
| Pattern Match | 多位置模板比對、局部峰值、NMS 與排序 | 重複外觀結構 |

### 4.3 Detector 功能

| Detector | 主要用途 | 核心方法 | 對應 Recipe |
|---|---|---|---|
| `401` | 負極旋轉矩形異常 | Gaussian、Morphology、Adaptive Mean、旋轉矩形 | `PRODUCT_A_NEGATIVE_401_AOI_01.yaml` |
| `401-1` | 圓形輪廓異常 | Adaptive Mean、面積、圓度、填充比 | `PRODUCT_A_CIRCLE_401_1_AOI_01.yaml` |
| `401-2` | 輪廓內白像素比例異常 | Adaptive Mean、Contour、白像素比例 | `PRODUCT_A_WHITE_RATIO_401_2_AOI_01.yaml` |
| `900` | 內外框四邊間距異常 | 外框 threshold、內框 adaptive mask、框配對與邊距 | `PRODUCT_A_FRAME_900_AOI_01.yaml` |

所有 Detector 都輸出統一格式，包括 ID、PASS／NG、score、缺陷類型、bbox、area、confidence 與 metadata。Reporter、GUI 和 Aggregator 因此不需了解個別演算法細節。

### 4.4 PASS／NG 判定與座標追蹤

系統先取得每個 Tile 的 Detector 結果，再依 Recipe decision 規則決定整張影像的最終結果。`bbox_local` 會轉為原圖的 `bbox_global`，因此報表與 Overlay 可對應實際影像位置。

主要摘要包含：

- 最終 PASS／NG／ERROR。
- Tile 總數與 NG Tile 數。
- 缺陷數量與各 Detector 結果。
- 純檢測、報表及端到端耗時。
- 實際 backend：CPU、CUDA 或 CPU FALLBACK。

### 4.5 單張、批次與監控模式

- **單張檢測**：適合工程調機、抽測與缺陷定位。
- **Batch Folder**：可掃描整個資料夾與子資料夾，適合離線驗證歷史資料。
- **Monitor Folder**：監控新影像，確認檔案寫入穩定後自動檢測，可將完成影像搬移到 processed 目錄。

Batch 預設依 CPU 核心數與影像數選擇 worker 上限，並控制 OpenCV thread budget。GPU 工作使用 bounded 單一 queue，避免多個 worker 同時爭用 GPU 或無限制累積 VRAM。

### 4.6 GUI 操作介面

GUI 主要畫面包括執行檢測、檢測結果、Recipe 設計、監控模式與批量數據圖表。

操作權限分為：

- **OP**：產線操作，啟動時預設進入此模式。
- **Engineer**：工程調機，預設密碼 `1234`。
- **Admin**：完整參數權限，預設密碼 `5678`。

目前密碼功能定位為本機防誤操作，不是正式帳號、加密儲存或資安稽核系統。

介面另支援實際 backend chip、fallback 原因、非阻塞提示、Recipe dirty／validation 狀態、NG 上下筆導覽、J／K／Enter 快捷鍵、批次篩選、固定規則散佈圖取樣，以及工作環境與視窗狀態保存。

### 4.7 輸出與追溯

| 輸出 | 內容與用途 |
|---|---|
| Overlay | 原圖標示 Tile 與缺陷 bbox，供人工判讀 |
| NG Tiles | 裁切 NG 區域，旁附 dataset sidecar JSON |
| CSV | 扁平化缺陷資料，方便 Excel、MES 或統計工具使用 |
| Matrix CSV | 依 row／column 顯示 NG 分布 |
| JSON | 保存完整 Pipeline、Tile、Detector、metadata 與 provenance |
| Logs | 輪替日誌，用於問題追查 |
| Debug Images | 選配的 preprocess 中間圖，預設關閉 |

追溯資料包含原始 Recipe SHA-256、runtime override 後的 effective Recipe SHA-256，以及執行時 Git commit／dirty 狀態。NG sidecar 另保存人工複判欄位，方便未來建立正式標註資料集。

### 4.8 CPU／GPU／CUDA 模式

| 模式 | 行為 |
|---|---|
| `cpu` | 完全不要求或載入 CUDA，固定使用 CPU |
| `auto` | 優先嘗試 CUDA；無法使用時完整回退 CPU |
| `cuda` | CUDA 必須成功，禁止隱藏 CPU fallback |

Detector 先建立 backend-neutral `PreprocessPlan`，CPU executor 以 OpenCV 定義正確語意，CUDA executor 執行可支援的通用 operators。任一 GPU 步驟失敗時，系統會從 Detector 起點完整重跑 CPU，不會把部分 GPU 中間結果和 CPU 結果混用。

GPU backend 具備 persistent context、buffer reuse、CUDA stream、DAG／multi-output plan、ROI batch 與執行時間分項。YAML、少量 contour 幾何、GUI、CSV／JSON、PNG 編碼和磁碟 I/O 原則上保留 CPU，因為這些工作搬到 GPU 不一定有收益。

### 4.9 效能觀測

系統分開記錄 image load、tiling、Detector、findContours、幾何分析、aggregation、reporting 與 end-to-end 時間。CUDA metrics 可再拆分 context、allocation、H2D、kernel、synchronize、D2H 與各 operator event。

這樣可避免直接比較不同量測範圍，例如把純 Detector 時間和 GUI 使用者等待時間當成同一項數據。Detector 401 已有專用 profiler 與離線分析器，可比較 cold、warm、pipeline、GUI 和 CUDA events。

## 五、驗證介紹

### 5.1 驗證策略

驗證分為六個層次：

1. **單元測試**：測試 Recipe、Tiler、Detector、Reporter、GUI 元件與工具函式。
2. **契約與回歸測試**：固定 PASS／NG、bbox、area、metadata、fallback 與結果排序。
3. **隨機測試**：使用 Hypothesis 產生影像與合法 PreprocessPlan，對照 OpenCV reference。
4. **Smoke Test**：驗證 CLI、GUI offscreen 及打包版可啟動並完成基本流程。
5. **CUDA 靜態／假 DLL 驗證**：檢查 ABI、source manifest、舊 DLL 相容、錯誤與 OOM fallback。
6. **GPU 實機驗收**：檢查真實 CUDA 等價、穩定性、VRAM、median／P95 與端到端效能。

### 5.2 本機標準驗證命令

```powershell
.\env\Scripts\python.exe -m unittest discover -s tests -v
.\env\Scripts\python.exe -m compileall main.py gui_launcher.py core detectors gui gpu
.\env\Scripts\python.exe gpu\preflight_cuda_build.py
git diff --check
```

截至本報告基準版本，本機完整測試為 **142 tests，全部通過**。此外，五份 production Recipe 已具備可重現的合成 PASS／NG golden regression，並檢查 final result、defect count、bbox 容差、area、confidence、metadata 與順序。

### 5.3 CPU 與 fallback 驗證

CPU-only 已驗證為完整支援模式，無 NVIDIA GPU 或 CUDA DLL 時仍可使用 GUI、CLI、Batch 與 Monitor。測試矩陣涵蓋：

- DLL 不存在。
- ABI 版本不相容。
- 無 CUDA device。
- context 建立失敗。
- operator 不支援。
- 執行錯誤與模擬 OOM。
- fallback 開啟時完整 CPU 重跑。
- fallback 關閉或 strict CUDA 時明確失敗。

### 5.4 CUDA 編譯與驗證工具

- `gpu/preflight_cuda_build.py`：靜態核對 header、source、ABI exports、build manifest 與 smoke coverage。
- `gpu/build_cuda_dll.ps1`：在 x64 Native Tools PowerShell 建置 CUDA DLL、LIB 與測試程式。
- `gpu/test_cuda_api.cu`：C++ ABI／device／primitive smoke test。
- `gpu/validate_cuda_dll.py`：CPU／GPU 像素等價、crossover、morphology、persistent context、stress 與 benchmark。
- `gpu/production_manifest.example.yaml`：五份 Recipe 的 PASS／NG 正式驗收清單範例。

### 5.5 打包驗證

Windows 執行檔使用 PyInstaller 與受版控的 `VisionFlow AOI.spec` 建置。Packaged smoke 包含：

- 內含 Recipe 並可建立 MainWindow。
- CPU-only Pipeline 可執行。
- 缺少 CUDA DLL 且 fallback 開啟時，結果與 CPU 一致且 GPU call count 為 0。
- strict CUDA 在 DLL 不存在時明確失敗。

無 GPU 電腦的 CPU-compatible package 已完成實測；有 NVIDIA GPU 的完整打包版矩陣仍待正式驗收。

### 5.6 CI 持續整合

| Workflow | 用途 |
|---|---|
| `windows-ci.yml` | 一般 Windows tests、compileall、Recipe／CLI／GUI smoke 與 CUDA 靜態檢查 |
| `rtx3090-validation.yml` | 受信任 self-hosted RTX 3090 的 CUDA runtime、等價、VRAM 與 benchmark |
| `rtx-heartbeat.yml` | 監測 RTX workflow 是否超過 48 小時未成功 |
| `weekly-packaging.yml` | 每週建立 PyInstaller package 並執行 packaged smoke |

GPU runner 只允許受信任工作，不讓外部 fork PR 直接在可接觸本機資源的 self-hosted runner 執行。

### 5.7 GPU 實機結果與判讀

2026-07-21 使用者在具 NVIDIA GPU 的機台確認：

- GPU 模式可以正常執行。
- CPU 模式可以正常執行。
- 本次觀察兩者整體耗時相差無幾，GPU 尚無明顯加速。

此結果證明兩種模式可運作，但還不能視為完整 benchmark。正式效能結論仍需固定 image／Recipe、相同輸出條件、cold 1 次、warm 多次、median、P95、Driver／Toolkit／GPU 資訊與 fallback 確認。專案的 GPU production 門檻為純檢測 median 與 P95 優於 CPU，目標至少 **1.5 倍**；未達門檻前 GPU 預設維持關閉。

### 5.8 尚待驗證事項

- 五份 production Recipe 的真實 PASS／NG 樣本 CPU／GPU 等價。
- RTX 3090 的完整 primitive／plan 像素容差矩陣。
- 固定資料集的 CPU／GPU median、P95 與端到端 speedup。
- 10、100、1000 張壓測後的 VRAM 穩定與無 crash／error。
- 真實 CUDA error／OOM 注入後的 context 重用與資源釋放。
- 有 NVIDIA GPU 的 packaged GUI 完整驗收。
- 正式標註資料集、誤判率、漏判率與量產接受門檻。

## 六、版本控制介紹

### 6.1 Git 管理方式

專案使用 Git 與 GitHub 管理程式碼，主要分支為 `main`，遠端為 `origin/main`。截至本報告基準共有 **150 筆 commit**，已建立以下版本標籤：

```text
gui-exe-2026-06-23
v0.1.0
v0.2.0
v0.3.0
v1.0.0
v1.1.0
```

版本控制的目的不只是備份程式碼，也用於：

- 記錄每次功能、修正與驗證結果。
- 透過 commit hash 追蹤檢測結果由哪一版程式產生。
- 使用 tag 固定對外發布版本。
- 讓 CI 對每次變更自動驗證。
- 比較版本差異並在必要時定位回歸來源。

### 6.2 版本與發布流程

標準流程如下：

```text
確認 Todo／需求
      ↓
修改程式、測試或文件
      ↓
執行完整驗證
      ↓
只 stage 本次相關檔案
      ↓
檢查 staged diff
      ↓
commit 至 main
      ↓
push origin/main
      ↓
CI 驗證
      ↓
正式版本建立 tag、Windows ZIP 與 checksum
```

專案不使用 `git add .` 將所有檔案一次加入，而是明確指定本次檔案，避免誤提交 logs、validation outputs、DLL build products、ZIP 或使用者尚未完成的變更。

### 6.3 文件與版本一致性

- `Todo.md`：唯一 roadmap，包含 CPU、GPU、GUI、CI、打包與 RTX 驗收。
- `README.md`：使用者操作、架構、參數與驗證說明。
- `AGENT.md`：維護規範、模組責任與必跑驗證。
- Release ZIP：對應 tag 與 checksum，不提交產生過程中的暫存 artifacts。
- JSON／NG sidecar：保存 build commit 與 Recipe hash，支援結果追溯。

只有真正完成的項目才可勾選；需要 RTX 或真實影像的工作，即使 source 與模擬測試已完成，也必須保持未完成直到硬體驗收。

### 6.4 版本控制帶來的效益

- **可追溯**：知道缺陷結果、Recipe 與執行程式版本。
- **可重現**：依 tag、lock file 與 Recipe hash 重建相同環境。
- **可稽核**：commit 說明每次改動原因與驗證證據。
- **可維護**：功能拆成小型 commit，較容易找出回歸。
- **可發布**：每個正式版本有 tag、ZIP 與 checksum。

## 七、安裝、執行與打包

### 7.1 安裝環境

```powershell
py -m venv env
.\env\Scripts\python.exe -m pip install -r requirements.lock.txt
```

專案鎖定 Python 3.13 與完整 Windows dependency lock，使本機、CI、RTX runner 與打包環境降低套件版本漂移。

### 7.2 啟動 GUI

```powershell
.\env\Scripts\python.exe main.py --gui
```

### 7.3 CLI 單張檢測

```powershell
.\env\Scripts\python.exe main.py `
  --image C:\path\to\image.png `
  --recipe recipes\PRODUCT_A_FRAME_900_AOI_01.yaml `
  --output outputs
```

PASS 結束碼為 `0`，NG 為 `2`，配方、影像或 runtime error 為其他非零值。

### 7.4 Windows 打包

```powershell
.\build_exe.ps1
.\dist\VisionFlow AOI\VisionFlow AOI.exe --smoke-test
```

有 CUDA DLL 時可條件式收錄；沒有 DLL 時仍能建立 CPU-compatible package。

## 八、目前成果、限制與風險

### 8.1 已完成成果

- 四種傳統 CV Detector 與五份 production Recipe。
- CLI、GUI、單張、Batch、Monitor 與 Recipe Designer。
- 四種切圖策略與座標回推。
- Overlay、NG Tiles、CSV、Matrix CSV、JSON、Debug Images 與 Logs。
- CPU reference、CUDA optional backend 與安全 fallback。
- 142 項自動化測試、合成 golden dataset 與 Hypothesis fuzzing。
- Windows CI、RTX workflow、heartbeat 與每週打包。
- PyInstaller CPU-compatible package 與 packaged smoke。
- Git tag、commit provenance、Recipe hash 與 NG dataset sidecar。

### 8.2 目前限制

- 正式 production 真實影像資料集尚未建立完成。
- GPU 能執行，但目前實測未顯示明顯速度優勢。
- 部分 contour、幾何、報表和磁碟 I/O 仍在 CPU，端到端速度不等於 kernel 速度。
- GPU 的完整等價、壓測、VRAM 與有 GPU 打包版驗收仍待完成。
- GUI 密碼是防誤操作機制，不是企業級身分驗證。
- 尚未整合 MES、資料庫、帳號稽核與正式模型管理。

### 8.3 主要風險與控制方式

| 風險 | 控制方式 |
|---|---|
| GPU 結果與 CPU 不一致 | CPU 作為 reference、像素容差與完整 Detector 重跑 |
| GPU 失敗造成誤判 | `auto` 完整 fallback；`cuda` 嚴格失敗 |
| Recipe 誤設 | 嚴格 schema、GUI validation、dirty tracking |
| 套件版本漂移 | Python 3.13 與 `requirements.lock.txt` |
| 修改造成回歸 | 142 tests、golden cases、CI 與 smoke tests |
| 結果無法追溯 | Recipe hash、Git commit、bbox、metadata 與 sidecar |
| 大量資料造成 UI／記憶體壓力 | model/view、bounded updates、結果壓縮與 deterministic sampling |

## 九、後續發展建議

1. 建立可追蹤的 production PASS／NG 真實影像資料集與人工標註流程。
2. 用固定影像與 Recipe 完成 CPU／GPU cold、warm、median、P95、VRAM 與 Driver 紀錄。
3. 依 profiler 證據決定優化 ROI batch、Morphology、同步或傳輸，不以「GPU 理論較快」直接更改 production。
4. 完成 1000 張壓測、CUDA error／OOM 注入及有 GPU packaged GUI 驗收。
5. 定義誤判率、漏判率、速度、穩定性與可追溯性的正式驗收門檻。
6. 視產線需求整合 MES、Lot／OP／Station ID、資料庫及權限稽核。
7. 未來導入 AI Detector 時，共用 GPU scheduler、VRAM budget、session lifecycle、fallback 與 provenance。

## 十、報告結論

VisionFlow AOI 已由單一影像處理腳本發展成具備配方、模組化 Detector、GUI、批次／監控、報表追溯、CPU／GPU routing、自動驗證、CI 與版本發布能力的 AOI 框架。

目前最重要的成果是「可執行、可追溯、可驗證、可安全回退」：CPU 模式完整可用，GPU 模式亦已在實機確認能執行，但效能與 CPU 相近，因此專案沒有把 GPU 可執行誤當成 GPU 已加速。下一階段應以固定 production dataset 與可重現 benchmark 為核心，完成等價、穩定與至少 1.5 倍效能門檻後，再決定 GPU 的量產預設策略。

---

### 簡報建議頁面

若要將本文件製作成投影片，建議拆成以下 10 頁：

1. 專案背景與目標。
2. 系統架構與資料流程。
3. Recipe 與四種切圖方式。
4. 四個 Detector 介紹。
5. GUI、單張、Batch 與 Monitor。
6. 輸出與追溯。
7. CPU／GPU／fallback 架構。
8. 驗證矩陣、142 tests 與 CI。
9. Git 版本控制、tag 與發布流程。
10. GPU 實機結果、限制與後續規劃。
