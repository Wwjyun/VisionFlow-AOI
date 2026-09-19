# VisionFlow AOI v1.7.1

這是 Windows x64 CUDA-enabled 修正版，包含完整 `VisionFlow AOI` PyInstaller 資料夾與以 CUDA 13.3、MSVC x64、`sm_86` 編譯的 `gpu/visionflow_cuda.dll`（SHA-256 `38433800568FAB7BBD8E7007A970ADE20B11B2960FEDD319829C78B8167B2345`）。

本版未修改任何 CUDA source/header，沿用 v1.6.3 已於 RTX 3090 驗證的同一個 DLL；檢測流程、Detector、Recipe 語意與 PASS/NG 判定皆未變更。修正的是第一次把相機綁定帶到實機時發現的相機無法使用問題。

## 修正：實機 Sapera LT 8.60 的 `SapBufferWithTrash` 建構子

相機機台回報：

```text
E-0301 Sapera API 缺少必要成員：SapBufferWithTrash(System.Int32, SapAcquisition, SapBuffer+MemoryType)
```

參考程式是在另一個 Sapera 版本上編譯的，8.60 沒有完全相同多載的這個建構子；而 v1.7.0 的 API 自檢把它列為必要成員，因此相機被判定不可用。

- API 自檢不再要求這個建構子。會用到的 buffer 成員（`Clear`、`Width`、`Height`、`GetParameter`、`ReadRect`）改在兩個 buffer 類別共用的基底類別 `SapBuffer` 上檢查。
- 建立 buffer 時以 .NET 反射挑選這個 Sapera 版本實際提供的建構子，順序為 `SapBufferWithTrash`（可回報落在 trash buffer 的 frame）→ `SapBuffer`（沒有 trash 回報）。兩者都接受「`(int count, SapAcquisition, MemoryType...)`，記憶體型別參數可重複」的形狀，因此多一個 trash 記憶體型別的版本也能用。
- 找不到可用建構子時才以 `E-0503` 失敗；改用 `SapBuffer` 時 CCD 頁與診斷報告會顯示一行說明，如實告知不會回報 trash frame。
- `E-0301` 現在除了缺少的成員，也會列出該成員在機台組件裡**實際可用的簽名**（最多 6 筆），現場抄回一次即可判斷差異，不必來回猜測。

已在兩個模擬 8.60 形狀的假 assembly 上驗證（獨立行程、真實 pythonnet）：多一個記憶體型別參數的版本仍使用 `SapBufferWithTrash`，完全沒有可用 trash 建構子的版本退回 `SapBuffer`；兩者 API 自檢都沒有 `E-0301`，S1–S8 全部 8 PASS。

## 新增：現場可用的自我檢查與啟動失敗回報

- 打包版新增 `--self-check`：逐一匯入每個執行期模組（含 Sapera、LSI-8181、GUI 診斷模組）、載入 .NET Framework 執行期、載入 LSI-8181 DLL、尋找並載入 `SapClassBasic.dll` 並執行 API 自檢，逐步列出 PASS／FAIL 與失敗原因；結果以對話框顯示，同時寫入機台 `outputs\logs\camera\self-check-*.txt`。這是「缺模組」問題的一次性答案。
- 啟動時發生任何例外（缺少 Python 模組、原生 DLL 找不到、Windows 錯誤 126 等）改為顯示對話框摘要並把完整堆疊寫到 `outputs\logs\camera\startup-error-*.txt`。視窗程式沒有主控台，先前這類失敗只能靠 PyInstaller 的預設訊息，或直接無訊息結束。
- 打包 `--smoke-test` 新增「每個執行期模組都能匯入」的檢查，缺少模組在**建置端**就會失敗，不必等實機。
- 修正打包遺漏：`devices.ccd_settings_import` 先前沒有被打包（尚無呼叫端，PyInstaller 因此略過），已列入 spec。`Pillow`／`plotly` 屬於獨立工具的相依，不在本程式的自我檢查清單內。

## 現場操作

1. 管理模式登入 → 左側「CCD 控制」→ 頁面下方「Sapera 診斷」面板（OP 模式看不到、工程模式看得到面板但不能執行）。
2. 按「執行相機診斷」：S1–S8 會在工作執行緒執行，完成後在頁面上逐行顯示短碼，並可「匯出診斷」把報告寫到 `outputs\logs\camera\`。
3. 需要一次看清所有缺少的模組時，用 `.\VisionFlow AOI.exe --self-check`，或直接跑 `--sapera-diagnose`（視窗版以對話框顯示，不需主控台）。

## 相容性與限制

- **Sapera LT 相機與米輪仍未完成實機驗收**：本版修掉的是「API 自檢把相機判成不可用」的阻擋點，並以假 assembly 驗證到 S1–S8 全 PASS；真實 Sapera LT 8.60、Xtium-CL MX4 與 Linea Mono 16K 上的結果仍需要在相機機台確認（含 16384 寬、Mono8 與 48 kHz 線速率）。
- CUDA DLL 與 v1.7.0 相同（同一個 SHA-256），其他 GPU／無 GPU 電腦驗收仍待目標環境。
- `xx_ccd` `settings.ini` 匯入仍只有解析模組，GUI 套用動作未接線。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
