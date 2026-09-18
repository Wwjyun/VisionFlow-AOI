# VisionFlow AOI v1.7.0

這是 Windows x64 CUDA-enabled 更新，包含完整 `VisionFlow AOI` PyInstaller 資料夾與以 CUDA 13.3、MSVC x64、`sm_86` 編譯的 `gpu/visionflow_cuda.dll`（SHA-256 `38433800568FAB7BBD8E7007A970ADE20B11B2960FEDD319829C78B8167B2345`）。

本版沒有修改任何 CUDA source/header，因此沿用 v1.6.3 已於 RTX 3090 驗證的同一個 DLL，GPU 行為、ABI v1 與 optional exports 與 v1.6.3 完全相同。檢測流程、Detector、Recipe 語意、PASS/NG 判定與輸出格式皆未變更。

## Sapera LT 線掃相機綁定

- 新增 `pythonnet` 相機綁定：`SapClassBasic.dll` 一律從相機機台自己的 Sapera LT 安裝載入，**不隨程式打包、不隨程式散佈**；找不到 Sapera LT、pythonnet 或硬體時 GUI、CLI、批次與監控照常啟動，CCD 控制頁顯示含可抄寫短碼的不可用原因。
- 只使用 Sapera 核心類別（`SapManager`、`SapLocation`、`SapAcquisition`、`SapAcqDevice`、`SapBuffer`、`SapAcqToBuf` 與通知事件），只移植參考程式已實機確認的參數寫入路徑與觸發互斥規則，不重寫 CCF、不做 feature 探測。
- 載入前先以 .NET 反射核對整份 API manifest（缺成員時拒絕存取硬體並列出缺少的成員與簽名），並顯示 managed DLL 與 Sapera runtime 版本；兩者版本不符時以「Sapera runtime 版本不符」提示兩個版本號，相機仍可使用，不會誤報成沒有擷取卡。
- 相機參數維持兩層：機台層（Sapera server／resource、CCF、米輪卡片與 CMP0–7）存在機台設定檔，產品層（曝光、增益、影像長度、內部線速率、觸發、自動存圖）存在 Recipe 選用 `camera` 區段。
- 產線硬體已記錄為 Teledyne DALSA **Xtium-CL MX4**（`OR-Y4C0-XMX00`）擷取卡 ＋ **Linea Mono 16K**（`LA-HM-16K05A-00-R`）Camera Link 線掃相機：預期 frame 為 16384 × `CROP_HEIGHT`、8-bit 單色（CCF 需為 Mono8）、線速率最高 48 kHz。server／resource／CCF 一律現場列舉選取後存入機台設定檔。

## 現場診斷 `--sapera-diagnose`

- 新增 S1–S8 分步診斷：找到 Sapera 安裝與版本、載入 `SapClassBasic.dll`、API 自檢、列舉 server／resource 與 CCF、建立並釋放各 Sapera 物件、連線並寫入參數後讀回、`Snap` 一張並檢查尺寸與灰階統計、斷線與清理。
- 每一步輸出一行可人工抄寫的短結果（例如 `S6 FAIL E-0602 Exposure 寫入失敗`），前一步失敗時後續標示略過；流程本身永不因例外中斷。機台檔案帶不出來，因此短碼就是回報方式。
- 完整報告（含逐次 Sapera 呼叫）寫到機台 `outputs/logs/camera/`；30 個錯誤碼與機台處理動作對照見 `docs/sapera-diagnose.md`。
- 打包版是視窗程式、沒有主控台，因此 `--sapera-diagnose` 的結果以對話框顯示；由原始碼執行時直接印在主控台。GUI 管理模式在「CCD 控制」頁可執行同一套流程並匯出診斷報告。

## CCD 控制畫面

- 「選擇 Sapera 位置…」對話框列舉 server／Acq／AcqDevice 與 CCF 檔；只找到一個 AcqDevice 時自動選取並提示，多個時要求明確選擇，Sapera 不可用時顯示原因與短碼且不阻擋取消。
- 新增 admin-only「Sapera 診斷」面板：managed／runtime 版本、缺少的 API 成員與完整簽名、逐行診斷短碼與總結、「執行相機診斷」與「匯出診斷」。OP 看不到整個面板，工程模式維持原有權限。

## 設定匯入與測試環境

- 新增 `xx_ccd` `settings.ini` 一次性匯入解析（機台層與產品層），未匯入的欄位各產生一條繁中說明；GUI 端的「套用匯入結果」動作尚未接線，目前僅提供模組與測試。
- 測試套件預設把相機與米輪的 vendor DLL 查詢指向不存在的路徑，因此在相機機台上執行測試也不會連真實卡片或寫入硬體。

## 驗證

- 完整 817 項 unit tests、`compileall`、CUDA source／ABI preflight、`git diff --check`、CLI 合成圖 smoke（PASS）、GUI offscreen smoke 全部通過。
- Sapera 部分另有 5 個測試模組：以可注入的假 interop 驗證連線與寫入順序、四種觸發組合、`Snap`／`Grab`／`Freeze`、擷取中 Stop 不 `Freeze`、清理失敗不外傳；以 `csc.exe` 編譯的假 `SapClassBasic` assembly 經**真實 pythonnet** 載入驗證整份 API manifest 反射、overload 選擇、`SapBuffer.ReadRect` pitch 裁切逐像素、`EndOfFrame` 跨執行緒交接與外部觸發事件；另涵蓋診斷 S1–S8 的每個失敗與略過路徑、GUI 位置對話框／版本不符／API 自檢顯示／診斷 worker／匯出／權限。
- 打包版 `--smoke-test` 新增三項檢查並通過：無相機環境 CCD 顯示不可用、`--sapera-diagnose` 停在 S1 並寫出報告且不崩潰、凍結版內可匯入 pythonnet 並完成 .NET Framework runtime 載入。

## 相容性與限制

- CUDA DLL 以 RTX 3090／compute capability 8.6 驗證；其他 GPU 必須支援包內 `sm_86` binary。沒有相容 NVIDIA GPU 時可選「僅 CPU」或「GPU 優先，失敗改用 CPU」。
- **Sapera LT 相機與米輪尚未實機驗證**：本版只在本機以假 Sapera 層與編譯的假 assembly 完成自動測試，S1–S8 在真實 Sapera LT 8.60、Xtium-CL MX4 與 Linea 16K 上的結果仍待相機機台執行（現場請先跑 `--sapera-diagnose` 並抄回短碼）。相機直連監控的大 frame 記憶體與長時間壓測同樣待實機。
- `settings.ini` 匯入僅完成解析模組，GUI 套用動作未接線。
- 相機機台需安裝 Sapera LT 8.60 與 .NET Framework 4.7.2 以上；`SapClassBasic.dll`、`LSI8181_64.dll` 與驅動由現場安裝，程式只打包 pythonnet（`pythonnet==3.1.0`）。
- 效能數據來自固定 seed 合成影像；真實產線影像與判定門檻仍需在目標產線驗收。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
