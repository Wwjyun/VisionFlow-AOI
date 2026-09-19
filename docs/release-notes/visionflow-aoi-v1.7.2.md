# VisionFlow AOI v1.7.2

這是 Windows x64 CUDA-enabled 修正版，包含完整 `VisionFlow AOI` PyInstaller 資料夾與以 CUDA 13.3、MSVC x64、`sm_86` 編譯的 `gpu/visionflow_cuda.dll`（SHA-256 `38433800568FAB7BBD8E7007A970ADE20B11B2960FEDD319829C78B8167B2345`）。

本版未修改任何 CUDA source/header（沿用 v1.6.3 起同一個 DLL），檢測流程、Detector、Recipe 語意與 PASS/NG 判定皆未變更。修正與新增的都是相機機台第二次回報的兩個阻擋點，以及現場回報用的數字短碼。

## 修正：S6 `SapAcquisition` 建立失敗（server 選到 `System`）

實機第二次回報 S6 失敗。根因是「選擇 Sapera 位置」對話框把 Sapera 回報的**所有** server 都列出來並預設選第一個，而第一個通常是主機虛擬 server `System`；它沒有 Acq resource（擷取卡），把這個值存進機台設定檔後，`SapAcquisition.Create()` 必然失敗。

- 對話框只列出**有 Acq resource** 的 server（真正的擷取卡），並優先選回已儲存的 server；被略過的 server 會以一行說明列出（例如「已略過沒有 Acq resource 的 server：System」）。
- 連線前新增檢查：選到的 server 若沒有 Acq resource，直接以 `E-0402` 拒絕並說明「Sapera 的 System 是主機虛擬 server；請選擇擷取卡」，不再等到 `SapAcquisition` 失敗。
- `E-0502` 的訊息改為包含**完整 CCF 路徑**、`SapAcquisition.Create()` 的結果，以及建構子丟出的 .NET 例外文字（例如 `COMException: CCF 與擷取卡不符`），讓現場一次就能判斷是 CCF、卡被佔用還是位置選錯。

## 修正：米輪 `LSI8181_64.dll` 不能用

- CCD 頁（管理模式）新增「瀏覽 LSI DLL」，可直接指定 `LSI8181_64.dll`（例如原廠安裝資料夾），路徑存進**機台設定檔**，不必在機台上設定環境變數；設定後立即重新載入，不必重開程式。**打包版必須用完整路徑或把 DLL 放在 EXE 同一資料夾**：凍結版不會用檔名搜尋系統路徑（凍結時找不到會直接失敗），所以「瀏覽」是必要功能而不只是方便功能；`--self-check` 的訊息也會這樣說明。
- 載入失敗時畫面直接顯示具體原因，而不是只有一句「找不到」：搜尋順序、Windows 載入器錯誤碼（126 缺少相依／193 位元數不符／5 存取被拒／127 版本不符）、檔案位元數，以及**缺少哪一個相依 DLL**（以標準函式庫讀 PE 匯入表，不需要額外工具）。
- `--self-check` 會輸出同一份完整診斷（搜尋順序、位元數、匯入清單、缺少的相依、環境變數是否指向別的檔案）。
- 相機機台無法把檔案帶出，因此這些資訊都必須在畫面上可讀；診斷結果同時寫入 `outputs\logs\camera\`。

## 新增：純數字短碼（現場回報用）

診斷的每一行除了原本的繁中短碼，另加一組純數字 **`<步驟 2 位><錯誤碼 4 位>`**：

```text
010000 020000 030000 040000 050000 060602 079999 089999
```

`060602` ＝ 第 6 步失敗、錯誤碼 `E-0602`；`010000` ＝ 第 1 步通過；`9999` ＝ SKIP、`9998` ＝ 沒有錯誤碼的 FAIL。數字優先顯示在 GUI 診斷面板、`--sapera-diagnose` 對話框與報告檔最上方，現場只需抄一列數字。對照表與抄寫表在 `docs/sapera-diagnose.md`。

## 現場操作

1. 管理模式 → 「CCD 控制」→「Sapera 位置」→「選擇 Sapera 位置…」，選 **Xtium-CL_MX4_1**（`System` 不會再出現在清單）與 CCF，按「套用到 CCD 頁」。
2. 「Sapera 診斷」→「執行相機診斷」，把最上面那一列**數字短碼**抄回。
3. 米輪若顯示不可用，先看畫面顯示的原因；必要時按「瀏覽 LSI DLL」指向原廠的 `LSI8181_64.dll`，再按「連線」。
4. 一次要看全部缺少的模組：`.\VisionFlow AOI.exe --self-check`。

## 相容性與限制

- **Sapera LT 相機與米輪仍未完成實機驗收**：本版修掉的是位置選擇與 DLL 診斷的阻擋點，並以假 assembly／假 interop 覆蓋到 S1–S8；真實 Xtium-CL MX4、Linea Mono 16K、CCF（Mono8、寬 16384）與實體米輪卡仍需要在相機機台確認。
- CUDA DLL 與 v1.7.0／v1.7.1 相同（同一個 SHA-256），其他 GPU／無 GPU 電腦驗收仍待目標環境。
- `xx_ccd` `settings.ini` 匯入仍只有解析模組，GUI 套用動作未接線。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
